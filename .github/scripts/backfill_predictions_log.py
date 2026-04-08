#!/usr/bin/env python3
"""
backfill_predictions_log.py — One-shot historical backfill for v1.5 analytics log.

WHAT THIS DOES
--------------
Walks every existing s3://mlb-hit-predictor-${ENV}-bucket/predictions/YYYY-MM-DD/
folder, reads the legacy all_predictions.csv, reconstructs the analytics-grade
parquet log row-for-row, and (optionally) joins in the legacy daily_log.json
top-N hit results so the dashboard isn't empty on day one.

This script is IDEMPOTENT — re-running it will overwrite each partition with
the same content. Safe to run multiple times.

WHEN TO RUN
-----------
Once, immediately after deploying the v1.5 code (predictions_log.py +
patched daily_predictions.py + patched check_actuals.py) and the CFN patch.
Run from your local machine with valid AWS credentials, or as a one-shot
ECS task.

USAGE
-----
    # Default: backfill the entire 2026 season from Opening Day
    python3 scripts/backfill_predictions_log.py

    # Specific date range
    python3 scripts/backfill_predictions_log.py --start 2026-03-25 --end 2026-04-01

    # Dry run — list what would be written but don't write anything
    python3 scripts/backfill_predictions_log.py --dry-run

    # Skip the actuals enrichment step (faster, no MLB API calls)
    python3 scripts/backfill_predictions_log.py --no-actuals

NOTES
-----
- The legacy daily_log.json only stores top-1/2/5/10 *tier* boolean flags
  (i.e. "did at least one of the top 5 get a hit"), not per-player results.
  For per-player actual_hit values across the full top-100 we have to call
  the MLB Stats API just like check_actuals does. The script does that by
  default. Pass --no-actuals to skip and rely on the next morning's
  scheduled actuals check to populate yesterday's row.
- The script uses the same predictions_log module as production, so any
  schema changes there are picked up automatically.
- Cost estimate: For ~30 days of history × ~150 predictions/day × 1 API
  call each = ~4500 calls to statsapi.mlb.com. The MLB Stats API is free
  and unmetered but the script paces itself with a small sleep between
  calls to be polite.
"""

from __future__ import annotations

import os
import io
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional

import boto3
import pandas as pd

# This script lives in scripts/ and imports from the parent mlb_predictor package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlb_predictor.predictions_log import (  # noqa: E402
    append_predictions,
    append_actuals,
    LOG_PREFIX,
    S3_BUCKET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill")

OPENING_DAY_2026 = "2026-03-25"
LEGACY_PREDICTIONS_PREFIX = "predictions/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_legacy_dates(s3_client) -> list[str]:
    """
    List every YYYY-MM-DD subfolder under predictions/ that has an
    all_predictions.csv. Skips 'latest/' and any malformed keys.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    dates = set()
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=LEGACY_PREDICTIONS_PREFIX):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            # We're looking for predictions/YYYY-MM-DD/all_predictions.csv
            if not key.endswith("/all_predictions.csv"):
                continue
            parts = key.split("/")
            if len(parts) < 3:
                continue
            date_part = parts[1]
            if date_part == "latest":
                continue
            # Validate it's actually a date
            try:
                datetime.strptime(date_part, "%Y-%m-%d")
                dates.add(date_part)
            except ValueError:
                continue
    return sorted(dates)


def load_legacy_predictions(s3_client, date_str: str) -> Optional[pd.DataFrame]:
    """Read predictions/YYYY-MM-DD/all_predictions.csv into a DataFrame."""
    key = f"{LEGACY_PREDICTIONS_PREFIX}{date_str}/all_predictions.csv"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except s3_client.exceptions.NoSuchKey:
        log.warning(f"  no all_predictions.csv for {date_str}")
        return None
    except Exception as e:
        log.warning(f"  failed to read all_predictions.csv for {date_str}: {e}")
        return None


def partition_already_exists(s3_client, date_str: str) -> bool:
    """Check if predictions_log/dt=YYYY-MM-DD/predictions.parquet already exists."""
    key = f"{LOG_PREFIX}/dt={date_str}/predictions.parquet"
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def infer_model_version_for_date(date_str: str) -> str:
    """
    Tag historical partitions with the model version that was deployed on
    that date, so dashboards can correctly attribute backtest results.

    Best-effort: use the version notes from daily_predictions.py history.
    Adjust the cutoffs below if you backfill across older deploys.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    if d < datetime.strptime("2026-04-07", "%Y-%m-%d").date():
        # Pre-v1.5 deploy era — tag as v1.4 (the version that wrote these CSVs)
        return "v1.4"
    return "v1.5"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backfill v1.5 analytics prediction log")
    parser.add_argument("--start", default=OPENING_DAY_2026,
                        help="Start date YYYY-MM-DD (default: 2026 Opening Day)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be written, don't actually write")
    parser.add_argument("--no-actuals", action="store_true",
                        help="Skip the per-player actuals enrichment step")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite partitions that already exist")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Sleep between MLB API calls during actuals (default 0.05s)")
    args = parser.parse_args()

    end_date = args.end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"Backfill window: {args.start} to {end_date}")
    log.info(f"Bucket: s3://{S3_BUCKET}/")
    log.info(f"Dry run: {args.dry_run} | Actuals: {not args.no_actuals} | Force: {args.force}")

    s3 = boto3.client("s3")

    # Discover what's actually present in S3
    log.info("Discovering existing predictions/ folders...")
    available = list_legacy_dates(s3)
    log.info(f"Found {len(available)} dates with all_predictions.csv")

    # Filter to requested window
    in_window = [d for d in available if args.start <= d <= end_date]
    log.info(f"{len(in_window)} dates fall in the requested window")

    if not in_window:
        log.warning("Nothing to do.")
        return

    # Stats
    written = 0
    skipped_existing = 0
    failed = 0
    actuals_enriched = 0

    for i, date_str in enumerate(in_window, 1):
        log.info(f"[{i}/{len(in_window)}] {date_str}")

        if not args.force and partition_already_exists(s3, date_str):
            log.info(f"  partition already exists — skipping (use --force to overwrite)")
            skipped_existing += 1
            continue

        df = load_legacy_predictions(s3, date_str)
        if df is None or len(df) == 0:
            log.warning(f"  empty predictions — skipping")
            failed += 1
            continue

        # Ensure game_date column is set
        if "game_date" not in df.columns:
            df["game_date"] = date_str

        model_version = infer_model_version_for_date(date_str)
        log.info(f"  loaded {len(df)} predictions, tagging as {model_version}")

        if args.dry_run:
            log.info(f"  [dry-run] would write parquet partition for {date_str}")
            written += 1
            continue

        # Step 1: Write the predictions partition
        key = append_predictions(date_str, df, model_version=model_version)
        if not key:
            log.error(f"  append_predictions returned None — failure")
            failed += 1
            continue
        written += 1

        # Step 2: Enrich with actuals (one MLB API call per top-100 row)
        if not args.no_actuals:
            log.info(f"  enriching with actuals...")
            t0 = time.time()
            try:
                append_actuals(date_str)
                actuals_enriched += 1
                elapsed = time.time() - t0
                log.info(f"  actuals enrichment complete in {elapsed:.1f}s")
            except Exception as e:
                log.error(f"  actuals enrichment failed: {e}")

            # Be polite to statsapi.mlb.com between dates
            if args.sleep > 0:
                time.sleep(args.sleep)

    # Summary
    log.info("=" * 60)
    log.info("BACKFILL COMPLETE")
    log.info(f"  Written:           {written}")
    log.info(f"  Skipped existing:  {skipped_existing}")
    log.info(f"  Actuals enriched:  {actuals_enriched}")
    log.info(f"  Failed:            {failed}")
    log.info("=" * 60)
    log.info("")
    log.info("Next step: validate in Athena with this query:")
    log.info("")
    log.info("  SELECT dt, COUNT(*) AS rows,")
    log.info("         SUM(CAST(actual_hit AS int)) AS hits")
    log.info(f"  FROM beatthestreak_analytics_{os.environ.get('ENVIRONMENT', 'production')}.predictions_log")
    log.info("  WHERE is_top_1 = true")
    log.info("  GROUP BY dt ORDER BY dt DESC LIMIT 30;")


if __name__ == "__main__":
    main()