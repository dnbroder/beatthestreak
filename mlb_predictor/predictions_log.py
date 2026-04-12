"""
predictions_log.py — Analytics-grade prediction log for QuickSight / Athena.

WHY THIS EXISTS
---------------
The legacy daily_log.json (one record per day, with top1/top2/top5/top10 boolean
flags) is great for the README auto-update but is the wrong shape for a BI tool.
QuickSight wants one row per prediction so you can:

  - Compute hit rates for any top-N at any time window with a single SQL filter.
  - Compare model versions side-by-side (shadow deploys, A/B tests).
  - Slice by team, opponent, lineup slot, pitcher hand, optimal-zone flag, etc.
  - Track confidence calibration over time (predicted_prob vs. actual_hit).

This module writes a flat parquet log to:

    s3://mlb-hit-predictor-{env}-bucket/predictions_log/dt=YYYY-MM-DD/predictions.parquet

partitioned by `dt` so Athena scans are cheap. Schema is intentionally wide and
flat — no nested types, no maps — so Glue auto-discovery and QuickSight ingestion
are frictionless.

USAGE
-----
At prediction time (called from daily_predictions.write_predictions_to_s3):

    from mlb_predictor.predictions_log import append_predictions
    append_predictions(date_str, full_pred_df, model_version="v1.4")

After actuals check (called from check_actuals.run_actuals_check):

    from mlb_predictor.predictions_log import append_actuals
    append_actuals(date_str)

NOTES
-----
- Pure addition. Does not touch daily_log.json or summary.json.
- Idempotent: appending the same date twice will overwrite the prior parquet.
- Falls back to CSV if pyarrow is not installed (with a warning). Parquet is
  strongly preferred for Athena cost reasons but CSV still works.
- The actuals join is done by player_id when available, else by normalized name.
"""

from __future__ import annotations

import io
import os
import json
import logging
from datetime import datetime
from typing import Optional

import boto3
import pandas as pd
import numpy as np

logger = logging.getLogger("beatthestreak")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = f"mlb-hit-predictor-{ENVIRONMENT}-bucket"
LOG_PREFIX = "predictions_log"

# Columns we want in the analytics log, in order. Anything missing from the
# source DataFrame will be created as NaN/None so the schema is stable across
# dates and model versions.
LOG_COLUMNS = [
    # Identity
    "game_date",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "venue",
    "is_home",
    # Lineup / matchup context
    "lineup_position",
    "bat_side",
    "opp_pitcher",
    "opp_pitcher_hand",
    "is_doubleheader",
    "game_number",
    "is_confirmed_starter",
    # Model output
    "rank",
    "predicted_prob",
    "confidence_pct",
    "model_version",
    # Top-N flags (precomputed for QuickSight ease)
    "is_top_1",
    "is_top_2",
    "is_top_5",
    "is_top_10",
    "is_top_25",
    "is_top_50",
    # Key features (for slicing in dashboards / SHAP-style debugging)
    "batting_avg_season",
    "avg_last_30",
    "aabbh",
    "aabbh_ratio",
    "in_optimal_zone",
    "platoon_advantage",
    "pitcher_era",
    "pitcher_whip",
    "park_factor",
    # Actuals (filled in by append_actuals after games complete)
    "actual_hit",
    "actual_ab_count",
    "actual_hit_count",
    "actuals_resolved_at",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _s3_key_for(date_str: str) -> str:
    """S3 key for a given date's parquet partition."""
    return f"{LOG_PREFIX}/dt={date_str}/predictions.parquet"


def _s3_csv_fallback_key_for(date_str: str) -> str:
    return f"{LOG_PREFIX}/dt={date_str}/predictions.csv"


def _s3_client():
    return boto3.client("s3")


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _write_dataframe(df: pd.DataFrame, date_str: str) -> str:
    """
    Write a frame to the partition for `date_str`. Prefers parquet, falls back
    to CSV with a warning if pyarrow is unavailable.
    Returns the s3 key written.
    """
    s3 = _s3_client()

    if _has_pyarrow():
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        key = _s3_key_for(date_str)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/x-parquet",
        )
        logger.info(f"[predictions_log] wrote parquet: s3://{S3_BUCKET}/{key} ({len(df)} rows)")
        # v1.5.2 HEALTH: emit enrichment metrics as a single grepable line.
        # Counts rows where actual_hit has been resolved (not None) and how
        # many of those are top-1 picks. Zero-resolved-actuals days are a
        # leading indicator that the MLB Stats API lookup failed globally.
        try:
            if "actual_hit" in df.columns:
                resolved = int(df["actual_hit"].notna().sum())
            else:
                resolved = 0
            top1_resolved = 0
            if "is_top_1" in df.columns and "actual_hit" in df.columns:
                top1_resolved = int(df[df["is_top_1"] == True]["actual_hit"].notna().sum())
            logger.info(
                f"[predictions_log] [HEALTH] write_metrics date={date_str} "
                f"n_rows={len(df)} n_rows_with_actuals={resolved} "
                f"n_top_1_resolved={top1_resolved}"
            )
            if len(df) > 0 and resolved == 0:
                logger.warning(
                    f"[predictions_log] [HEALTH] WARNING: zero resolved actuals "
                    f"for {date_str} despite {len(df)} rows — investigate"
                )
        except Exception as _metrics_err:
            logger.info(f"[predictions_log] [HEALTH] metrics calc failed: {_metrics_err}")
        return key
    else:
        logger.warning(
            "[predictions_log] pyarrow not installed — falling back to CSV. "
            "Install pyarrow for cheaper Athena scans."
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        key = _s3_csv_fallback_key_for(date_str)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=buf.getvalue(),
            ContentType="text/csv",
        )
        logger.info(f"[predictions_log] wrote csv: s3://{S3_BUCKET}/{key} ({len(df)} rows)")
        return key


def _read_existing(date_str: str) -> Optional[pd.DataFrame]:
    """
    Read the existing partition for `date_str` if present, else None.
    Tries parquet first, then CSV fallback.
    """
    s3 = _s3_client()

    # Try parquet
    if _has_pyarrow():
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=_s3_key_for(date_str))
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except s3.exceptions.NoSuchKey:
            pass
        except Exception as e:
            logger.warning(f"[predictions_log] could not read existing parquet for {date_str}: {e}")

    # Try CSV fallback
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=_s3_csv_fallback_key_for(date_str))
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None


def _normalize_name(name: str) -> str:
    """Lower-case, strip punctuation/accents for fuzzy join with actuals API."""
    if not isinstance(name, str):
        return ""
    return (
        name.lower()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )


def _coerce_to_log_schema(df: pd.DataFrame, model_version: str) -> pd.DataFrame:
    """
    Project an arbitrary prediction DataFrame onto LOG_COLUMNS, filling
    missing columns with NaN/None and adding derived top-N flags.
    """
    out = pd.DataFrame()

    # Map source -> log column where the source has a different name
    source_aliases = {
        "game_date": ["game_date", "date"],
        "predicted_prob": ["predicted_prob", "probability"],
    }

    for col in LOG_COLUMNS:
        if col in df.columns:
            out[col] = df[col].values
        elif col in source_aliases:
            picked = next((c for c in source_aliases[col] if c in df.columns), None)
            out[col] = df[picked].values if picked else None
        else:
            out[col] = None

    # Derive predicted_prob from confidence_pct if not present
    if out["predicted_prob"].isna().all() and "confidence_pct" in df.columns:
        out["predicted_prob"] = (df["confidence_pct"].astype(float) / 100.0).values

    # Derive top-N flags from rank
    if "rank" in df.columns:
        rank_int = pd.to_numeric(df["rank"], errors="coerce")
        out["is_top_1"] = (rank_int <= 1).fillna(False).astype(bool).values
        out["is_top_2"] = (rank_int <= 2).fillna(False).astype(bool).values
        out["is_top_5"] = (rank_int <= 5).fillna(False).astype(bool).values
        out["is_top_10"] = (rank_int <= 10).fillna(False).astype(bool).values
        out["is_top_25"] = (rank_int <= 25).fillna(False).astype(bool).values
        out["is_top_50"] = (rank_int <= 50).fillna(False).astype(bool).values

    # Stamp model version
    out["model_version"] = model_version

    # Make sure types are friendly to parquet/Athena
    bool_cols = ["is_home", "is_doubleheader", "is_confirmed_starter",
                 "in_optimal_zone", "platoon_advantage",
                 "is_top_1", "is_top_2", "is_top_5", "is_top_10",
                 "is_top_25", "is_top_50"]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)

    # Numeric coercion (silently leave NaN where invalid)
    numeric_cols = ["lineup_position", "rank", "predicted_prob", "confidence_pct",
                    "batting_avg_season", "avg_last_30", "aabbh", "aabbh_ratio",
                    "pitcher_era", "pitcher_whip", "park_factor", "game_number",
                    "actual_ab_count", "actual_hit_count"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # actual_hit stays nullable boolean (NaN until resolved)
    return out


# ---------------------------------------------------------------------------
# Public API: append_predictions
# ---------------------------------------------------------------------------
def append_predictions(
    date_str: str,
    predictions_df: pd.DataFrame,
    model_version: str = "v1.4",
) -> Optional[str]:
    """
    Write a day's full ranked predictions to the analytics log.

    Idempotent: re-running for the same date overwrites the partition.

    Args:
        date_str: 'YYYY-MM-DD'
        predictions_df: full ranked DataFrame from score_batters() — should
                        have a 'rank' column and as many of the LOG_COLUMNS
                        feature columns as possible.
        model_version: tag for shadow / A-B comparisons in dashboards.

    Returns:
        S3 key written, or None on error (errors logged but not raised, to
        avoid taking down the prediction pipeline if logging breaks).
    """
    try:
        if predictions_df is None or len(predictions_df) == 0:
            logger.warning(f"[predictions_log] empty predictions_df for {date_str} — skipping")
            return None

        df = predictions_df.copy()

        # v1.5.4 HEALTH: schema drift guard. If upstream daily_predictions.py
        # starts producing columns that LOG_COLUMNS doesn't know about (e.g.
        # a Phase 2 pitcher feature added without updating this schema),
        # we'd silently drop them at the parquet write. Log the unexpected
        # columns so the omission is visible in CloudWatch.
        known = set(LOG_COLUMNS) | {"date", "game_date", "player_name", "team",
                                    "opponent", "venue", "opp_pitcher",
                                    "opp_pitcher_hand", "bat_side", "rank",
                                    "confidence_pct", "game_pk"}
        unexpected = [c for c in df.columns if c not in known]
        if unexpected:
            logger.warning(
                f"[predictions_log] [HEALTH] WARNING: schema_drift "
                f"unexpected_columns={unexpected} — these will be dropped from "
                f"the parquet. Update LOG_COLUMNS if these should be persisted."
            )

        # Ensure game_date is set on every row
        if "game_date" not in df.columns and "date" in df.columns:
            df["game_date"] = df["date"]
        if "game_date" not in df.columns:
            df["game_date"] = date_str

        log_df = _coerce_to_log_schema(df, model_version=model_version)

        # actuals columns explicit None at write time
        log_df["actual_hit"] = None
        log_df["actual_ab_count"] = None
        log_df["actual_hit_count"] = None
        log_df["actuals_resolved_at"] = None

        return _write_dataframe(log_df, date_str)

    except Exception as e:
        logger.error(f"[predictions_log] append_predictions failed for {date_str}: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API: append_actuals
# ---------------------------------------------------------------------------
def append_actuals(date_str: str) -> Optional[str]:
    """
    Re-read the partition for `date_str`, fetch real game results from the
    MLB Stats API for every logged prediction, and write the partition back
    with actual_hit / actual_ab_count / actual_hit_count filled in.

    This is intended to be called from check_actuals.run_actuals_check after
    the legacy daily_log.json update has succeeded. It is idempotent — if
    called twice, the second call just refreshes the same values.

    Returns the S3 key written, or None on error / no data.
    """
    try:
        existing = _read_existing(date_str)
        if existing is None or len(existing) == 0:
            logger.warning(f"[predictions_log] no log partition found for {date_str} — nothing to enrich")
            return None

        # Lazy import so the prediction-time path doesn't pay for requests
        import requests

        BASE_URL = "https://statsapi.mlb.com/api/v1"

        def _fetch_for_player(player_id, player_name) -> dict:
            """Return dict with hits/abs/hit_bool, or empty dict on failure."""
            try:
                # Prefer player_id if we have it (avoids name-collision bugs)
                pid = None
                if pd.notna(player_id):
                    try:
                        pid = int(player_id)
                    except (TypeError, ValueError):
                        pid = None

                if pid is None and isinstance(player_name, str):
                    r = requests.get(
                        f"{BASE_URL}/people/search",
                        params={"names": player_name, "sportId": 1},
                        timeout=10,
                    )
                    if r.ok:
                        people = r.json().get("people", [])
                        if people:
                            pid = people[0]["id"]

                if pid is None:
                    return {}

                r = requests.get(
                    f"{BASE_URL}/people/{pid}/stats",
                    params={
                        "stats": "gameLog",
                        "group": "hitting",
                        "season": date_str[:4],
                        "startDate": date_str,
                        "endDate": date_str,
                    },
                    timeout=10,
                )
                if not r.ok:
                    return {}

                splits = []
                for stat_group in r.json().get("stats", []):
                    splits.extend(stat_group.get("splits", []))
                if not splits:
                    return {}

                # If a player had two games (DH), sum across splits
                total_h = sum(int(s.get("stat", {}).get("hits", 0) or 0) for s in splits)
                total_ab = sum(int(s.get("stat", {}).get("atBats", 0) or 0) for s in splits)
                return {
                    "hits": total_h,
                    "abs": total_ab,
                    "hit_bool": total_h >= 1,
                }
            except Exception as e:
                logger.warning(f"[predictions_log] actuals fetch failed for {player_name}: {e}")
                return {}

        # We don't need to query every player — only players who matter for
        # any top-N tier are interesting for the dashboard. But for the
        # batter-vs-pitcher backtest later we *do* want full coverage.
        # Cap at top 100 by rank to keep API load reasonable.
        if "rank" in existing.columns:
            target = existing[existing["rank"].fillna(9999).astype(int) <= 100].copy()
        else:
            target = existing.copy()

        logger.info(f"[predictions_log] resolving actuals for {len(target)} rows on {date_str}")

        results = []
        for _, row in target.iterrows():
            res = _fetch_for_player(row.get("player_id"), row.get("player_name"))
            results.append(res)

        # Map results back to existing by index
        target_indices = target.index.tolist()
        for idx, res in zip(target_indices, results):
            if res:
                existing.at[idx, "actual_hit"] = bool(res["hit_bool"])
                existing.at[idx, "actual_ab_count"] = int(res["abs"])
                existing.at[idx, "actual_hit_count"] = int(res["hits"])

        existing["actuals_resolved_at"] = datetime.utcnow().isoformat()

        # Re-assert numeric/bool dtypes for the columns we just touched, but
        # do NOT pass through _coerce_to_log_schema — that helper is for the
        # write path and unconditionally nulls the actuals columns.
        existing["actual_ab_count"] = pd.to_numeric(existing["actual_ab_count"], errors="coerce")
        existing["actual_hit_count"] = pd.to_numeric(existing["actual_hit_count"], errors="coerce")
        # actual_hit stays as object/nullable bool — parquet handles None fine.

        return _write_dataframe(existing, date_str)

    except Exception as e:
        logger.error(f"[predictions_log] append_actuals failed for {date_str}: {e}")
        return None