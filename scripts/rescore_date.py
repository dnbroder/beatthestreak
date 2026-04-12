#!/usr/bin/env python3
"""
rescore_date.py — Re-run /predict/daily for a single historical date through
the current (v1.5.1+) pipeline, then resolve actuals.

Use case:
  - A historical day's predictions were corrupted by a bug that has since
    been fixed (e.g. 2026-04-02 had duplicated players from the doubleheader
    detection miss; v1.5.1 added output-side dedup).
  - This script re-invokes the full daily_predictions pipeline for that date,
    which overwrites predictions/{date}/*.csv, predictions/latest/*.csv,
    the entry in predictions/latest/daily_log.json, and the v1.5+ parquet
    at predictions_log/dt={date}/predictions.parquet — all as side effects
    of run_daily_predictions.write_to_s3=True.
  - It then invokes the actuals checker Lambda to resolve the fresh picks
    against real game results.

NOTE: This rewrites history. The new entry reflects what the CURRENT pipeline
would have produced given today's state of the MLB Stats API. It is NOT
bit-identical to what ran live on that date (pitcher ERAs drift, lineup
confirmation may differ slightly). It IS internally consistent and free of
the upstream bug.

Usage:
    python3 scripts/rescore_date.py --date 2026-04-02
    python3 scripts/rescore_date.py --date 2026-04-02 --skip-actuals
"""

import argparse
import json
import sys
import time

import boto3

from mlb_predictor.daily_predictions import run_daily_predictions


def rescore(date_str: str, skip_actuals: bool = False) -> int:
    print(f"[RESCORE] Re-running predictions for {date_str}...")
    print(f"[RESCORE] This will overwrite:")
    print(f"[RESCORE]   - s3://.../predictions/{date_str}/all_predictions.csv")
    print(f"[RESCORE]   - s3://.../predictions/{date_str}/top10_picks.csv")
    print(f"[RESCORE]   - s3://.../predictions/latest/* (if {date_str} is most recent)")
    print(f"[RESCORE]   - {date_str} entry in predictions/latest/daily_log.json")
    print(f"[RESCORE]   - s3://.../predictions_log/dt={date_str}/predictions.parquet")

    start = time.time()
    df = run_daily_predictions(date_str=date_str, write_to_s3=True)
    elapsed = time.time() - start

    if df.empty:
        print(f"[RESCORE] ERROR: pipeline returned empty dataframe for {date_str}")
        return 1

    top1 = df.iloc[0]
    print(f"[RESCORE] Re-scored in {elapsed:.0f}s — new top pick: "
          f"{top1['player_name']} ({top1['confidence_pct']}%)")

    # Check for duplicates in top 10 as a sanity check that the v1.5.1 dedup
    # actually did its job
    top10_names = df.head(10)["player_name"].tolist()
    if len(set(top10_names)) != len(top10_names):
        print(f"[RESCORE] WARNING: top 10 still contains duplicates: {top10_names}")
        print(f"[RESCORE] This should not happen with v1.5.1+ code. Investigate.")
        return 2
    print(f"[RESCORE] Top 10 uniqueness check: PASSED ({len(set(top10_names))} unique players)")

    if skip_actuals:
        print(f"[RESCORE] Skipping actuals resolution (--skip-actuals flag set)")
        print(f"[RESCORE] Run manually: aws lambda invoke --function-name "
              f"production-actuals-checker --payload '{{\"date\":\"{date_str}\"}}' /tmp/out.json")
        return 0

    # The daily_log.json entry now has top1_hit=None after the rescore (the
    # pipeline resets it on re-entry for the date). Invoke check_actuals to
    # resolve against real game results.
    print(f"[RESCORE] Invoking production-actuals-checker for {date_str}...")
    lam = boto3.client("lambda")
    resp = lam.invoke(
        FunctionName="production-actuals-checker",
        Payload=json.dumps({"trigger": "rescore", "date": date_str}).encode(),
    )
    payload = json.loads(resp["Payload"].read())
    status = resp.get("StatusCode")
    print(f"[RESCORE] Lambda returned StatusCode={status}, body={payload}")

    if status != 200:
        print(f"[RESCORE] ERROR: actuals Lambda returned non-200 status")
        return 3

    # Verify the resolution landed
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(
            Bucket="mlb-hit-predictor-production-bucket",
            Key="predictions/latest/daily_log.json",
        )
        daily_log = json.loads(obj["Body"].read())
        entry = next((d for d in daily_log if d["date"] == date_str), None)
        if entry:
            print(f"[RESCORE] Post-actuals verification for {date_str}:")
            print(f"[RESCORE]   top_pick: {entry.get('top_pick')} ({entry.get('confidence')}%)")
            print(f"[RESCORE]   top1_hit: {entry.get('top1_hit')}")
            print(f"[RESCORE]   top2_hit: {entry.get('top2_hit')}")
            print(f"[RESCORE]   top5_hit: {entry.get('top5_hit')}")
            print(f"[RESCORE]   top10_hit: {entry.get('top10_hit')}")
            if entry.get("top1_hit") is None:
                print(f"[RESCORE] WARNING: top1_hit still None — actuals did not resolve")
                print(f"[RESCORE] Check CloudWatch logs for production-actuals-checker")
                return 4
        else:
            print(f"[RESCORE] ERROR: no entry for {date_str} in daily_log.json after rescore")
            return 5
    except Exception as e:
        print(f"[RESCORE] ERROR verifying daily_log.json: {e}")
        return 6

    print(f"[RESCORE] {date_str} re-score complete and actuals resolved.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-score a historical date through the current prediction pipeline",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to re-score in YYYY-MM-DD format (e.g. 2026-04-02)",
    )
    parser.add_argument(
        "--skip-actuals",
        action="store_true",
        help="Skip the actuals resolution step (useful if you want to inspect "
             "the re-scored picks before resolving against real results)",
    )
    args = parser.parse_args()
    sys.exit(rescore(args.date, skip_actuals=args.skip_actuals))