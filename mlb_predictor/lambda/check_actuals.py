"""
check_actuals.py — Fill in previous day's hit/miss results in daily_log.json

Runs each morning BEFORE /predict/daily fires.
Reads yesterday's top picks from daily_log.json, checks actual game results
from the MLB Stats API, updates top1/top2/top5/top10_hit fields, and
recalculates running summary stats.

Schedule: Run at 10:00 AM ET daily (before the 11:30 AM prediction trigger)
EventBridge cron: cron(0 14 * 3-10 ? *)

Can also be triggered manually:
    python3 check_actuals.py --date 2026-03-27
"""

import os
import json
import logging
import argparse
from datetime import datetime, timedelta

import boto3
import requests

logger = logging.getLogger("beatthestreak")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = f"mlb-hit-predictor-{ENVIRONMENT}-bucket"
BASE_URL = "https://statsapi.mlb.com/api/v1"


def log(msg):
    print(f"[ACTUALS] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fetch actual game results from MLB Stats API
# ---------------------------------------------------------------------------
def get_player_hit_result(player_name: str, date_str: str) -> bool | None:
    """
    Check if a player got at least one hit on a given date.
    Returns True (hit), False (no hit), or None (game not found / postponed).
    """
    try:
        # Search for player by name
        resp = requests.get(
            f"{BASE_URL}/people/search",
            params={"names": player_name, "sportId": 1},
            timeout=10,
        )
        if not resp.ok:
            return None

        people = resp.json().get("people", [])
        if not people:
            log(f"  Player not found: {player_name}")
            return None

        player_id = people[0]["id"]

        # Get game log for the date
        resp = requests.get(
            f"{BASE_URL}/people/{player_id}/stats",
            params={
                "stats": "gameLog",
                "group": "hitting",
                "season": date_str[:4],
                "startDate": date_str,
                "endDate": date_str,
            },
            timeout=10,
        )
        if not resp.ok:
            return None

        splits = []
        for stat_group in resp.json().get("stats", []):
            splits.extend(stat_group.get("splits", []))

        if not splits:
            log(f"  No game log for {player_name} on {date_str} — possibly no game / postponed")
            return None

        hits = splits[0].get("stat", {}).get("hits", 0)
        result = hits >= 1
        log(f"  {player_name}: {hits} hit(s) → {'✅ HIT' if result else '❌ MISS'}")
        return result

    except Exception as e:
        log(f"  Error checking {player_name}: {e}")
        return None


def check_tier_hit(picks: list, date_str: str) -> bool | None:
    """
    Returns True if at least one player in the picks list got a hit.
    Returns None if all results are unknown (postponed / not found).
    """
    results = []
    for name in picks:
        result = get_player_hit_result(name, date_str)
        if result is True:
            return True  # Short circuit — at least one hit
        if result is not None:
            results.append(result)

    if not results:
        return None  # All unknown
    return False  # All checked, none got a hit


# ---------------------------------------------------------------------------
# Update streak and summary stats
# ---------------------------------------------------------------------------
def update_summary(summary: dict, daily_log: list) -> dict:
    """
    Recalculate all summary stats from the daily log.
    Only counts days where top1_hit is not None (has been resolved).
    """
    resolved = [d for d in daily_log if d.get("top1_hit") is not None]

    if not resolved:
        return summary

    # Hit rates
    for tier, key in [("top1", "top1"), ("top2", "top2"), ("top5", "top5"), ("top10", "top10")]:
        hits = sum(1 for d in resolved if d.get(f"{tier}_hit") is True)
        misses = sum(1 for d in resolved if d.get(f"{tier}_hit") is False)
        total = hits + misses
        summary[f"{key}_hits"] = hits
        summary[f"{key}_misses"] = misses
        summary[f"{key}_hit_rate_pct"] = round(hits / total * 100, 1) if total > 0 else 0.0

    summary["dates_predicted"] = len(resolved)

    # Current and longest streak (top-1 only)
    current_streak = 0
    longest_streak = 0
    running = 0
    for day in sorted(resolved, key=lambda x: x["date"]):
        if day.get("top1_hit") is True:
            running += 1
            longest_streak = max(longest_streak, running)
        else:
            running = 0
    current_streak = running

    summary["current_streak"] = current_streak
    summary["longest_streak"] = longest_streak

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_actuals_check(date_str: str = None):
    """
    Check actuals for a given date (default: yesterday).
    """
    if date_str is None:
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

    log(f"Checking actuals for {date_str}...")

    s3 = boto3.client("s3")

    # Load daily log
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="predictions/latest/daily_log.json")
        daily_log = json.loads(obj["Body"].read())
    except Exception:
        log("No daily_log.json found — nothing to check")
        return

    # Find yesterday's entry
    entry = next((d for d in daily_log if d["date"] == date_str), None)
    if not entry:
        log(f"No entry found for {date_str} — predictions may not have run that day")
        return

    if entry.get("top1_hit") is not None:
        log(f"Actuals already filled for {date_str} — skipping")
        return

    log(f"Checking results for: {entry.get('top_pick')} and top 2/5/10...")

    # Check top-1
    top1_name = entry.get("top_pick")
    if top1_name:
        entry["top1_hit"] = get_player_hit_result(top1_name, date_str)

    # Check top-2
    top2_picks = entry.get("top2_picks", [])
    if top2_picks:
        entry["top2_hit"] = check_tier_hit(top2_picks, date_str)

    # Check top-5
    top5_picks = entry.get("top5_picks", [])
    if top5_picks:
        entry["top5_hit"] = check_tier_hit(top5_picks, date_str)

    # Check top-10
    top10_picks = entry.get("top10_picks", [])
    if top10_picks:
        entry["top10_hit"] = check_tier_hit(top10_picks, date_str)

    log(f"Results: top1={entry['top1_hit']} top2={entry['top2_hit']} "
        f"top5={entry['top5_hit']} top10={entry['top10_hit']}")

    # Update entry in log
    daily_log = [d if d["date"] != date_str else entry for d in daily_log]

    # Load and update summary
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="predictions/latest/summary.json")
        summary = json.loads(obj["Body"].read())
    except Exception:
        summary = {"season": date_str[:4]}

    summary = update_summary(summary, daily_log)

    log(f"Updated summary: top1={summary.get('top1_hit_rate_pct')}%, "
        f"streak={summary.get('current_streak')}, "
        f"longest={summary.get('longest_streak')}")

    # Write back to S3
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="predictions/latest/daily_log.json",
        Body=json.dumps(daily_log, indent=2),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="predictions/latest/summary.json",
        Body=json.dumps(summary, indent=2),
        ContentType="application/json",
    )

    # Also write dated results file
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"predictions/{date_str}/results.json",
        Body=json.dumps(entry, indent=2),
        ContentType="application/json",
    )

    log(f"S3 updated — actuals check complete for {date_str}")


# ---------------------------------------------------------------------------
# Lambda handler (for EventBridge trigger)
# ---------------------------------------------------------------------------
def handler(event, context):
    """Lambda entry point."""
    date_str = event.get("date")  # Optional override
    run_actuals_check(date_str)
    return {"statusCode": 200, "body": "Actuals check complete"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check actuals for a given date")
    parser.add_argument("--date", default=None, help="Date to check (YYYY-MM-DD, default: yesterday)")
    args = parser.parse_args()
    run_actuals_check(args.date)