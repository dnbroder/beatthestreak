"""
backtester.py – Run predictions across historical dates and validate results.

Pipeline:
  1. Generate list of game dates for a season
  2. For each date, run daily predictions
  3. Check actual results (did the picks get hits?)
  4. Compute hit rates for top-N picks
  5. Simulate streak attempts
  6. Write comprehensive results to S3
"""

import io
import sys
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import boto3

from mlb_predictor.historical_data import _safe_get, BASE_URL
from mlb_predictor.daily_predictions import run_daily_predictions
from mlb_predictor.model import load_model

logger = logging.getLogger("beatthestreak")


def log(msg):
    """Print with flush so it always shows in container logs immediately."""
    print(f"[BACKTEST] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Get actual game results for a date
# ---------------------------------------------------------------------------
def get_actual_results(date_str: str) -> dict:
    """
    Pull actual game results for a date.
    Returns dict of {player_id: {"hits": N, "at_bats": N, "got_hit": 0/1}}
    """
    data = _safe_get(
        f"{BASE_URL}/schedule",
        params={
            "sportId": 1,
            "date": date_str,
            "gameType": "R",
            "hydrate": "scoreboard",
        }
    )
    if not data:
        log(f"  No schedule data for {date_str}")
        return {}

    # Get boxscores for each game to find individual player results
    results = {}
    games_checked = 0
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk = game.get("gamePk")
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue

            box = _safe_get(f"{BASE_URL}/game/{game_pk}/boxscore")
            if not box:
                continue

            games_checked += 1
            # Small delay between boxscore fetches to be nice to the API
            time.sleep(0.3)

            for side in ["home", "away"]:
                team_data = box.get("teams", {}).get(side, {})
                players = team_data.get("players", {})

                for player_key, player_data in players.items():
                    if not player_key.startswith("ID"):
                        continue

                    pid = int(player_key.replace("ID", ""))
                    stats = player_data.get("stats", {})
                    batting = stats.get("batting", {})

                    abs_ = batting.get("atBats", 0)
                    hits = batting.get("hits", 0)

                    if abs_ > 0:  # Only include players who actually batted
                        # For doubleheaders: if player already has a hit, keep it
                        if pid in results and results[pid]["got_hit"] == 1:
                            continue
                        results[pid] = {
                            "hits": hits,
                            "at_bats": abs_,
                            "got_hit": 1 if hits > 0 else 0,
                        }

    log(f"  Actuals: {games_checked} games, {len(results)} batters with ABs")
    return results


# ---------------------------------------------------------------------------
# Get all game dates in a season
# ---------------------------------------------------------------------------
def get_season_game_dates(season: int, interval: int = 3) -> list:
    """
    Get regular season game dates, sampled at interval.
    Args:
        season: Year
        interval: Pick every Nth game date (3 = every 3rd day with games)
    """
    log(f"Fetching game dates for {season} season...")

    all_dates = set()

    # Pull month by month
    for month in range(4, 11):  # April through October (March lacks AABBH history)
        start = f"{season}-{month:02d}-01"
        end_month = month + 1
        end = f"{season}-{end_month:02d}-01" if month < 12 else f"{season}-12-31"

        data = _safe_get(
            f"{BASE_URL}/schedule",
            params={
                "sportId": 1,
                "startDate": start,
                "endDate": end,
                "gameType": "R",
            }
        )
        if not data:
            continue

        for date_entry in data.get("dates", []):
            games = date_entry.get("games", [])
            # Only include dates with multiple games (full slate)
            if len(games) >= 4:
                all_dates.add(date_entry["date"])

    sorted_dates = sorted(all_dates)
    sampled = sorted_dates[::interval]

    log(f"Found {len(sorted_dates)} game dates, sampled {len(sampled)} (every {interval} days)")
    return sampled


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------
def run_backtest(
    season: int = 2024,
    interval: int = 3,
    max_dates: int = None,
    top_n_picks: list = None,
    write_to_s3: bool = True,
) -> dict:
    """
    Full backtest: run predictions for historical dates and validate.

    Args:
        season: Year to backtest
        interval: Sample every Nth game date
        max_dates: Cap number of dates (for testing)
        top_n_picks: List of N values to evaluate (default [1, 2, 3, 5, 10])
        write_to_s3: Upload results to S3

    Returns:
        Dict with hit rates, streak results, and daily detail
    """
    if top_n_picks is None:
        top_n_picks = [1, 2, 3, 5, 10]

    log(f"=== BACKTEST START: {season} season, every {interval} days ===")

    # Check model is loaded
    model, feature_cols = load_model()
    if model is None:
        log("ERROR: No trained model. Run /train first.")
        return {"error": "No trained model"}
    log(f"Model loaded with {len(feature_cols)} features")

    # Get dates to test
    dates = get_season_game_dates(season, interval=interval)
    if max_dates:
        dates = dates[:max_dates]

    if not dates:
        log("ERROR: No dates to test")
        return {"error": "No game dates found"}

    log(f"Will test {len(dates)} dates: {dates[0]} to {dates[-1]}")

    # Track results
    daily_results = []
    streak_tracker = {n: {"current": 0, "longest": 0} for n in top_n_picks}
    running_hits = 0
    running_total = 0

    for i, date_str in enumerate(dates):
        log(f"")
        log(f"--- [{i+1}/{len(dates)}] {date_str} ---")

        # Rate limit: pause between dates to avoid MLB API throttling
        if i > 0:
            time.sleep(2)

        # Run predictions (don't write each day to S3)
        try:
            log(f"  Generating predictions...")
            predictions = run_daily_predictions(
                date_str=date_str,
                season=season,
                write_to_s3=False,
            )
        except Exception as e:
            log(f"  ERROR generating predictions: {e}")
            log(f"  Traceback: {traceback.format_exc()}")
            continue

        if predictions is None or (hasattr(predictions, 'empty') and predictions.empty):
            log(f"  No predictions for {date_str}, skipping")
            continue

        log(f"  Got {len(predictions)} predictions")

        # Get actual results
        log(f"  Fetching actual boxscores...")
        try:
            actuals = get_actual_results(date_str)
        except Exception as e:
            log(f"  ERROR fetching actuals: {e}")
            log(f"  Traceback: {traceback.format_exc()}")
            continue

        if not actuals:
            log(f"  No actual results for {date_str}, skipping")
            continue

        # Evaluate each top-N threshold
        day_entry = {
            "date": date_str,
            "total_batters_scored": len(predictions),
        }

        for n in top_n_picks:
            top_picks = predictions.head(n)
            hits = 0
            misses = 0
            pick_details = []

            for _, pick in top_picks.iterrows():
                pid = pick["player_id"]
                actual = actuals.get(pid, {})
                got_hit = actual.get("got_hit", None)

                pick_detail = {
                    "player": pick["player_name"],
                    "team": pick["team"],
                    "confidence": round(float(pick["confidence_pct"]), 1),
                    "got_hit": got_hit,
                    "actual_hits": actual.get("hits", "N/A"),
                    "actual_abs": actual.get("at_bats", "N/A"),
                }

                if "aabbh_ratio" in pick:
                    pick_detail["aabbh_ratio"] = round(float(pick.get("aabbh_ratio", 0)), 2)
                if "in_optimal_zone" in pick:
                    pick_detail["in_optimal_zone"] = int(pick.get("in_optimal_zone", 0))

                pick_details.append(pick_detail)

                if got_hit == 1:
                    hits += 1
                elif got_hit == 0:
                    misses += 1
                # got_hit is None means player wasn't in actuals (pinch runner, etc.)

            total_evaluated = hits + misses
            day_entry[f"top{n}_hits"] = hits
            day_entry[f"top{n}_total"] = total_evaluated
            day_entry[f"top{n}_rate"] = round(hits / max(total_evaluated, 1), 3)
            day_entry[f"top{n}_picks"] = pick_details

            # Streak tracking (for top-1 and top-2 simulations)
            if n <= 2:
                all_hit = (misses == 0 and total_evaluated > 0)
                if all_hit:
                    streak_tracker[n]["current"] += 1
                    streak_tracker[n]["longest"] = max(
                        streak_tracker[n]["longest"],
                        streak_tracker[n]["current"]
                    )
                else:
                    streak_tracker[n]["current"] = 0

        daily_results.append(day_entry)

        # Log the top-1 pick result prominently
        top1_pick = day_entry.get("top1_picks", [{}])[0] if day_entry.get("top1_picks") else {}
        top1_result = "HIT" if day_entry.get("top1_rate", 0) == 1.0 else "MISS"
        running_hits += day_entry.get("top1_hits", 0)
        running_total += day_entry.get("top1_total", 0)
        running_pct = round(running_hits / max(running_total, 1) * 100, 1)

        log(f"  TOP-1: {top1_pick.get('player', '???')} -> {top1_result} "
            f"({top1_pick.get('actual_hits', '?')}-for-{top1_pick.get('actual_abs', '?')})")
        log(f"  Top-5: {day_entry.get('top5_rate', 0):.0%} | "
            f"Top-10: {day_entry.get('top10_rate', 0):.0%}")
        log(f"  Running top-1: {running_hits}/{running_total} = {running_pct}%")

    # ---------------------------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------------------------
    log(f"")
    log(f"{'='*60}")
    log(f"=== BACKTEST COMPLETE ===")
    log(f"{'='*60}")

    num_dates = len(daily_results)
    summary = {
        "season": season,
        "dates_tested": num_dates,
        "interval": interval,
    }

    for n in top_n_picks:
        total_hits = sum(d.get(f"top{n}_hits", 0) for d in daily_results)
        total_eval = sum(d.get(f"top{n}_total", 0) for d in daily_results)
        hit_rate = round(total_hits / max(total_eval, 1), 4)

        summary[f"top{n}_hit_rate"] = hit_rate
        summary[f"top{n}_total_hits"] = total_hits
        summary[f"top{n}_total_evaluated"] = total_eval

        if n <= 2:
            summary[f"top{n}_longest_streak"] = streak_tracker[n]["longest"]

        log(f"Top-{n} hit rate: {hit_rate:.1%} ({total_hits}/{total_eval})")

    if 1 in top_n_picks:
        log(f"Longest streak (top-1 pick): {streak_tracker[1]['longest']}")
    if 2 in top_n_picks:
        log(f"Longest streak (top-2 picks): {streak_tracker[2]['longest']}")

    results = {
        "summary": summary,
        "daily_results": daily_results,
    }

    # Write to S3
    if write_to_s3:
        log(f"Writing results to S3...")
        _write_backtest_to_s3(results, season)

    log(f"=== DONE ===")
    return results


def _write_backtest_to_s3(results: dict, season: int):
    """Write backtest results to S3."""
    try:
        import os
        import json
        from datetime import datetime

        env = os.environ.get("ENVIRONMENT", "production")
        bucket = f"mlb-hit-predictor-{env}-bucket"
        s3 = boto3.client("s3")

        # Timestamp for unique run identification
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_prefix = f"backtest/{season}/{run_ts}"

        # Summary CSV
        summary_df = pd.DataFrame([results["summary"]])
        csv_buf = io.StringIO()
        summary_df.to_csv(csv_buf, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=f"{run_prefix}/summary.csv",
            Body=csv_buf.getvalue(),
            ContentType="text/csv",
        )

        # Daily results CSV
        daily_rows = []
        for day in results["daily_results"]:
            row = {
                "date": day["date"],
                "total_scored": day["total_batters_scored"],
            }
            for n in [1, 2, 3, 5, 10]:
                row[f"top{n}_hits"] = day.get(f"top{n}_hits", "")
                row[f"top{n}_total"] = day.get(f"top{n}_total", "")
                row[f"top{n}_rate"] = day.get(f"top{n}_rate", "")
            daily_rows.append(row)

        daily_df = pd.DataFrame(daily_rows)
        csv_buf = io.StringIO()
        daily_df.to_csv(csv_buf, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=f"{run_prefix}/daily_results.csv",
            Body=csv_buf.getvalue(),
            ContentType="text/csv",
        )

        # Full JSON (with pick details)
        s3.put_object(
            Bucket=bucket,
            Key=f"{run_prefix}/full_results.json",
            Body=json.dumps(results, default=str),
            ContentType="application/json",
        )

        # Also write to "latest" for easy access
        s3.put_object(
            Bucket=bucket,
            Key=f"backtest/{season}/latest/summary.csv",
            Body=pd.DataFrame([results["summary"]]).to_csv(index=False),
            ContentType="text/csv",
        )

        log(f"Results written to s3://{bucket}/{run_prefix}/")

    except Exception as e:
        log(f"ERROR writing to S3: {e}")
        log(f"Traceback: {traceback.format_exc()}")