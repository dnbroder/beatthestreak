#!/usr/bin/env python3
"""
scripts/backtest_phase2.py — Head-to-head backtest: V1 baseline vs V2 Phase 2.

This is the SHIP GATE for Phase 2. If V2 doesn't beat V1 by at least +1.5pp
on 2024 top-1 hit rate, Phase 2 doesn't promote to production.

Methodology:
  1. Load both locally-trained models (hit_predictor_v1_baseline.pkl
     and hit_predictor_v2_phase2.pkl from models_local/)
  2. Load the cached 2021-2024 training dataset
  3. Filter to 2024 dates only — this is our held-out test window
  4. For each game date, score every batter through BOTH models
  5. Pick top-1 for each model, check if they got a hit
  6. Compute top-1, top-2, top-5, top-10 hit rates for each model
  7. Compute longest streak for each
  8. Compare side-by-side and emit PASS/FAIL verdict

The backtest uses the same train/test time-split discipline as production
model.py — the 2024 data is chronologically after the training window, so
no leakage. Phase 2 features are already in the cached dataset (computed
with strict no-leakage discipline in pitcher_features.py).

Usage:
    python3 scripts/backtest_phase2.py
    python3 scripts/backtest_phase2.py --test-year 2024
    python3 scripts/backtest_phase2.py --test-year 2023  # cross-year sanity
    python3 scripts/backtest_phase2.py --output-json /tmp/backtest_results.json

SHIP GATE:
    V2 top-1 hit rate MUST be >= V1 top-1 hit rate + 1.5pp
    Otherwise we iterate on features before promoting.

Logging conventions:
  - [BACKTEST] routine output
  - [BACKTEST]   [HEALTH] grepable metrics
  - [BACKTEST]   [HEALTH] WARNING anomalies
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd


logger = logging.getLogger("beatthestreak")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


# Match paths used by train_phase2.py
LOCAL_DIR = os.path.expanduser("~/beatthestreak/models_local")
DATA_CACHE_PATH = os.path.join(LOCAL_DIR, "phase2_training_data.parquet")
MODEL_V1_PATH = os.path.join(LOCAL_DIR, "hit_predictor_v1_baseline.pkl")
MODEL_V2_PATH = os.path.join(LOCAL_DIR, "hit_predictor_v2_phase2.pkl")
FEATURE_COLS_V1_PATH = os.path.join(LOCAL_DIR, "feature_cols_v1.pkl")
FEATURE_COLS_V2_PATH = os.path.join(LOCAL_DIR, "feature_cols_v2.pkl")

# SHIP GATE — the critical threshold
SHIP_GATE_LIFT_PP = 1.5


def log(msg):
    logger.info(f"[BACKTEST] {msg}")


def health(msg):
    logger.info(f"[BACKTEST]   [HEALTH] {msg}")


def warn(msg):
    logger.warning(f"[BACKTEST]   [HEALTH] WARNING: {msg}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_local_model(model_path: str, cols_path: str):
    """Load a locally-trained model + feature column list."""
    if not os.path.exists(model_path):
        warn(f"model_not_found path={model_path}")
        return None, None
    if not os.path.exists(cols_path):
        warn(f"cols_not_found path={cols_path}")
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path, "rb") as f:
        cols = pickle.load(f)

    size_kb = os.path.getsize(model_path) / 1024
    log(f"  Loaded model from {model_path} ({size_kb:.1f} KB)")
    log(f"  Feature count: {len(cols)}")
    return model, cols


def load_backtest_data(test_year: int) -> Optional[pd.DataFrame]:
    """Load the cached training dataset, filter to test year."""
    if not os.path.exists(DATA_CACHE_PATH):
        warn(f"dataset_not_found path={DATA_CACHE_PATH} — "
             f"run train_phase2.py first to build the cache")
        return None

    log(f"Loading dataset from {DATA_CACHE_PATH}...")
    df = pd.read_parquet(DATA_CACHE_PATH)
    log(f"  Total rows in cache: {len(df):,}")

    # Filter to test year
    if "season" in df.columns:
        test_df = df[df["season"] == test_year].copy()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        test_df = df[df["date"].dt.year == test_year].copy()
    else:
        warn("no_season_or_date_column — can't filter to test year")
        return None

    log(f"  Test year {test_year}: {len(test_df):,} rows")
    if len(test_df) == 0:
        warn(f"no_data_for_test_year year={test_year}")
        return None

    return test_df


# ---------------------------------------------------------------------------
# Backtest simulation
# ---------------------------------------------------------------------------
def score_and_rank_daily(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    label: str,
) -> pd.DataFrame:
    """
    Score every row through the model, then rank within each game date.

    Returns a DataFrame with columns: date, player_id, player_name,
    confidence_pct, daily_rank, got_hit.
    """
    # Fill any missing features with 0 (defensive — shouldn't happen if
    # training matched this dataset, but graceful fallback prevents crashes)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        warn(f"missing_features_at_scoring model={label} cols={missing}")
        for c in missing:
            df[c] = 0

    X = df[feature_cols].fillna(df[feature_cols].median())
    probs = model.predict_proba(X)[:, 1]

    scored = df[["date", "player_id", "player_name", "got_hit"]].copy()
    scored["confidence_pct"] = probs * 100
    scored["model_label"] = label

    # Rank within each date
    scored = scored.sort_values(["date", "confidence_pct"], ascending=[True, False])
    scored["daily_rank"] = scored.groupby("date").cumcount() + 1

    return scored


def compute_top_n_hit_rate(
    scored: pd.DataFrame,
    n: int,
) -> dict:
    """
    For each date, check if at least one of the top-N picks got a hit.
    Returns aggregate hit rate + daily breakdown.

    'Hit' at top-N means: at least one of the top N picks recorded a hit.
    This matches the Beat-the-Streak contest semantics where you pick 1-2
    batters per day and need at least one to hit.
    """
    top_n_by_date = scored[scored["daily_rank"] <= n].copy()

    # Per-date: did ANY top-N pick get a hit?
    daily = top_n_by_date.groupby("date").agg(
        top_n_any_hit=("got_hit", "max"),  # at least one hit = 1
        top_n_picks_count=("got_hit", "count"),
    ).reset_index()

    hits = int(daily["top_n_any_hit"].sum())
    total = len(daily)
    hit_rate = hits / total if total > 0 else 0.0

    return {
        "n": n,
        "total_dates": total,
        "hits": hits,
        "misses": total - hits,
        "hit_rate": hit_rate,
        "hit_rate_pct": round(hit_rate * 100, 2),
        "daily": daily,
    }


def compute_streak(top_1_daily: pd.DataFrame) -> dict:
    """
    Walk the top-1 daily results in chronological order, computing longest
    consecutive streak of hits.
    """
    sorted_daily = top_1_daily.sort_values("date").reset_index(drop=True)
    longest = 0
    current = 0
    streak_start = None
    longest_start = None
    longest_end = None

    for _, row in sorted_daily.iterrows():
        if row["top_n_any_hit"] == 1:
            if current == 0:
                streak_start = row["date"]
            current += 1
            if current > longest:
                longest = current
                longest_start = streak_start
                longest_end = row["date"]
        else:
            current = 0

    return {
        "longest_streak": longest,
        "longest_streak_start": str(longest_start) if longest_start else None,
        "longest_streak_end": str(longest_end) if longest_end else None,
    }


# ---------------------------------------------------------------------------
# Side-by-side comparison + verdict
# ---------------------------------------------------------------------------
def print_comparison_and_verdict(
    v1_results: dict,
    v2_results: dict,
    v1_streak: dict,
    v2_streak: dict,
) -> str:
    """Print side-by-side table and return verdict string."""
    log("")
    log("=" * 70)
    log("HEAD-TO-HEAD COMPARISON")
    log("=" * 70)
    log("")
    log(f"{'Metric':<24} {'V1 Baseline':>14} {'V2 Phase 2':>14} {'Delta':>10}")
    log("-" * 65)

    for n, v1_key, v2_key in [(1, "top_1", "top_1"),
                               (2, "top_2", "top_2"),
                               (5, "top_5", "top_5"),
                               (10, "top_10", "top_10")]:
        v1_rate = v1_results[v1_key]["hit_rate_pct"]
        v2_rate = v2_results[v2_key]["hit_rate_pct"]
        delta = v2_rate - v1_rate
        sign = "+" if delta >= 0 else ""
        log(f"{'Top-' + str(n) + ' Hit Rate':<24} {v1_rate:>13.2f}% "
            f"{v2_rate:>13.2f}% {sign}{delta:>7.2f}pp")

    log("")

    # Streak comparison
    log(f"{'Longest Streak':<24} {v1_streak['longest_streak']:>14} "
        f"{v2_streak['longest_streak']:>14} "
        f"{v2_streak['longest_streak'] - v1_streak['longest_streak']:>+10}")

    # Date counts
    v1_dates = v1_results["top_1"]["total_dates"]
    v2_dates = v2_results["top_1"]["total_dates"]
    log(f"{'Dates Evaluated':<24} {v1_dates:>14} {v2_dates:>14} {'':>10}")

    log("")
    log("=" * 70)
    log("SHIP GATE VERDICT")
    log("=" * 70)

    top_1_lift_pp = v2_results["top_1"]["hit_rate_pct"] - v1_results["top_1"]["hit_rate_pct"]
    log(f"")
    log(f"Ship gate threshold: V2 must beat V1 by ≥ {SHIP_GATE_LIFT_PP}pp on top-1 hit rate")
    log(f"Actual V2 vs V1 lift: {top_1_lift_pp:+.2f}pp")
    log(f"")

    if top_1_lift_pp >= SHIP_GATE_LIFT_PP:
        log("█████████████████████████████████████████████████████████████████████")
        log(f"   PASS — Phase 2 is cleared for production promotion")
        log(f"   V2 top-1 hit rate {v2_results['top_1']['hit_rate_pct']:.2f}% "
            f"exceeds V1 baseline {v1_results['top_1']['hit_rate_pct']:.2f}% "
            f"by {top_1_lift_pp:+.2f}pp")
        log("█████████████████████████████████████████████████████████████████████")
        verdict = "PASS"
    elif top_1_lift_pp > 0:
        log("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
        log(f"   MARGINAL — Phase 2 improves but misses the {SHIP_GATE_LIFT_PP}pp gate")
        log(f"   V2 top-1 {v2_results['top_1']['hit_rate_pct']:.2f}% vs "
            f"V1 {v1_results['top_1']['hit_rate_pct']:.2f}% ({top_1_lift_pp:+.2f}pp)")
        log(f"   Recommendation: investigate feature importance, consider")
        log(f"   feature refinement, or add batter-side variance feature.")
        log("▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒")
        verdict = "MARGINAL"
    else:
        log("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
        log(f"   FAIL — Phase 2 does NOT improve on V1 baseline")
        log(f"   V2 top-1 {v2_results['top_1']['hit_rate_pct']:.2f}% vs "
            f"V1 {v1_results['top_1']['hit_rate_pct']:.2f}% ({top_1_lift_pp:+.2f}pp)")
        log(f"   DO NOT PROMOTE. Investigate why new features hurt performance:")
        log(f"   1. Check feature importance — are Phase 2 features dominant?")
        log(f"   2. Check for feature leakage or data quality issues")
        log(f"   3. Consider whether interaction term is miscalibrated")
        log("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
        verdict = "FAIL"

    log("")
    health(f"ship_gate_verdict verdict={verdict} "
           f"v1_top1={v1_results['top_1']['hit_rate_pct']:.2f} "
           f"v2_top1={v2_results['top_1']['hit_rate_pct']:.2f} "
           f"lift_pp={top_1_lift_pp:.2f} "
           f"gate_pp={SHIP_GATE_LIFT_PP}")

    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-year", type=int, default=2024,
                        help="Year to use as held-out test set (default: 2024)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write full results to this JSON file")
    parser.add_argument("--show-top-days", type=int, default=10,
                        help="Show top N daily results in detail (default 10)")
    args = parser.parse_args()

    run_start = time.time()
    log("=" * 70)
    log("PHASE 2 BACKTEST — V1 Baseline vs V2 Phase 2")
    log("=" * 70)
    log(f"Test year: {args.test_year}")
    log(f"Ship gate: V2 must beat V1 by ≥ {SHIP_GATE_LIFT_PP}pp on top-1 hit rate")
    log("")

    # Load models
    log("Loading models...")
    log("V1 baseline:")
    v1_model, v1_cols = load_local_model(MODEL_V1_PATH, FEATURE_COLS_V1_PATH)
    if v1_model is None:
        log("ERROR: V1 baseline not found. Run:")
        log("  python3 -m mlb_predictor.train_phase2")
        return 1

    log("V2 Phase 2:")
    v2_model, v2_cols = load_local_model(MODEL_V2_PATH, FEATURE_COLS_V2_PATH)
    if v2_model is None:
        log("ERROR: V2 Phase 2 not found. Run:")
        log("  python3 -m mlb_predictor.train_phase2")
        return 1

    # Load test data
    log("")
    log("Loading test data...")
    test_df = load_backtest_data(args.test_year)
    if test_df is None:
        return 1

    unique_dates = test_df["date"].nunique() if "date" in test_df.columns else 0
    unique_batters = test_df["player_id"].nunique()
    log(f"  Unique game dates: {unique_dates}")
    log(f"  Unique batters:    {unique_batters}")
    log(f"  Total rows:        {len(test_df):,}")
    log(f"  Overall hit rate:  {test_df['got_hit'].mean():.2%}")

    # Score through both models
    log("")
    log("Scoring all batters through V1 baseline...")
    v1_scored = score_and_rank_daily(test_df, v1_model, v1_cols, "v1")
    log(f"  V1 scored: {len(v1_scored):,} predictions across {v1_scored['date'].nunique()} dates")

    log("Scoring all batters through V2 Phase 2...")
    v2_scored = score_and_rank_daily(test_df, v2_model, v2_cols, "v2")
    log(f"  V2 scored: {len(v2_scored):,} predictions across {v2_scored['date'].nunique()} dates")

    # Compute top-N hit rates
    log("")
    log("Computing top-N hit rates...")
    v1_results = {}
    v2_results = {}
    for n in [1, 2, 5, 10]:
        v1_results[f"top_{n}"] = compute_top_n_hit_rate(v1_scored, n)
        v2_results[f"top_{n}"] = compute_top_n_hit_rate(v2_scored, n)
        log(f"  Top-{n}: V1={v1_results[f'top_{n}']['hit_rate_pct']:.2f}% "
            f"V2={v2_results[f'top_{n}']['hit_rate_pct']:.2f}%")

    # Compute streaks (top-1 only)
    v1_streak = compute_streak(v1_results["top_1"]["daily"])
    v2_streak = compute_streak(v2_results["top_1"]["daily"])
    log(f"  Streak: V1={v1_streak['longest_streak']} V2={v2_streak['longest_streak']}")

    # Show a detailed sample of top days for visual inspection
    log("")
    log("-" * 70)
    log(f"SAMPLE: First {args.show_top_days} game dates (V1 vs V2 top-1 pick)")
    log("-" * 70)
    dates_sorted = sorted(v1_scored["date"].unique())[:args.show_top_days]
    log(f"{'Date':<12} {'V1 Pick':<22} {'V1 Hit':>7} {'V2 Pick':<22} {'V2 Hit':>7}")
    for d in dates_sorted:
        v1_top = v1_scored[(v1_scored["date"] == d) & (v1_scored["daily_rank"] == 1)]
        v2_top = v2_scored[(v2_scored["date"] == d) & (v2_scored["daily_rank"] == 1)]
        if not v1_top.empty and not v2_top.empty:
            v1_row = v1_top.iloc[0]
            v2_row = v2_top.iloc[0]
            v1_hit = "✓" if v1_row["got_hit"] == 1 else "✗"
            v2_hit = "✓" if v2_row["got_hit"] == 1 else "✗"
            log(f"{str(d):<12} {v1_row['player_name']:<22} {v1_hit:>7} "
                f"{v2_row['player_name']:<22} {v2_hit:>7}")

    # Verdict
    verdict = print_comparison_and_verdict(v1_results, v2_results, v1_streak, v2_streak)

    elapsed = time.time() - run_start
    log("")
    log(f"Backtest complete in {elapsed:.0f}s")
    health(f"backtest_complete duration_sec={elapsed:.0f} "
           f"test_year={args.test_year} verdict={verdict}")

    # Optional JSON output
    if args.output_json:
        # Strip the daily DataFrames out — they're big and not needed for the summary
        def trim_results(res):
            return {
                k: {kk: vv for kk, vv in v.items() if kk != "daily"}
                for k, v in res.items()
            }

        payload = {
            "metadata": {
                "test_year": args.test_year,
                "ship_gate_lift_pp": SHIP_GATE_LIFT_PP,
                "duration_sec": round(elapsed, 1),
                "verdict": verdict,
            },
            "v1_baseline": {
                **trim_results(v1_results),
                "streak": v1_streak,
            },
            "v2_phase2": {
                **trim_results(v2_results),
                "streak": v2_streak,
            },
            "deltas": {
                f"top_{n}_lift_pp": round(
                    v2_results[f"top_{n}"]["hit_rate_pct"]
                    - v1_results[f"top_{n}"]["hit_rate_pct"], 2
                )
                for n in [1, 2, 5, 10]
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log(f"Full results written to: {args.output_json}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())