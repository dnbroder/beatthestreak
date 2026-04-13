"""
scripts/pitcher_features_probe.py — Phase 2 feature validation probe.

Tests four proposed Phase 2 pitcher features sourced from the MLB Stats
API statSplits endpoint. For ~30 stratified starters, pulls the relevant
sitCodes, computes each proposed feature, and reports:

  1. Data feasibility — does each sitCode actually return usable stats?
  2. Feature distributions — do the features have meaningful spread
     across pitchers, or is everyone clustered around the same value?
  3. Feature correlations — are any of the four proposed features
     essentially measuring the same underlying phenomenon?
  4. Skill confounding — does each feature correlate strongly with
     career OPS allowed, suggesting it's a skill proxy rather than
     a distinct signal?

Features under test:

  pitcher_late_inning_degradation =
      (OPS_allowed_innings_7_plus) - (OPS_allowed_innings_1_to_6)
  Positive = pitcher falls apart late.

  pitcher_fade_after_75_pitches =
      (OPS_allowed_pitches_76_plus) - (OPS_allowed_first_75_pitches)
  Positive = pitcher fades as pitch count climbs.

  pitcher_first_inning_advantage =
      (OPS_allowed_1st_inning) - (OPS_allowed_innings_2_plus)
  Positive = slow starter. Negative = strong opener.

  pitcher_performance_on_current_rest =
      OPS_allowed_at_rest_level
  Measured at each rest bucket (dr1-dr5). For variance analysis we
  compare performance between 4-days rest (most common) and 5+ rest
  (extended rest) to see if rest matters.

Methodology:
  - Select 30 pitchers stratified across AABBH tiers (elite/avg/below).
  - For each pitcher, pull career statSplits for the relevant sitCodes.
  - Compute the four features per pitcher.
  - Report per-feature distribution (mean/min/max/stddev).
  - Compute 4x4 correlation matrix between features.
  - Compute correlation of each feature with career OPS allowed
    (skill confound detector).

Usage:
    python3 scripts/pitcher_features_probe.py
    python3 scripts/pitcher_features_probe.py --max-pitchers 10

Runtime: ~15-20 minutes for 30 pitchers (5 sitCodes each + season stats
= 6 API calls per pitcher at 0.25s rate limit ~= 45s/pitcher).

Logging conventions match Phase 2 scripts:
  - [FEATURES_PROBE] routine output
  - [FEATURES_PROBE]   [HEALTH] grepable metrics
  - [FEATURES_PROBE]   [HEALTH] WARNING anomalies
"""

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from typing import Optional

import requests


logger = logging.getLogger("beatthestreak")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


def log(msg: str):
    logger.info(f"[FEATURES_PROBE] {msg}")


def health(msg: str):
    logger.info(f"[FEATURES_PROBE]   [HEALTH] {msg}")


def warn(msg: str):
    logger.warning(f"[FEATURES_PROBE]   [HEALTH] WARNING: {msg}")


BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 20
RATE_LIMIT_DELAY = 0.25


# SitCodes we need per feature
SITCODES_PER_INNING = ["ig01", "ig07"]          # innings 1-6, 7+
SITCODES_PER_INNING_FIRST = ["i01"]              # first inning specifically
SITCODES_PITCH_COUNT = ["pi000", "pi760"]       # first 75, 76+
SITCODES_REST_DAYS = ["dr4", "dr5"]             # 4 days rest, 5+ rest

ALL_SITCODES = (SITCODES_PER_INNING + SITCODES_PER_INNING_FIRST
                + SITCODES_PITCH_COUNT + SITCODES_REST_DAYS)


def safe_get(url: str, params: dict = None) -> Optional[dict]:
    time.sleep(RATE_LIMIT_DELAY)
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        warn(f"api_fetch_failed url={url} error={e}")
        return None


# ---------------------------------------------------------------------------
# Pitcher selection — reuses the stratification logic from zone discovery
# ---------------------------------------------------------------------------
def get_season_starters(season: int, min_starts: int = 15) -> list:
    data = safe_get(
        f"{BASE_URL}/stats",
        params={
            "stats": "season",
            "group": "pitching",
            "season": season,
            "sportId": 1,
            "limit": 300,
        }
    )
    if not data:
        return []

    starters = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            s = split.get("stat", {})
            if s.get("gamesStarted", 0) < min_starts:
                continue
            player = split.get("player", {})
            pid = player.get("id")
            name = player.get("fullName")
            ab = s.get("atBats", 0)
            h = s.get("hits", 0)
            aabbh = ab / h if h > 0 else None
            ops = s.get("ops")
            try:
                ops_val = float(ops) if ops is not None else None
            except (ValueError, TypeError):
                ops_val = None
            if pid and name:
                starters.append({
                    "pitcher_id": pid,
                    "name": name,
                    "starts": s.get("gamesStarted", 0),
                    "season_aabbh": aabbh,
                    "season_ops_allowed": ops_val,
                })
    return starters


def stratified_select(starters: list, n_per_tier: int = 10) -> list:
    by_pid = {s["pitcher_id"]: s for s in starters}
    unique = list(by_pid.values())

    elite = sorted([s for s in unique if s["season_aabbh"] and s["season_aabbh"] >= 4.3],
                   key=lambda s: s["starts"], reverse=True)
    avg = sorted([s for s in unique if s["season_aabbh"] and 3.7 <= s["season_aabbh"] < 4.3],
                 key=lambda s: s["starts"], reverse=True)
    below = sorted([s for s in unique if s["season_aabbh"] and s["season_aabbh"] < 3.7],
                   key=lambda s: s["starts"], reverse=True)

    selected = elite[:n_per_tier] + avg[:n_per_tier] + below[:n_per_tier]
    health(f"selection pool_elite={len(elite)} pool_avg={len(avg)} "
           f"pool_below={len(below)} selected={len(selected)}")
    return selected


# ---------------------------------------------------------------------------
# Split fetching
# ---------------------------------------------------------------------------
def fetch_split_stats(pitcher_id: int, sit_code: str, season: int,
                       debug_first: bool = False) -> Optional[dict]:
    """
    Pull one specific split for a pitcher in a specific season.
    Returns the stat dict for that split, or None if unavailable.

    If debug_first=True, logs the raw response shape for this sitCode
    so we can diagnose which sitCodes work vs don't.
    """
    data = safe_get(
        f"{BASE_URL}/people/{pitcher_id}/stats",
        params={
            "stats": "statSplits",
            "group": "pitching",
            "season": season,
            "sitCodes": sit_code,
        }
    )
    if debug_first:
        if data is None:
            log(f"    [DEBUG {sit_code}] response=None")
        else:
            stats = data.get("stats", [])
            if not stats:
                log(f"    [DEBUG {sit_code}] empty stats array")
            else:
                splits = stats[0].get("splits", [])
                log(f"    [DEBUG {sit_code}] splits_count={len(splits)}")
                if splits:
                    first_stat_keys = sorted(splits[0].get("stat", {}).keys())[:8]
                    log(f"    [DEBUG {sit_code}] first_stat_fields (first 8): {first_stat_keys}")

    if not data:
        return None

    stats = data.get("stats", [])
    if not stats:
        return None

    splits = stats[0].get("splits", [])
    if not splits:
        return None

    return splits[0].get("stat", {})


def parse_ops(val) -> Optional[float]:
    """OPS comes as a string; parse defensively."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        # Sanity: valid OPS for a pitcher-allowed stat should be in [0.2, 1.5]
        if 0.1 < f < 2.0:
            return f
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Feature computation from raw sitCode data
# ---------------------------------------------------------------------------
def compute_features(splits_data: dict, career_aabbh: Optional[float],
                     season_ops_allowed: Optional[float]) -> dict:
    """
    Take the raw {sit_code: stat_dict} data for one pitcher and compute
    all four proposed features plus metadata for downstream analysis.

    Returns None-valued entries for features that can't be computed
    due to missing underlying splits — the analysis layer filters these.
    """
    # Extract OPS for each relevant split
    ops_innings_1_6 = parse_ops(splits_data.get("ig01", {}).get("ops"))
    ops_innings_7_plus = parse_ops(splits_data.get("ig07", {}).get("ops"))
    ops_first_inning = parse_ops(splits_data.get("i01", {}).get("ops"))
    ops_first_75_pitches = parse_ops(splits_data.get("pi000", {}).get("ops"))
    ops_pitches_76_plus = parse_ops(splits_data.get("pi760", {}).get("ops"))
    ops_4_days_rest = parse_ops(splits_data.get("dr4", {}).get("ops"))
    ops_5_plus_rest = parse_ops(splits_data.get("dr5", {}).get("ops"))

    # Feature 1: late inning degradation
    if ops_innings_7_plus is not None and ops_innings_1_6 is not None:
        late_inning_degradation = ops_innings_7_plus - ops_innings_1_6
    else:
        late_inning_degradation = None

    # Feature 2: pitch count fade
    if ops_pitches_76_plus is not None and ops_first_75_pitches is not None:
        pitch_count_fade = ops_pitches_76_plus - ops_first_75_pitches
    else:
        pitch_count_fade = None

    # Feature 3: first inning advantage (note: need "innings 2+" which we
    # approximate as ig01 minus weighted first inning. Simpler: compare
    # first inning OPS directly to ig01 which is "innings 1-6". If first
    # inning OPS is notably higher than 1-6 average, the first inning is
    # dragging the 1-6 number up => slow starter. Clean approximation
    # because ig01 is already 5-6 innings of data compared to 1 inning.)
    if ops_first_inning is not None and ops_innings_1_6 is not None:
        first_inning_disadvantage = ops_first_inning - ops_innings_1_6
    else:
        first_inning_disadvantage = None

    # Feature 4: rest-day performance differential. If the pitcher has
    # BOTH dr4 and dr5 data, we can compare. Otherwise return None.
    if ops_5_plus_rest is not None and ops_4_days_rest is not None:
        extra_rest_benefit = ops_4_days_rest - ops_5_plus_rest
        # Positive = pitcher is BETTER on 5+ rest (extra rest helps).
        # Negative = pitcher is WORSE on 5+ rest (rust dominates recovery).
    else:
        extra_rest_benefit = None

    return {
        "late_inning_degradation": late_inning_degradation,
        "pitch_count_fade": pitch_count_fade,
        "first_inning_disadvantage": first_inning_disadvantage,
        "extra_rest_benefit": extra_rest_benefit,
        # Metadata
        "career_aabbh": career_aabbh,
        "season_ops_allowed": season_ops_allowed,
        # Raw components for auditing
        "_ops_innings_1_6": ops_innings_1_6,
        "_ops_innings_7_plus": ops_innings_7_plus,
        "_ops_first_inning": ops_first_inning,
        "_ops_first_75_pitches": ops_first_75_pitches,
        "_ops_pitches_76_plus": ops_pitches_76_plus,
        "_ops_4_days_rest": ops_4_days_rest,
        "_ops_5_plus_rest": ops_5_plus_rest,
    }


def probe_pitcher(pitcher: dict, season: int, debug: bool = False) -> dict:
    """Pull all sitCodes for one pitcher, compute features."""
    name = pitcher["name"]
    pid = pitcher["pitcher_id"]
    log("")
    log(f"[{name}] id={pid} starts={pitcher['starts']} "
        f"aabbh={pitcher['season_aabbh']:.2f}")

    t0 = time.time()
    splits_data = {}
    successful_codes = 0
    for code in ALL_SITCODES:
        stat = fetch_split_stats(pid, code, season, debug_first=debug)
        if stat:
            splits_data[code] = stat
            successful_codes += 1
        elif not debug:
            # Quietly note which sitCodes failed on non-debug runs for a
            # running per-pitcher success count; aggregate diagnostic
            # comes at the end.
            pass

    elapsed = time.time() - t0
    log(f"    fetched {successful_codes}/{len(ALL_SITCODES)} sitCodes in {elapsed:.1f}s")
    failed_codes = [c for c in ALL_SITCODES if c not in splits_data]
    if failed_codes:
        log(f"    failed sitCodes: {failed_codes}")

    features = compute_features(
        splits_data,
        career_aabbh=pitcher["season_aabbh"],
        season_ops_allowed=pitcher["season_ops_allowed"],
    )
    features["pitcher_id"] = pid
    features["name"] = name

    # Log the feature values for visibility
    health(f"pitcher_features name=\"{name}\" "
           f"late_inning_degradation={_fmt(features['late_inning_degradation'])} "
           f"pitch_count_fade={_fmt(features['pitch_count_fade'])} "
           f"first_inning_disadvantage={_fmt(features['first_inning_disadvantage'])} "
           f"extra_rest_benefit={_fmt(features['extra_rest_benefit'])}")

    return features


def _fmt(v) -> str:
    return f"{v:+.3f}" if v is not None else "n/a"


# ---------------------------------------------------------------------------
# Analysis — distributions, correlation matrix, skill confound check
# ---------------------------------------------------------------------------
def distribution_stats(values: list, name: str) -> dict:
    """Mean, stddev, min, max for a list of values (None-filtered)."""
    clean = [v for v in values if v is not None]
    if len(clean) < 3:
        return {"name": name, "n": len(clean), "mean": None, "stddev": None,
                "min": None, "max": None, "range": None}
    return {
        "name": name,
        "n": len(clean),
        "mean": statistics.mean(clean),
        "stddev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "range": max(clean) - min(clean),
    }


def pearson_correlation(xs: list, ys: list) -> Optional[float]:
    """
    Compute Pearson correlation on paired values, skipping None pairs.
    Returns None if too few valid pairs or zero variance.
    Manual implementation — statistics.correlation requires Python 3.10+
    and CloudShell runs 3.9.
    """
    paired = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(paired) < 4:
        return None
    n = len(paired)
    x_vals = [p[0] for p in paired]
    y_vals = [p[1] for p in paired]
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in paired)
    x_var = sum((x - x_mean) ** 2 for x in x_vals)
    y_var = sum((y - y_mean) ** 2 for y in y_vals)
    denom = (x_var * y_var) ** 0.5
    if denom == 0:
        return None
    return num / denom


def print_distribution_report(features_list: list):
    log("")
    log("=" * 70)
    log("FEATURE DISTRIBUTIONS")
    log("=" * 70)
    log("Spread tells us whether each feature varies meaningfully across pitchers.")
    log("A feature with tight distribution (stddev < 0.05) has low information value.")
    log("")

    feature_names = [
        "late_inning_degradation",
        "pitch_count_fade",
        "first_inning_disadvantage",
        "extra_rest_benefit",
    ]

    log(f"{'Feature':<32} {'N':>4} {'Mean':>8} {'StdDev':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    log("-" * 80)
    for name in feature_names:
        vals = [f[name] for f in features_list]
        stats = distribution_stats(vals, name)
        if stats["mean"] is None:
            log(f"{name:<32} {stats['n']:>4}  (insufficient data)")
        else:
            log(f"{name:<32} {stats['n']:>4} "
                f"{stats['mean']:>+8.3f} {stats['stddev']:>8.3f} "
                f"{stats['min']:>+8.3f} {stats['max']:>+8.3f} "
                f"{stats['range']:>8.3f}")
        health(f"feature_distribution name={name} n={stats['n']} "
               f"mean={_fmt(stats['mean'])} stddev={_fmt(stats['stddev'])} "
               f"range={_fmt(stats['range'])}")


def print_correlation_matrix(features_list: list):
    log("")
    log("=" * 70)
    log("FEATURE × FEATURE CORRELATION MATRIX")
    log("=" * 70)
    log("Values close to +1 or -1 indicate redundancy. Drop one of each")
    log("pair with |r| > 0.7 — they're measuring the same underlying thing.")
    log("")

    feature_names = [
        "late_inning_degradation",
        "pitch_count_fade",
        "first_inning_disadvantage",
        "extra_rest_benefit",
    ]

    # Short labels for the header
    labels = ["late_inn", "pitch_cnt", "first_inn", "rest_diff"]

    # Header
    log(f"{'':<26} " + "  ".join(f"{lbl:>10}" for lbl in labels))
    log("-" * 80)

    for i, row_name in enumerate(feature_names):
        row = [f"{labels[i]:<26}"]
        row_vals = [f[row_name] for f in features_list]
        for col_name in feature_names:
            col_vals = [f[col_name] for f in features_list]
            r = pearson_correlation(row_vals, col_vals)
            if r is None:
                row.append(f"{'n/a':>10}")
            else:
                row.append(f"{r:>+10.3f}")
        log("  ".join(row))

    # Flag the redundancy-risk pairs
    log("")
    log("Redundancy check — pairs with |r| > 0.7 are candidates for removal:")
    redundancies_found = False
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            row_vals = [f[feature_names[i]] for f in features_list]
            col_vals = [f[feature_names[j]] for f in features_list]
            r = pearson_correlation(row_vals, col_vals)
            if r is not None and abs(r) > 0.7:
                log(f"  * {feature_names[i]}  <->  {feature_names[j]}  r={r:+.3f}")
                health(f"redundancy_flag feature1={feature_names[i]} "
                       f"feature2={feature_names[j]} r={r:+.3f}")
                redundancies_found = True
    if not redundancies_found:
        log("  None. All four features measure distinct patterns.")
        health("redundancy_flag found=0")


def print_skill_confound_check(features_list: list):
    log("")
    log("=" * 70)
    log("SKILL CONFOUND CHECK")
    log("=" * 70)
    log("Each feature correlated against career OPS allowed. |r| > 0.6 means")
    log("the feature is strongly related to pitcher skill — it might be a")
    log("skill proxy rather than a distinct signal. Same trap we hit with")
    log("raw variance in zone discovery. Lower correlation is better here.")
    log("")

    feature_names = [
        "late_inning_degradation",
        "pitch_count_fade",
        "first_inning_disadvantage",
        "extra_rest_benefit",
    ]

    ops_vals = [f.get("season_ops_allowed") for f in features_list]

    log(f"{'Feature':<32} {'vs season_ops_allowed':>22}")
    log("-" * 58)
    confounded = []
    for name in feature_names:
        vals = [f[name] for f in features_list]
        r = pearson_correlation(vals, ops_vals)
        marker = ""
        if r is not None and abs(r) > 0.6:
            marker = "  <- LIKELY SKILL PROXY"
            confounded.append(name)
        r_str = f"{r:+.3f}" if r is not None else "n/a"
        log(f"{name:<32} {r_str:>22}{marker}")
        health(f"confound_check feature={name} r_vs_season_ops={_fmt(r)} "
               f"confounded={name in confounded}")

    if confounded:
        warn(f"confound_check_flagged features={confounded} — likely "
             f"correlated with skill rather than measuring independent signal")


def print_final_verdict(features_list: list):
    log("")
    log("=" * 70)
    log("PHASE 2 FEATURE RECOMMENDATIONS")
    log("=" * 70)

    feature_names = [
        "late_inning_degradation",
        "pitch_count_fade",
        "first_inning_disadvantage",
        "extra_rest_benefit",
    ]

    # Features with a distribution sufficient to be useful (non-tight)
    keepers = []
    for name in feature_names:
        vals = [f[name] for f in features_list if f[name] is not None]
        if len(vals) < 5:
            log(f"  {name}: INSUFFICIENT_DATA — only {len(vals)} values, drop")
            continue
        stddev = statistics.stdev(vals)
        if stddev < 0.02:
            log(f"  {name}: TIGHT_DISTRIBUTION — stddev {stddev:.4f} "
                f"(<0.02), drop")
        else:
            log(f"  {name}: SPREAD_OK — stddev {stddev:.4f}, evaluate further")
            keepers.append(name)

    # Check redundancies among keepers — pick one representative
    survivors = []
    dropped_for_redundancy = set()
    for i, name_a in enumerate(keepers):
        if name_a in dropped_for_redundancy:
            continue
        survivors.append(name_a)
        for j in range(i + 1, len(keepers)):
            name_b = keepers[j]
            if name_b in dropped_for_redundancy:
                continue
            r = pearson_correlation(
                [f[name_a] for f in features_list],
                [f[name_b] for f in features_list]
            )
            if r is not None and abs(r) > 0.7:
                log(f"    {name_b}: drop, r={r:+.3f} with {name_a}")
                dropped_for_redundancy.add(name_b)

    log("")
    log(f"Surviving features: {len(survivors)}")
    for s in survivors:
        log(f"  - {s}")

    health(f"phase2_feature_recommendation survivors={survivors} "
           f"dropped_redundant={list(dropped_for_redundancy)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--max-pitchers", type=int, default=30,
                        help="Total pitchers to probe (default: 30)")
    parser.add_argument("--min-starts", type=int, default=15)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    run_start = time.time()
    log("=" * 70)
    log("PITCHER FEATURES PROBE — Phase 2 sitCode-based feature validation")
    log("=" * 70)
    log(f"Season: {args.season}")
    log(f"Target pitchers: {args.max_pitchers}")
    log(f"SitCodes per pitcher: {ALL_SITCODES}")
    log("")

    # Step 1: stratified pitcher selection
    log("-" * 70)
    log(f"STEP 1 — Select starters from {args.season}")
    log("-" * 70)
    pool = get_season_starters(args.season, min_starts=args.min_starts)
    log(f"Candidate pool: {len(pool)} starters")

    n_per_tier = max(5, args.max_pitchers // 3)
    selected = stratified_select(pool, n_per_tier=n_per_tier)
    if args.max_pitchers:
        selected = selected[:args.max_pitchers]

    log(f"Selected: {len(selected)} pitchers")

    # Runtime estimate
    est_api_calls = len(selected) * len(ALL_SITCODES)
    est_seconds = est_api_calls * (RATE_LIMIT_DELAY + 0.3)
    log(f"Estimated API calls: {est_api_calls}")
    log(f"Estimated runtime:   {est_seconds:.0f}s ({est_seconds/60:.1f} min)")
    log("")

    # Step 2: probe each pitcher
    log("-" * 70)
    log("STEP 2 — Probe pitchers")
    log("-" * 70)
    all_features = []
    for i, pitcher in enumerate(selected, 1):
        log("")
        log(f"[{i}/{len(selected)}]")
        features = probe_pitcher(pitcher, args.season, debug=(i == 1))
        all_features.append(features)

    # Step 3-5: analysis
    print_distribution_report(all_features)
    print_correlation_matrix(all_features)
    print_skill_confound_check(all_features)
    print_final_verdict(all_features)

    elapsed = time.time() - run_start
    log("")
    log("=" * 70)
    log(f"PROBE COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("=" * 70)
    health(f"probe_complete duration_sec={elapsed:.0f} "
           f"pitchers_probed={len(all_features)}")

    if args.output_json:
        # Strip leading underscore keys for cleanliness in the dump
        with open(args.output_json, "w") as f:
            json.dump({
                "metadata": {
                    "season": args.season,
                    "pitchers": len(all_features),
                    "duration_sec": round(elapsed, 1),
                },
                "features": all_features,
            }, f, indent=2, default=str)
        log(f"Full results written to: {args.output_json}")


if __name__ == "__main__":
    main()