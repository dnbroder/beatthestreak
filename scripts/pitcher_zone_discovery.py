"""
scripts/pitcher_zone_discovery_v2.py — Phase 2 empirical analysis (v2).

v2 changes from v1:
  - Adds DE-CONFOUNDED variance analysis using normalized stddev
    (coefficient of variation = stddev / career_aabbh). Removes the
    entanglement between raw variance and pitcher skill level that v1
    flagged as a confound.
  - v1's raw-stddev tiering stays in the output for comparison — we
    want to SEE the de-confounding effect, not just replace it silently.
  - Zone discovery logic on the ratio axis is unchanged from v1.

The v1 run showed:
  - Hypothesis B confirmed on the ratio axis: 0.5-1.0 bucket is vulnerable
    zone at 25.94% hit rate (spread 3.19pp).
  - RAW variance analysis flagged "low variance = more hittable" at -2.32pp
    spread, but career AABBH difference (low tier 3.96 vs high tier 4.32)
    suggested the signal was confounded with pitcher skill.

The v2 question: after controlling for skill level, does variance per se
still carry signal, and in which direction?

Finds the pitcher-side equivalent of the batter 0.5-1.0 optimal zone by
bucketing per-game pitcher aabbh_ratio values and measuring the actual
hit rate allowed in each bucket. Also measures per-pitcher AABBH variance
as a passive second analysis — pitchers with high game-to-game variance
may be more hittable independent of their mean AABBH, because a lot can
happen to a pitcher in the 4-6 days between starts (mechanics, velocity,
fatigue, off-field) that we can't observe directly.

Tests two competing hypotheses on the ratio axis:

  Hypothesis A (mirror the batter):
    Overdue pitchers (aabbh_ratio > 1.0) regress toward the mean.
    They've been unusually hit-proof → opponents break out.
    → Vulnerable zone is HIGH ratios (> 1.0).

  Hypothesis B (anti-mirror):
    A pitcher on a hitless-AB streak is "in the zone" — commanding.
    High ratio = currently dominant, not due.
    → Vulnerable zone is LOW ratios (< 0.5), when they've been giving
      up hits frequently.

Plus a passive third question on the variance axis:

  Variance hypothesis:
    Pitchers with high game-to-game AABBH variance are more volatile
    and therefore more hittable on average — their floor matters more
    than their ceiling for a BtS pick.
    → High-stddev pitchers allow a higher pooled hit rate than
      low-stddev pitchers, independent of mean AABBH.

Methodology:
  1. Dynamically select starters from 2020-2024 who had enough workload
     to generate stable AABBH (≥15 starts in any of those seasons).
  2. Stratify selection across AABBH tiers: elite / average / below-avg.
     40-60 pitchers total.
  3. For each pitcher, pull their full 2020-2024 gameLog.
  4. For each GAME, compute aabbh_ratio using only PRIOR games
     (current_hitless_abs / career_aabbh_to_date).
  5. Bucket games by ratio: [0-0.5), [0.5-1.0), [1.0-1.5), [1.5-2.0), [2.0+).
  6. For each bucket, compute average hit rate allowed in those games.
  7. Also collect per-game AABBH values per pitcher, compute stddev,
     bucket pitchers by variance tier, report hit rates per tier.
  8. Highest hit rate bucket = pitcher's vulnerable zone.

No-leakage discipline: every ratio uses strictly PRIOR games. Matches
the shift(1) pattern from the batter feature pipeline.

Usage:
    python3 scripts/pitcher_zone_discovery.py
    python3 scripts/pitcher_zone_discovery.py --max-pitchers 20 --seasons 2022 2023
    python3 scripts/pitcher_zone_discovery.py --output-json /tmp/zone_results.json

Runtime: ~60 minutes for the full 40-60 pitcher × 5 season sweep.
Runtime: ~15-20 minutes for a 20-pitcher × 3-season smaller sample.
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


# ---------------------------------------------------------------------------
# Logging setup — same convention as pitcher_gamelog_audit.py and the
# mlb_predictor/* modules.
# ---------------------------------------------------------------------------
logger = logging.getLogger("beatthestreak")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


def log(msg: str):
    logger.info(f"[ZONE_DISCOVERY] {msg}")


def health(msg: str):
    logger.info(f"[ZONE_DISCOVERY]   [HEALTH] {msg}")


def warn(msg: str):
    logger.warning(f"[ZONE_DISCOVERY]   [HEALTH] WARNING: {msg}")


BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 20
RATE_LIMIT_DELAY = 0.25  # seconds between API calls — matches historical_data.py

# Bucket edges for aabbh_ratio. Keep edges aligned with the batter side
# so the two analyses compare cleanly.
BUCKETS = [
    ("0.0-0.5",  0.0, 0.5),
    ("0.5-1.0",  0.5, 1.0),
    ("1.0-1.5",  1.0, 1.5),
    ("1.5-2.0",  1.5, 2.0),
    ("2.0+",     2.0, float("inf")),
]


# ---------------------------------------------------------------------------
# Inning-pitched parsing — "6.2" means 6 full + 2 outs, not 6.2 innings.
# Used for workload sanity checks, not for AABBH math.
# ---------------------------------------------------------------------------
def parse_ip(ip_value) -> float:
    """Convert MLB IP notation ('6.2') to real innings (6.667)."""
    if ip_value is None:
        return 0.0
    try:
        s = str(ip_value)
        if "." in s:
            whole, frac = s.split(".")
            outs_map = {"0": 0.0, "1": 0.333, "2": 0.667}
            return float(whole) + outs_map.get(frac, 0.0)
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# API helpers — lightweight wrappers with rate limiting.
# ---------------------------------------------------------------------------
def safe_get(url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    """GET with retry + rate limiting. Mirrors historical_data._safe_get."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                warn(f"api_fetch_exhausted url={url} error={e}")
    return None


def fetch_pitcher_gamelog(pitcher_id: int, season: int) -> list:
    """
    Return sorted list of game dicts for one pitcher-season.
    Each dict: date, at_bats, hits, innings_pitched, games_started, is_starter.
    Only starts are kept — relief outings have different workload profiles
    and pollute AABBH 'streak' calculations.
    """
    data = safe_get(
        f"{BASE_URL}/people/{pitcher_id}/stats",
        params={"stats": "gameLog", "group": "pitching", "season": season}
    )
    if not data:
        return []

    games = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            s = split.get("stat", {})
            if s.get("gamesStarted", 0) != 1:
                continue  # relief only
            games.append({
                "date": split.get("date", ""),
                "game_pk": split.get("game", {}).get("gamePk"),
                "at_bats": s.get("atBats", 0),
                "hits": s.get("hits", 0),
                "innings_pitched": parse_ip(s.get("inningsPitched", 0)),
                "batters_faced": s.get("battersFaced", 0),
                "earned_runs": s.get("earnedRuns", 0),
            })

    return sorted(games, key=lambda g: g["date"])


# ---------------------------------------------------------------------------
# Dynamic pitcher selection — query the leaderboard instead of hardcoding.
# ---------------------------------------------------------------------------
def get_season_starters(season: int, min_starts: int = 15) -> list:
    """
    Return a list of (pitcher_id, name) for pitchers with min_starts or
    more in the given season.

    Uses the /stats endpoint with group=pitching and leaderCategories to
    find qualifying starters.
    """
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
            starts = s.get("gamesStarted", 0)
            if starts < min_starts:
                continue
            player = split.get("player", {})
            pid = player.get("id")
            name = player.get("fullName")
            if pid and name:
                # Also pull season ABs + hits to compute career AABBH
                # proxy for stratification.
                ab = s.get("atBats", 0)
                h = s.get("hits", 0)
                aabbh = ab / h if h > 0 else None
                starters.append({
                    "pitcher_id": pid,
                    "name": name,
                    "season": season,
                    "starts": starts,
                    "season_aabbh": aabbh,
                })
    return starters


def stratified_select(starters: list, n_elite: int, n_avg: int, n_below: int) -> list:
    """
    Select pitchers stratified by season AABBH tier. Tier definitions:
      elite:    aabbh >= 4.3 (hard to hit — 1 hit per ~4.3+ ABs)
      average:  3.7 <= aabbh < 4.3
      below:    aabbh < 3.7

    Note: higher AABBH = harder to hit for a pitcher (fewer hits per AB).
    So 'elite' means HIGH AABBH, 'below-average' means LOW AABBH.
    """
    # Dedupe by pitcher_id, keeping the most recent season's data
    by_pid = {}
    for s in starters:
        pid = s["pitcher_id"]
        if pid not in by_pid or s["season"] > by_pid[pid]["season"]:
            by_pid[pid] = s
    unique = list(by_pid.values())

    elite = [s for s in unique if s["season_aabbh"] and s["season_aabbh"] >= 4.3]
    avg   = [s for s in unique if s["season_aabbh"] and 3.7 <= s["season_aabbh"] < 4.3]
    below = [s for s in unique if s["season_aabbh"] and s["season_aabbh"] < 3.7]

    # Sort each tier by starts descending — workhorses first for stable AABBH
    elite.sort(key=lambda s: s["starts"], reverse=True)
    avg.sort(key=lambda s: s["starts"], reverse=True)
    below.sort(key=lambda s: s["starts"], reverse=True)

    selected = elite[:n_elite] + avg[:n_avg] + below[:n_below]

    health(f"stratification_pool elite={len(elite)} avg={len(avg)} below={len(below)}")
    health(f"stratification_selected elite_taken={min(n_elite, len(elite))} "
           f"avg_taken={min(n_avg, len(avg))} below_taken={min(n_below, len(below))} "
           f"total={len(selected)}")

    return selected


# ---------------------------------------------------------------------------
# Core analysis: per-game aabbh_ratio with no leakage.
# ---------------------------------------------------------------------------
def analyze_pitcher_career(
    pitcher_id: int,
    name: str,
    seasons: list,
    bucket_totals: dict,
    pitcher_summary: list,
) -> None:
    """
    Pull one pitcher's full career across given seasons, compute
    per-game aabbh_ratio with no leakage, and increment global buckets.

    Mutates bucket_totals in place — each key is a bucket label, each
    value is {'games': int, 'ab_sum': int, 'hit_sum': int, 'hit_rate_sum': float}.

    Also appends a per-pitcher summary row to pitcher_summary with:
      - career AABBH (end of career window)
      - per-game AABBH values (for variance analysis)
      - per-game AABBH mean / stddev / min / max / iqr
      - per-pitcher aggregate hit rate allowed (for variance tier analysis)
    """
    all_games = []
    for season in seasons:
        games = fetch_pitcher_gamelog(pitcher_id, season)
        for g in games:
            g["season"] = season
        all_games.extend(games)

    if len(all_games) < 10:
        warn(f"insufficient_career_data pitcher={name} id={pitcher_id} "
             f"games={len(all_games)} — skipping")
        return

    all_games.sort(key=lambda g: g["date"])

    # Running totals — each game uses PRIOR totals only (no leakage).
    cumulative_ab = 0
    cumulative_hits = 0
    current_hitless_abs = 0
    games_bucketed = 0
    games_skipped_insufficient_history = 0

    per_pitcher_bucket_counts = defaultdict(int)
    career_aabbh_end = None

    # Per-game AABBH collection for variance analysis. Each game's "AABBH"
    # is just at_bats / hits for that single outing — the game's own hit
    # frequency. Games with zero hits allowed produce infinite AABBH so we
    # cap those at a sentinel value for stddev purposes.
    per_game_aabbh = []
    per_game_hit_rates = []  # for variance-tier hit-rate computation

    for g in all_games:
        ab = g["at_bats"]
        h = g["hits"]

        # To compute aabbh_ratio for THIS game, use cumulative state BEFORE this game.
        # Need enough history for the AABBH denominator to be stable — require
        # at least 50 career ABs faced before the ratio is computed.
        if cumulative_hits > 0 and cumulative_ab >= 50:
            career_aabbh = cumulative_ab / cumulative_hits
            aabbh_ratio = current_hitless_abs / career_aabbh if career_aabbh > 0 else 0

            # Find the right bucket for this ratio
            for label, lo, hi in BUCKETS:
                if lo <= aabbh_ratio < hi:
                    # Hit rate allowed in THIS game (the outcome we're measuring)
                    hit_rate = h / ab if ab > 0 else 0
                    bucket_totals[label]["games"] += 1
                    bucket_totals[label]["ab_sum"] += ab
                    bucket_totals[label]["hit_sum"] += h
                    bucket_totals[label]["hit_rate_sum"] += hit_rate
                    per_pitcher_bucket_counts[label] += 1
                    games_bucketed += 1
                    break
        else:
            games_skipped_insufficient_history += 1

        # Collect per-game AABBH for variance analysis. Only games with
        # meaningful workload (≥5 ABs faced, i.e. the pitcher actually
        # worked, not a 1-batter cameo). Shutout starts (0 hits) get a
        # capped value of 30.0 to represent "maximally hard to hit" without
        # polluting stddev with literal infinities.
        if ab >= 5:
            game_aabbh = ab / h if h > 0 else 30.0
            per_game_aabbh.append(game_aabbh)
            per_game_hit_rates.append(h / ab)

        # Update cumulative state AFTER bucketing (post-game)
        cumulative_ab += ab
        cumulative_hits += h
        if h > 0:
            # Got a hit — reset hitless streak, carry over unhit ABs from this game
            current_hitless_abs = max(0, ab - h)
        else:
            current_hitless_abs += ab

    if cumulative_hits > 0:
        career_aabbh_end = cumulative_ab / cumulative_hits

    # Variance stats — only meaningful if we have enough games.
    variance_stats = {
        "game_aabbh_mean": None,
        "game_aabbh_stddev": None,
        "game_aabbh_min": None,
        "game_aabbh_max": None,
        "game_aabbh_p25": None,
        "game_aabbh_p75": None,
        "aggregate_hit_rate_allowed": None,
        "games_measured": len(per_game_aabbh),
    }
    if len(per_game_aabbh) >= 10:
        sorted_vals = sorted(per_game_aabbh)
        variance_stats["game_aabbh_mean"] = round(statistics.mean(per_game_aabbh), 3)
        variance_stats["game_aabbh_stddev"] = round(statistics.stdev(per_game_aabbh), 3)
        variance_stats["game_aabbh_min"] = round(min(per_game_aabbh), 3)
        variance_stats["game_aabbh_max"] = round(max(per_game_aabbh), 3)
        # Percentiles via index — cheap and accurate enough for n ≥ 10
        p25_idx = max(0, int(len(sorted_vals) * 0.25))
        p75_idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * 0.75))
        variance_stats["game_aabbh_p25"] = round(sorted_vals[p25_idx], 3)
        variance_stats["game_aabbh_p75"] = round(sorted_vals[p75_idx], 3)
        # Aggregate hit rate for variance-tier analysis
        variance_stats["aggregate_hit_rate_allowed"] = round(
            statistics.mean(per_game_hit_rates), 4
        )

    pitcher_summary.append({
        "pitcher_id": pitcher_id,
        "name": name,
        "total_games": len(all_games),
        "games_bucketed": games_bucketed,
        "games_skipped": games_skipped_insufficient_history,
        "career_aabbh_end": round(career_aabbh_end, 3) if career_aabbh_end else None,
        "bucket_distribution": dict(per_pitcher_bucket_counts),
        **variance_stats,
    })

    stddev_str = (f"{variance_stats['game_aabbh_stddev']:.2f}"
                  if variance_stats['game_aabbh_stddev'] is not None else "n/a")
    health(f"pitcher_analyzed name=\"{name}\" id={pitcher_id} "
           f"games={len(all_games)} bucketed={games_bucketed} "
           f"career_aabbh={round(career_aabbh_end, 3) if career_aabbh_end else 'n/a'} "
           f"game_aabbh_stddev={stddev_str}")


# ---------------------------------------------------------------------------
# Running-results printer — shows bucket state mid-run.
# ---------------------------------------------------------------------------
def print_running_results(bucket_totals: dict, pitcher_num: int, total_pitchers: int):
    """Print current bucket hit rates as a mid-run snapshot."""
    total_games = sum(b["games"] for b in bucket_totals.values())
    if total_games == 0:
        return

    log("")
    log(f"--- Running results after pitcher {pitcher_num}/{total_pitchers} "
        f"({total_games:,} games bucketed) ---")
    log(f"{'Bucket':<10} {'Games':>7}  {'Avg Hit Rate':>14}  {'AB Sum':>9}  {'Hit Sum':>9}  {'Pool Rate':>10}")
    log("-" * 70)
    for label, _, _ in BUCKETS:
        b = bucket_totals[label]
        n = b["games"]
        if n == 0:
            log(f"{label:<10} {n:>7}  {'(no data)':>14}")
            continue
        # Two ways to compute the rate — average of per-game rates
        # (treats each start as equal), vs pooled (treats each AB as equal).
        # Report both for transparency; pooled is less noisy.
        avg_rate = b["hit_rate_sum"] / n
        pooled_rate = b["hit_sum"] / b["ab_sum"] if b["ab_sum"] > 0 else 0
        log(f"{label:<10} {n:>7}  {avg_rate*100:>13.2f}%  "
            f"{b['ab_sum']:>9,}  {b['hit_sum']:>9,}  {pooled_rate*100:>9.2f}%")


# ---------------------------------------------------------------------------
# Final report + hypothesis verdict
# ---------------------------------------------------------------------------
def print_final_report(bucket_totals: dict, pitcher_summary: list, elapsed: float):
    log("")
    log("=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)

    total_games = sum(b["games"] for b in bucket_totals.values())
    total_ab = sum(b["ab_sum"] for b in bucket_totals.values())
    total_hits = sum(b["hit_sum"] for b in bucket_totals.values())
    overall_rate = total_hits / total_ab if total_ab > 0 else 0

    log(f"Pitchers analyzed:       {len(pitcher_summary)}")
    log(f"Total games bucketed:    {total_games:,}")
    log(f"Total ABs faced:         {total_ab:,}")
    log(f"Total hits allowed:      {total_hits:,}")
    log(f"Overall pooled hit rate: {overall_rate*100:.2f}%   (baseline)")
    log(f"Analysis duration:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("")

    log("Hit rate by aabbh_ratio bucket:")
    log(f"{'Bucket':<10} {'Games':>7}  {'Pooled Rate':>12}  {'vs Baseline':>13}")
    log("-" * 55)

    bucket_pooled_rates = {}
    for label, _, _ in BUCKETS:
        b = bucket_totals[label]
        n = b["games"]
        if n == 0 or b["ab_sum"] == 0:
            log(f"{label:<10} {n:>7}  {'(no data)':>12}")
            bucket_pooled_rates[label] = None
            continue
        pooled_rate = b["hit_sum"] / b["ab_sum"]
        bucket_pooled_rates[label] = pooled_rate
        delta = (pooled_rate - overall_rate) * 100
        sign = "+" if delta >= 0 else ""
        log(f"{label:<10} {n:>7}  {pooled_rate*100:>11.2f}%  {sign}{delta:>11.2f}pp")

    # Hypothesis verdict
    log("")
    log("=" * 70)
    log("HYPOTHESIS VERDICT")
    log("=" * 70)

    valid_rates = {k: v for k, v in bucket_pooled_rates.items() if v is not None}
    if len(valid_rates) < 2:
        warn("insufficient_bucket_coverage — can't distinguish hypotheses")
        return

    # Hypothesis A: vulnerable zone is HIGH ratios (> 1.0)
    # Hypothesis B: vulnerable zone is LOW ratios (< 0.5)
    highest_bucket = max(valid_rates.items(), key=lambda kv: kv[1])
    lowest_bucket = min(valid_rates.items(), key=lambda kv: kv[1])

    log(f"Highest hit rate:  {highest_bucket[0]} at {highest_bucket[1]*100:.2f}%")
    log(f"Lowest hit rate:   {lowest_bucket[0]} at {lowest_bucket[1]*100:.2f}%")
    spread = (highest_bucket[1] - lowest_bucket[1]) * 100
    log(f"Spread:            {spread:.2f}pp")
    log("")

    if spread < 1.0:
        log("WEAK SIGNAL — bucket spread under 1pp. Pitcher aabbh_ratio may not")
        log("carry enough predictive signal to justify the feature. Consider")
        log("dropping pitcher_in_optimal_zone from Phase 2 and keeping only")
        log("the continuous pitcher_aabbh_ratio feature.")
        verdict = "WEAK"
    elif highest_bucket[0] in ("1.0-1.5", "1.5-2.0", "2.0+"):
        log("HYPOTHESIS A CONFIRMED — vulnerable zone is HIGH ratios.")
        log("Pitchers on long hitless-AB streaks regress toward the mean.")
        log("An overdue pitcher IS more hittable in their next start.")
        log(f"Recommended pitcher_in_optimal_zone threshold: {highest_bucket[0]}")
        verdict = "A"
    elif highest_bucket[0] in ("0.0-0.5", "0.5-1.0"):
        log("HYPOTHESIS B CONFIRMED — vulnerable zone is LOW ratios.")
        log("Pitchers who've been giving up hits recently continue to be")
        log("hit. This is the INVERSE of the batter side — a pitcher with a")
        log("short hitless-AB streak is NOT in the zone, they're struggling.")
        log(f"Recommended pitcher_in_optimal_zone threshold: {highest_bucket[0]}")
        log("IMPORTANT: the feature logic is INVERTED vs batter side.")
        verdict = "B"
    else:
        verdict = "MIXED"

    health(f"analysis_complete verdict={verdict} "
           f"highest_bucket={highest_bucket[0]} highest_rate={highest_bucket[1]*100:.2f} "
           f"lowest_bucket={lowest_bucket[0]} lowest_rate={lowest_bucket[1]*100:.2f} "
           f"spread_pp={spread:.2f}")

    # -----------------------------------------------------------------
    # Variance analysis — passive second question: does AABBH stddev
    # predict hit rate allowed independent of the ratio-bucket signal?
    # -----------------------------------------------------------------
    log("")
    log("=" * 70)
    log("VARIANCE ANALYSIS (passive — does per-game AABBH stddev carry signal?)")
    log("=" * 70)

    # Only pitchers with enough games to have a meaningful stddev
    measurable = [p for p in pitcher_summary
                  if p.get("game_aabbh_stddev") is not None]
    if len(measurable) < 6:
        warn(f"variance_analysis_insufficient n={len(measurable)} — "
             f"need ≥6 pitchers with stddev data, skipping variance report")
        return

    # Sort by stddev, split into three equal tiers: low / medium / high variance
    measurable.sort(key=lambda p: p["game_aabbh_stddev"])
    n = len(measurable)
    low_end = n // 3
    high_start = 2 * n // 3
    low_tier = measurable[:low_end]
    mid_tier = measurable[low_end:high_start]
    high_tier = measurable[high_start:]

    log(f"Pitchers with variance data: {n}")
    log(f"  Low-variance tier  (n={len(low_tier)}): "
        f"stddev {low_tier[0]['game_aabbh_stddev']:.2f} to "
        f"{low_tier[-1]['game_aabbh_stddev']:.2f}")
    log(f"  Mid-variance tier  (n={len(mid_tier)}): "
        f"stddev {mid_tier[0]['game_aabbh_stddev']:.2f} to "
        f"{mid_tier[-1]['game_aabbh_stddev']:.2f}")
    log(f"  High-variance tier (n={len(high_tier)}): "
        f"stddev {high_tier[0]['game_aabbh_stddev']:.2f} to "
        f"{high_tier[-1]['game_aabbh_stddev']:.2f}")
    log("")

    # Aggregate hit rate per tier. We already computed aggregate_hit_rate_allowed
    # per pitcher — mean that across the tier.
    def tier_stats(tier, label):
        rates = [p["aggregate_hit_rate_allowed"] for p in tier
                 if p.get("aggregate_hit_rate_allowed") is not None]
        if not rates:
            return None
        mean_rate = statistics.mean(rates)
        stddev_values = [p["game_aabbh_stddev"] for p in tier]
        mean_stddev = statistics.mean(stddev_values)
        career_aabbh_values = [p["career_aabbh_end"] for p in tier
                               if p.get("career_aabbh_end") is not None]
        mean_career_aabbh = statistics.mean(career_aabbh_values) if career_aabbh_values else None
        return {
            "label": label,
            "n": len(tier),
            "mean_stddev": mean_stddev,
            "mean_career_aabbh": mean_career_aabbh,
            "mean_hit_rate_allowed": mean_rate,
        }

    stats_low = tier_stats(low_tier, "Low-variance")
    stats_mid = tier_stats(mid_tier, "Mid-variance")
    stats_high = tier_stats(high_tier, "High-variance")

    log(f"{'Tier':<16} {'N':>4} {'Avg Stddev':>11} {'Avg AABBH':>11} {'Avg Hit Rate':>14}")
    log("-" * 65)
    for s in [stats_low, stats_mid, stats_high]:
        if s is None:
            continue
        aabbh_str = f"{s['mean_career_aabbh']:.2f}" if s['mean_career_aabbh'] else "n/a"
        log(f"{s['label']:<16} {s['n']:>4} {s['mean_stddev']:>11.2f} "
            f"{aabbh_str:>11} {s['mean_hit_rate_allowed']*100:>13.2f}%")

    # Interpretation
    if stats_low and stats_high:
        variance_spread = (stats_high["mean_hit_rate_allowed"]
                           - stats_low["mean_hit_rate_allowed"]) * 100
        log("")
        log(f"High-vs-low variance hit rate spread: {variance_spread:+.2f}pp")

        # Also check if the variance tiers are confounded by career AABBH.
        # If high-variance pitchers just happen to be worse pitchers (lower
        # career AABBH), the signal we're seeing is really career AABBH not
        # variance. Flag if there's a big AABBH gap between tiers.
        if stats_low["mean_career_aabbh"] and stats_high["mean_career_aabbh"]:
            aabbh_gap = stats_low["mean_career_aabbh"] - stats_high["mean_career_aabbh"]
            log(f"Career AABBH difference (low - high): {aabbh_gap:+.2f}")
            if abs(aabbh_gap) > 0.5:
                warn(f"variance_confounded_with_career_aabbh gap={aabbh_gap:+.2f} — "
                     f"high-variance and low-variance pitchers have very different "
                     f"mean skill levels. The variance signal may be skill signal "
                     f"in disguise. Follow-up: partial out career AABBH before "
                     f"concluding variance is a useful feature.")

        if abs(variance_spread) < 1.0:
            log("")
            log("VERDICT — Variance carries weak signal (<1pp spread).")
            log("Not worth adding pitcher_aabbh_stddev as a feature. The signal")
            log("in this data lives in the ratio axis, not the variance axis.")
            variance_verdict = "WEAK"
        elif variance_spread > 0:
            log("")
            log("VERDICT — HIGH variance correlates with HIGHER hit rate allowed.")
            log("Volatile pitchers give up more hits on average. This matches")
            log("Dan's hypothesis that variance matters on the pitcher side.")
            log(f"Recommended feature: pitcher_aabbh_game_stddev")
            log(f"Recommended interpretation: higher stddev = more hittable")
            variance_verdict = "HIGH_VARIANCE_MORE_HITTABLE"
        else:
            log("")
            log("VERDICT — LOW variance correlates with HIGHER hit rate allowed.")
            log("Consistent pitchers give up more hits on average. Counterintuitive.")
            log("Possible explanation: consistent but mediocre pitchers (knuckleballers,")
            log("junk-ballers) have tight distributions around a hittable mean, while")
            log("volatile power pitchers swing between unhittable and hit-hard.")
            variance_verdict = "LOW_VARIANCE_MORE_HITTABLE"

        health(f"variance_analysis_complete verdict={variance_verdict} "
               f"spread_pp={variance_spread:+.2f} n_pitchers={n}")

    # -----------------------------------------------------------------
    # De-confounded variance analysis (v2 addition).
    # Uses normalized stddev = game_aabbh_stddev / career_aabbh
    # (coefficient of variation). Removes the entanglement with pitcher
    # skill level that v1 flagged as a confound.
    # -----------------------------------------------------------------
    log("")
    log("=" * 70)
    log("DE-CONFOUNDED VARIANCE ANALYSIS (v2 — normalized by career AABBH)")
    log("=" * 70)
    log("Coefficient of variation = game_aabbh_stddev / career_aabbh_end")
    log("Measures: how volatile is this pitcher relative to their own mean?")
    log("")

    # Compute normalized stddev per pitcher, filter out those missing data
    normalized = []
    for p in pitcher_summary:
        if (p.get("game_aabbh_stddev") is not None
                and p.get("career_aabbh_end") is not None
                and p["career_aabbh_end"] > 0):
            p["normalized_stddev"] = p["game_aabbh_stddev"] / p["career_aabbh_end"]
            normalized.append(p)

    if len(normalized) < 6:
        warn(f"normalized_variance_insufficient n={len(normalized)} — "
             f"need >=6 pitchers with both stddev and career_aabbh, skipping")
    else:
        # Sort by normalized stddev, split into three equal tiers
        normalized.sort(key=lambda p: p["normalized_stddev"])
        nn = len(normalized)
        low_end_n = nn // 3
        high_start_n = 2 * nn // 3
        low_n = normalized[:low_end_n]
        mid_n = normalized[low_end_n:high_start_n]
        high_n = normalized[high_start_n:]

        log(f"Pitchers with normalized variance data: {nn}")
        log(f"  Low-CV tier  (n={len(low_n)}):  "
            f"CV {low_n[0]['normalized_stddev']:.2f} to "
            f"{low_n[-1]['normalized_stddev']:.2f}")
        log(f"  Mid-CV tier  (n={len(mid_n)}):  "
            f"CV {mid_n[0]['normalized_stddev']:.2f} to "
            f"{mid_n[-1]['normalized_stddev']:.2f}")
        log(f"  High-CV tier (n={len(high_n)}): "
            f"CV {high_n[0]['normalized_stddev']:.2f} to "
            f"{high_n[-1]['normalized_stddev']:.2f}")
        log("")

        def cv_tier_stats(tier, label):
            rates = [p["aggregate_hit_rate_allowed"] for p in tier
                     if p.get("aggregate_hit_rate_allowed") is not None]
            if not rates:
                return None
            return {
                "label": label,
                "n": len(tier),
                "mean_cv": statistics.mean([p["normalized_stddev"] for p in tier]),
                "mean_career_aabbh": statistics.mean(
                    [p["career_aabbh_end"] for p in tier
                     if p.get("career_aabbh_end") is not None]),
                "mean_hit_rate_allowed": statistics.mean(rates),
            }

        cv_low = cv_tier_stats(low_n, "Low-CV")
        cv_mid = cv_tier_stats(mid_n, "Mid-CV")
        cv_high = cv_tier_stats(high_n, "High-CV")

        log(f"{'Tier':<12} {'N':>4} {'Avg CV':>8} {'Avg AABBH':>11} "
            f"{'Avg Hit Rate':>14}")
        log("-" * 60)
        for s in [cv_low, cv_mid, cv_high]:
            if s is None:
                continue
            log(f"{s['label']:<12} {s['n']:>4} {s['mean_cv']:>8.3f} "
                f"{s['mean_career_aabbh']:>11.2f} "
                f"{s['mean_hit_rate_allowed']*100:>13.2f}%")

        if cv_low and cv_high:
            cv_spread = (cv_high["mean_hit_rate_allowed"]
                         - cv_low["mean_hit_rate_allowed"]) * 100
            aabbh_gap_cv = cv_low["mean_career_aabbh"] - cv_high["mean_career_aabbh"]

            log("")
            log(f"High-CV vs low-CV hit rate spread: {cv_spread:+.2f}pp")
            log(f"Career AABBH difference (low_cv - high_cv): {aabbh_gap_cv:+.2f}")

            # Is de-confounding actually working? If the AABBH gap between
            # CV tiers is close to zero, we've successfully controlled for
            # skill and any remaining spread is true variance effect.
            log("")
            if abs(aabbh_gap_cv) < 0.2:
                log("DE-CONFOUNDING SUCCESSFUL — CV tiers have near-identical")
                log(f"mean skill (AABBH gap {aabbh_gap_cv:+.2f} < 0.2). The")
                log(f"{cv_spread:+.2f}pp spread represents TRUE variance signal")
                log("independent of pitcher quality.")
                deconfound_ok = True
            elif abs(aabbh_gap_cv) < 0.4:
                log("DE-CONFOUNDING PARTIAL — CV tiers still have some skill")
                log(f"asymmetry (AABBH gap {aabbh_gap_cv:+.2f}). Signal is")
                log("meaningfully less confounded than raw variance but not fully clean.")
                deconfound_ok = True
            else:
                log("DE-CONFOUNDING FAILED — CV tiers still differ substantially")
                log(f"in mean skill (AABBH gap {aabbh_gap_cv:+.2f}). The")
                log("coefficient-of-variation normalization didn't resolve the")
                log("confound. Would need a residual-based approach (Option B).")
                deconfound_ok = False

            log("")
            if not deconfound_ok:
                cv_verdict = "CONFOUND_UNRESOLVED"
            elif abs(cv_spread) < 1.0:
                log("VERDICT — De-confounded variance has WEAK signal (<1pp).")
                log("Pitcher AABBH variance does NOT carry predictive power once")
                log("skill is controlled for. The v1 raw-variance effect was")
                log("confounded signal, not true signal. Do NOT add variance")
                log("features to Phase 2. Keep the feature set lean.")
                cv_verdict = "WEAK_AFTER_DECONFOUND"
            elif cv_spread > 0:
                log("VERDICT — HIGH normalized variance correlates with HIGHER")
                log("hit rate allowed, even after controlling for skill. This is")
                log("the original hypothesis confirmed in clean form: volatile")
                log("pitchers are genuinely more hittable on average.")
                log("Recommended Phase 2 feature: pitcher_aabbh_cv")
                log("Formula: stddev_of_per_game_aabbh / career_aabbh")
                cv_verdict = "HIGH_CV_MORE_HITTABLE"
            else:
                log("VERDICT — LOW normalized variance correlates with HIGHER")
                log("hit rate allowed, even after controlling for skill.")
                log("Consistent pitchers give up more hits. Counter to the original")
                log("hypothesis but de-confounded so this is a real effect.")
                log("Recommended Phase 2 feature: pitcher_aabbh_cv (inverted)")
                log("Lower CV value = higher expected hit rate.")
                cv_verdict = "LOW_CV_MORE_HITTABLE"

            health(f"deconfounded_variance_complete verdict={cv_verdict} "
                   f"spread_pp={cv_spread:+.2f} aabbh_gap={aabbh_gap_cv:+.2f} "
                   f"n_pitchers={nn}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=[2020, 2021, 2022, 2023, 2024],
                        help="Seasons to analyze")
    parser.add_argument("--n-elite", type=int, default=15,
                        help="Number of elite pitchers to select (AABBH >= 4.3)")
    parser.add_argument("--n-avg", type=int, default=20,
                        help="Number of average pitchers (3.7 <= AABBH < 4.3)")
    parser.add_argument("--n-below", type=int, default=15,
                        help="Number of below-average pitchers (AABBH < 3.7)")
    parser.add_argument("--max-pitchers", type=int, default=None,
                        help="Hard cap on total pitchers (for faster testing)")
    parser.add_argument("--min-starts", type=int, default=15,
                        help="Minimum starts per season to be selectable")
    parser.add_argument("--selection-season", type=int, default=2024,
                        help="Season to query for the pitcher selection pool")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write full results as JSON to this path")
    parser.add_argument("--progress-every", type=int, default=5,
                        help="Print running results every N pitchers")
    args = parser.parse_args()

    run_start = time.time()
    log("=" * 70)
    log("PITCHER AABBH OPTIMAL ZONE DISCOVERY")
    log("=" * 70)
    log(f"Seasons:            {args.seasons}")
    log(f"Target selection:   {args.n_elite} elite + {args.n_avg} avg + "
        f"{args.n_below} below-avg = {args.n_elite + args.n_avg + args.n_below} pitchers")
    if args.max_pitchers:
        log(f"Hard cap:           {args.max_pitchers} pitchers")
    log(f"Min starts/season:  {args.min_starts}")
    log(f"Selection pool:     {args.selection_season} season leaderboard")
    log("")

    # Step 1: select pitchers
    log("-" * 70)
    log(f"STEP 1 — Select starters from {args.selection_season}")
    log("-" * 70)
    pool = get_season_starters(args.selection_season, min_starts=args.min_starts)
    log(f"Candidate pool: {len(pool)} starters with ≥{args.min_starts} starts")
    health(f"selection_pool_size n={len(pool)} season={args.selection_season}")

    if len(pool) < 30:
        warn(f"selection_pool_small n={len(pool)} — stratification may be weak")

    selected = stratified_select(pool, args.n_elite, args.n_avg, args.n_below)
    if args.max_pitchers:
        selected = selected[:args.max_pitchers]

    log(f"Selected: {len(selected)} pitchers for analysis")
    log("")
    log("Pitcher list:")
    for i, p in enumerate(selected, 1):
        aabbh = p.get("season_aabbh")
        log(f"  {i:>2}. {p['name']:<28} "
            f"aabbh={aabbh:.2f}" if aabbh else f"  {i:>2}. {p['name']}")
    log("")

    # Step 2: ETA estimate
    est_games_per_pitcher = len(args.seasons) * 28
    est_api_calls = len(selected) * len(args.seasons)
    est_seconds = est_api_calls * RATE_LIMIT_DELAY + est_api_calls * 0.3  # API latency
    log(f"Estimated API calls: {est_api_calls}")
    log(f"Estimated runtime:   {est_seconds:.0f}s ({est_seconds/60:.1f} min)")
    log("")

    # Step 3: career analysis loop with running reports
    log("-" * 70)
    log(f"STEP 2 — Career analysis across {len(args.seasons)} seasons")
    log("-" * 70)
    bucket_totals = {label: {"games": 0, "ab_sum": 0, "hit_sum": 0, "hit_rate_sum": 0.0}
                     for label, _, _ in BUCKETS}
    pitcher_summary = []

    for i, p in enumerate(selected, 1):
        pitcher_start = time.time()
        log("")
        log(f"[{i}/{len(selected)}] Analyzing {p['name']} (season_aabbh={p['season_aabbh']:.2f})")
        analyze_pitcher_career(
            p["pitcher_id"], p["name"], args.seasons,
            bucket_totals, pitcher_summary
        )
        pitcher_elapsed = time.time() - pitcher_start
        log(f"    ({pitcher_elapsed:.1f}s)")

        # Running results snapshot
        if i % args.progress_every == 0 or i == len(selected):
            print_running_results(bucket_totals, i, len(selected))

            # ETA update based on actual pace
            total_elapsed = time.time() - run_start
            avg_per_pitcher = total_elapsed / i
            remaining = (len(selected) - i) * avg_per_pitcher
            if remaining > 0:
                log(f"    ETA: {remaining:.0f}s ({remaining/60:.1f} min) remaining")

    # Final report
    elapsed = time.time() - run_start
    print_final_report(bucket_totals, pitcher_summary, elapsed)

    # Optional JSON output
    if args.output_json:
        payload = {
            "run_metadata": {
                "seasons": args.seasons,
                "pitchers_analyzed": len(pitcher_summary),
                "duration_sec": round(elapsed, 1),
            },
            "bucket_totals": bucket_totals,
            "pitcher_summary": pitcher_summary,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        log("")
        log(f"Full results written to: {args.output_json}")


if __name__ == "__main__":
    main()