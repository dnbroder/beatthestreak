"""
pitcher_features.py — Phase 2 pitcher-side feature computation.

This module is the single source of truth for every pitcher feature
added in Phase 2. It handles both:

  1. Training-time computation — called from feature_engineering.py when
     building the training dataset. Batch-caches one pitcher at a time
     across all seasons they appeared in.

  2. Scoring-time lookup — called from daily_predictions.py when scoring
     today's batters. Fetches features for the opposing pitcher in each
     game, with per-run in-memory caching to avoid re-fetching for every
     batter.

Features computed (11 total):

  AABBH family (8 features, from pitcher gameLog):
    pitcher_aabbh_career        — career ABs_faced / hits_allowed
    pitcher_aabbh_recent        — same, last 5 starts only
    pitcher_current_hitless_abs — consecutive ABs since last hit allowed
    pitcher_aabbh_ratio         — current_hitless_abs / career_aabbh
    pitcher_in_optimal_zone     — 1 if 0.5 <= ratio < 1.0 (INVERTED vs batter side)
    pitcher_aabbh_cv            — per-game AABBH stddev / career AABBH
    aabbh_interaction           — batter_aabbh_ratio × pitcher_aabbh_ratio
    expected_pa_against_starter — lineup_position / IP_per_start approximation

  SitCode family (3 features, from MLB Stats API statSplits):
    pitcher_late_inning_degradation = OPS(innings 7+) - OPS(innings 1-6)
    pitcher_pitch_count_fade        = OPS(pitches 76+) - OPS(first 75)
    pitcher_first_inning_disadvantage = OPS(1st inning) - OPS(innings 1-6)

EMPIRICAL PROVENANCE (so future-you knows why these exist):
  - Zone discovery v1 (38 pitchers, 5 seasons, 3,636 games) confirmed
    Hypothesis B on the ratio axis: vulnerable zone is LOW ratios
    (0.5-1.0 bucket at 25.94% hit rate vs 24.20% baseline, spread 3.19pp).
    The feature logic is INVERTED vs batter side — for pitchers, being
    "in the zone" means STRUGGLING, not dominating.
  - Zone discovery v2 (same sample, de-confounded variance) showed that
    after controlling for pitcher skill via coefficient of variation,
    low-CV pitchers allowed 2.09pp more hits than high-CV pitchers.
    Counterintuitive but real: consistent pitchers are reliably hittable.
  - Features probe (23 pitchers, 2024) validated all three sitCode
    features have (a) meaningful spread, (b) low redundancy (max |r|=0.31),
    (c) acceptable skill-confound independence (max |r|=0.51).
  - dr4/dr5 rest-day sitCodes returned empty on all 23 probe pitchers.
    Rest-days feature dropped — not available via statSplits endpoint.

Logging conventions:
  - [PITCHER_FEATURES] routine output
  - [PITCHER_FEATURES]   [HEALTH] grepable metrics
  - [PITCHER_FEATURES]   [HEALTH] WARNING anomalies
"""

import logging
import statistics
import time
from typing import Optional

import requests

logger = logging.getLogger("beatthestreak")


BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.25

# Career AABBH needs enough history to be stable before we compute ratios.
MIN_CAREER_ABS_FACED = 50

# SitCodes we query for the OPS-based splits.
SITCODE_INNINGS_1_6 = "ig01"
SITCODE_INNINGS_7_PLUS = "ig07"
SITCODE_FIRST_INNING = "i01"
SITCODE_FIRST_75_PITCHES = "pi000"
SITCODE_PITCHES_76_PLUS = "pi760"

# Default OPS values to impute when a split is missing. League average
# starter OPS against is ~.720 per recent seasons; use as the "no signal"
# baseline so the delta features (degradation, fade, disadvantage) drop
# to zero instead of NaN.
DEFAULT_OPS_ALLOWED = 0.720


def log(msg: str):
    logger.info(f"[PITCHER_FEATURES] {msg}")


def health(msg: str):
    logger.info(f"[PITCHER_FEATURES]   [HEALTH] {msg}")


def warn(msg: str):
    logger.warning(f"[PITCHER_FEATURES]   [HEALTH] WARNING: {msg}")


# ---------------------------------------------------------------------------
# Low-level API fetching. Kept separate from historical_data.py's helpers so
# this module can be used standalone (e.g. in backtest scripts that don't
# load the full production pipeline).
# ---------------------------------------------------------------------------
def _safe_get(url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    """GET with retry + rate limiting. Matches historical_data._safe_get."""
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


def _parse_ip(ip_value) -> float:
    """MLB IP notation: '6.2' = 6 full innings + 2 outs = 6.667 actual IP."""
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


def _parse_ops(val) -> Optional[float]:
    """OPS comes as a string; parse defensively with sanity bounds."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        # Sanity bounds: a pitcher's OPS allowed should be in [0.2, 1.5] in
        # any legitimate split. Outside this range is a data problem.
        if 0.1 < f < 2.0:
            return f
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Raw data fetchers — one per data source
# ---------------------------------------------------------------------------
def fetch_pitcher_gamelog(pitcher_id: int, season: int) -> list:
    """
    Return sorted list of START dicts for one pitcher-season.
    Relief appearances are filtered out — their workload profile is
    fundamentally different and would pollute AABBH calculations.

    Each dict: date, at_bats, hits, innings_pitched, season.
    """
    data = _safe_get(
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
                continue  # filter relief
            games.append({
                "date": split.get("date", ""),
                "season": season,
                "at_bats": s.get("atBats", 0),
                "hits": s.get("hits", 0),
                "innings_pitched": _parse_ip(s.get("inningsPitched", 0)),
                "batters_faced": s.get("battersFaced", 0),
            })

    return sorted(games, key=lambda g: g["date"])


def fetch_pitcher_sitcodes(pitcher_id: int, season: int) -> dict:
    """
    Fetch all five required sitCode splits for one pitcher-season in a single
    API call (the API supports comma-separated sitCodes). Returns a dict
    keyed by sitCode → stat dict.

    Empty dict if all fetches fail. Per-sitCode missing handled gracefully
    by the caller (compute_sitcode_features uses DEFAULT_OPS_ALLOWED).
    """
    all_codes = ",".join([
        SITCODE_INNINGS_1_6,
        SITCODE_INNINGS_7_PLUS,
        SITCODE_FIRST_INNING,
        SITCODE_FIRST_75_PITCHES,
        SITCODE_PITCHES_76_PLUS,
    ])
    data = _safe_get(
        f"{BASE_URL}/people/{pitcher_id}/stats",
        params={
            "stats": "statSplits",
            "group": "pitching",
            "season": season,
            "sitCodes": all_codes,
        }
    )
    if not data:
        return {}

    # MLB API returns splits in a single stats[0].splits list with each
    # split tagged by its sitCode in split.split.code.
    result = {}
    stats = data.get("stats", [])
    if not stats:
        return {}
    for split in stats[0].get("splits", []):
        code = split.get("split", {}).get("code")
        if code:
            result[code] = split.get("stat", {})
    return result


# ---------------------------------------------------------------------------
# Feature computation — AABBH family
# ---------------------------------------------------------------------------
def compute_aabbh_features_as_of_date(
    games_prior: list,
    as_of_date: str,
    batter_aabbh_ratio: Optional[float] = None,
    lineup_position: int = 5,
) -> dict:
    """
    Compute all 8 AABBH-family features for a pitcher AS OF a given date,
    using only games BEFORE that date. No leakage.

    Args:
        games_prior: pre-sorted list of all starts for this pitcher up to
                     and EXCLUDING the target date. Must be sorted by date.
        as_of_date:  ISO date string — used only for logging/debugging.
        batter_aabbh_ratio: the batter's aabbh_ratio on the same date,
                     needed for the interaction feature. Pass None if not
                     available (interaction → 0).
        lineup_position: 1-9, defaults to 5 (middle of order) if unknown.

    Returns:
        dict of 8 features. Missing values default to neutral/zero so the
        model never sees NaN.
    """
    # Filter games strictly BEFORE as_of_date
    prior = [g for g in games_prior if g["date"] < as_of_date]

    # Compute career totals
    career_ab = sum(g["at_bats"] for g in prior)
    career_hits = sum(g["hits"] for g in prior)
    career_ip = sum(g["innings_pitched"] for g in prior)
    n_starts = len(prior)

    # If we don't have enough history, return zeros/defaults for everything.
    # The model will still train/score — it just won't have meaningful
    # pitcher-side signal for this prediction.
    if career_ab < MIN_CAREER_ABS_FACED or career_hits == 0 or n_starts == 0:
        return {
            "pitcher_aabbh_career": 4.0,  # league-avg default
            "pitcher_aabbh_recent": 4.0,
            "pitcher_current_hitless_abs": 0,
            "pitcher_aabbh_ratio": 0.0,
            "pitcher_in_optimal_zone": 0,
            "pitcher_aabbh_cv": 0.25,  # neutral CV default
            "aabbh_interaction": 0.0,
            "expected_pa_against_starter": 2.5,  # league-avg PAs at pos 5
        }

    # Career AABBH
    aabbh_career = career_ab / career_hits

    # Recent AABBH — last 5 starts only
    recent_games = prior[-5:]
    recent_ab = sum(g["at_bats"] for g in recent_games)
    recent_hits = sum(g["hits"] for g in recent_games)
    if recent_hits > 0:
        aabbh_recent = recent_ab / recent_hits
    else:
        # 5 starts with zero hits is extremely rare — use career as fallback
        aabbh_recent = aabbh_career

    # Current hitless AB streak — walk backwards from most recent start
    current_hitless_abs = 0
    for g in reversed(prior):
        if g["hits"] > 0:
            # Add just the unhit ABs from this game, then stop
            current_hitless_abs += max(0, g["at_bats"] - g["hits"])
            break
        else:
            current_hitless_abs += g["at_bats"]

    # AABBH ratio — "how deep into a drought is this pitcher?"
    # Remember: for pitchers, LOW ratio (0.5-1.0) = struggling = MORE hittable.
    # This is INVERTED from the batter side where HIGH ratio = overdue = more likely to hit.
    aabbh_ratio = current_hitless_abs / aabbh_career if aabbh_career > 0 else 0.0

    # In optimal zone — INVERTED logic vs batter side
    # Zone discovery v1 confirmed vulnerable zone is 0.5-1.0 (25.94% hit rate).
    # For a batter facing a pitcher in this zone → MORE likely to get a hit.
    in_optimal_zone = 1 if 0.5 <= aabbh_ratio < 1.0 else 0

    # Per-game AABBH coefficient of variation
    # Zone discovery v2 confirmed LOW-CV pitchers are MORE hittable (-2.09pp
    # spread even after de-confounding for skill via normalization).
    # Formula: stddev_of_per_game_aabbh / career_aabbh
    per_game_aabbh = []
    for g in prior:
        if g["at_bats"] >= 5:  # meaningful workload only
            # Shutout starts get a cap of 30.0 so stddev isn't infinity
            game_aabbh = g["at_bats"] / g["hits"] if g["hits"] > 0 else 30.0
            per_game_aabbh.append(game_aabbh)

    if len(per_game_aabbh) >= 3:
        game_stddev = statistics.stdev(per_game_aabbh)
        aabbh_cv = game_stddev / aabbh_career
    else:
        aabbh_cv = 0.25  # neutral default

    # Interaction term: batter overdue × pitcher struggling = double signal
    if batter_aabbh_ratio is not None:
        aabbh_interaction = batter_aabbh_ratio * aabbh_ratio
    else:
        aabbh_interaction = 0.0

    # Expected plate appearances against this starter
    # Rough approximation: (lineup_position + (IP_per_start × 9 / 27))
    # In practice what matters is whether a top-order batter gets 3 vs 4
    # appearances vs a 7th-slot batter getting 2 vs 3. IP/start modulates
    # the cutoff.
    ip_per_start = career_ip / n_starts if n_starts > 0 else 6.0
    # A starter going 6 IP faces the top of the order 3 times and the
    # bottom 2 times. Approximate: if IP >= 6, pos 1-3 get 3 PA, pos 4-9 get 2-3.
    # Simplified linear model:
    total_batters_faced = ip_per_start * 3.1  # ~3.1 BF/IP for starters
    pa_for_this_slot = total_batters_faced / 9.0 if lineup_position >= 7 else \
                      min(4.0, total_batters_faced / 9.0 + 0.5)
    expected_pa = round(pa_for_this_slot, 2)

    return {
        "pitcher_aabbh_career": round(aabbh_career, 3),
        "pitcher_aabbh_recent": round(aabbh_recent, 3),
        "pitcher_current_hitless_abs": int(current_hitless_abs),
        "pitcher_aabbh_ratio": round(aabbh_ratio, 3),
        "pitcher_in_optimal_zone": int(in_optimal_zone),
        "pitcher_aabbh_cv": round(aabbh_cv, 3),
        "aabbh_interaction": round(aabbh_interaction, 3),
        "expected_pa_against_starter": expected_pa,
    }


# ---------------------------------------------------------------------------
# Feature computation — SitCode family
# ---------------------------------------------------------------------------
def compute_sitcode_features(sitcode_data: dict) -> dict:
    """
    Compute the 3 sitCode-based features from raw MLB Stats API split data.

    Args:
        sitcode_data: dict returned by fetch_pitcher_sitcodes() — keyed by
                      sitCode → stat dict.

    Returns:
        dict of 3 features. Missing splits default to DEFAULT_OPS_ALLOWED
        on both sides of the delta so features drop to 0 rather than NaN.
    """
    def get_ops(code):
        stat = sitcode_data.get(code, {})
        parsed = _parse_ops(stat.get("ops"))
        return parsed if parsed is not None else DEFAULT_OPS_ALLOWED

    ops_innings_1_6 = get_ops(SITCODE_INNINGS_1_6)
    ops_innings_7_plus = get_ops(SITCODE_INNINGS_7_PLUS)
    ops_first_inning = get_ops(SITCODE_FIRST_INNING)
    ops_first_75 = get_ops(SITCODE_FIRST_75_PITCHES)
    ops_76_plus = get_ops(SITCODE_PITCHES_76_PLUS)

    late_inning_degradation = ops_innings_7_plus - ops_innings_1_6
    pitch_count_fade = ops_76_plus - ops_first_75
    first_inning_disadvantage = ops_first_inning - ops_innings_1_6

    return {
        "pitcher_late_inning_degradation": round(late_inning_degradation, 4),
        "pitcher_pitch_count_fade": round(pitch_count_fade, 4),
        "pitcher_first_inning_disadvantage": round(first_inning_disadvantage, 4),
    }


# ---------------------------------------------------------------------------
# Unified accessor — one call to rule them all
# ---------------------------------------------------------------------------
def get_all_pitcher_features(
    pitcher_id: int,
    season: int,
    as_of_date: str,
    batter_aabbh_ratio: Optional[float] = None,
    lineup_position: int = 5,
    gamelog_cache: Optional[dict] = None,
    sitcode_cache: Optional[dict] = None,
) -> dict:
    """
    One-stop shop: given a pitcher ID, season, and date, returns ALL 11
    Phase 2 pitcher features.

    Caching:
        gamelog_cache and sitcode_cache are optional dicts that the caller
        can provide to avoid re-fetching data. This is critical for the
        daily prediction pipeline where the same opposing pitcher appears
        for every batter in a lineup (9 batters × 1 pitcher = 1 fetch
        instead of 9).

        Cache key format:
          gamelog_cache: (pitcher_id, season) -> list of games
          sitcode_cache: (pitcher_id, season) -> sitcode dict

    Args:
        pitcher_id: MLB Stats API player ID
        season: MLB season year
        as_of_date: ISO date — only games BEFORE this date count
        batter_aabbh_ratio: for interaction feature, None if unknown
        lineup_position: 1-9, defaults to 5
        gamelog_cache, sitcode_cache: optional per-run caches

    Returns:
        Dict of 11 features (8 AABBH + 3 sitCode).
    """
    if pitcher_id is None:
        # No pitcher identified (e.g. TBD starter, API gap) — return defaults
        warn(f"no_pitcher_id as_of_date={as_of_date} — using default features")
        return _default_pitcher_features()

    # Fetch or cache pitcher gamelog (for AABBH features)
    cache_key = (pitcher_id, season)
    if gamelog_cache is not None and cache_key in gamelog_cache:
        games = gamelog_cache[cache_key]
    else:
        games = fetch_pitcher_gamelog(pitcher_id, season)
        # ALSO pull prior season for continuity — early-season predictions
        # often don't have enough current-season starts for a stable AABBH.
        prior_season_games = fetch_pitcher_gamelog(pitcher_id, season - 1)
        games = prior_season_games + games
        if gamelog_cache is not None:
            gamelog_cache[cache_key] = games

    # Compute AABBH family
    aabbh_features = compute_aabbh_features_as_of_date(
        games, as_of_date, batter_aabbh_ratio, lineup_position
    )

    # Fetch or cache sitCode splits (for OPS-based features)
    if sitcode_cache is not None and cache_key in sitcode_cache:
        sitcodes = sitcode_cache[cache_key]
    else:
        sitcodes = fetch_pitcher_sitcodes(pitcher_id, season)
        if sitcode_cache is not None:
            sitcode_cache[cache_key] = sitcodes

    # Compute sitCode family
    sitcode_features = compute_sitcode_features(sitcodes)

    # Merge and return
    all_features = {**aabbh_features, **sitcode_features}

    # Logging — one per-pitcher line for traceability, grepable
    health(f"pitcher_features_computed pitcher_id={pitcher_id} season={season} "
           f"as_of={as_of_date} career_aabbh={aabbh_features['pitcher_aabbh_career']} "
           f"ratio={aabbh_features['pitcher_aabbh_ratio']} "
           f"in_zone={aabbh_features['pitcher_in_optimal_zone']} "
           f"cv={aabbh_features['pitcher_aabbh_cv']} "
           f"late_inn={sitcode_features['pitcher_late_inning_degradation']} "
           f"fade={sitcode_features['pitcher_pitch_count_fade']} "
           f"first_inn={sitcode_features['pitcher_first_inning_disadvantage']}")

    return all_features


def _default_pitcher_features() -> dict:
    """Neutral defaults when we have no pitcher data at all."""
    return {
        "pitcher_aabbh_career": 4.0,
        "pitcher_aabbh_recent": 4.0,
        "pitcher_current_hitless_abs": 0,
        "pitcher_aabbh_ratio": 0.0,
        "pitcher_in_optimal_zone": 0,
        "pitcher_aabbh_cv": 0.25,
        "aabbh_interaction": 0.0,
        "expected_pa_against_starter": 2.5,
        "pitcher_late_inning_degradation": 0.0,
        "pitcher_pitch_count_fade": 0.0,
        "pitcher_first_inning_disadvantage": 0.0,
    }


# ---------------------------------------------------------------------------
# Phase 2 feature column names — importable from other modules to keep
# the list consistent across training and scoring.
# ---------------------------------------------------------------------------
PHASE2_PITCHER_FEATURE_COLUMNS = [
    "pitcher_aabbh_career",
    "pitcher_aabbh_recent",
    "pitcher_current_hitless_abs",
    "pitcher_aabbh_ratio",
    "pitcher_in_optimal_zone",
    "pitcher_aabbh_cv",
    "aabbh_interaction",
    "expected_pa_against_starter",
    "pitcher_late_inning_degradation",
    "pitcher_pitch_count_fade",
    "pitcher_first_inning_disadvantage",
]