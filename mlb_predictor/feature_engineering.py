"""
feature_engineering.py – Transform raw game logs into model-ready features.

Features:
  BATTER:
    - batting_avg_season: Season batting average up to this game
    - avg_last_7 / avg_last_14 / avg_last_30: Rolling hit averages
    - k_rate: Strikeout rate (season to date)
    - bb_rate: Walk rate (season to date)
    - multi_hit_freq: % of games with 2+ hits (season to date)
    - avg_abs_per_game: Average at-bats per game
    - lineup_position: Batting order slot (1-9)
    - bat_side: R/L/S
    - games_between_hits: Average games between hitless games (secret sauce)
    - hit_streak: Current consecutive games with a hit

  AABBH (secret sauce v2):
    - aabbh: Average At Bats Between Hits
    - current_hitless_abs: Consecutive ABs since last hit
    - aabbh_ratio: current_hitless_abs / aabbh (overdue signal)
    - in_optimal_zone: 1 if aabbh_ratio is in the 0.5-1.0 sweet spot (81.8% hit rate)

  ERA TIER (v1.2):
    - batter_hit_rate_vs_era_tier: hit rate vs today's pitcher's ERA tier
    - batter_ab_vs_era_tier: sample size for that tier
    - era_tier_delta: tier-specific rate minus overall rate (specialist signal)

  PITCHER (basic):
    - pitcher_era: Opposing starter's season ERA
    - pitcher_whip: Opposing starter's WHIP
    - pitcher_hand: R/L
    - pitcher_k_per_9: Strikeouts per 9 innings
    - pitcher_recent_era: ERA over last 3-5 starts

  PHASE 2 PITCHER (v2.0 — 11 new features):
    AABBH family:
      - pitcher_aabbh_career: Career ABs faced / hits allowed (higher = harder to hit)
      - pitcher_aabbh_recent: Same, last 5 starts only
      - pitcher_current_hitless_abs: ABs allowed since last hit
      - pitcher_aabbh_ratio: current_hitless_abs / career_aabbh
      - pitcher_in_optimal_zone: 1 if 0.5 <= ratio < 1.0 (INVERTED from batter)
      - pitcher_aabbh_cv: per-game AABBH stddev / career_aabbh (LOW = more hittable)
      - aabbh_interaction: batter_aabbh_ratio × pitcher_aabbh_ratio
      - expected_pa_against_starter: lineup_pos × recent IP/start
    SitCode family:
      - pitcher_late_inning_degradation: OPS(innings 7+) - OPS(innings 1-6)
      - pitcher_pitch_count_fade: OPS(pitches 76+) - OPS(first 75)
      - pitcher_first_inning_disadvantage: OPS(1st inn) - OPS(innings 1-6)

  MATCHUP:
    - platoon_advantage: 1 if batter has platoon advantage, 0 otherwise
    - batter_vs_pitcher_avg: Career avg vs this pitcher (if >= 5 AB)
    - batter_vs_pitcher_abs: Sample size of matchup

  ENVIRONMENT:
    - is_home: 1/0
    - is_night: 1/0
    - park_factor: Venue run factor
    - temperature: Game-time temp (when available)

  TARGET:
    - got_hit: 1 if hits >= 1, else 0

  WEIGHT:
    - sample_weight: Recency weight for training
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("beatthestreak")


# ---------------------------------------------------------------------------
# Rolling / cumulative stat helpers
# ---------------------------------------------------------------------------
def _rolling_avg(group: pd.DataFrame, window: int, col: str = "hits", min_periods: int = 1) -> pd.Series:
    """
    Compute a rolling average within a player-season group.
    Uses shift(1) so we only use PRIOR games (no data leakage).
    """
    return (
        group[col]
        .shift(1)
        .rolling(window=window, min_periods=min_periods)
        .mean()
    )


def _cumulative_rate(group: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """
    Compute cumulative rate (e.g., K rate = cumulative Ks / cumulative ABs).
    Uses shift(1) to avoid leakage.
    """
    cum_num = group[numerator].shift(1).cumsum()
    cum_den = group[denominator].shift(1).cumsum()
    return (cum_num / cum_den).replace([np.inf, -np.inf], np.nan)


def _cumulative_avg(group: pd.DataFrame) -> pd.Series:
    """Season-to-date batting average using prior games only."""
    cum_hits = group["hits"].shift(1).cumsum()
    cum_abs = group["at_bats"].shift(1).cumsum()
    return (cum_hits / cum_abs).replace([np.inf, -np.inf], np.nan)


def _multi_hit_frequency(group: pd.DataFrame) -> pd.Series:
    """Fraction of prior games with 2+ hits."""
    multi = (group["hits"] >= 2).astype(int).shift(1)
    return multi.expanding(min_periods=1).mean()


def _avg_abs_per_game(group: pd.DataFrame) -> pd.Series:
    """Average at-bats per game using prior games."""
    return group["at_bats"].shift(1).expanding(min_periods=1).mean()


def _games_between_hits(group: pd.DataFrame) -> pd.Series:
    """
    Secret sauce: average number of games between hitless games.
    Higher = more consistent hitter (rarely goes hitless).
    Computed as: games_played / (hitless_games + 1)

    A guy who plays 30 games and only goes hitless 3 times = 30/4 = 7.5
    A guy who plays 30 games and goes hitless 15 times = 30/16 = 1.875
    """
    got_hit = (group["hits"] >= 1).astype(int).shift(1)
    games_played = got_hit.expanding(min_periods=1).count()
    hitless_games = (1 - got_hit).expanding(min_periods=1).sum()
    return games_played / (hitless_games + 1)


def _aabbh(group: pd.DataFrame) -> pd.Series:
    """
    AABBH: Average At Bats Between Hits.
    Total ABs / Total Hits using prior games only.
    Lower = more frequent hitter (gets a hit every N at-bats).

    A .333 hitter has AABBH of ~3.0 (hit every 3 ABs)
    A .250 hitter has AABBH of ~4.0 (hit every 4 ABs)
    """
    cum_abs = group["at_bats"].shift(1).cumsum()
    cum_hits = group["hits"].shift(1).cumsum()
    return (cum_abs / cum_hits).replace([np.inf, -np.inf], np.nan)


def _current_hitless_abs(group: pd.DataFrame) -> pd.Series:
    """
    Count consecutive at-bats without a hit heading into this game.
    Resets to 0 after any game with a hit.

    Uses shift(1) so we're counting the drought BEFORE this game.
    """
    hits = group["hits"].shift(1).fillna(0)
    abs_ = group["at_bats"].shift(1).fillna(0)

    hitless_abs = []
    running_count = 0
    for h, ab in zip(hits, abs_):
        if h > 0:
            # Got a hit last game — reset, but carry over unhit ABs
            # e.g., 1-for-4 means 3 ABs after the last hit
            running_count = int(ab - h)
        else:
            # Hitless game — add all ABs to the drought
            running_count += int(ab)
        hitless_abs.append(running_count)

    return pd.Series(hitless_abs, index=group.index)


def _aabbh_ratio(aabbh: pd.Series, current_hitless_abs: pd.Series) -> pd.Series:
    """
    AABBH Ratio: current_hitless_abs / AABBH.

    < 1.0 = within normal rhythm
    = 1.0 = right at their average
    > 1.0 = overdue for a hit (regression to mean candidate)

    Higher ratio = stronger BTS pick signal.
    """
    return (current_hitless_abs / aabbh).replace([np.inf, -np.inf], np.nan)


def _in_optimal_zone(aabbh_ratio: pd.Series) -> pd.Series:
    """
    Optimal AABBH zone indicator.

    Backtest analysis (160 picks across 18 dates) showed:
      - aabbh_ratio 0.5 - 1.0: 81.8% hit rate (n=77)
      - aabbh_ratio 0.0 - 0.5: 75.9% hit rate (n=29)
      - aabbh_ratio 1.0 - 1.5: 73.7% hit rate (n=19)
      - aabbh_ratio 3.0 - 4.0: 55.6% hit rate (n=9)

    The sweet spot is 0.5-1.0: the player is "due but not overdue."
    Their hitless drought is between half and equal to their normal rhythm.

    Returns 1 if in optimal zone, 0 otherwise.
    """
    return ((aabbh_ratio >= 0.5) & (aabbh_ratio < 1.0)).astype(int)


def _current_hit_streak(group: pd.DataFrame) -> pd.Series:
    """
    Current consecutive games with at least 1 hit.
    Uses shift(1) — streak heading INTO this game.
    """
    got_hit = (group["hits"] >= 1).astype(int).shift(1).fillna(0)

    streaks = []
    current_streak = 0
    for h in got_hit:
        if h == 1:
            current_streak += 1
        else:
            current_streak = 0
        streaks.append(current_streak)

    return pd.Series(streaks, index=group.index)


# ---------------------------------------------------------------------------
# ERA Tier features (v1.2)
# ---------------------------------------------------------------------------
def _era_tier_features_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ERA tier features for a player-season group.

    For each game, uses only PRIOR games to compute the batter's hit rate
    against the ERA tier of the current game's opposing pitcher.
    No data leakage — strictly lookback.

    Requires 'pitcher_era' column in the group (opposing pitcher ERA,
    already present from historical_data.py enrichment).
    """
    from mlb_predictor.era_tier_features import classify_era_tier

    hit_rates = []
    ab_counts = []
    deltas = []

    for i in range(len(group)):
        if i < 3:
            # Not enough prior games for meaningful tier stats
            hit_rates.append(0.0)
            ab_counts.append(0)
            deltas.append(0.0)
            continue

        # Prior games only (no leakage)
        prior = group.iloc[:i]

        # Current game's opposing pitcher ERA
        opp_era = group.iloc[i].get("pitcher_era")
        if opp_era is None or (isinstance(opp_era, float) and np.isnan(opp_era)):
            hit_rates.append(0.0)
            ab_counts.append(0)
            deltas.append(0.0)
            continue

        target_tier = classify_era_tier(opp_era)

        # Classify prior games' opposing pitchers into tiers
        prior_tiers = prior["pitcher_era"].apply(
            lambda x: classify_era_tier(x) if pd.notna(x) else "average"
        )

        # Overall stats from prior games
        total_abs = prior["at_bats"].sum()
        total_hits = prior["hits"].sum()
        overall_rate = total_hits / max(total_abs, 1)

        # Tier-specific stats from prior games
        tier_mask = prior_tiers == target_tier
        tier_abs = prior.loc[tier_mask, "at_bats"].sum()
        tier_hits = prior.loc[tier_mask, "hits"].sum()

        if tier_abs > 0:
            tier_rate = tier_hits / tier_abs
            delta = tier_rate - overall_rate
        else:
            # No ABs against this tier yet — fall back to overall
            tier_rate = overall_rate
            delta = 0.0

        hit_rates.append(round(tier_rate, 4))
        ab_counts.append(int(tier_abs))
        deltas.append(round(delta, 4))

    return pd.DataFrame({
        "batter_hit_rate_vs_era_tier": hit_rates,
        "batter_ab_vs_era_tier": ab_counts,
        "era_tier_delta": deltas,
    }, index=group.index)


# ---------------------------------------------------------------------------
# Platoon advantage
# ---------------------------------------------------------------------------
def compute_platoon_advantage(bat_side: str, pitcher_hand: str) -> int:
    """
    1 if batter has platoon advantage:
      - Righty batter vs Lefty pitcher
      - Lefty batter vs Righty pitcher
      - Switch hitter always gets 1 (they pick the favorable side)
    """
    if bat_side == "S":
        return 1
    if bat_side == "R" and pitcher_hand == "L":
        return 1
    if bat_side == "L" and pitcher_hand == "R":
        return 1
    return 0


# ---------------------------------------------------------------------------
# Recency weights
# ---------------------------------------------------------------------------
def compute_sample_weight(season: int) -> float:
    """
    Assign training weight based on season recency.
      2020-2024: weight 1.0  (full weight, current era)
      2015-2019: weight 0.7  (moderate)
      2010-2014: weight 0.4  (lighter)
      2000-2009: weight 0.2  (historical context)
    """
    if season >= 2020:
        return 1.0
    elif season >= 2015:
        return 0.7
    elif season >= 2010:
        return 0.4
    else:
        return 0.2


# ---------------------------------------------------------------------------
# Main feature engineering pipeline
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw game log DataFrame and add all engineered features.
    Input must have columns from historical_data.build_season_game_logs().

    Returns DataFrame ready for model training.

    Phase 2 note: if the input df already has Phase 2 pitcher feature
    columns (from enrich_pitcher_features=True in historical_data.py),
    this function will also recompute aabbh_interaction now that batter
    aabbh_ratio is known. See "Phase 2 interaction recompute" below.
    """
    logger.info(f"Engineering features for {len(df)} game log rows...")

    if df.empty:
        return df

    # Ensure proper types and sorting
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["player_id", "season", "date"]).reset_index(drop=True)

    # ----- Target variable -----
    df["got_hit"] = (df["hits"] >= 1).astype(int)

    # ----- Per-player-season features (rolling/cumulative) -----
    logger.info("Computing rolling and cumulative stats...")

    grouped = df.groupby(["player_id", "season"])

    # Rolling averages (hits per game as proxy — will refine)
    df["avg_last_7"] = grouped.apply(
        lambda g: _rolling_avg(g, 7, "hits"), include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    df["avg_last_14"] = grouped.apply(
        lambda g: _rolling_avg(g, 14, "hits"), include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    df["avg_last_30"] = grouped.apply(
        lambda g: _rolling_avg(g, 30, "hits"), include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Season-to-date batting average
    df["batting_avg_season"] = grouped.apply(
        _cumulative_avg, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Strikeout rate
    df["k_rate"] = grouped.apply(
        lambda g: _cumulative_rate(g, "strikeouts", "at_bats"), include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Walk rate
    df["bb_rate"] = grouped.apply(
        lambda g: _cumulative_rate(g, "walks", "plate_appearances"), include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Multi-hit frequency
    df["multi_hit_freq"] = grouped.apply(
        _multi_hit_frequency, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Average ABs per game
    df["avg_abs_per_game"] = grouped.apply(
        _avg_abs_per_game, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Secret sauce: games between hits
    df["games_between_hits"] = grouped.apply(
        _games_between_hits, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Secret sauce v2: AABBH (Average At Bats Between Hits)
    df["aabbh"] = grouped.apply(
        _aabbh, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # Current hitless AB streak
    df["current_hitless_abs"] = grouped.apply(
        _current_hitless_abs, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # AABBH Ratio: are they overdue? (> 1.0 = overdue for a hit)
    df["aabbh_ratio"] = _aabbh_ratio(df["aabbh"], df["current_hitless_abs"])

    # Optimal zone indicator (0.5-1.0 aabbh_ratio = 81.8% hit rate)
    df["in_optimal_zone"] = _in_optimal_zone(df["aabbh_ratio"])

    # Current hit streak
    df["hit_streak"] = grouped.apply(
        _current_hit_streak, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # ----- ERA Tier features (v1.2) -----
    if "pitcher_era" in df.columns:
        logger.info("Computing ERA tier features...")
        era_tier_results = grouped.apply(
            _era_tier_features_for_group, include_groups=False
        ).reset_index(level=[0, 1], drop=True)

        df["batter_hit_rate_vs_era_tier"] = era_tier_results["batter_hit_rate_vs_era_tier"]
        df["batter_ab_vs_era_tier"] = era_tier_results["batter_ab_vs_era_tier"]
        df["era_tier_delta"] = era_tier_results["era_tier_delta"]
    else:
        logger.warning("pitcher_era not in dataset — skipping ERA tier features")
        df["batter_hit_rate_vs_era_tier"] = 0.0
        df["batter_ab_vs_era_tier"] = 0
        df["era_tier_delta"] = 0.0

    # ----- Phase 2 interaction recompute -----
    # historical_data.py computes aabbh_interaction with batter_aabbh_ratio=None
    # (it isn't known until this stage). Recompute now that we have both sides.
    if "pitcher_aabbh_ratio" in df.columns:
        logger.info("[PHASE2] Recomputing aabbh_interaction now that batter "
                    "aabbh_ratio is available...")
        df["aabbh_interaction"] = (
            df["aabbh_ratio"].fillna(0) * df["pitcher_aabbh_ratio"].fillna(0)
        ).round(3)

        # Phase 2 sanity logging: report feature presence and null rates
        phase2_cols = [
            "pitcher_aabbh_career", "pitcher_aabbh_recent",
            "pitcher_current_hitless_abs", "pitcher_aabbh_ratio",
            "pitcher_in_optimal_zone", "pitcher_aabbh_cv",
            "aabbh_interaction", "expected_pa_against_starter",
            "pitcher_late_inning_degradation", "pitcher_pitch_count_fade",
            "pitcher_first_inning_disadvantage",
        ]
        present_cols = [c for c in phase2_cols if c in df.columns]
        logger.info(f"[PHASE2]   [HEALTH] phase2_features_present "
                    f"count={len(present_cols)}/{len(phase2_cols)} "
                    f"cols={present_cols}")

        # Null rate per feature — high null rate means enrichment had gaps
        for col in present_cols:
            null_rate = df[col].isna().mean() * 100
            if null_rate > 10:
                logger.warning(f"[PHASE2]   [HEALTH] WARNING: high_null_rate "
                               f"feature={col} pct={null_rate:.1f}%")

    # ----- Binary / categorical features -----
    df["is_home"] = df["is_home"].astype(int)

    # Bat side encoding (R=0, L=1, S=2)
    bat_side_map = {"R": 0, "L": 1, "S": 2, "U": 0}
    df["bat_side_encoded"] = df["bat_side"].map(bat_side_map).fillna(0).astype(int)

    # Pitcher hand encoding (R=0, L=1, U=0)
    if "pitcher_hand" in df.columns:
        pitcher_hand_map = {"R": 0, "L": 1, "U": 0}
        df["pitcher_hand_encoded"] = df["pitcher_hand"].map(pitcher_hand_map).fillna(0).astype(int)

    # Ensure platoon_advantage exists (should come from data enrichment)
    if "platoon_advantage" not in df.columns:
        df["platoon_advantage"] = 0

    # Ensure park_factor exists
    if "park_factor" not in df.columns:
        df["park_factor"] = 1.0

    # Fill missing pitcher stats with league averages
    if "pitcher_era" in df.columns:
        df["pitcher_era"] = df["pitcher_era"].fillna(df["pitcher_era"].median())
    if "pitcher_whip" in df.columns:
        df["pitcher_whip"] = df["pitcher_whip"].fillna(df["pitcher_whip"].median())
    if "pitcher_k_per_9" in df.columns:
        df["pitcher_k_per_9"] = df["pitcher_k_per_9"].fillna(df["pitcher_k_per_9"].median())

    # Phase 2: fill missing pitcher-feature columns with sensible defaults
    # (not medians — defaults are domain-aware, e.g. aabbh_career=4.0)
    phase2_defaults = {
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
    for col, default_val in phase2_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)

    # ----- Recency weight -----
    df["sample_weight"] = df["season"].apply(compute_sample_weight)

    # ----- Drop first few games per player-season (insufficient history) -----
    game_num = grouped.cumcount() + 1
    df["game_num_in_season"] = game_num

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature columns for model training
# ---------------------------------------------------------------------------

# Baseline (v1.x) feature columns — pre-Phase 2
FEATURE_COLUMNS_V1 = [
    # Batter features
    "batting_avg_season",
    "avg_last_7",
    "avg_last_14",
    "avg_last_30",
    "k_rate",
    "bb_rate",
    "multi_hit_freq",
    "avg_abs_per_game",
    "bat_side_encoded",
    "games_between_hits",
    "hit_streak",
    "game_num_in_season",

    # AABBH features (secret sauce v2)
    "aabbh",
    "current_hitless_abs",
    "aabbh_ratio",
    "in_optimal_zone",

    # ERA Tier features (v1.2)
    "batter_hit_rate_vs_era_tier",
    "batter_ab_vs_era_tier",
    "era_tier_delta",

    # Matchup features
    "is_home",
    "platoon_advantage",

    # Pitcher features
    "pitcher_era",
    "pitcher_whip",
    "pitcher_k_per_9",
    "pitcher_hand_encoded",

    # Environment
    "park_factor",
]

# Phase 2 pitcher-side feature additions (11 new columns)
FEATURE_COLUMNS_PHASE2 = [
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

# Full Phase 2 feature list = v1 baseline + 11 new columns = 37 features
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS_V1 + FEATURE_COLUMNS_PHASE2

# Default FEATURE_COLUMNS points to v1 for backward compatibility with
# existing production code. Phase 2 training uses FEATURE_COLUMNS_V2 explicitly.
FEATURE_COLUMNS = FEATURE_COLUMNS_V1

# Columns we can train with even before pitcher/park enrichment
BASIC_FEATURE_COLUMNS = [
    "batting_avg_season",
    "avg_last_7",
    "avg_last_14",
    "avg_last_30",
    "k_rate",
    "bb_rate",
    "multi_hit_freq",
    "avg_abs_per_game",
    "bat_side_encoded",
    "games_between_hits",
    "hit_streak",
    "game_num_in_season",
    "is_home",
]

TARGET_COLUMN = "got_hit"
WEIGHT_COLUMN = "sample_weight"