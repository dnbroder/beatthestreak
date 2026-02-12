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

  PITCHER:
    - pitcher_era: Opposing starter's season ERA
    - pitcher_whip: Opposing starter's WHIP
    - pitcher_hand: R/L
    - pitcher_k_per_9: Strikeouts per 9 innings
    - pitcher_recent_era: ERA over last 3-5 starts

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

    # Current hit streak
    df["hit_streak"] = grouped.apply(
        _current_hit_streak, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

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
FEATURE_COLUMNS = [
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