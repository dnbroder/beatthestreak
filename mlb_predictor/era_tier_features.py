"""
era_tier_features.py – Batter performance by opposing pitcher ERA tier.

Adds three features:
  1. batter_hit_rate_vs_era_tier: batter's hit rate against the ERA tier
     of today's opposing pitcher (rolling, current season)
  2. batter_ab_vs_era_tier: sample size for that tier (model learns reliability)
  3. era_tier_delta: difference between tier-specific and overall hit rate
     (captures the "specialist" signal — positive = better against this tier)

ERA Tiers:
  - elite:        ERA < 3.00
  - good:         3.00 <= ERA < 3.75
  - average:      3.75 <= ERA < 4.50
  - below_avg:    ERA >= 4.50

The module provides:
  - classify_era_tier(): classify a pitcher ERA into a tier
  - compute_era_tier_features(): given a batter's game log with opposing
    pitcher ERAs, compute the three features for a given tier

Integration:
  - In historical_data.py: when building game logs, include opp_pitcher_era
    for each game row
  - In feature_engineering.py: call compute_era_tier_features() and add
    the 3 features to FEATURE_COLUMNS
  - In daily_predictions.py: compute the features at prediction time using
    the batter's game log and today's opposing pitcher ERA
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("beatthestreak")


# ---------------------------------------------------------------------------
# ERA Tier classification
# ---------------------------------------------------------------------------
ERA_TIERS = {
    "elite":     (0.0, 3.00),
    "good":      (3.00, 3.75),
    "average":   (3.75, 4.50),
    "below_avg": (4.50, 99.0),
}


def classify_era_tier(era: float) -> str:
    """
    Classify a pitcher's ERA into a tier.

    Args:
        era: Pitcher's earned run average

    Returns:
        Tier name: "elite", "good", "average", or "below_avg"
    """
    if era is None or np.isnan(era):
        return "average"  # Default for unknown

    for tier_name, (low, high) in ERA_TIERS.items():
        if low <= era < high:
            return tier_name

    return "below_avg"  # Fallback for extreme values


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_era_tier_features(
    game_log: pd.DataFrame,
    opposing_pitcher_era: float,
) -> dict:
    """
    Compute ERA tier features for a batter given their game log.

    The game log must have columns:
      - at_bats: int
      - hits: int
      - opp_pitcher_era: float (ERA of the opposing pitcher in that game)

    Args:
        game_log: DataFrame of batter's games with opp_pitcher_era column
        opposing_pitcher_era: ERA of TODAY's opposing pitcher

    Returns:
        dict with:
          - batter_hit_rate_vs_era_tier: hit rate against this ERA tier
          - batter_ab_vs_era_tier: total ABs against this tier (sample size)
          - era_tier_delta: tier hit rate minus overall hit rate
    """
    # Classify today's opposing pitcher
    target_tier = classify_era_tier(opposing_pitcher_era)

    # If no game log or missing opp_pitcher_era column, return defaults
    if game_log is None or game_log.empty or "opp_pitcher_era" not in game_log.columns:
        return {
            "batter_hit_rate_vs_era_tier": 0.0,
            "batter_ab_vs_era_tier": 0,
            "era_tier_delta": 0.0,
        }

    # Classify each game's opposing pitcher
    game_log = game_log.copy()
    game_log["era_tier"] = game_log["opp_pitcher_era"].apply(classify_era_tier)

    # Filter to games against the target tier
    tier_games = game_log[game_log["era_tier"] == target_tier]

    # Overall stats (for delta)
    total_abs = game_log["at_bats"].sum()
    total_hits = game_log["hits"].sum()
    overall_hit_rate = total_hits / max(total_abs, 1)

    # Tier-specific stats
    tier_abs = tier_games["at_bats"].sum() if len(tier_games) > 0 else 0
    tier_hits = tier_games["hits"].sum() if len(tier_games) > 0 else 0

    if tier_abs > 0:
        tier_hit_rate = tier_hits / tier_abs
    else:
        # No ABs against this tier — fall back to overall hit rate
        tier_hit_rate = overall_hit_rate

    # Delta: positive = batter is BETTER against this tier than average
    era_tier_delta = tier_hit_rate - overall_hit_rate if tier_abs > 0 else 0.0

    return {
        "batter_hit_rate_vs_era_tier": round(tier_hit_rate, 4),
        "batter_ab_vs_era_tier": int(tier_abs),
        "era_tier_delta": round(era_tier_delta, 4),
    }


# ---------------------------------------------------------------------------
# Batch computation for training data
# ---------------------------------------------------------------------------
def add_era_tier_features_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ERA tier features to a full training dataset.

    For training, we compute the features using a LOOKBACK approach:
    for each game row, we use only the batter's PRIOR games in that season
    to compute their tier-specific hit rate against the tier of the
    current game's opposing pitcher.

    This prevents data leakage (we never use future games).

    Expects columns:
      - player_id
      - season (or year)
      - at_bats
      - hits
      - opp_pitcher_era (ERA of opposing pitcher in this game)
      - game_date or date (for ordering)

    Returns:
        DataFrame with 3 new columns added
    """
    if "opp_pitcher_era" not in df.columns:
        logger.warning("opp_pitcher_era not in dataset — filling ERA tier features with 0")
        df["batter_hit_rate_vs_era_tier"] = 0.0
        df["batter_ab_vs_era_tier"] = 0
        df["era_tier_delta"] = 0.0
        return df

    # Sort by player and date
    date_col = "game_date" if "game_date" in df.columns else "date"
    season_col = "season" if "season" in df.columns else "year"

    df = df.sort_values(["player_id", date_col]).copy()

    # Initialize columns
    df["batter_hit_rate_vs_era_tier"] = 0.0
    df["batter_ab_vs_era_tier"] = 0
    df["era_tier_delta"] = 0.0

    # Group by player + season and compute rolling features
    for (pid, season), group in df.groupby(["player_id", season_col]):
        indices = group.index.tolist()

        for i, idx in enumerate(indices):
            if i < 3:
                # Not enough prior games
                continue

            # Prior games only (no leakage)
            prior = group.iloc[:i]

            # Today's opposing pitcher ERA
            opp_era = group.loc[idx, "opp_pitcher_era"]

            # Compute features
            features = compute_era_tier_features(prior, opp_era)

            df.loc[idx, "batter_hit_rate_vs_era_tier"] = features["batter_hit_rate_vs_era_tier"]
            df.loc[idx, "batter_ab_vs_era_tier"] = features["batter_ab_vs_era_tier"]
            df.loc[idx, "era_tier_delta"] = features["era_tier_delta"]

    return df