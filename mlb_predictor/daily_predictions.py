"""
daily_predictions.py – Build and score daily predictions for Beat the Streak.

Pipeline:
  1. Pull today's (or any date's) scheduled games with starting pitchers
  2. Get each batter's recent game logs to build features
  3. Score every batter through the trained model
  4. Rank by confidence, output top picks
  5. Write results to S3 for QuickSight
"""

import os
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import boto3

from mlb_predictor.historical_data import (
    _safe_get,
    get_player_info,
    get_pitcher_season_stats,
    get_park_factor,
    BASE_URL,
)
from mlb_predictor.feature_engineering import (
    compute_platoon_advantage,
    FEATURE_COLUMNS,
)
from mlb_predictor.model import load_model

logger = logging.getLogger("beatthestreak")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = f"mlb-hit-predictor-{ENVIRONMENT}-bucket"


# ---------------------------------------------------------------------------
# Step 1: Get scheduled games for a date
# ---------------------------------------------------------------------------
def get_games_for_date(date_str: str) -> list:
    """
    Pull scheduled games for a date.
    Args:
        date_str: "YYYY-MM-DD"
    Returns:
        List of game dicts with teams, venue, starting pitchers
    """
    logger.info(f"Fetching games for {date_str}...")

    data = _safe_get(
        f"{BASE_URL}/schedule",
        params={
            "sportId": 1,
            "date": date_str,
            "gameType": "R",
            "hydrate": "probablePitcher,venue,lineups",
        }
    )
    if not data:
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status == "Final" or status == "Live" or status == "Preview":
                away = game.get("teams", {}).get("away", {})
                home = game.get("teams", {}).get("home", {})

                games.append({
                    "game_pk": game.get("gamePk"),
                    "game_date": date_str,
                    "game_time": game.get("gameDate", ""),
                    "status": status,
                    "venue_name": game.get("venue", {}).get("name", ""),
                    "home_team": home.get("team", {}).get("name", ""),
                    "home_team_id": home.get("team", {}).get("id"),
                    "away_team": away.get("team", {}).get("name", ""),
                    "away_team_id": away.get("team", {}).get("id"),
                    "home_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "home_pitcher_name": home.get("probablePitcher", {}).get("fullName", "TBD"),
                    "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "away_pitcher_name": away.get("probablePitcher", {}).get("fullName", "TBD"),
                })

    logger.info(f"Found {len(games)} games for {date_str}")
    return games


# ---------------------------------------------------------------------------
# Step 2: Get lineups / rosters for each game
# ---------------------------------------------------------------------------
def get_game_lineups(game_pk: int) -> list:
    """
    Get batting lineups for a game. If lineups aren't posted yet,
    fall back to active roster.
    Returns list of dicts with player_id, name, team, is_home, lineup_pos.
    """
    # Try boxscore for lineups (works for live/final games)
    data = _safe_get(f"{BASE_URL}/game/{game_pk}/boxscore")

    batters = []
    if data:
        for side, is_home in [("home", True), ("away", False)]:
            team_data = data.get("teams", {}).get(side, {})
            team_name = team_data.get("team", {}).get("name", "")

            batting_order = team_data.get("battingOrder", [])
            players = team_data.get("players", {})

            if batting_order:
                for pos_idx, pid in enumerate(batting_order):
                    player_key = f"ID{pid}"
                    player_data = players.get(player_key, {})
                    person = player_data.get("person", {})
                    position = player_data.get("position", {})

                    if position.get("abbreviation") == "P":
                        continue

                    batters.append({
                        "player_id": pid,
                        "player_name": person.get("fullName", "Unknown"),
                        "team": team_name,
                        "is_home": is_home,
                        "lineup_position": pos_idx + 1,
                    })

    # If no lineups from boxscore, try roster
    if not batters:
        batters = _get_roster_batters(game_pk)

    return batters


def _get_roster_batters(game_pk: int) -> list:
    """Fallback: get batters from game linescore/roster."""
    data = _safe_get(f"{BASE_URL}/game/{game_pk}/linescore")
    if not data:
        return []

    # For preview games, pull team rosters instead
    game_data = _safe_get(f"{BASE_URL}/schedule", params={"gamePk": game_pk})
    if not game_data:
        return []

    batters = []
    for date_entry in game_data.get("dates", []):
        for game in date_entry.get("games", []):
            for side, is_home in [("home", True), ("away", False)]:
                team_info = game.get("teams", {}).get(side, {}).get("team", {})
                team_id = team_info.get("id")
                team_name = team_info.get("name", "")

                if team_id:
                    roster = _safe_get(
                        f"{BASE_URL}/teams/{team_id}/roster",
                        params={"rosterType": "active"}
                    )
                    if roster:
                        for idx, entry in enumerate(roster.get("roster", [])):
                            pos = entry.get("position", {}).get("abbreviation", "")
                            if pos == "P":
                                continue
                            person = entry.get("person", {})
                            batters.append({
                                "player_id": person.get("id"),
                                "player_name": person.get("fullName", "Unknown"),
                                "team": team_name,
                                "is_home": is_home,
                                "lineup_position": idx + 1,
                            })
    return batters


# ---------------------------------------------------------------------------
# Step 3: Build features for each batter
# ---------------------------------------------------------------------------
def build_batter_features(
    player_id: int,
    season: int,
    game_date: str,
) -> dict:
    """
    Build feature vector for a single batter based on their recent history.
    Pulls game log up to (but not including) game_date.
    """
    data = _safe_get(
        f"{BASE_URL}/people/{player_id}/stats",
        params={"stats": "gameLog", "group": "hitting", "season": season}
    )
    if not data:
        return {}

    # Parse game logs
    games = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            stat = split.get("stat", {})
            g_date = split.get("date", "")

            # Only use games BEFORE the prediction date
            if g_date >= game_date:
                continue

            games.append({
                "date": g_date,
                "at_bats": stat.get("atBats", 0),
                "hits": stat.get("hits", 0),
                "walks": stat.get("baseOnBalls", 0),
                "strikeouts": stat.get("strikeOuts", 0),
                "plate_appearances": stat.get("plateAppearances", 0),
            })

    if len(games) < 7:
        return {}  # Not enough history

    df = pd.DataFrame(games).sort_values("date")

    total_abs = df["at_bats"].sum()
    total_hits = df["hits"].sum()
    total_ks = df["strikeouts"].sum()
    total_bbs = df["walks"].sum()
    total_pas = df["plate_appearances"].sum()
    num_games = len(df)

    # Rolling averages (hits per game)
    last_7 = df.tail(7)["hits"].mean() if len(df) >= 7 else df["hits"].mean()
    last_14 = df.tail(14)["hits"].mean() if len(df) >= 14 else df["hits"].mean()
    last_30 = df.tail(30)["hits"].mean() if len(df) >= 30 else df["hits"].mean()

    # Multi-hit frequency
    multi_hit_games = (df["hits"] >= 2).sum()
    multi_hit_freq = multi_hit_games / num_games

    # Games between hits (secret sauce)
    got_hit = (df["hits"] >= 1).astype(int)
    hitless_games = (1 - got_hit).sum()
    games_between_hits = num_games / (hitless_games + 1)

    # Current hit streak (counting backwards from most recent game)
    hit_streak = 0
    for h in reversed(got_hit.tolist()):
        if h == 1:
            hit_streak += 1
        else:
            break

    features = {
        "batting_avg_season": total_hits / max(total_abs, 1),
        "avg_last_7": last_7,
        "avg_last_14": last_14,
        "avg_last_30": last_30,
        "k_rate": total_ks / max(total_abs, 1),
        "bb_rate": total_bbs / max(total_pas, 1),
        "multi_hit_freq": multi_hit_freq,
        "avg_abs_per_game": total_abs / num_games,
        "games_between_hits": games_between_hits,
        "hit_streak": hit_streak,
        "game_num_in_season": num_games,
    }

    return features


# ---------------------------------------------------------------------------
# Step 4: Full daily prediction pipeline
# ---------------------------------------------------------------------------
def run_daily_predictions(
    date_str: str = None,
    season: int = None,
    write_to_s3: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: games → lineups → features → model → ranked picks → S3.

    Args:
        date_str: "YYYY-MM-DD" (defaults to today)
        season: MLB season year (defaults to year from date_str)
        write_to_s3: Whether to upload results to S3

    Returns:
        DataFrame of ranked predictions
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    if season is None:
        season = int(date_str[:4])

    logger.info(f"=== Running daily predictions for {date_str} ===")

    # Load trained model
    model, feature_cols = load_model()
    if model is None:
        logger.error("No trained model found. Run /train first.")
        return pd.DataFrame()

    # Step 1: Get games
    games = get_games_for_date(date_str)
    if not games:
        logger.warning(f"No games found for {date_str}")
        return pd.DataFrame()

    # Cache for pitcher stats and player info
    pitcher_cache = {}
    player_info_cache = {}

    all_predictions = []

    for game in games:
        game_pk = game["game_pk"]
        venue = game["venue_name"]
        park_factor = get_park_factor(venue)

        logger.info(f"Processing: {game['away_team']} @ {game['home_team']} ({venue})")

        # Step 2: Get lineups
        batters = get_game_lineups(game_pk)
        if not batters:
            logger.warning(f"No lineup data for game {game_pk}")
            continue

        for batter in batters:
            pid = batter["player_id"]
            is_home = batter["is_home"]

            # Determine opposing pitcher
            if is_home:
                opp_pitcher_id = game["away_pitcher_id"]
                opp_pitcher_name = game["away_pitcher_name"]
                opponent = game["away_team"]
            else:
                opp_pitcher_id = game["home_pitcher_id"]
                opp_pitcher_name = game["home_pitcher_name"]
                opponent = game["home_team"]

            # Step 3: Build batter features
            features = build_batter_features(pid, season, date_str)
            if not features:
                continue

            # Get player bio for bat side
            if pid not in player_info_cache:
                player_info_cache[pid] = get_player_info(pid)
            bio = player_info_cache[pid]
            bat_side = bio.get("bat_side", "U")

            # Bat side encoding
            bat_side_map = {"R": 0, "L": 1, "S": 2, "U": 0}
            features["bat_side_encoded"] = bat_side_map.get(bat_side, 0)
            features["is_home"] = 1 if is_home else 0

            # Pitcher features
            if opp_pitcher_id:
                if opp_pitcher_id not in pitcher_cache:
                    pitcher_cache[opp_pitcher_id] = get_pitcher_season_stats(
                        opp_pitcher_id, season
                    )
                p_stats = pitcher_cache[opp_pitcher_id]
            else:
                p_stats = {}

            features["pitcher_era"] = p_stats.get("pitcher_era", 4.50)
            features["pitcher_whip"] = p_stats.get("pitcher_whip", 1.30)
            features["pitcher_k_per_9"] = p_stats.get("pitcher_k_per_9", 8.5)

            # Pitcher handedness
            opp_hand = "U"
            if opp_pitcher_id:
                if opp_pitcher_id not in player_info_cache:
                    player_info_cache[opp_pitcher_id] = get_player_info(opp_pitcher_id)
                opp_hand = player_info_cache[opp_pitcher_id].get("throw_side", "U")

            pitcher_hand_map = {"R": 0, "L": 1, "U": 0}
            features["pitcher_hand_encoded"] = pitcher_hand_map.get(opp_hand, 0)

            # Platoon advantage
            features["platoon_advantage"] = compute_platoon_advantage(bat_side, opp_hand)

            # Park factor
            features["park_factor"] = park_factor

            # Build prediction row
            row = {
                "player_id": pid,
                "player_name": batter["player_name"],
                "team": batter["team"],
                "opponent": opponent,
                "game_pk": game_pk,
                "venue": venue,
                "bat_side": bat_side,
                "lineup_position": batter.get("lineup_position", 0),
                "opp_pitcher": opp_pitcher_name,
                "opp_pitcher_hand": opp_hand,
                "date": date_str,
                **features,
            }
            all_predictions.append(row)

    if not all_predictions:
        logger.warning("No predictions generated")
        return pd.DataFrame()

    pred_df = pd.DataFrame(all_predictions)

    # Step 4: Score through model
    # Fill any missing feature columns with defaults
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0

    # Handle NaN in feature columns
    for col in feature_cols:
        if pred_df[col].isna().any():
            median_val = pred_df[col].median()
            pred_df[col] = pred_df[col].fillna(median_val if not pd.isna(median_val) else 0)

    X = pred_df[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]
    pred_df["confidence_pct"] = (probabilities * 100).round(2)

    # Rank and sort
    pred_df = pred_df.sort_values("confidence_pct", ascending=False).reset_index(drop=True)
    pred_df["rank"] = range(1, len(pred_df) + 1)

    logger.info(f"Generated {len(pred_df)} predictions for {date_str}")
    if len(pred_df) > 0:
        logger.info("=== TOP 10 PICKS ===")
        for _, row in pred_df.head(10).iterrows():
            logger.info(
                f"  #{int(row['rank'])} {row['player_name']} ({row['team']}) "
                f"vs {row['opp_pitcher']} ({row['opp_pitcher_hand']}) "
                f"@ {row['venue']} — {row['confidence_pct']}%"
            )

    # Step 5: Write to S3
    if write_to_s3:
        write_predictions_to_s3(pred_df, date_str)

    return pred_df


# ---------------------------------------------------------------------------
# Step 5: Write to S3
# ---------------------------------------------------------------------------
def write_predictions_to_s3(df: pd.DataFrame, date_str: str):
    """Write daily predictions to S3 as CSV for QuickSight."""
    try:
        s3 = boto3.client("s3")

        # Full predictions file
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        full_key = f"predictions/{date_str}/all_predictions.csv"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=full_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv",
        )
        logger.info(f"Full predictions written to s3://{S3_BUCKET}/{full_key}")

        # Top 10 picks (cleaner for QuickSight dashboard)
        top10 = df.head(10)[[
            "rank", "player_name", "team", "opponent", "venue",
            "bat_side", "lineup_position", "opp_pitcher", "opp_pitcher_hand",
            "confidence_pct", "batting_avg_season", "avg_last_30",
            "games_between_hits", "platoon_advantage", "pitcher_era",
            "park_factor", "date",
        ]].copy()

        # Round floats for readability
        float_cols = ["confidence_pct", "batting_avg_season", "avg_last_30",
                      "games_between_hits", "pitcher_era", "park_factor"]
        for col in float_cols:
            if col in top10.columns:
                top10[col] = top10[col].round(3)

        top10_buffer = io.StringIO()
        top10.to_csv(top10_buffer, index=False)

        top10_key = f"predictions/{date_str}/top10_picks.csv"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=top10_key,
            Body=top10_buffer.getvalue(),
            ContentType="text/csv",
        )
        logger.info(f"Top 10 picks written to s3://{S3_BUCKET}/{top10_key}")

        # Also write to a "latest" location for QuickSight to always point at
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="predictions/latest/top10_picks.csv",
            Body=top10_buffer.getvalue(),
            ContentType="text/csv",
        )
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="predictions/latest/all_predictions.csv",
            Body=csv_buffer.getvalue(),
            ContentType="text/csv",
        )
        logger.info("Latest predictions updated")

    except Exception as e:
        logger.error(f"Failed to write to S3: {e}")