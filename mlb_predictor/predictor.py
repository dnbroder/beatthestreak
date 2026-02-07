import os
import logging
import boto3
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger("beatthestreak")

# Feature extraction

def get_live_game_data() -> pd.DataFrame:
    """
    Pull today's games from the MLB Stats API, enrich each game with
    pitching / park / team features for the prediction model.
    """
    logger.info("Fetching live MLB game data...")
    url = (
        "https://statsapi.mlb.com/api/v1/schedule?sportId=1"
        "&hydrate=team,linescore,flags,"
        "game(content(summary,media(epg)),tickets),"
        "broadcasts(all),tickets"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch schedule: {e}")
        return pd.DataFrame()

    json_data = response.json()
    games = []

    for date_block in json_data.get("dates", []):
        for game in date_block.get("games", []):
            try:
                game_pk = game["gamePk"]
                home_team = game["teams"]["home"]["team"]["name"]
                away_team = game["teams"]["away"]["team"]["name"]
                venue = game.get("venue", {}).get("name", "Unknown")
                game_time = game["gameDate"]
                status = game.get("status", {}).get("abstractGameState", "Unknown")

                # ---- pitcher ERA from boxscore (safe) ----
                pitcher_era = 4.50  # sensible default
                try:
                    box_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
                    box_resp = requests.get(box_url, timeout=10)
                    box_data = box_resp.json()

                    home_pitchers = box_data["teams"]["home"].get("pitchers", [])
                    if home_pitchers:
                        pid = home_pitchers[0]
                        player_key = f"ID{pid}"
                        pitcher_stats = (
                            box_data["teams"]["home"]
                            .get("players", {})
                            .get(player_key, {})
                            .get("stats", {})
                            .get("pitching", {})
                        )
                        era_val = pitcher_stats.get("era", "4.50")
                        pitcher_era = float(era_val) if era_val not in ("", "-", None) else 4.50
                except Exception as box_err:
                    logger.debug(f"Boxscore fetch failed for game {game_pk}: {box_err}")

                # ---- placeholder features (TODO: wire up real sources) ----
                temperature = 80          # TODO: weather API
                park_avg_runs = 9.0       # TODO: venue lookup table
                team_batting_avg = 0.260  # TODO: team season stats
                team_vs_pitcher_avg = 0.270  # TODO: batter-vs-pitcher splits

                games.append({
                    "game_pk": game_pk,
                    "home_team": home_team,
                    "away_team": away_team,
                    "venue": venue,
                    "game_time": game_time,
                    "status": status,
                    "pitcher_era": pitcher_era,
                    "temperature": temperature,
                    "park_avg_runs": park_avg_runs,
                    "team_batting_avg": team_batting_avg,
                    "team_vs_pitcher_avg": team_vs_pitcher_avg,
                })
            except Exception as e:
                logger.warning(f"Skipping game: {e}")
                continue

    logger.info(f"Retrieved {len(games)} games with features")
    return pd.DataFrame(games)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "pitcher_era",
    "temperature",
    "park_avg_runs",
    "team_batting_avg",
    "team_vs_pitcher_avg",
]


def make_over_under_prediction() -> pd.DataFrame:
    """
    Train a quick logistic-regression model on today's game features and
    return predictions.  Writes results to /tmp and optionally to S3.
    """
    logger.info("Starting over/under prediction model")
    df = get_live_game_data()

    if df.empty:
        logger.warning("No live game data available.")
        return pd.DataFrame()

    # --- need at least 4 games to do a meaningful train/test split ---
    if len(df) < 4:
        logger.warning(f"Only {len(df)} games today - too few to split. Returning raw data.")
        return df

    era_median = df["pitcher_era"].median()
    df["over"] = (df["pitcher_era"] > era_median).astype(int)

    if df["over"].nunique() < 2:
        logger.warning("Only one class in target variable - skipping training.")
        return pd.DataFrame()

    X = df[FEATURE_COLS]
    y = df["over"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Over/Under model accuracy: {accuracy:.2f}")

    output_df = X_test.copy()
    output_df["predicted_over"] = predictions
    output_df["actual_over"] = y_test.values

    # ---- persist locally ----
    local_path = "/tmp/over_under_predictions.csv"
    output_df.to_csv(local_path, index=False)
    logger.info(f"Saved predictions to {local_path}")

    # ---- persist to S3 (optional) ----
    bucket = os.environ.get("S3_BUCKET_NAME")
    if bucket:
        try:
            s3 = boto3.client("s3")
            key = "over_under_predictions.csv"
            s3.upload_file(local_path, bucket, key)
            logger.info(f"Uploaded to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
    else:
        logger.info("S3_BUCKET_NAME not set - skipping S3 upload")

    return output_df