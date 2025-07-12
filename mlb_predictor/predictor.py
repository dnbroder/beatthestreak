import requests
import pandas as pd
import logging
from typing import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import boto3
import os

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_live_game_data():
    logger.info("Fetching live MLB game data...")
    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=team,linescore,flags,game(content(summary,media(epg)),tickets),broadcasts(all),tickets"
    response = requests.get(url)
    response.raise_for_status()
    json_data = response.json()

    games = []
    for date in json_data.get("dates", []):
        for game in date.get("games", []):
            try:
                game_pk = game["gamePk"]
                home_team = game["teams"]["home"]["team"]["name"]
                away_team = game["teams"]["away"]["team"]["name"]
                venue = game.get("venue", {}).get("name", "Unknown")
                game_time = game["gameDate"]

                # Fetch boxscore to get pitcher info
                boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
                boxscore_resp = requests.get(boxscore_url)
                boxscore_data = boxscore_resp.json()

                home_pitcher = boxscore_data["teams"]["home"]["pitchers"][0]
                away_pitcher = boxscore_data["teams"]["away"]["pitchers"][0]

                pitcher_era = boxscore_data["teams"]["home"]["players"][f"ID{home_pitcher}"]["stats"]["pitching"].get("era", 4.50)

                # Simulated or default values until real integration is expanded
                temperature = 80  # TODO: Integrate real-time weather API
                park_avg_runs = 9.0  # TODO: Lookup based on venue
                team_batting_avg = 0.260  # TODO: Fetch from team stats
                team_vs_pitcher_avg = 0.270  # TODO: Compute from hitter-vs-pitcher matchups

                games.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "venue": venue,
                    "game_time": game_time,
                    "pitcher_era": pitcher_era,
                    "temperature": temperature,
                    "park_avg_runs": park_avg_runs,
                    "team_batting_avg": team_batting_avg,
                    "team_vs_pitcher_avg": team_vs_pitcher_avg
                })
            except Exception as e:
                logger.warning(f"Failed to process game: {e}")
                continue

    logger.info(f"Retrieved {len(games)} live games with features")
    return pd.DataFrame(games)

def make_over_under_prediction():
    logger.info("Starting over/under prediction model")
    df = get_live_game_data()

    if df.empty:
        logger.warning("No live game data available.")
        return pd.DataFrame()

    # Dummy binary target based on ERA for testing (to be replaced by real labels later)
    df["over"] = df["pitcher_era"].apply(lambda x: 1 if x > 4.00 else 0)

    # Check if we have at least two classes to train on
    if df["over"].nunique() < 2:
        logger.warning("Insufficient class diversity in target variable 'over'. Skipping training.")
        return pd.DataFrame()

    X = df[[
        "pitcher_era",
        "temperature",
        "park_avg_runs",
        "team_batting_avg",
        "team_vs_pitcher_avg"
    ]]
    y = df["over"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Over/Under prediction complete. Accuracy: {accuracy:.2f}")

    output_df = X_test.copy()
    output_df["predicted_over"] = predictions

    local_path = "/tmp/over_under_predictions.csv"
    output_df.to_csv(local_path, index=False)
    logger.info(f"Saved over/under prediction results to {local_path}")

    try:
        s3 = boto3.client("s3")
        bucket = os.environ.get("S3_BUCKET_NAME")
        key = "over_under_predictions.csv"
        s3.upload_file(local_path, bucket, key)
        logger.info(f"Uploaded over/under results to S3 at s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")

    return output_df
