import requests
import pandas as pd
import logging
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from typing import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_data(source: Literal["mlb", "statsapi"]) -> pd.DataFrame:
    try:
        if source == "mlb":
            url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1"
        elif source == "statsapi":
            url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
        else:
            raise ValueError(f"Unsupported source: {source}")

        logger.info(f"Requesting data from {source} API...")
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()

        if "dates" in json_data:
            data = pd.json_normalize(json_data["dates"])
        elif "teams" in json_data:
            data = pd.json_normalize(json_data["teams"])
        else:
            logger.warning("Response JSON does not contain expected keys.")
            return pd.DataFrame()

        logger.info(f"Successfully loaded data from {source}. Rows: {len(data)}")
        return data

    except requests.RequestException as e:
        logger.error(f"Request failed for source '{source}': {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        return pd.DataFrame()

def make_basic_prediction():
    logger.info("Starting basic prediction logic")
    
    # Simulate some dummy structured data for prediction
    data = pd.DataFrame({
        "hits": [5, 7, 3, 8, 6, 9],
        "runs": [3, 5, 2, 6, 4, 7],
        "home_game": [1, 0, 1, 0, 1, 0],
        "win": [1, 1, 0, 1, 0, 1]  # Target variable
    })

    X = data[["hits", "runs", "home_game"]]
    y = data["win"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Prediction complete. Accuracy: {accuracy:.2f}")

    # Save prediction results
    output_df = X_test.copy()
    output_df["predicted_win"] = predictions
    local_file = "/tmp/prediction_results.csv"
    output_df.to_csv(local_file, index=False)
    logger.info(f"Saved prediction results to {local_file}")

    # Upload to S3
    try:
        s3 = boto3.client("s3")
        bucket_name = "mlb-hit-predictor-production-bucket"  # <-- replace this
        s3.upload_file(local_file, bucket_name, "prediction_results.csv")
        logger.info("✅ Successfully uploaded prediction_results.csv to S3")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")

    return output_df
