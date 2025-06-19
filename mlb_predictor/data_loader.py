import requests
import pandas as pd
import logging
from typing import Literal

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler for ECS logs
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_data(source: Literal["mlb", "statsapi"]) -> pd.DataFrame:
    """
    Load MLB data from a supported free API.
    Supported sources:
      - "mlb": mlb-data API (via MLB's open data)
      - "statsapi": MLB Stats API (https://statsapi.mlb.com)
    """
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

        # Attempt to flatten relevant JSON part into a DataFrame
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
