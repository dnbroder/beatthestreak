"""
data_loader.py – Fetch reference data from the MLB Stats API.
"""

import requests
import pandas as pd
import logging
from typing import Literal

logger = logging.getLogger("beatthestreak")


def load_data(source: Literal["mlb", "statsapi"]) -> pd.DataFrame:
    """
    Load MLB data from a supported free API.
      - "mlb"      -> today's schedule
      - "statsapi" -> current teams
    """
    try:
        if source == "mlb":
            url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1"
        elif source == "statsapi":
            url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
        else:
            raise ValueError(f"Unsupported source: {source}")

        logger.info(f"Requesting data from {source} API: {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        json_data = response.json()

        if "dates" in json_data:
            data = pd.json_normalize(json_data["dates"])
        elif "teams" in json_data:
            data = pd.json_normalize(json_data["teams"])
        else:
            logger.warning("Response JSON does not contain expected keys.")
            return pd.DataFrame()

        logger.info(f"Loaded {len(data)} rows from {source}")
        return data

    except requests.RequestException as e:
        logger.error(f"Request failed for source '{source}': {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return pd.DataFrame()