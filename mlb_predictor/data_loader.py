import pandas as pd

def load_data(source: str) -> pd.DataFrame:
    if source.endswith(".csv"):
        return pd.read_csv(source)
    raise ValueError("Unsupported data source format")


# BeatTheStreak/mlb_predictor/utils.py
import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Clean and format data
    return data.fillna(0)
