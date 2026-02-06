import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Clean and format data
    return data.fillna(0)
