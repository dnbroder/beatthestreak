import pandas as pd
import xgboost as xgb
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants for file paths
DATA_PATH = 'data/processed/final_dataset.csv'
MODEL_PATH = 'models/mlb_hit_predictor.pkl'

def load_data(path):
    """Load the processed dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def train_model(X_train, y_train):
    """Train an XGBoost classifier."""
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Print classification metrics."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Model Evaluation:\n", report)

def save_model(model, path):
    """Save the trained model to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def main():
    print("🔄 Loading data...")
    df = load_data(DATA_PATH)

    print("🔍 Preparing features and target...")
    X = df.drop(columns=['hit'])  # 'hit' should be your target column
    y = df['hit']

    print("🧪 Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training model...")
    model = train_model(X_train, y_train)

    print("📊 Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("💾 Saving model...")
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
