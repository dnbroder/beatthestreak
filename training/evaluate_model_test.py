import pandas as pd
import pickle
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Constants for file paths
MODEL_PATH = 'models/mlb_hit_predictor.pkl'
TEST_DATA_PATH = 'data/processed/evaluation_dataset.csv'

def load_model(path):
    """Load the trained model from a file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_test_data(path):
    """Load the evaluation dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test dataset not found at {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=['hit'])  # 'hit' should be the target column
    y = df['hit']
    return X, y

def evaluate(model, X, y_true):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X)
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred))
    print("✅ Accuracy:", accuracy_score(y_true, y_pred))
    print("✅ Precision:", precision_score(y_true, y_pred))
    print("✅ Recall:", recall_score(y_true, y_pred))
    print("✅ F1 Score:", f1_score(y_true, y_pred))

def main():
    print("📦 Loading model...")
    model = load_model(MODEL_PATH)

    print("🧪 Loading evaluation dataset...")
    X_eval, y_eval = load_test_data(TEST_DATA_PATH)

    print("🔍 Evaluating model...")
    evaluate(model, X_eval, y_eval)

if __name__ == "__main__":
    main()
