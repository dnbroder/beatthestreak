import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# File paths
DATA_PATH = 'data/processed/final_dataset.csv'
NEW_DATA_PATH = 'data/processed/daily_update.csv'
MODEL_PATH = 'models/mlb_hit_predictor.pkl'

def load_and_merge_datasets(historical_path, new_data_path):
    """Load historical and new data, merge them, and remove duplicates."""
    if not os.path.exists(historical_path):
        raise FileNotFoundError(f"Historical dataset not found at {historical_path}")
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"New daily dataset not found at {new_data_path}")
    
    df_existing = pd.read_csv(historical_path)
    df_new = pd.read_csv(new_data_path)

    combined_df = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
    combined_df.to_csv(historical_path, index=False)
    return combined_df

def retrain_model(df):
    """Train a new model on the updated dataset."""
    X = df.drop(columns=['hit'])  # 'hit' is the target variable
    y = df['hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Updated Model Performance:")
    print(classification_report(y_test, y_pred))

    return model

def save_model(model, path):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def main():
    print("🔁 Running feedback loop...")
    updated_df = load_and_merge_datasets(DATA_PATH, NEW_DATA_PATH)

    print("🧠 Retraining model on updated data...")
    updated_model = retrain_model(updated_df)

    print("💾 Saving updated model...")
    save_model(updated_model, MODEL_PATH)
    print("✅ Feedback loop completed successfully.")

if __name__ == "__main__":
    main()
