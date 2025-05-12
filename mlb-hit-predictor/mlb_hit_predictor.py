# # MLB Hit Predictor

## 1. Import Libraries
import pandas as pd
import numpy as np
import requests
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

## 2. Generate Dummy MLB Data (for local testing)
np.random.seed(42)
n_samples = 5000
data = {
    'batter': np.random.randint(100000, 999999, n_samples),
    'pitcher': np.random.randint(100000, 999999, n_samples),
    'batter_hand': np.random.choice(['R', 'L', 'S'], n_samples),
    'pitcher_hand': np.random.choice(['R', 'L'], n_samples),
    'is_home_game': np.random.randint(0, 2, n_samples),
    'inning': np.random.randint(1, 10, n_samples),
    'pitch_type': np.random.choice(['FF', 'SL', 'CU', 'CH'], n_samples),
    'batter_recent_avg': np.round(np.random.uniform(0.150, 0.400, n_samples), 3),
    'pitcher_recent_era': np.round(np.random.uniform(2.00, 6.00, n_samples), 2),
    'avg_time_between_hits': np.random.randint(1, 10, n_samples),
    'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain', 'Fog'], n_samples),
    'temperature': np.random.randint(50, 100, n_samples),
    'is_night_game': np.random.randint(0, 2, n_samples),
    'team_avg_vs_pitcher': np.round(np.random.uniform(0.200, 0.350, n_samples), 3),
    'hit': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

## 3. Encode Categorical Features
for col in ['batter_hand', 'pitcher_hand', 'pitch_type', 'weather_condition']:
    df[col] = df[col].astype('category').cat.codes

## 4. Split into Train/Test Sets
X = df.drop(columns=['hit'])
y = df['hit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 5. Train the XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

## 6. Evaluate Model Performance
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

## 7. Visualize Feature Importances
plt.figure(figsize=(10, 6))
importance = model.feature_importances_
plt.barh(X.columns, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()

## 8. Save the Trained Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/mlb_hit_predictor.pkl")

## 9. Historical Stat Utilities
def get_multi_year_stats(player_ids, start_season=2021, end_season=2024):
    all_stats = []
    for season in range(start_season, end_season + 1):
        for pid in player_ids:
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            params = {
                "stats": "season",
                "group": "hitting",
                "season": season
            }
            res = requests.get(url, params=params).json()
            try:
                stats = res["stats"][0]["splits"][0]["stat"]
                stats["player_id"] = pid
                stats["season"] = season
                all_stats.append(stats)
            except (KeyError, IndexError):
                continue
    return pd.DataFrame(all_stats)

def get_batter_vs_pitcher(batter_id, pitcher_id):
    url = f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats"
    params = {
        "stats": "vsPlayer",
        "opposingPlayerId": pitcher_id,
        "group": "hitting"
    }
    res = requests.get(url, params=params).json()
    try:
        return res["stats"][0]["splits"][0]["stat"]
    except (KeyError, IndexError):
        return {}

## 10. Define a Prediction Function
def predict_hit(input_data):
    input_df = pd.DataFrame([input_data])
    for col in ['batter_hand', 'pitcher_hand', 'pitch_type', 'weather_condition']:
        input_df[col] = pd.Series(input_df[col]).astype('category').cat.codes
    model = joblib.load("models/mlb_hit_predictor.pkl")
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]
    return proba, pred

## 11. Run an Example Prediction
example = {
    'batter': 123456,
    'pitcher': 654321,
    'batter_hand': 'R',
    'pitcher_hand': 'L',
    'is_home_game': 1,
    'inning': 1,
    'pitch_type': 'FF',
    'batter_recent_avg': 0.325,
    'pitcher_recent_era': 3.80,
    'avg_time_between_hits': 2,
    'weather_condition': 'Clear',
    'temperature': 72,
    'is_night_game': 0,
    'team_avg_vs_pitcher': 0.275
}

proba, pred = predict_hit(example)
print(f"Predicted Probability of Hit: {proba:.2%}")
print(f"Predicted Outcome: {'Hit' if pred == 1 else 'No Hit'}")

## 12. [Optional] Lambda-Compatible Entry Point
def lambda_handler(event, context):
    input_data = event['input']
    proba, pred = predict_hit(input_data)
    return {
        'statusCode': 200,
        'body': {
            'prediction': 'Hit' if pred == 1 else 'No Hit',
            'probability': round(proba, 4)
        }
    }
