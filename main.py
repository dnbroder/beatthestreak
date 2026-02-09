"""
BeatTheStreak API – main.py
Entrypoint for uvicorn (main:app).
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import logging
import os

from mlb_predictor.predictor import get_live_game_data, make_over_under_prediction
from mlb_predictor.data_loader import load_data
from mlb_predictor.historical_data import (
    build_season_game_logs,
    build_multi_season_dataset,
)
from mlb_predictor.feature_engineering import (
    engineer_features,
    BASIC_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
)
from mlb_predictor.model import train_model, predict_daily, load_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")

app = FastAPI(
    title="BeatTheStreak API",
    version="0.2.0",
    description=f"Environment: {ENVIRONMENT}",
)


@app.on_event("startup")
async def startup_event():
    logger.info(f"ECS container started. Environment: {ENVIRONMENT}")


@app.get("/")
def root():
    return {
        "app": "BeatTheStreak",
        "version": "0.2.0",
        "environment": ENVIRONMENT,
        "endpoints": {
            "GET /":              "This page",
            "GET /health":        "Health check",
            "GET /teams":         "MLB teams",
            "GET /games":         "Today's games with features",
            "POST /predict":      "Legacy over/under prediction",
            "GET /test":          "Mock pipeline test",
            "POST /train":        "Train XGBoost model (single season)",
            "POST /train/full":   "Train XGBoost model (multi-season)",
            "GET /predict/daily": "Today's hit predictions (ranked)",
            "POST /backtest":     "Run streak simulation",
        },
    }


@app.get("/health")
def health():
    logger.info("Health check hit")
    return {"status": "healthy", "environment": ENVIRONMENT}


@app.get("/teams")
def teams():
    logger.info("/teams endpoint hit")
    df = load_data("statsapi")
    if df.empty:
        return JSONResponse(status_code=503, content={"error": "Could not fetch team data"})
    return df.to_dict(orient="records")


@app.get("/games")
def games():
    logger.info("/games endpoint hit")
    df = get_live_game_data()
    if df.empty:
        return JSONResponse(status_code=503, content={"error": "No live game data available"})
    return df.to_dict(orient="records")


@app.post("/predict")
def trigger_prediction():
    logger.info("/predict endpoint hit")
    try:
        df = make_over_under_prediction()
        if df is None or (hasattr(df, "empty") and df.empty):
            return JSONResponse(
                status_code=503,
                content={"message": "Not enough data to generate predictions"},
            )
        return {
            "message": "Prediction executed successfully.",
            "num_rows": len(df),
            "predictions": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})


# ---------------------------------------------------------------------------
# /test – mock pipeline
# ---------------------------------------------------------------------------
MOCK_GAMES = [
    {"game_pk": 99901, "home_team": "New York Yankees", "away_team": "Boston Red Sox",
     "venue": "Yankee Stadium", "game_time": "2025-07-15T23:05:00Z", "status": "Final",
     "pitcher_era": 3.21, "temperature": 85, "park_avg_runs": 9.6,
     "team_batting_avg": 0.268, "team_vs_pitcher_avg": 0.285},
    {"game_pk": 99902, "home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants",
     "venue": "Dodger Stadium", "game_time": "2025-07-15T01:10:00Z", "status": "Final",
     "pitcher_era": 2.87, "temperature": 78, "park_avg_runs": 8.4,
     "team_batting_avg": 0.272, "team_vs_pitcher_avg": 0.260},
    {"game_pk": 99903, "home_team": "Chicago Cubs", "away_team": "St. Louis Cardinals",
     "venue": "Wrigley Field", "game_time": "2025-07-15T23:20:00Z", "status": "Final",
     "pitcher_era": 4.55, "temperature": 82, "park_avg_runs": 9.8,
     "team_batting_avg": 0.255, "team_vs_pitcher_avg": 0.275},
    {"game_pk": 99904, "home_team": "Houston Astros", "away_team": "Texas Rangers",
     "venue": "Minute Maid Park", "game_time": "2025-07-16T00:10:00Z", "status": "Final",
     "pitcher_era": 3.78, "temperature": 73, "park_avg_runs": 9.2,
     "team_batting_avg": 0.265, "team_vs_pitcher_avg": 0.290},
    {"game_pk": 99905, "home_team": "Atlanta Braves", "away_team": "Philadelphia Phillies",
     "venue": "Truist Park", "game_time": "2025-07-15T23:20:00Z", "status": "Final",
     "pitcher_era": 5.12, "temperature": 90, "park_avg_runs": 9.0,
     "team_batting_avg": 0.258, "team_vs_pitcher_avg": 0.245},
    {"game_pk": 99906, "home_team": "Seattle Mariners", "away_team": "Oakland Athletics",
     "venue": "T-Mobile Park", "game_time": "2025-07-16T02:40:00Z", "status": "Final",
     "pitcher_era": 3.45, "temperature": 68, "park_avg_runs": 7.8,
     "team_batting_avg": 0.248, "team_vs_pitcher_avg": 0.270},
    {"game_pk": 99907, "home_team": "Minnesota Twins", "away_team": "Cleveland Guardians",
     "venue": "Target Field", "game_time": "2025-07-15T23:40:00Z", "status": "Final",
     "pitcher_era": 4.92, "temperature": 76, "park_avg_runs": 9.4,
     "team_batting_avg": 0.262, "team_vs_pitcher_avg": 0.280},
    {"game_pk": 99908, "home_team": "Tampa Bay Rays", "away_team": "Baltimore Orioles",
     "venue": "Tropicana Field", "game_time": "2025-07-15T22:50:00Z", "status": "Final",
     "pitcher_era": 3.10, "temperature": 72, "park_avg_runs": 8.1,
     "team_batting_avg": 0.250, "team_vs_pitcher_avg": 0.255},
]


@app.get("/test")
def test_pipeline():
    logger.info("/test endpoint hit - running mock pipeline")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    try:
        df = pd.DataFrame(MOCK_GAMES)
        era_median = df["pitcher_era"].median()
        df["over"] = (df["pitcher_era"] > era_median).astype(int)

        feature_cols = ["pitcher_era", "temperature", "park_avg_runs",
                        "team_batting_avg", "team_vs_pitcher_avg"]
        X = df[feature_cols]
        y = df["over"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        logger.info(f"Mock pipeline complete. Accuracy: {accuracy:.2f}")
        output = X_test.copy()
        output["predicted_over"] = predictions
        output["actual_over"] = y_test.values

        return {
            "status": "pass",
            "message": "End-to-end pipeline executed successfully with mock data.",
            "mock_games": len(df),
            "accuracy": round(accuracy, 4),
            "predictions": output.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"Mock pipeline failed: {e}")
        return JSONResponse(status_code=500, content={"status": "fail", "error": str(e)})


# ---------------------------------------------------------------------------
# Train XGBoost hit prediction model
# ---------------------------------------------------------------------------
@app.post("/train")
def train_hit_model(
    season: int = Query(default=2024, description="Season to train on"),
    max_players: int = Query(default=None, description="Limit players (for testing)"),
):
    """Train on a single season."""
    logger.info(f"/train endpoint hit - season={season}, max_players={max_players}")
    try:
        game_logs = build_season_game_logs(season, max_players=max_players)
        if game_logs.empty:
            return JSONResponse(status_code=503, content={"error": f"No data for {season}"})

        featured = engineer_features(game_logs)
        model, metrics = train_model(featured)

        if model is None:
            return JSONResponse(status_code=503, content={"error": "Training failed", **metrics})

        return {"status": "success", "message": f"Model trained on {season}", "metrics": metrics}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/train/full")
def train_full_model(
    start_year: int = Query(default=2000, description="Start season"),
    end_year: int = Query(default=2024, description="End season"),
    max_players: int = Query(default=None, description="Limit players per season"),
):
    """Train on multiple seasons with recency weighting."""
    logger.info(f"/train/full endpoint hit - {start_year} to {end_year}")
    try:
        game_logs = build_multi_season_dataset(
            start_year=start_year, end_year=end_year,
            max_players_per_season=max_players,
        )
        if game_logs.empty:
            return JSONResponse(status_code=503, content={"error": "No data retrieved"})

        featured = engineer_features(game_logs)
        model, metrics = train_model(featured)

        if model is None:
            return JSONResponse(status_code=503, content={"error": "Training failed", **metrics})

        return {
            "status": "success",
            "message": f"Model trained on {start_year}-{end_year}",
            "total_rows": len(featured),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Full training failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Daily predictions
# ---------------------------------------------------------------------------
@app.get("/predict/daily")
def daily_predictions():
    """Today's hit predictions ranked by confidence."""
    logger.info("/predict/daily endpoint hit")

    model, feature_cols = load_model()
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "No trained model. Run POST /train first."},
        )

    try:
        return JSONResponse(
            status_code=501,
            content={
                "message": "Daily predictions coming soon — model is trained and ready.",
                "next_step": "Wire up real-time feature engineering for today's batters.",
            },
        )
    except Exception as e:
        logger.error(f"Daily prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})