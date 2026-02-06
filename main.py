"""
BeatTheStreak API – main.py
Entrypoint for uvicorn (main:app).
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mlb_predictor.predictor import get_live_game_data, make_over_under_prediction
from mlb_predictor.data_loader import load_data
import logging

# ---------------------------------------------------------------------------
# Logging – visible in CloudWatch via awslogs driver
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
app = FastAPI(title="BeatTheStreak API", version="0.1.0")


@app.on_event("startup")
async def startup_event():
    logger.info("ECS container started. Running initial prediction.")
    try:
        make_over_under_prediction()
        logger.info("Initial prediction completed successfully.")
    except Exception as e:
        logger.error(f"Failed to run prediction on startup: {e}")
        # Don't re-raise — let the container stay up even if first prediction fails


@app.get("/")
def root():
    """Landing page – confirms the API is alive and lists endpoints."""
    return {
        "app": "BeatTheStreak",
        "version": "0.1.0",
        "endpoints": {
            "GET /":         "This page",
            "GET /health":   "Container health check",
            "GET /teams":    "Current MLB teams",
            "GET /games":    "Today's games with model features",
            "POST /predict": "Run over/under prediction model",
        },
    }


@app.get("/health")
def health():
    """
    Lightweight health check for ECS container health + ALB target group.
    Does NOT call the MLB API — just confirms the process is responding.
    """
    logger.info("Health check hit")
    return {"status": "healthy"}


@app.get("/teams")
def teams():
    """Return current MLB teams from the Stats API."""
    logger.info("/teams endpoint hit")
    df = load_data("statsapi")
    if df.empty:
        return JSONResponse(status_code=503, content={"error": "Could not fetch team data"})
    return df.to_dict(orient="records")


@app.get("/games")
def games():
    """Return today's games with feature columns used by the model."""
    logger.info("/games endpoint hit")
    df = get_live_game_data()
    if df.empty:
        return JSONResponse(status_code=503, content={"error": "No live game data available"})
    return df.to_dict(orient="records")


@app.post("/predict")
def trigger_prediction():
    """Run the over/under prediction model on today's games."""
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
        return JSONResponse(
            status_code=500,
            content={"message": "Prediction failed.", "error": str(e)},
        )