"""
BeatTheStreak API – main.py
Entrypoint for uvicorn (main:app).
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import logging
import os
from datetime import datetime

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
from mlb_predictor.model import train_model, load_model
from mlb_predictor.daily_predictions import run_daily_predictions
from mlb_predictor.backtester import run_backtest
from mlb_predictor.shap_analysis import run_shap_analysis
from mlb_predictor.hyperparameter_tuning import run_hyperparameter_tuning

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

# Configure the "beatthestreak" logger used by all mlb_predictor modules
# Without this, logger.info() calls in historical_data, model, feature_engineering,
# daily_predictions, backtester, etc. are silently dropped (no handlers attached).
bt_logger = logging.getLogger("beatthestreak")
bt_logger.setLevel(logging.INFO)
if not bt_logger.handlers:
    bt_logger.addHandler(handler)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")

app = FastAPI(
    title="BeatTheStreak API",
    version="0.4.0",
    description=f"Environment: {ENVIRONMENT}",
)


@app.on_event("startup")
async def startup_event():
    logger.info(f"ECS container started. Environment: {ENVIRONMENT}")


@app.get("/")
def root():
    return {
        "app": "BeatTheStreak",
        "version": "0.4.0",
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
            "POST /train/tuned":  "Train with hyperparameter optimization",
            "GET /predict/daily": "Today's hit predictions (ranked)",
            "POST /backtest":     "Run streak simulation",
            "POST /backtest-async": "Run full-season backtest (background, results to S3)",
            "POST /shap":         "Run SHAP feature importance analysis",
            "POST /run-script":   "Execute ad-hoc analysis script from S3",
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
# /test - mock pipeline
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
# Train with Hyperparameter Tuning
# ---------------------------------------------------------------------------
@app.post("/train/tuned")
def train_tuned_model(
    start_year: int = Query(default=2021, description="Start season"),
    end_year: int = Query(default=2023, description="End season"),
    max_players: int = Query(default=None, description="Limit players per season"),
    n_iter: int = Query(default=60, description="Number of random search iterations"),
    cv_folds: int = Query(default=5, description="Cross-validation folds"),
):
    """
    Train XGBoost with hyperparameter tuning via RandomizedSearchCV.

    Tests n_iter random parameter combinations across cv_folds cross-validation
    folds. Picks the best, retrains on full data, calibrates probabilities.

    This replaces the trained model — run backtest after to validate.
    """
    logger.info(f"/train/tuned endpoint hit - {start_year} to {end_year}, n_iter={n_iter}")
    try:
        game_logs = build_multi_season_dataset(
            start_year=start_year, end_year=end_year,
            max_players_per_season=max_players,
        )
        if game_logs.empty:
            return JSONResponse(status_code=503, content={"error": "No data retrieved"})

        featured = engineer_features(game_logs)
        model, metrics = run_hyperparameter_tuning(
            featured,
            n_iter=n_iter,
            cv_folds=cv_folds,
        )

        if model is None:
            return JSONResponse(status_code=503, content={"error": "Tuning failed"})

        return {
            "status": "success",
            "message": f"Tuned model trained on {start_year}-{end_year}",
            "total_rows": len(featured),
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Tuned training failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Daily predictions
# ---------------------------------------------------------------------------
@app.get("/predict/daily")
def daily_predictions(
    date: str = Query(default=None, description="Date YYYY-MM-DD (default: today)"),
    season: int = Query(default=None, description="Season year (default: from date)"),
    write_s3: bool = Query(default=True, description="Write results to S3"),
):
    """
    Full daily prediction pipeline.
    Pulls games, lineups, builds features, scores batters, ranks picks.
    Writes top 10 + full predictions to S3 for QuickSight.
    """
    logger.info(f"/predict/daily endpoint hit - date={date}")

    try:
        predictions = run_daily_predictions(
            date_str=date,
            season=season,
            write_to_s3=write_s3,
        )

        if predictions.empty:
            return JSONResponse(
                status_code=404,
                content={"error": f"No predictions generated for {date or 'today'}. "
                         "Are there games scheduled?"},
            )

        # Return top 10 in response
        top10 = predictions.head(10)

        # Build clean response
        picks = []
        for _, row in top10.iterrows():
            picks.append({
                "rank": int(row["rank"]),
                "player": row["player_name"],
                "team": row["team"],
                "opponent": row["opponent"],
                "venue": row.get("venue", ""),
                "bat_side": row.get("bat_side", ""),
                "lineup_pos": int(row.get("lineup_position", 0)),
                "vs_pitcher": row.get("opp_pitcher", ""),
                "pitcher_hand": row.get("opp_pitcher_hand", ""),
                "confidence": float(row["confidence_pct"]),
                "batting_avg": round(float(row.get("batting_avg_season", 0)), 3),
                "last_30_avg": round(float(row.get("avg_last_30", 0)), 3),
                "consistency": round(float(row.get("games_between_hits", 0)), 2),
                "aabbh": round(float(row.get("aabbh", 0)), 2),
                "aabbh_ratio": round(float(row.get("aabbh_ratio", 0)), 2),
                "platoon_adv": int(row.get("platoon_advantage", 0)),
                "pitcher_era": round(float(row.get("pitcher_era", 0)), 2),
                "park_factor": round(float(row.get("park_factor", 1.0)), 2),
            })

        return {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "total_batters_scored": len(predictions),
            "top_10_picks": picks,
            "s3_written": write_s3,
        }

    except Exception as e:
        logger.error(f"Daily prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@app.post("/backtest")
def backtest_model(
    season: int = Query(default=2024, description="Season to backtest"),
    interval: int = Query(default=3, description="Test every Nth game date"),
    max_dates: int = Query(default=None, description="Cap dates (for testing)"),
    write_s3: bool = Query(default=True, description="Write results to S3"),
):
    """
    Run backtest across a season: predict -> check actuals -> compute hit rates + streaks.
    WARNING: This is a long-running operation. Use max_dates for quick tests.
    """
    logger.info(f"/backtest endpoint hit - season={season}, interval={interval}, max_dates={max_dates}")

    try:
        results = run_backtest(
            season=season,
            interval=interval,
            max_dates=max_dates,
            write_to_s3=write_s3,
        )

        if "error" in results:
            return JSONResponse(status_code=503, content=results)

        # Return summary (daily detail is in S3)
        summary = results["summary"]

        # Add a quick daily overview (just hit/miss per day, not full pick details)
        daily_overview = []
        for day in results["daily_results"]:
            daily_overview.append({
                "date": day["date"],
                "top1_hit": day.get("top1_rate", None) == 1.0 if day.get("top1_total", 0) > 0 else None,
                "top5_rate": day.get("top5_rate", 0),
                "top10_rate": day.get("top10_rate", 0),
            })

        return {
            "status": "success",
            "summary": summary,
            "daily_overview": daily_overview,
            "s3_written": write_s3,
        }

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------
@app.post("/shap")
def shap_analysis(
    s3_key: str = Query(
        default=None,
        description="S3 key for full_results.json (e.g. backtest/2024/20260214_004946/full_results.json)"
    ),
    write_s3: bool = Query(default=True, description="Write SHAP results to S3"),
):
    """
    Run SHAP explainability analysis on the trained model.

    If s3_key is provided, loads backtest results and analyzes misses.
    Otherwise, runs feature importance only.

    Returns:
      - Global feature importance ranked by SHAP values
      - Miss analysis (which features drove incorrect predictions)
      - Hit vs miss comparison
      - Optimal zone validation
      - Actionable recommendations
    """
    logger.info(f"/shap endpoint hit - s3_key={s3_key}")

    try:
        results = run_shap_analysis(
            backtest_json_path=s3_key,
            write_to_s3=write_s3,
        )

        if "error" in results:
            return JSONResponse(status_code=503, content=results)

        return {
            "status": "success",
            "global_importance": results.get("global_importance", {}),
            "miss_analysis_summary": {
                "total_misses": results.get("miss_analysis", {}).get("total_misses", 0),
                "most_common_miss_drivers": results.get("miss_analysis", {}).get("most_common_miss_drivers", {}),
            },
            "optimal_zone": results.get("optimal_zone_analysis", {}),
            "recommendations": results.get("recommendations", []),
            "s3_written": write_s3,
        }

    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Run S3 Script (for ad-hoc analysis)
# ---------------------------------------------------------------------------
@app.post("/run-script")
def run_s3_script(
    script: str = Query(default="aabbh_curve.py", description="Script filename in S3 bucket"),
):
    """
    Pull a Python script from S3 and execute it. Returns stdout.
    """
    import subprocess
    import boto3

    logger.info(f"/run-script endpoint hit - script={script}")

    try:
        bucket = os.environ.get("S3_BUCKET", "mlb-hit-predictor-test-bucket")
        s3 = boto3.client("s3")
        local_path = f"/tmp/{script}"

        s3.download_file(bucket, script, local_path)
        logger.info(f"Downloaded s3://{bucket}/{script} to {local_path}")

        result = subprocess.run(
            ["python3", local_path],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
        )

        return {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Async Backtest (fire-and-forget, results go to S3)
# ---------------------------------------------------------------------------
@app.post("/backtest-async")
def backtest_async(
    season: int = Query(default=2024, description="Season to backtest"),
    interval: int = Query(default=3, description="Test every Nth game date"),
    max_dates: int = Query(default=None, description="Cap dates (for testing)"),
):
    """
    Launch backtest in background thread. Results written to S3 when complete.
    Returns immediately with status 'started'.
    Check results at: s3://mlb-hit-predictor-{env}-bucket/backtest/{season}/
    """
    import threading

    logger.info(f"/backtest-async endpoint hit - season={season}, interval={interval}, max_dates={max_dates}")

    def _run():
        try:
            logger.info("Background backtest STARTED")
            results = run_backtest(
                season=season,
                interval=interval,
                max_dates=max_dates,
                write_to_s3=True,
            )
            if "error" in results:
                logger.error(f"Background backtest FAILED: {results['error']}")
            else:
                summary = results.get("summary", {})
                logger.info(
                    f"Background backtest COMPLETE: "
                    f"{summary.get('dates_tested', 0)} dates, "
                    f"top1 hit rate = {summary.get('top1_hit_rate', 'N/A')}"
                )
        except Exception as e:
            logger.error(f"Background backtest EXCEPTION: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    env = os.environ.get("ENVIRONMENT", "production")
    return {
        "status": "started",
        "message": f"Backtest running in background for {season} season (every {interval} days). "
                   f"Results will be written to S3 when complete.",
        "s3_location": f"s3://mlb-hit-predictor-{env}-bucket/backtest/{season}/",
        "check_results": f"Look for summary.csv, daily_results.csv, and full_results.json",
    }