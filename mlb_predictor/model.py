"""
model.py – XGBoost model for predicting whether a batter gets at least 1 hit.

Training uses recency-weighted samples (2000-2024).
Output is a probability (confidence %) for each batter.

v2.1 Changes:
  - S3 checkpoint saves at each major training step (model never lost on container recycle)
  - Training status flag written to S3 (training_status.json)
  - Memory usage logged at key points
  - Calibration OOM protection — falls back to uncalibrated model gracefully
  - Verbose logging throughout
"""

import os
import io
import json
import logging
import pickle
import time
import traceback
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    log_loss,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV

from mlb_predictor.feature_engineering import (
    FEATURE_COLUMNS,
    BASIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)

logger = logging.getLogger("beatthestreak")

MODEL_DIR = "/tmp/models"
MODEL_PATH = os.path.join(MODEL_DIR, "hit_predictor.pkl")
CALIBRATED_MODEL_PATH = os.path.join(MODEL_DIR, "hit_predictor_calibrated.pkl")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = f"mlb-hit-predictor-{ENVIRONMENT}-bucket"


def log(msg):
    """Print with flush for immediate container log visibility."""
    print(f"[MODEL] {msg}", flush=True)


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Linux returns KB
    except Exception:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if "VmRSS" in line:
                        return int(line.split()[1]) / 1024
        except Exception:
            return 0.0
    return 0.0


def _log_memory(label: str):
    """Log current memory usage."""
    mb = _get_memory_mb()
    log(f"Memory [{label}]: {mb:.0f} MB")


def _write_status(s3_client, status: str, detail: str = ""):
    """Write training status to S3 so it's visible even if container dies."""
    try:
        payload = {
            "status": status,
            "detail": detail,
            "timestamp": time.strftime("%Y-%m-%d %Human:%M:%S UTC", time.gmtime()),
            "environment": ENVIRONMENT,
        }
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key="models/training_status.json",
            Body=json.dumps(payload, indent=2),
            ContentType="application/json",
        )
    except Exception as e:
        log(f"WARNING: Could not write training status to S3: {e}")


def _save_model_to_s3(s3_client, local_path: str, s3_key: str):
    """Upload a model file to S3."""
    try:
        with open(local_path, "rb") as f:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=f.read(),
                ContentType="application/octet-stream",
            )
        size_kb = os.path.getsize(local_path) / 1024
        log(f"S3 upload: s3://{S3_BUCKET}/{s3_key} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        log(f"ERROR: S3 upload failed for {s3_key}: {e}")
        return False


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_model(
    df: pd.DataFrame,
    feature_cols: list = None,
    test_size: float = 0.2,
    calibrate: bool = True,
) -> Tuple[object, dict]:
    """
    Train XGBoost classifier on engineered features.

    Args:
        df: DataFrame with features + target + sample_weight
        feature_cols: Which features to use. Defaults to BASIC_FEATURE_COLUMNS.
        test_size: Fraction for holdout test set
        calibrate: If True, calibrate probabilities using Platt scaling

    Returns:
        (model, metrics_dict)
    """
    import boto3
    s3 = boto3.client("s3")

    log("=" * 60)
    log("MODEL TRAINING START")
    log("=" * 60)
    _log_memory("start")
    train_start = time.time()

    _write_status(s3, "started", "Feature selection and data preparation")

    if feature_cols is None:
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        if len(available) < 5:
            available = [c for c in BASIC_FEATURE_COLUMNS if c in df.columns]
        feature_cols = available

    log(f"Features selected: {len(feature_cols)}")
    for i, col in enumerate(feature_cols):
        log(f"  {i+1:2d}. {col}")

    # ----- Prep data -----
    log("")
    log("--- Data Preparation ---")
    log(f"Raw input: {len(df):,} rows, {len(df.columns)} columns")
    _log_memory("before clean")

    required_cols = feature_cols + [TARGET_COLUMN]
    if WEIGHT_COLUMN in df.columns:
        required_cols.append(WEIGHT_COLUMN)

    clean = df.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    dropped_nan = len(df) - len(clean)
    log(f"Dropped {dropped_nan:,} rows with NaN values ({dropped_nan/len(df)*100:.1f}%)")

    if "game_num_in_season" in clean.columns:
        before = len(clean)
        clean = clean[clean["game_num_in_season"] >= 7]
        dropped_early = before - len(clean)
        log(f"Dropped {dropped_early:,} early-season rows (< 7 games)")

    log(f"Clean training data: {len(clean):,} rows")
    _log_memory("after clean")

    if len(clean) < 100:
        log("ERROR: Not enough clean data to train. Need at least 100 rows.")
        _write_status(s3, "failed", "Insufficient data")
        return None, {"error": "Insufficient data"}

    if "season" in clean.columns:
        season_counts = clean["season"].value_counts().sort_index()
        log(f"Seasons in dataset:")
        for season, count in season_counts.items():
            log(f"  {int(season)}: {count:,} rows")

    X = clean[feature_cols]
    y = clean[TARGET_COLUMN]
    weights = clean[WEIGHT_COLUMN] if WEIGHT_COLUMN in clean.columns else None

    hit_rate_overall = y.mean()
    log(f"Overall hit rate: {hit_rate_overall:.1%} ({int(y.sum()):,} hits / {len(y):,} games)")

    if weights is not None:
        weight_dist = clean[WEIGHT_COLUMN].value_counts().sort_index()
        log(f"Sample weight distribution:")
        for w, count in weight_dist.items():
            log(f"  weight={w}: {count:,} rows")

    # ----- Train/test split -----
    log("")
    log("--- Train/Test Split ---")
    _write_status(s3, "in_progress", "Train/test split")

    if "date" in clean.columns:
        clean_sorted = clean.sort_values("date")
        split_idx = int(len(clean_sorted) * (1 - test_size))
        train_data = clean_sorted.iloc[:split_idx]
        test_data = clean_sorted.iloc[split_idx:]

        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data[TARGET_COLUMN]
        y_test = test_data[TARGET_COLUMN]
        w_train = train_data[WEIGHT_COLUMN] if weights is not None else None
        w_test = test_data[WEIGHT_COLUMN] if weights is not None else None

        if "date" in train_data.columns:
            log(f"Time-based split (chronological)")
            log(f"  Train period: {train_data['date'].min()} to {train_data['date'].max()}")
            log(f"  Test period:  {test_data['date'].min()} to {test_data['date'].max()}")
    else:
        split = train_test_split(
            X, y, weights,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
        if weights is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = split
        else:
            X_train, X_test, y_train, y_test = split[:4]
            w_train, w_test = None, None
        log(f"Random stratified split")

    log(f"  Train: {len(X_train):,} rows (hit rate: {y_train.mean():.1%})")
    log(f"  Test:  {len(X_test):,} rows (hit rate: {y_test.mean():.1%})")
    _log_memory("after split")

    # ----- XGBoost model -----
    log("")
    log("--- XGBoost Training ---")
    _write_status(s3, "in_progress", f"XGBoost training ({len(X_train):,} rows)")

    hit_rate = y_train.mean()
    scale_pos_weight = (1 - hit_rate) / hit_rate if hit_rate > 0 else 1.0
    log(f"Class balance: scale_pos_weight = {scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
    )

    log(f"Hyperparameters:")
    log(f"  n_estimators:      500 (early stopping at 50)")
    log(f"  max_depth:         6")
    log(f"  learning_rate:     0.05")
    log(f"  subsample:         0.8")
    log(f"  colsample_bytree:  0.8")
    log(f"  min_child_weight:  5")
    log(f"Training XGBoost on {len(X_train):,} rows...")

    fit_start = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    fit_time = time.time() - fit_start

    best_iteration = getattr(model, 'best_iteration', 'N/A')
    best_score = getattr(model, 'best_score', 'N/A')
    log(f"XGBoost training complete in {fit_time:.1f}s")
    log(f"  Best iteration: {best_iteration}")
    log(f"  Best eval score: {best_score}")
    _log_memory("after XGBoost fit")

    # ----- Save uncalibrated checkpoint to S3 immediately -----
    # This ensures we always have a usable model even if calibration OOMs
    log("")
    log("--- Saving Uncalibrated Checkpoint ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, "hit_predictor_checkpoint.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    _save_model_to_s3(s3, checkpoint_path, "models/hit_predictor_checkpoint.pkl")
    _save_model_to_s3(s3, os.path.join(MODEL_DIR, "feature_cols.pkl"), "models/feature_cols.pkl")
    _write_status(s3, "in_progress", "Uncalibrated checkpoint saved — starting calibration")
    log(f"Checkpoint saved to S3 — model is safe even if calibration fails")

    # ----- Calibration (makes probabilities more accurate) -----
    final_model = model
    calibrated = False
    if calibrate and len(X_test) > 50:
        log("")
        log("--- Probability Calibration ---")
        _log_memory("before calibration")
        log("Calibrating with Platt scaling (sigmoid, 5-fold CV)...")
        log(f"Note: calibration refits model 5x — peak memory usage is here")
        cal_start = time.time()
        try:
            cal_params = model.get_params()
            cal_params.pop('early_stopping_rounds', None)
            cal_base = xgb.XGBClassifier(**cal_params)
            log(f"Refitting base model without early stopping for calibration...")
            cal_base.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            _log_memory("after cal_base fit")
            log(f"Running 5-fold CV calibration...")
            cal_model = CalibratedClassifierCV(cal_base, cv=5, method="sigmoid")
            cal_model.fit(X_train, y_train, sample_weight=w_train)
            final_model = cal_model
            calibrated = True
            cal_time = time.time() - cal_start
            log(f"Calibration complete in {cal_time:.1f}s")
            _log_memory("after calibration")
        except MemoryError as e:
            log(f"WARNING: Calibration OOM — using uncalibrated model")
            log(f"  Error: {e}")
            log(f"  The uncalibrated checkpoint is already saved to S3")
            final_model = model
            calibrated = False
        except Exception as e:
            log(f"WARNING: Calibration failed — using uncalibrated model")
            log(f"  Error: {e}")
            log(f"  Traceback: {traceback.format_exc()}")
            final_model = model
            calibrated = False

    # ----- Evaluate -----
    log("")
    log("--- Model Evaluation ---")
    _write_status(s3, "in_progress", "Evaluating model")

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "log_loss": round(log_loss(y_test, y_prob), 4),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "hit_rate_test": round(y_test.mean(), 4),
        "calibrated": calibrated,
        "features_used": feature_cols,
    }

    log(f"  Accuracy:   {metrics['accuracy']:.4f}")
    log(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    log(f"  Precision:  {metrics['precision']:.4f}")
    log(f"  Recall:     {metrics['recall']:.4f}")
    log(f"  Log Loss:   {metrics['log_loss']:.4f}")
    log(f"  Calibrated: {'Yes' if calibrated else 'No (OOM or error — using uncalibrated)'}")

    log(f"  Probability distribution (test set):")
    log(f"    Mean:   {y_prob.mean():.3f}")
    log(f"    Median: {np.median(y_prob):.3f}")
    log(f"    Min:    {y_prob.min():.3f}")
    log(f"    Max:    {y_prob.max():.3f}")
    log(f"    Std:    {y_prob.std():.3f}")

    # Confidence tiers
    log(f"  Confidence tiers:")
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
        count = (y_prob >= threshold).sum()
        log(f"    >= {threshold:.0%}: {count} batters")

    # ----- Feature importance -----
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        metrics["feature_importance"] = sorted_imp
        log("")
        log("--- Feature Importance ---")
        for i, (feat, imp) in enumerate(sorted_imp.items()):
            bar = "█" * int(imp * 100)
            log(f"  {i+1:2d}. {feat:30s} {imp:.4f} {bar}")

    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            metrics[k] = float(v)

    # ----- Save final model locally and to S3 -----
    log("")
    log("--- Saving Final Model ---")
    _write_status(s3, "in_progress", "Saving final model to S3")
    _log_memory("before final save")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    model_size = os.path.getsize(MODEL_PATH)
    log(f"Model saved locally to {MODEL_PATH} ({model_size / 1024:.1f} KB)")

    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    log(f"Feature columns saved ({len(feature_cols)} features)")

    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Upload final model + metadata to S3
    log("Uploading final model to S3...")
    model_saved = _save_model_to_s3(s3, MODEL_PATH, "models/hit_predictor.pkl")
    _save_model_to_s3(s3, os.path.join(MODEL_DIR, "feature_cols.pkl"), "models/feature_cols.pkl")
    _save_model_to_s3(s3, metrics_path, "models/metrics.json")

    if model_saved:
        log(f"Final model saved to s3://{S3_BUCKET}/models/hit_predictor.pkl")
    else:
        log(f"WARNING: S3 upload failed — model only saved locally at {MODEL_PATH}")

    total_time = time.time() - train_start
    log("")
    log("=" * 60)
    log("MODEL TRAINING COMPLETE")
    log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log(f"  Features:   {len(feature_cols)}")
    log(f"  Train rows: {len(X_train):,}")
    log(f"  Test rows:  {len(X_test):,}")
    log(f"  Accuracy:   {metrics['accuracy']:.4f}")
    log(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    log(f"  Calibrated: {'Yes' if calibrated else 'No'}")
    log(f"  S3:         {'✓ saved' if model_saved else '✗ FAILED'}")
    log("=" * 60)
    _log_memory("complete")

    _write_status(s3, "complete",
        f"accuracy={metrics['accuracy']:.4f} auc={metrics['auc_roc']:.4f} "
        f"calibrated={calibrated} time={total_time:.0f}s")

    return final_model, metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def load_model() -> Tuple[Optional[object], Optional[list]]:
    """
    Load saved model and feature column order.
    Tries local /tmp first, then falls back to S3.
    """
    # Try local first
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
        logger.info(f"Model loaded from local: {MODEL_PATH}")
        return model, feature_cols
    except FileNotFoundError:
        pass

    # Fall back to S3
    logger.info("Local model not found — loading from S3...")
    try:
        import boto3
        s3 = boto3.client("s3")

        os.makedirs(MODEL_DIR, exist_ok=True)

        # Download model
        s3.download_file(S3_BUCKET, "models/hit_predictor.pkl", MODEL_PATH)
        logger.info(f"Downloaded model from s3://{S3_BUCKET}/models/hit_predictor.pkl")

        # Download feature cols
        feature_cols_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
        s3.download_file(S3_BUCKET, "models/feature_cols.pkl", feature_cols_path)

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(feature_cols_path, "rb") as f:
            feature_cols = pickle.load(f)

        logger.info(f"Model loaded from S3 ({len(feature_cols)} features)")
        return model, feature_cols

    except Exception as e:
        logger.error(f"Failed to load model from S3: {e}")
        return None, None


def predict_daily(
    df: pd.DataFrame,
    model=None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Generate hit predictions for today's batters.
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model()
        if model is None:
            logger.error("No model available for prediction")
            return pd.DataFrame()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing features (filling with median): {missing}")
        for col in missing:
            df[col] = df[col].median() if col in df.columns else 0

    clean = df.dropna(subset=[c for c in feature_cols if c in df.columns])
    if clean.empty:
        logger.warning("No valid rows for prediction")
        return pd.DataFrame()

    X = clean[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]

    output = clean[["player_id", "player_name", "team", "opponent", "date"]].copy()
    if "game_pk" in clean.columns:
        output["game_pk"] = clean["game_pk"]
    if "bat_side" in clean.columns:
        output["bat_side"] = clean["bat_side"]

    output["confidence_pct"] = (probabilities * 100).round(2)
    output = output.sort_values("confidence_pct", ascending=False).reset_index(drop=True)
    output["rank"] = range(1, len(output) + 1)

    logger.info(f"Generated predictions for {len(output)} batters")
    if len(output) > 0:
        top = output.iloc[0]
        logger.info(f"Top pick: {top['player_name']} ({top['confidence_pct']}%)")

    return output


# ---------------------------------------------------------------------------
# Streak simulation (for backtesting)
# ---------------------------------------------------------------------------
def simulate_streak(
    predictions_by_date: dict,
    actuals: pd.DataFrame,
    pick_count: int = 1,
) -> dict:
    """
    Simulate playing Beat the Streak using model predictions.
    """
    logger.info(f"Simulating streak with top-{pick_count} picks...")

    current_streak = 0
    longest_streak = 0
    total_correct = 0
    total_games = 0
    daily_log = []

    for date in sorted(predictions_by_date.keys()):
        preds = predictions_by_date[date]
        if preds.empty:
            continue

        picks = preds.head(pick_count)
        day_success = True

        for _, pick in picks.iterrows():
            pid = pick["player_id"]
            actual = actuals[
                (actuals["player_id"] == pid) &
                (actuals["date"] == date)
            ]

            if actual.empty:
                continue

            got_hit = actual.iloc[0]["got_hit"]
            total_games += 1

            if got_hit == 1:
                total_correct += 1
            else:
                day_success = False

        if day_success and len(picks) > 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

        daily_log.append({
            "date": date,
            "picks": picks["player_name"].tolist(),
            "confidence": picks["confidence_pct"].tolist(),
            "streak_after": current_streak,
        })

    results = {
        "longest_streak": longest_streak,
        "total_correct": total_correct,
        "total_games": total_games,
        "hit_rate": round(total_correct / max(total_games, 1), 4),
        "daily_log": daily_log,
    }

    logger.info(f"Simulation complete: longest streak = {longest_streak}, "
                f"hit rate = {results['hit_rate']}")
    return results