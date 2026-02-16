"""
model.py – XGBoost model for predicting whether a batter gets at least 1 hit.

Training uses recency-weighted samples (2000-2024).
Output is a probability (confidence %) for each batter.
"""

import os
import logging
import pickle
import time
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


def log(msg):
    """Print with flush for immediate container log visibility."""
    print(f"[MODEL] {msg}", flush=True)


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
    log("=" * 60)
    log("MODEL TRAINING START")
    log("=" * 60)
    train_start = time.time()

    if feature_cols is None:
        # Use whichever columns actually exist in the data
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

    # Drop rows with NaN in features or target
    required_cols = feature_cols + [TARGET_COLUMN]
    if WEIGHT_COLUMN in df.columns:
        required_cols.append(WEIGHT_COLUMN)

    clean = df.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    dropped_nan = len(df) - len(clean)
    log(f"Dropped {dropped_nan:,} rows with NaN values ({dropped_nan/len(df)*100:.1f}%)")

    # Drop very early season games (< 7 games played) — not enough history
    if "game_num_in_season" in clean.columns:
        before = len(clean)
        clean = clean[clean["game_num_in_season"] >= 7]
        dropped_early = before - len(clean)
        log(f"Dropped {dropped_early:,} early-season rows (< 7 games)")

    log(f"Clean training data: {len(clean):,} rows")

    if len(clean) < 100:
        log("ERROR: Not enough clean data to train. Need at least 100 rows.")
        return None, {"error": "Insufficient data"}

    # Dataset stats
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

    # ----- XGBoost model -----
    log("")
    log("--- XGBoost Training ---")

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
    log(f"Training XGBoost...")

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

    # ----- Calibration (makes probabilities more accurate) -----
    final_model = model
    if calibrate and len(X_test) > 50:
        log("")
        log("--- Probability Calibration ---")
        log("Calibrating with Platt scaling (sigmoid, 5-fold CV)...")
        cal_start = time.time()
        try:
            # Clone model without early_stopping_rounds for calibration
            cal_params = model.get_params()
            cal_params.pop('early_stopping_rounds', None)
            cal_base = xgb.XGBClassifier(**cal_params)
            log(f"Refitting base model without early stopping for calibration...")
            cal_base.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            cal_model = CalibratedClassifierCV(cal_base, cv=5, method="sigmoid")
            cal_model.fit(X_train, y_train, sample_weight=w_train)
            final_model = cal_model
            cal_time = time.time() - cal_start
            log(f"Calibration complete in {cal_time:.1f}s")
        except Exception as e:
            log(f"WARNING: Calibration failed, using uncalibrated model: {e}")

    # ----- Evaluate -----
    log("")
    log("--- Model Evaluation ---")

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
        "features_used": feature_cols,
    }

    log(f"  Accuracy:  {metrics['accuracy']:.4f}")
    log(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    log(f"  Precision: {metrics['precision']:.4f}")
    log(f"  Recall:    {metrics['recall']:.4f}")
    log(f"  Log Loss:  {metrics['log_loss']:.4f}")

    # Probability distribution
    log(f"  Probability distribution (test set):")
    log(f"    Mean:   {y_prob.mean():.3f}")
    log(f"    Median: {np.median(y_prob):.3f}")
    log(f"    Min:    {y_prob.min():.3f}")
    log(f"    Max:    {y_prob.max():.3f}")
    log(f"    Std:    {y_prob.std():.3f}")

    # ----- Feature importance -----
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        metrics["feature_importance"] = sorted_imp
        log("")
        log("--- Feature Importance (all features) ---")
        for i, (feat, imp) in enumerate(sorted_imp.items()):
            bar = "█" * int(imp * 100)
            log(f"  {i+1:2d}. {feat:30s} {imp:.4f} {bar}")

    # Ensure all metrics are JSON-serializable (no numpy types)
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            metrics[k] = float(v)

    # ----- Save model -----
    log("")
    log("--- Saving Model ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    model_size = os.path.getsize(MODEL_PATH)
    log(f"Model saved to {MODEL_PATH} ({model_size / 1024:.1f} KB)")

    # Save feature column order (needed for prediction)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    log(f"Feature columns saved ({len(feature_cols)} features)")

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
    log(f"  Calibrated: {'Yes' if final_model != model else 'No'}")
    log("=" * 60)

    return final_model, metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def load_model() -> Tuple[Optional[object], Optional[list]]:
    """Load saved model and feature column order."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
        return model, feature_cols
    except FileNotFoundError:
        logger.warning("No saved model found.")
        return None, None


def predict_daily(
    df: pd.DataFrame,
    model=None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Generate hit predictions for today's batters.

    Args:
        df: Today's batter data with engineered features
        model: Trained model (loads from disk if None)
        feature_cols: Feature columns used during training

    Returns:
        DataFrame sorted by confidence (highest first) with columns:
          player_name, team, opponent, confidence_pct, rank, features...
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model()
        if model is None:
            logger.error("No model available for prediction")
            return pd.DataFrame()

    # Ensure all feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing features (filling with median): {missing}")
        for col in missing:
            df[col] = df[col].median() if col in df.columns else 0

    # Drop rows where we can't compute features
    clean = df.dropna(subset=[c for c in feature_cols if c in df.columns])
    if clean.empty:
        logger.warning("No valid rows for prediction")
        return pd.DataFrame()

    X = clean[feature_cols]

    # Predict probability of getting a hit
    probabilities = model.predict_proba(X)[:, 1]

    # Build output
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

    Args:
        predictions_by_date: {date: DataFrame of ranked predictions}
        actuals: DataFrame with player_id, date, got_hit columns
        pick_count: How many players to pick per day (1 or 2)

    Returns:
        Dict with longest_streak, total_correct, total_games, daily_log
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

        # Pick top N players
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