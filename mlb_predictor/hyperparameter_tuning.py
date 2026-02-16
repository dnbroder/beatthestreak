"""
hyperparameter_tuning.py – XGBoost hyperparameter optimization via cross-validation.

Uses RandomizedSearchCV to find optimal parameters, then retrains on the full
dataset with the best params. Saves the tuned model to the same path as the
standard model so backtesting picks it up automatically.

Run via: POST /train/tuned?start_year=2021&end_year=2023
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    classification_report, log_loss,
)
from xgboost import XGBClassifier
import pickle
import boto3

from mlb_predictor.feature_engineering import FEATURE_COLUMNS
from mlb_predictor.model import MODEL_PATH

logger = logging.getLogger("beatthestreak")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = f"mlb-hit-predictor-{ENVIRONMENT}-bucket"


def log(msg):
    """Print with flush for immediate container log visibility."""
    print(f"[TUNE] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators": [200, 300, 400, 500, 600, 800],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 7, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 5.0],
    "scale_pos_weight": [1.0, 1.5, 2.0, 2.5],
}

# Total possible combinations
_total_combos = 1
for v in PARAM_GRID.values():
    _total_combos *= len(v)


def run_hyperparameter_tuning(
    featured_df: pd.DataFrame,
    n_iter: int = 60,
    cv_folds: int = 5,
    write_to_s3: bool = True,
) -> tuple:
    """
    Run hyperparameter tuning on the featured dataset.

    Args:
        featured_df: DataFrame with engineered features and 'got_hit' target
        n_iter: Number of random parameter combinations to try
        cv_folds: Number of cross-validation folds
        write_to_s3: Whether to write results to S3

    Returns:
        (model, metrics_dict) — tuned model and performance metrics
    """
    log("=" * 60)
    log("HYPERPARAMETER TUNING START")
    log("=" * 60)
    tune_start = time.time()

    log(f"  Search iterations: {n_iter} of {_total_combos:,} possible combinations")
    log(f"  CV folds: {cv_folds}")
    log(f"  Total model fits: {n_iter * cv_folds}")
    log(f"  Dataset: {len(featured_df):,} rows")

    # Prepare data
    feature_cols = [c for c in FEATURE_COLUMNS if c in featured_df.columns]
    X = featured_df[feature_cols].copy()
    y = featured_df["got_hit"].copy()

    # Fill NaN
    nan_before = X.isna().sum().sum()
    X = X.fillna(0)
    if nan_before > 0:
        log(f"  Filled {nan_before:,} NaN values with 0")

    log(f"  Features: {len(feature_cols)}")
    log(f"  Target distribution: hits={int(y.sum()):,}, misses={int((1-y).sum()):,}")
    log(f"  Hit rate: {y.mean():.1%}")

    if "season" in featured_df.columns:
        season_counts = featured_df["season"].value_counts().sort_index()
        log(f"  Seasons:")
        for season, count in season_counts.items():
            log(f"    {int(season)}: {count:,} rows")

    log("")
    log("Search space:")
    for param, values in PARAM_GRID.items():
        log(f"  {param:25s}: {values}")

    # Base estimator
    base_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # Stratified K-Fold to maintain hit/miss ratio in each fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Randomized search
    log("")
    log(f"Starting RandomizedSearchCV...")
    log(f"Estimated time: {n_iter * cv_folds * 0.5:.0f}–{n_iter * cv_folds * 2:.0f}s "
        f"({n_iter * cv_folds * 0.5 / 60:.0f}–{n_iter * cv_folds * 2 / 60:.0f} min)")

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",  # AUC is better than accuracy for ranking
        random_state=42,
        verbose=1,
        n_jobs=-1,
        return_train_score=True,
    )

    search_start = time.time()
    search.fit(X, y)
    search_time = time.time() - search_start

    # Results
    best_params = search.best_params_
    best_score = search.best_score_

    log("")
    log("=" * 60)
    log(f"SEARCH COMPLETE in {search_time:.0f}s ({search_time/60:.1f} min)")
    log("=" * 60)
    log("")
    log("BEST PARAMETERS FOUND:")
    for param, value in sorted(best_params.items()):
        log(f"  {param:25s} = {value}")
    log(f"  Best CV AUC: {best_score:.4f}")

    # Show top 10 parameter combos
    log("")
    log("Top 10 parameter combinations:")
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    for i, (_, row) in enumerate(results_df.head(10).iterrows()):
        log(f"  Rank {int(row['rank_test_score']):2d}: "
            f"AUC={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f}), "
            f"train_AUC={row['mean_train_score']:.4f}")

    # Check for overfitting
    top_result = results_df.iloc[0]
    train_auc = top_result["mean_train_score"]
    test_auc = top_result["mean_test_score"]
    overfit_gap = train_auc - test_auc

    log("")
    log(f"--- Overfit Analysis ---")
    log(f"  Train AUC: {train_auc:.4f}")
    log(f"  Test AUC:  {test_auc:.4f}")
    log(f"  Gap:       {overfit_gap:.4f}")
    if overfit_gap > 0.05:
        log("  WARNING: Possible overfitting detected (gap > 5%)")
    elif overfit_gap > 0.03:
        log("  CAUTION: Moderate gap (3-5%), monitor on OOS data")
    else:
        log("  OK: Minimal overfitting")

    # AUC distribution across all iterations
    all_scores = results_df["mean_test_score"]
    log("")
    log(f"--- Score Distribution (all {n_iter} iterations) ---")
    log(f"  Mean AUC:   {all_scores.mean():.4f}")
    log(f"  Median AUC: {all_scores.median():.4f}")
    log(f"  Min AUC:    {all_scores.min():.4f}")
    log(f"  Max AUC:    {all_scores.max():.4f}")
    log(f"  Std:        {all_scores.std():.4f}")

    # Retrain on FULL dataset with best params
    log("")
    log("--- Retraining on Full Dataset ---")
    log(f"Using best parameters on all {len(X):,} rows...")

    retrain_start = time.time()
    best_xgb = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    best_xgb.fit(X, y)
    retrain_time = time.time() - retrain_start
    log(f"Retrain complete in {retrain_time:.1f}s")

    # Calibrate probabilities
    log("")
    log("--- Probability Calibration ---")
    log("Calibrating with isotonic regression (3-fold CV)...")
    cal_start = time.time()
    calibrated = CalibratedClassifierCV(best_xgb, cv=3, method="isotonic")
    calibrated.fit(X, y)
    cal_time = time.time() - cal_start
    log(f"Calibration complete in {cal_time:.1f}s")

    # Evaluate on training data (for comparison with previous runs)
    y_pred = calibrated.predict(X)
    y_prob = calibrated.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    logloss = log_loss(y, y_prob)

    log("")
    log("--- In-Sample Metrics (full dataset) ---")
    log(f"  Accuracy:   {accuracy:.4f}")
    log(f"  AUC:        {auc:.4f}")
    log(f"  Brier:      {brier:.4f}")
    log(f"  Log Loss:   {logloss:.4f}")

    # Probability distribution
    log(f"  Probability distribution:")
    log(f"    Mean:   {y_prob.mean():.3f}")
    log(f"    Median: {np.median(y_prob):.3f}")
    log(f"    Min:    {y_prob.min():.3f}")
    log(f"    Max:    {y_prob.max():.3f}")

    # Feature importance from best model
    if hasattr(best_xgb, "feature_importances_"):
        importance = dict(zip(feature_cols, [float(x) for x in best_xgb.feature_importances_]))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        log("")
        log("--- Feature Importance (tuned model) ---")
        for i, (feat, imp) in enumerate(sorted_imp):
            bar = "█" * int(imp * 100)
            log(f"  {i+1:2d}. {feat:30s} {imp:.4f} {bar}")

    # Save model
    log("")
    log("--- Saving Model ---")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    model_size = os.path.getsize(MODEL_PATH)
    with open(os.path.join(os.path.dirname(MODEL_PATH), "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    log(f"Tuned model saved to {MODEL_PATH} ({model_size / 1024:.1f} KB)")
    log(f"Feature columns saved ({len(feature_cols)} features)")

    # Build metrics dict
    metrics = {
        "tuning": {
            "n_iter": n_iter,
            "cv_folds": cv_folds,
            "best_cv_auc": round(best_score, 4),
            "train_auc": round(train_auc, 4),
            "test_auc": round(test_auc, 4),
            "overfit_gap": round(overfit_gap, 4),
            "search_time_seconds": round(search_time, 1),
        },
        "best_params": best_params,
        "full_dataset_metrics": {
            "accuracy": round(accuracy, 4),
            "auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "total_rows": len(X),
            "features": len(feature_cols),
        },
        "top_5_combos": [
            {
                "rank": int(row["rank_test_score"]),
                "mean_auc": round(row["mean_test_score"], 4),
                "std_auc": round(row["std_test_score"], 4),
                "params": row["params"],
            }
            for _, row in results_df.head(5).iterrows()
        ],
    }

    # Write to S3
    if write_to_s3:
        log("")
        log("--- S3 Upload ---")
        try:
            s3 = boto3.client("s3")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = f"tuning/{ts}/tuning_results.json"

            s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=json.dumps(metrics, indent=2, default=str),
                ContentType="application/json",
            )
            log(f"Results written to s3://{S3_BUCKET}/{key}")
        except Exception as e:
            log(f"ERROR: Failed to write to S3: {e}")

    total_time = time.time() - tune_start
    log("")
    log("=" * 60)
    log("HYPERPARAMETER TUNING COMPLETE")
    log(f"  Total time:    {total_time:.0f}s ({total_time/60:.1f} min)")
    log(f"  Search time:   {search_time:.0f}s")
    log(f"  Best CV AUC:   {best_score:.4f}")
    log(f"  Overfit gap:   {overfit_gap:.4f}")
    log(f"  Iterations:    {n_iter}")
    log(f"  Model fits:    {n_iter * cv_folds}")
    log("=" * 60)

    return calibrated, metrics