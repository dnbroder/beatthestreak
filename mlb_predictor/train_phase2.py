"""
train_phase2.py — Local-only Phase 2 training for CloudShell validation.

This module is INTENTIONALLY separate from model.py because:
  1. model.py writes to S3 and manages production model artifacts.
     We don't want "I'm just running a backtest" to touch production.
  2. CloudShell has limited memory (~1GB) vs ECS (2GB+). We skip
     calibration by default to stay under the memory ceiling.
  3. The whole point of this file is local iteration before promotion.
     Once Phase 2 clears the +1.5pp gate, we inline the feature
     changes into model.py for production and delete this file.

What it does:
  1. Pulls 2021-2024 training data with Phase 2 pitcher enrichment
  2. Runs feature_engineering.py end-to-end
  3. Trains XGBoost with identical hyperparameters to production model.py
  4. Saves locally to ~/beatthestreak/models_local/
  5. Returns model + metrics for the backtest script to consume

Usage:
    # Build dataset + train, save to models_local/
    python3 -m mlb_predictor.train_phase2

    # Skip dataset build, use cached parquet from prior run
    python3 -m mlb_predictor.train_phase2 --use-cached-data

    # Build dataset only, don't train (useful if you want to inspect data first)
    python3 -m mlb_predictor.train_phase2 --build-only

Logging conventions:
  - [TRAIN_P2] routine output
  - [TRAIN_P2]   [HEALTH] grepable metrics
  - [TRAIN_P2]   [HEALTH] WARNING anomalies
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    log_loss,
)

from mlb_predictor.historical_data import build_multi_season_dataset
from mlb_predictor.feature_engineering import (
    engineer_features,
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_PHASE2,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)


logger = logging.getLogger("beatthestreak")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


# Local-only paths. NEVER touches production S3.
LOCAL_DIR = os.path.expanduser("~/beatthestreak/models_local")
DATA_CACHE_PATH = os.path.join(LOCAL_DIR, "phase2_training_data.parquet")
MODEL_V1_PATH = os.path.join(LOCAL_DIR, "hit_predictor_v1_baseline.pkl")
MODEL_V2_PATH = os.path.join(LOCAL_DIR, "hit_predictor_v2_phase2.pkl")
FEATURE_COLS_V1_PATH = os.path.join(LOCAL_DIR, "feature_cols_v1.pkl")
FEATURE_COLS_V2_PATH = os.path.join(LOCAL_DIR, "feature_cols_v2.pkl")
METRICS_V1_PATH = os.path.join(LOCAL_DIR, "metrics_v1.json")
METRICS_V2_PATH = os.path.join(LOCAL_DIR, "metrics_v2.json")


def log(msg):
    logger.info(f"[TRAIN_P2] {msg}")


def health(msg):
    logger.info(f"[TRAIN_P2]   [HEALTH] {msg}")


def warn(msg):
    logger.warning(f"[TRAIN_P2]   [HEALTH] WARNING: {msg}")


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------
def build_phase2_dataset(
    start_year: int = 2021,
    end_year: int = 2024,
    max_players_per_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build the 2021-2024 training dataset with Phase 2 pitcher enrichment.
    Caches to parquet so subsequent runs can skip the ~30-45 min API pull.
    """
    log("=" * 70)
    log(f"STEP 1 — Building Phase 2 dataset ({start_year}-{end_year})")
    log("=" * 70)
    build_start = time.time()

    raw = build_multi_season_dataset(
        start_year=start_year,
        end_year=end_year,
        max_players_per_season=max_players_per_season,
        enrich_pitcher_features=True,
    )

    if raw.empty:
        warn("empty_dataset — build failed")
        return raw

    log("")
    log(f"Raw dataset: {len(raw):,} rows, {len(raw.columns)} columns")
    log("")
    log("Running feature engineering (including Phase 2 interaction recompute)...")
    engineered = engineer_features(raw)

    build_elapsed = time.time() - build_start
    log(f"Engineered dataset: {len(engineered):,} rows, {len(engineered.columns)} columns")
    log(f"Build elapsed: {build_elapsed:.0f}s ({build_elapsed/60:.1f} min)")

    # Cache to disk
    os.makedirs(LOCAL_DIR, exist_ok=True)
    engineered.to_parquet(DATA_CACHE_PATH, index=False)
    size_mb = os.path.getsize(DATA_CACHE_PATH) / 1024 / 1024
    log(f"Cached to: {DATA_CACHE_PATH} ({size_mb:.1f} MB)")

    # HEALTH: Phase 2 feature coverage
    phase2_present = [c for c in FEATURE_COLUMNS_PHASE2 if c in engineered.columns]
    health(f"dataset_built rows={len(engineered)} cols={len(engineered.columns)} "
           f"phase2_cols_present={len(phase2_present)}/{len(FEATURE_COLUMNS_PHASE2)} "
           f"duration_sec={build_elapsed:.0f}")

    if len(phase2_present) < len(FEATURE_COLUMNS_PHASE2):
        missing = set(FEATURE_COLUMNS_PHASE2) - set(phase2_present)
        warn(f"phase2_features_missing_from_dataset cols={list(missing)}")

    return engineered


def load_cached_dataset() -> Optional[pd.DataFrame]:
    """Load the previously-built training dataset from local parquet."""
    if not os.path.exists(DATA_CACHE_PATH):
        warn(f"no_cached_data path={DATA_CACHE_PATH}")
        return None
    log(f"Loading cached dataset from {DATA_CACHE_PATH}...")
    df = pd.read_parquet(DATA_CACHE_PATH)
    size_mb = os.path.getsize(DATA_CACHE_PATH) / 1024 / 1024
    log(f"Loaded {len(df):,} rows, {len(df.columns)} columns ({size_mb:.1f} MB)")
    health(f"dataset_loaded_from_cache rows={len(df)} cols={len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model_local(
    df: pd.DataFrame,
    feature_cols: list,
    label: str = "v2",
    test_size: float = 0.2,
    calibrate: bool = False,  # default OFF for CloudShell memory
) -> Tuple[object, dict]:
    """
    Train XGBoost classifier using local-only I/O — no S3.

    Args:
        df: DataFrame with features + target + sample_weight
        feature_cols: Which features to use (either V1 or V2 list)
        label: "v1" or "v2" for logging and file naming
        test_size: Fraction for holdout test set
        calibrate: Platt scaling — costs ~2x memory, default OFF

    Returns:
        (model, metrics_dict)
    """
    log("=" * 70)
    log(f"STEP 2 — Training {label} model ({len(feature_cols)} features)")
    log("=" * 70)
    train_start = time.time()

    # Validate feature availability
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        warn(f"missing_feature_columns cols={missing}")
        return None, {"error": f"missing columns: {missing}"}

    # Clean — drop rows with NaN in any required feature
    required = feature_cols + [TARGET_COLUMN]
    if WEIGHT_COLUMN in df.columns:
        required.append(WEIGHT_COLUMN)

    clean = df.dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    dropped_nan = len(df) - len(clean)
    log(f"Dropped {dropped_nan:,} rows with NaN values ({dropped_nan/len(df)*100:.1f}%)")

    # Drop early-season rows (< 7 games) to match production training
    if "game_num_in_season" in clean.columns:
        before = len(clean)
        clean = clean[clean["game_num_in_season"] >= 7]
        dropped_early = before - len(clean)
        log(f"Dropped {dropped_early:,} early-season rows (< 7 games)")

    log(f"Clean training data: {len(clean):,} rows")
    if len(clean) < 100:
        warn("insufficient_training_data rows<100")
        return None, {"error": "Insufficient data"}

    if "season" in clean.columns:
        log("Season distribution:")
        for season, count in clean["season"].value_counts().sort_index().items():
            log(f"  {int(season)}: {count:,} rows")

    X = clean[feature_cols]
    y = clean[TARGET_COLUMN]
    weights = clean[WEIGHT_COLUMN] if WEIGHT_COLUMN in clean.columns else None

    hit_rate_overall = y.mean()
    log(f"Overall hit rate: {hit_rate_overall:.1%}")

    # Time-based train/test split (chronological, matches production model.py)
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

        log(f"Time-based split (chronological)")
        log(f"  Train period: {train_data['date'].min()} to {train_data['date'].max()}")
        log(f"  Test period:  {test_data['date'].min()} to {test_data['date'].max()}")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        w_train, w_test = None, None
        log("Random stratified split (date column missing)")

    log(f"  Train: {len(X_train):,} rows (hit rate: {y_train.mean():.1%})")
    log(f"  Test:  {len(X_test):,} rows (hit rate: {y_test.mean():.1%})")

    # XGBoost — identical hyperparameters to production model.py
    hit_rate = y_train.mean()
    scale_pos_weight = (1 - hit_rate) / hit_rate if hit_rate > 0 else 1.0

    log("")
    log(f"Training XGBoost ({label})...")
    log(f"  Hyperparameters match production model.py:")
    log(f"    n_estimators=500, max_depth=6, learning_rate=0.05")
    log(f"    subsample=0.8, colsample_bytree=0.8, min_child_weight=5")
    log(f"    scale_pos_weight={scale_pos_weight:.3f} (class balance)")

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

    fit_start = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    fit_time = time.time() - fit_start

    best_iteration = getattr(model, "best_iteration", "N/A")
    best_score = getattr(model, "best_score", "N/A")
    log(f"XGBoost training complete in {fit_time:.1f}s")
    log(f"  Best iteration: {best_iteration}")
    log(f"  Best eval score: {best_score}")

    # Optional calibration — skipped by default on CloudShell
    final_model = model
    calibrated = False
    if calibrate and len(X_test) > 50:
        log("")
        log("Running Platt calibration (5-fold CV)...")
        try:
            from sklearn.calibration import CalibratedClassifierCV
            cal_params = model.get_params()
            cal_params.pop("early_stopping_rounds", None)
            cal_base = xgb.XGBClassifier(**cal_params)
            cal_base.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            cal_model = CalibratedClassifierCV(cal_base, cv=5, method="sigmoid")
            cal_model.fit(X_train, y_train, sample_weight=w_train)
            final_model = cal_model
            calibrated = True
            log(f"Calibration complete")
        except MemoryError:
            warn("calibration_oom — using uncalibrated model (CloudShell is tight on memory)")
        except Exception as e:
            warn(f"calibration_failed error={e} — using uncalibrated")
    else:
        log("Calibration skipped (calibrate=False, CloudShell memory-safe default)")

    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "label": label,
        "feature_count": len(feature_cols),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_test, y_prob)), 4),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "hit_rate_test": round(float(y_test.mean()), 4),
        "calibrated": calibrated,
        "training_time_sec": round(fit_time, 1),
        "features_used": feature_cols,
    }

    log("")
    log(f"Test set metrics ({label}):")
    log(f"  Accuracy:   {metrics['accuracy']:.4f}")
    log(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    log(f"  Precision:  {metrics['precision']:.4f}")
    log(f"  Recall:     {metrics['recall']:.4f}")
    log(f"  Log Loss:   {metrics['log_loss']:.4f}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
        sorted_imp = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
        metrics["feature_importance"] = sorted_imp

        log("")
        log(f"--- Feature Importance ({label}) ---")
        for i, (feat, imp) in enumerate(sorted_imp.items()):
            bar = "█" * max(1, int(imp * 100))
            flag = "  [PHASE2]" if feat in FEATURE_COLUMNS_PHASE2 else ""
            log(f"  {i+1:2d}. {feat:35s} {imp:.4f} {bar}{flag}")

    # Save locally
    os.makedirs(LOCAL_DIR, exist_ok=True)
    model_path = MODEL_V2_PATH if label == "v2" else MODEL_V1_PATH
    cols_path = FEATURE_COLS_V2_PATH if label == "v2" else FEATURE_COLS_V1_PATH
    metrics_path = METRICS_V2_PATH if label == "v2" else METRICS_V1_PATH

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(cols_path, "wb") as f:
        pickle.dump(feature_cols, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    log(f"")
    log(f"Saved locally:")
    log(f"  Model:    {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")
    log(f"  Features: {cols_path}")
    log(f"  Metrics:  {metrics_path}")

    total_time = time.time() - train_start
    log(f"")
    log(f"{label} training complete in {total_time:.0f}s")
    health(f"training_complete label={label} accuracy={metrics['accuracy']} "
           f"auc={metrics['auc_roc']} features={len(feature_cols)} "
           f"train_rows={len(X_train)} test_rows={len(X_test)} "
           f"duration_sec={total_time:.0f}")

    return final_model, metrics


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-cached-data", action="store_true",
                        help="Skip dataset build, use existing parquet cache")
    parser.add_argument("--build-only", action="store_true",
                        help="Only build the dataset, don't train models")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--calibrate", action="store_true",
                        help="Run Platt calibration (default OFF for CloudShell memory safety)")
    parser.add_argument("--max-players", type=int, default=None,
                        help="Cap players per season (for quick testing)")
    parser.add_argument("--skip-v1-baseline", action="store_true",
                        help="Only train Phase 2 model, skip v1 baseline comparison")
    args = parser.parse_args()

    overall_start = time.time()
    log("=" * 70)
    log("PHASE 2 LOCAL TRAINING — CloudShell backtest validation")
    log("=" * 70)
    log(f"Year range: {args.start_year}-{args.end_year}")
    log(f"Use cached data: {args.use_cached_data}")
    log(f"Build only: {args.build_only}")
    log(f"Calibrate: {args.calibrate}")
    log(f"Output dir: {LOCAL_DIR}")
    log("")

    # Step 1: Dataset
    if args.use_cached_data:
        df = load_cached_dataset()
        if df is None:
            log("No cache found. Run without --use-cached-data to build.")
            return 1
    else:
        df = build_phase2_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
            max_players_per_season=args.max_players,
        )
        if df.empty:
            return 1

    if args.build_only:
        log("")
        log("--build-only set, skipping training. Cached data ready at:")
        log(f"  {DATA_CACHE_PATH}")
        return 0

    # Step 2: Train v1 baseline (for head-to-head comparison)
    if not args.skip_v1_baseline:
        log("")
        log("Training V1 baseline (no Phase 2 features)...")
        v1_model, v1_metrics = train_model_local(
            df, FEATURE_COLUMNS_V1, label="v1", calibrate=args.calibrate
        )
    else:
        log("Skipping v1 baseline (--skip-v1-baseline)")
        v1_metrics = None

    # Step 3: Train v2 Phase 2 model
    log("")
    log("Training V2 Phase 2 (all 37 features)...")
    v2_model, v2_metrics = train_model_local(
        df, FEATURE_COLUMNS_V2, label="v2", calibrate=args.calibrate
    )

    # Side-by-side summary
    overall_time = time.time() - overall_start
    log("")
    log("=" * 70)
    log("PHASE 2 TRAINING SUMMARY")
    log("=" * 70)
    if v1_metrics:
        log(f"{'Metric':<14} {'V1 Baseline':>14} {'V2 Phase 2':>14} {'Delta':>10}")
        log("-" * 55)
        for metric_name in ["accuracy", "auc_roc", "precision", "recall", "log_loss"]:
            v1 = v1_metrics.get(metric_name, 0)
            v2 = v2_metrics.get(metric_name, 0)
            delta = v2 - v1
            sign = "+" if delta >= 0 else ""
            log(f"{metric_name:<14} {v1:>14.4f} {v2:>14.4f} {sign}{delta:>9.4f}")
    else:
        log(f"V2 Phase 2: accuracy={v2_metrics['accuracy']:.4f} "
            f"auc={v2_metrics['auc_roc']:.4f}")
    log("")
    log(f"Total training time: {overall_time:.0f}s ({overall_time/60:.1f} min)")
    log("")
    log("Models saved to:")
    if v1_metrics:
        log(f"  V1 baseline: {MODEL_V1_PATH}")
    log(f"  V2 Phase 2:  {MODEL_V2_PATH}")
    log("")
    log("Next step: run scripts/backtest_phase2.py to compare on held-out 2024")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())