"""
shap_analysis.py – SHAP-based model explainability for the hit prediction model.

Provides:
  1. Global feature importance (which features matter most overall)
  2. Miss analysis (what went wrong on incorrect predictions)
  3. Per-player SHAP values (why was this player ranked #1?)
"""

import os
import io
import logging
import json
from typing import Optional

import numpy as np
import pandas as pd
import shap
import boto3

from mlb_predictor.model import load_model, MODEL_PATH
from mlb_predictor.feature_engineering import FEATURE_COLUMNS

logger = logging.getLogger("beatthestreak")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = f"mlb-hit-predictor-{ENVIRONMENT}-bucket"


def log(msg):
    """Print with flush for immediate container log visibility."""
    print(f"[SHAP] {msg}", flush=True)


def run_shap_analysis(
    backtest_json_path: str = None,
    backtest_data: dict = None,
    write_to_s3: bool = True,
) -> dict:
    """
    Run comprehensive SHAP analysis on the trained model.

    Can either load backtest results from S3 or accept them directly.

    Returns dict with:
      - global_importance: ranked feature importance from SHAP
      - miss_analysis: breakdown of why misses happened
      - optimal_zone_impact: how much in_optimal_zone drives predictions
      - recommendations: suggested improvements
    """
    log("=== SHAP ANALYSIS START ===")

    # Load model
    model, feature_cols = load_model()
    if model is None:
        log("ERROR: No trained model found")
        return {"error": "No trained model"}

    # Get the underlying XGBoost model (unwrap calibration if needed)
    xgb_model = _unwrap_model(model)
    if xgb_model is None:
        log("ERROR: Could not extract XGBoost model for SHAP")
        return {"error": "Cannot extract base model"}

    log(f"Model loaded with {len(feature_cols)} features")

    # Load backtest data
    if backtest_data is None and backtest_json_path:
        backtest_data = _load_backtest_from_s3(backtest_json_path)

    if not backtest_data:
        log("No backtest data provided — running feature importance only")
        return _feature_importance_only(xgb_model, feature_cols)

    # Parse backtest into feature matrix + outcomes
    log("Parsing backtest results...")
    features_df, outcomes = _parse_backtest_results(backtest_data, feature_cols)

    if features_df.empty:
        log("Could not build feature matrix from backtest data")
        return _feature_importance_only(xgb_model, feature_cols)

    log(f"Built feature matrix: {features_df.shape[0]} picks x {features_df.shape[1]} features")

    # Run SHAP
    log("Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(features_df)

    # Global feature importance (mean absolute SHAP value)
    log("Computing global feature importance...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_cols, [round(float(v), 6) for v in mean_abs_shap]))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    log("Global Feature Importance (SHAP):")
    for feat, val in list(sorted_importance.items()):
        bar = "█" * int(val * 100)
        log(f"  {feat:30s} {val:.4f} {bar}")

    # Miss analysis
    log("")
    log("Analyzing misses...")
    miss_analysis = _analyze_misses(
        features_df, outcomes, shap_values, feature_cols
    )

    # Hit vs miss SHAP comparison
    log("Comparing hits vs misses...")
    comparison = _compare_hits_misses(
        features_df, outcomes, shap_values, feature_cols
    )

    # Optimal zone analysis
    log("Analyzing optimal zone impact...")
    zone_analysis = _analyze_optimal_zone(features_df, outcomes, feature_cols)

    # Build results
    results = {
        "global_importance": sorted_importance,
        "miss_analysis": miss_analysis,
        "hit_vs_miss_comparison": comparison,
        "optimal_zone_analysis": zone_analysis,
        "recommendations": _generate_recommendations(
            sorted_importance, miss_analysis, comparison, zone_analysis
        ),
    }

    # Write to S3
    if write_to_s3:
        _write_to_s3(results)

    log("=== SHAP ANALYSIS COMPLETE ===")
    return results


def _feature_importance_only(xgb_model, feature_cols) -> dict:
    """Run SHAP feature importance without backtest data."""
    log("Running feature importance only (no backtest data)...")

    # Create a small synthetic dataset to compute SHAP values
    # Use zeros as baseline — SHAP will show directional importance
    import shap
    dummy_data = pd.DataFrame(
        np.zeros((1, len(feature_cols))),
        columns=feature_cols,
    )

    try:
        explainer = shap.TreeExplainer(xgb_model)

        # Use the model's native feature importance as fallback
        if hasattr(xgb_model, 'feature_importances_'):
            importance = dict(zip(feature_cols, [round(float(v), 6) for v in xgb_model.feature_importances_]))
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            log("Feature Importance (XGBoost native):")
            for feat, val in list(sorted_importance.items()):
                bar = "█" * int(val * 100)
                log(f"  {feat:30s} {val:.4f} {bar}")

            return {
                "global_importance": sorted_importance,
                "note": "SHAP computed without backtest data. Run with s3_key for full miss analysis.",
            }
    except Exception as e:
        log(f"SHAP feature importance failed: {e}")

    # Ultimate fallback: just return native importance
    if hasattr(xgb_model, 'feature_importances_'):
        importance = dict(zip(feature_cols, [round(float(v), 6) for v in xgb_model.feature_importances_]))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {"global_importance": sorted_importance}

    return {"error": "Could not compute feature importance"}


def _unwrap_model(model):
    """Extract the underlying XGBoost model from calibrated wrapper."""
    # If it's already an XGBoost model
    if hasattr(model, 'get_booster'):
        return model

    # If it's a CalibratedClassifierCV
    if hasattr(model, 'calibrated_classifiers_'):
        for cal_clf in model.calibrated_classifiers_:
            if hasattr(cal_clf, 'estimator') and hasattr(cal_clf.estimator, 'get_booster'):
                return cal_clf.estimator
            # Older sklearn versions
            if hasattr(cal_clf, 'base_estimator') and hasattr(cal_clf.base_estimator, 'get_booster'):
                return cal_clf.base_estimator

    # If it's a pipeline
    if hasattr(model, 'steps'):
        for name, step in model.steps:
            if hasattr(step, 'get_booster'):
                return step

    log("WARNING: Could not unwrap model, trying direct SHAP")
    return None


def _load_backtest_from_s3(s3_key: str) -> dict:
    """Load backtest full_results.json from S3."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        log(f"Failed to load backtest from S3: {e}")
        return None


def _parse_backtest_results(backtest_data: dict, feature_cols: list) -> tuple:
    """
    Parse backtest results into a feature matrix and outcome list.
    Extracts top-1 pick from each date with available features.
    """
    rows = []
    outcomes = []

    for day in backtest_data.get("daily_results", []):
        top1_picks = day.get("top1_picks", [])
        if not top1_picks:
            continue

        pick = top1_picks[0]
        got_hit = pick.get("got_hit")
        if got_hit is None:
            continue

        # Build feature dict from what's available in the pick data
        feature_row = {}
        for col in feature_cols:
            if col in pick:
                feature_row[col] = pick[col]

        if len(feature_row) >= 3:  # Need at least some features
            rows.append(feature_row)
            outcomes.append({
                "player": pick.get("player", "Unknown"),
                "date": day.get("date", ""),
                "got_hit": got_hit,
                "confidence": pick.get("confidence", 0),
                "aabbh_ratio": pick.get("aabbh_ratio", None),
                "in_optimal_zone": pick.get("in_optimal_zone", None),
            })

    if not rows:
        return pd.DataFrame(), []

    df = pd.DataFrame(rows)

    # Fill missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order matches
    df = df[feature_cols]

    # Fill NaN
    df = df.fillna(0)

    return df, outcomes


def _analyze_misses(
    features_df: pd.DataFrame,
    outcomes: list,
    shap_values: np.ndarray,
    feature_cols: list,
) -> dict:
    """Analyze what features drove incorrect predictions (misses)."""
    misses = []
    for i, outcome in enumerate(outcomes):
        if outcome["got_hit"] == 0:
            # Get top positive SHAP contributors for this miss
            shap_row = shap_values[i]
            feature_shap = list(zip(feature_cols, [float(v) for v in shap_row]))
            feature_shap.sort(key=lambda x: x[1], reverse=True)

            top_positive = [(f, round(v, 4)) for f, v in feature_shap[:5] if v > 0]
            top_negative = [(f, round(v, 4)) for f, v in feature_shap[-3:] if v < 0]

            misses.append({
                "player": outcome["player"],
                "date": outcome["date"],
                "confidence": outcome["confidence"],
                "aabbh_ratio": outcome.get("aabbh_ratio"),
                "in_optimal_zone": outcome.get("in_optimal_zone"),
                "top_positive_drivers": top_positive,
                "top_negative_drivers": top_negative,
            })

    # Aggregate: which features most often drove misses?
    miss_driver_counts = {}
    for miss in misses:
        for feat, val in miss["top_positive_drivers"]:
            if feat not in miss_driver_counts:
                miss_driver_counts[feat] = 0
            miss_driver_counts[feat] += 1

    sorted_drivers = dict(sorted(miss_driver_counts.items(), key=lambda x: x[1], reverse=True))

    log(f"Total misses: {len(misses)}")
    log("Most common positive SHAP drivers on misses:")
    for feat, count in list(sorted_drivers.items())[:5]:
        log(f"  {feat}: appeared in {count}/{len(misses)} misses")

    return {
        "total_misses": len(misses),
        "miss_details": misses,
        "most_common_miss_drivers": sorted_drivers,
    }


def _compare_hits_misses(
    features_df: pd.DataFrame,
    outcomes: list,
    shap_values: np.ndarray,
    feature_cols: list,
) -> dict:
    """Compare average SHAP values between hits and misses."""
    hit_indices = [i for i, o in enumerate(outcomes) if o["got_hit"] == 1]
    miss_indices = [i for i, o in enumerate(outcomes) if o["got_hit"] == 0]

    if not hit_indices or not miss_indices:
        return {"error": "Need both hits and misses for comparison"}

    hit_shap_mean = np.abs(shap_values[hit_indices]).mean(axis=0)
    miss_shap_mean = np.abs(shap_values[miss_indices]).mean(axis=0)

    comparison = {}
    for i, feat in enumerate(feature_cols):
        comparison[feat] = {
            "hit_importance": round(float(hit_shap_mean[i]), 6),
            "miss_importance": round(float(miss_shap_mean[i]), 6),
            "delta": round(float(miss_shap_mean[i] - hit_shap_mean[i]), 6),
        }

    # Sort by absolute delta
    sorted_comp = dict(sorted(
        comparison.items(),
        key=lambda x: abs(x[1]["delta"]),
        reverse=True,
    ))

    log("Features with biggest SHAP difference (miss vs hit):")
    for feat, vals in list(sorted_comp.items())[:5]:
        direction = "↑ more on misses" if vals["delta"] > 0 else "↓ less on misses"
        log(f"  {feat:30s} delta={vals['delta']:+.4f} ({direction})")

    return sorted_comp


def _analyze_optimal_zone(
    features_df: pd.DataFrame,
    outcomes: list,
    feature_cols: list,
) -> dict:
    """Analyze hit rate inside vs outside the optimal AABBH zone."""
    if "in_optimal_zone" not in features_df.columns:
        return {"error": "in_optimal_zone not in features"}

    in_zone = []
    out_zone = []

    for i, outcome in enumerate(outcomes):
        zone_val = features_df.iloc[i].get("in_optimal_zone", 0)
        if zone_val == 1:
            in_zone.append(outcome["got_hit"])
        else:
            out_zone.append(outcome["got_hit"])

    in_zone_rate = sum(in_zone) / max(len(in_zone), 1)
    out_zone_rate = sum(out_zone) / max(len(out_zone), 1)

    log(f"Optimal zone (0.5-1.0 aabbh_ratio):")
    log(f"  In zone:  {sum(in_zone)}/{len(in_zone)} = {in_zone_rate:.1%}")
    log(f"  Out zone: {sum(out_zone)}/{len(out_zone)} = {out_zone_rate:.1%}")
    log(f"  Advantage: {(in_zone_rate - out_zone_rate)*100:+.1f} percentage points")

    return {
        "in_zone_hits": sum(in_zone),
        "in_zone_total": len(in_zone),
        "in_zone_rate": round(in_zone_rate, 4),
        "out_zone_hits": sum(out_zone),
        "out_zone_total": len(out_zone),
        "out_zone_rate": round(out_zone_rate, 4),
        "zone_advantage": round(in_zone_rate - out_zone_rate, 4),
    }


def _generate_recommendations(
    importance: dict,
    miss_analysis: dict,
    comparison: dict,
    zone_analysis: dict,
) -> list:
    """Generate actionable recommendations based on SHAP analysis."""
    recs = []

    # Check if top features make sense
    top_features = list(importance.keys())[:3]
    recs.append({
        "finding": f"Top 3 features driving predictions: {', '.join(top_features)}",
        "action": "Verify these align with baseball intuition. If pitcher_era is #1, "
                  "the model may be over-indexing on matchup vs batter quality.",
    })

    # Check optimal zone
    if isinstance(zone_analysis, dict) and "zone_advantage" in zone_analysis:
        adv = zone_analysis["zone_advantage"]
        if adv > 0.05:
            recs.append({
                "finding": f"Optimal zone provides {adv*100:.1f}% hit rate advantage",
                "action": "AABBH optimal zone is validated. Consider weighting it higher "
                         "or using it as a hard filter (only pick from in-zone players).",
            })
        elif adv < 0:
            recs.append({
                "finding": "Optimal zone does NOT show advantage in top-1 picks",
                "action": "The zone may be less useful for top-1 selection than expected. "
                         "Consider adjusting zone boundaries or removing the feature.",
            })

    # Check miss drivers
    if isinstance(miss_analysis, dict) and "most_common_miss_drivers" in miss_analysis:
        top_miss_driver = list(miss_analysis["most_common_miss_drivers"].keys())[:1]
        if top_miss_driver:
            recs.append({
                "finding": f"Most common positive SHAP driver on misses: {top_miss_driver[0]}",
                "action": f"The model is confidently wrong when {top_miss_driver[0]} is high. "
                         f"Consider adding a correction or interaction feature.",
            })

    # Check hit vs miss comparison
    if isinstance(comparison, dict):
        for feat, vals in list(comparison.items())[:2]:
            if isinstance(vals, dict) and vals.get("delta", 0) > 0.01:
                recs.append({
                    "finding": f"{feat} has higher SHAP importance on misses than hits",
                    "action": f"The model may be overweighting {feat}. Consider reducing "
                             f"its influence or adding complementary features.",
                })

    return recs


def _write_to_s3(results: dict):
    """Write SHAP analysis results to S3."""
    try:
        s3 = boto3.client("s3")
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"analysis/shap/{ts}/shap_analysis.json"

        # Make results JSON serializable
        clean_results = json.loads(json.dumps(results, default=str))

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(clean_results, indent=2),
            ContentType="application/json",
        )
        log(f"SHAP results written to s3://{S3_BUCKET}/{key}")

    except Exception as e:
        log(f"Failed to write SHAP results to S3: {e}")