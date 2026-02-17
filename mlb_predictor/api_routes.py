"""
api_routes.py – Public API endpoints for BeatTheStreak.

These are customer-facing routes served through API Gateway.
They return clean, documented JSON responses with no internal
model details exposed.

Internal endpoints (/train, /backtest, /shap) remain in main.py
and are NOT routed through API Gateway.

Usage:
  Include in main.py:
    from mlb_predictor.api_routes import api_router
    app.include_router(api_router)
"""

import os
import io
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import boto3
from fastapi import APIRouter, Query, HTTPException, Request

from mlb_predictor.model import load_model
from mlb_predictor.daily_predictions import run_daily_predictions

logger = logging.getLogger("beatthestreak")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
S3_BUCKET = f"mlb-hit-predictor-{ENVIRONMENT}-bucket"

api_router = APIRouter(prefix="/v1", tags=["Public API"])


# ---------------------------------------------------------------------------
# Tier-based field filtering
# ---------------------------------------------------------------------------
# API Gateway passes the usage plan name as a header.
# Basic = top 5, limited fields. Pro = top 10, full fields. Enterprise = everything.

BASIC_FIELDS = [
    "rank", "player_name", "team", "opponent", "confidence_pct",
]

PRO_FIELDS = [
    "rank", "player_name", "team", "opponent", "venue",
    "bat_side", "lineup_position", "opp_pitcher", "opp_pitcher_hand",
    "confidence_pct", "batting_avg_season", "avg_last_30",
    "aabbh_ratio", "in_optimal_zone", "platoon_advantage",
    "pitcher_era", "park_factor",
]

ENTERPRISE_FIELDS = PRO_FIELDS + [
    "aabbh", "current_hitless_abs", "games_between_hits",
    "k_rate", "bb_rate", "multi_hit_freq", "hit_streak",
    "avg_last_7", "avg_last_14", "avg_abs_per_game",
    "pitcher_whip", "pitcher_k_per_9",
    "batter_hit_rate_vs_era_tier", "era_tier_delta",
]


def _get_tier(request: Request) -> str:
    """Extract API tier from Gateway context or default to basic."""
    # API Gateway injects usage plan info via context
    tier = request.headers.get("X-API-Tier", "basic").lower()
    if tier not in ("basic", "pro", "enterprise"):
        tier = "basic"
    return tier


def _get_max_picks(tier: str) -> int:
    """Max picks returned per tier."""
    return {"basic": 5, "pro": 10, "enterprise": 50}.get(tier, 5)


def _filter_fields(df: pd.DataFrame, tier: str) -> list:
    """Filter response fields based on tier."""
    if tier == "enterprise":
        fields = ENTERPRISE_FIELDS
    elif tier == "pro":
        fields = PRO_FIELDS
    else:
        fields = BASIC_FIELDS

    available = [f for f in fields if f in df.columns]
    return df[available].to_dict(orient="records")


# ---------------------------------------------------------------------------
# GET /v1/picks/daily — Today's top predictions
# ---------------------------------------------------------------------------
@api_router.get("/picks/daily")
def get_daily_picks(
    request: Request,
    date: Optional[str] = Query(default=None, description="Date YYYY-MM-DD (default: today)"),
    top_n: Optional[int] = Query(default=None, description="Number of picks to return"),
):
    """
    Get today's ranked hit predictions.

    Returns batters most likely to get at least 1 hit, ranked by
    model confidence score. Fields returned depend on your API tier.

    **Basic:** Top 5 picks with name, team, opponent, confidence.
    **Pro:** Top 10 picks with full stats and matchup details.
    **Enterprise:** Up to 50 picks with all model features.
    """
    tier = _get_tier(request)
    max_picks = top_n or _get_max_picks(tier)
    max_picks = min(max_picks, _get_max_picks(tier))  # Can't exceed tier limit

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"[API] /v1/picks/daily | tier={tier} | date={date} | top_n={max_picks}")

    # Try S3 cache first (predictions already generated today)
    cached = _get_cached_predictions(date)
    if cached is not None:
        logger.info(f"[API] Serving {len(cached)} cached predictions for {date}")
        top = cached.head(max_picks)
        return {
            "date": date,
            "tier": tier,
            "picks": _filter_fields(top, tier),
            "total_scored": len(cached),
            "cached": True,
        }

    # Generate fresh predictions
    try:
        predictions = run_daily_predictions(
            date_str=date,
            write_to_s3=True,
        )

        if predictions.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions available for {date}. Are there games scheduled?",
            )

        top = predictions.head(max_picks)
        return {
            "date": date,
            "tier": tier,
            "picks": _filter_fields(top, tier),
            "total_scored": len(predictions),
            "cached": False,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] /v1/picks/daily failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction pipeline error")


# ---------------------------------------------------------------------------
# GET /v1/picks/{date} — Historical predictions
# ---------------------------------------------------------------------------
@api_router.get("/picks/{date}")
def get_historical_picks(
    request: Request,
    date: str,
    top_n: Optional[int] = Query(default=None, description="Number of picks to return"),
):
    """
    Get historical predictions for a past date.

    Returns cached predictions from S3. Only available for dates
    where the model has previously run.
    """
    tier = _get_tier(request)
    max_picks = top_n or _get_max_picks(tier)
    max_picks = min(max_picks, _get_max_picks(tier))

    logger.info(f"[API] /v1/picks/{date} | tier={tier} | top_n={max_picks}")

    cached = _get_cached_predictions(date)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {date}. Predictions are only "
                   "available for dates during the MLB season when the model ran.",
        )

    top = cached.head(max_picks)
    return {
        "date": date,
        "tier": tier,
        "picks": _filter_fields(top, tier),
        "total_scored": len(cached),
    }


# ---------------------------------------------------------------------------
# GET /v1/confidence/{player_id} — Single player lookup
# ---------------------------------------------------------------------------
@api_router.get("/confidence/{player_id}")
def get_player_confidence(
    request: Request,
    player_id: int,
    date: Optional[str] = Query(default=None, description="Date YYYY-MM-DD (default: today)"),
):
    """
    Get confidence score for a specific player.

    Returns the model's hit probability and feature breakdown for
    one batter on a given date. Useful for checking specific matchups.

    **Pro/Enterprise only:** Includes full feature details.
    **Basic:** Returns confidence score only.
    """
    tier = _get_tier(request)

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"[API] /v1/confidence/{player_id} | tier={tier} | date={date}")

    cached = _get_cached_predictions(date)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions available for {date}.",
        )

    player = cached[cached["player_id"] == player_id]
    if player.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Player {player_id} not found in {date} predictions. "
                   "Player may not have been in the starting lineup.",
        )

    row = player.iloc[0]

    if tier == "basic":
        return {
            "player_id": player_id,
            "player_name": row.get("player_name", ""),
            "date": date,
            "confidence_pct": float(row.get("confidence_pct", 0)),
            "rank": int(row.get("rank", 0)),
        }

    # Pro/Enterprise: full details
    result = {
        "player_id": player_id,
        "player_name": row.get("player_name", ""),
        "team": row.get("team", ""),
        "opponent": row.get("opponent", ""),
        "date": date,
        "confidence_pct": float(row.get("confidence_pct", 0)),
        "rank": int(row.get("rank", 0)),
        "matchup": {
            "opp_pitcher": row.get("opp_pitcher", ""),
            "opp_pitcher_hand": row.get("opp_pitcher_hand", ""),
            "pitcher_era": float(row.get("pitcher_era", 0)),
            "platoon_advantage": int(row.get("platoon_advantage", 0)),
            "park_factor": float(row.get("park_factor", 1.0)),
            "venue": row.get("venue", ""),
        },
        "batter_stats": {
            "batting_avg": float(row.get("batting_avg_season", 0)),
            "avg_last_30": float(row.get("avg_last_30", 0)),
            "aabbh_ratio": float(row.get("aabbh_ratio", 0)),
            "in_optimal_zone": bool(row.get("in_optimal_zone", 0)),
            "hit_streak": int(row.get("hit_streak", 0)),
        },
    }

    # Enterprise gets everything
    if tier == "enterprise":
        result["advanced"] = {
            "aabbh": float(row.get("aabbh", 0)),
            "current_hitless_abs": int(row.get("current_hitless_abs", 0)),
            "games_between_hits": float(row.get("games_between_hits", 0)),
            "k_rate": float(row.get("k_rate", 0)),
            "bb_rate": float(row.get("bb_rate", 0)),
            "multi_hit_freq": float(row.get("multi_hit_freq", 0)),
            "era_tier_delta": float(row.get("era_tier_delta", 0)),
        }

    return result


# ---------------------------------------------------------------------------
# GET /v1/health — Public health check
# ---------------------------------------------------------------------------
@api_router.get("/health")
def api_health():
    """API health check. No API key required."""
    model, feature_cols = load_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(feature_cols) if feature_cols else 0,
        "environment": ENVIRONMENT,
        "version": "1.2.0",
    }


# ---------------------------------------------------------------------------
# GET /v1/meta — API metadata
# ---------------------------------------------------------------------------
@api_router.get("/meta")
def api_metadata():
    """
    API metadata and documentation links.
    No API key required.
    """
    return {
        "name": "BeatTheStreak Hit Prediction API",
        "version": "1.2.0",
        "description": "ML-powered MLB hit predictions using patent-pending AABBH methodology",
        "endpoints": {
            "GET /v1/picks/daily": "Today's ranked hit predictions",
            "GET /v1/picks/{date}": "Historical predictions by date (YYYY-MM-DD)",
            "GET /v1/confidence/{player_id}": "Single player confidence score",
            "GET /v1/health": "API health and model status",
            "GET /v1/meta": "This endpoint",
        },
        "tiers": {
            "basic": {
                "picks_per_request": 5,
                "daily_limit": 100,
                "fields": "name, team, opponent, confidence",
            },
            "pro": {
                "picks_per_request": 10,
                "daily_limit": 1000,
                "fields": "Full matchup details + batter stats",
            },
            "enterprise": {
                "picks_per_request": 50,
                "daily_limit": 50000,
                "fields": "All model features + advanced analytics",
            },
        },
        "authentication": "API key required in x-api-key header",
        "rate_limits": "Varies by tier. See headers: X-RateLimit-Limit, X-RateLimit-Remaining",
    }


# ---------------------------------------------------------------------------
# S3 cache helper
# ---------------------------------------------------------------------------
def _get_cached_predictions(date_str: str) -> Optional[pd.DataFrame]:
    """
    Try to load predictions from S3 cache.
    Returns DataFrame or None if not cached.
    """
    try:
        s3 = boto3.client("s3")
        key = f"predictions/{date_str}/all_predictions.csv"

        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        csv_content = response["Body"].read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_content))

        if "confidence_pct" in df.columns:
            df = df.sort_values("confidence_pct", ascending=False).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)

        return df

    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.warning(f"[API] S3 cache read failed for {date_str}: {e}")
        return None