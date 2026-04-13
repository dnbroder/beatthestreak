"""
scripts/pitcher_splits_probe.py — Phase 2 splits endpoint validation.

Confirms the MLB Stats API returns usable "times through the order" split
data for pitchers, so we can add `pitcher_first_time_through_advantage`
(or similar) as a Phase 2 feature without committing to a full play-by-play
pipeline.

Tests the /people/{id}/stats?stats=statSplits endpoint with group=pitching
and inspects what split categories are available. We're specifically
looking for:
  - "Times Facing Opposing Lineup: 1st"
  - "Times Facing Opposing Lineup: 2nd"
  - "Times Facing Opposing Lineup: 3rd+"

If those categories exist and return per-batter stats (AB, hits, OPS),
we can compute first-time-through-advantage as a career-level feature
and join it to predictions at scoring time with a single API call per
pitcher per training run. One-time data pull, cached forever.

Also inspects adjacent split categories that might be useful:
  - By inning (1st inning, 2nd inning, etc.)
  - By pitch count threshold
  - By rest days
  - Home vs away
  - Day vs night

Usage:
    python3 scripts/pitcher_splits_probe.py
    python3 scripts/pitcher_splits_probe.py --pitcher-id 607074 --season 2024

Runtime: ~3-5 seconds per pitcher. Probes 3 pitchers by default for
cross-validation that the API behavior is consistent.

Logging conventions match the rest of the Phase 2 scripts:
  - [SPLITS_PROBE] routine output
  - [SPLITS_PROBE]   [HEALTH] grepable operational metrics
  - [SPLITS_PROBE]   [HEALTH] WARNING anomalies
"""

import argparse
import json
import logging
import sys
import time
from typing import Optional

import requests


logger = logging.getLogger("beatthestreak")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


def log(msg: str):
    logger.info(f"[SPLITS_PROBE] {msg}")


def health(msg: str):
    logger.info(f"[SPLITS_PROBE]   [HEALTH] {msg}")


def warn(msg: str):
    logger.warning(f"[SPLITS_PROBE]   [HEALTH] WARNING: {msg}")


BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.25


# Default probe set — one elite (Wheeler), one average (Nola), one below
# (Corbin). Matches the stratification tiers from zone discovery so we
# can see if splits behave differently by pitcher skill.
DEFAULT_PROBES = [
    (554430, "Zack Wheeler"),
    (605400, "Aaron Nola"),
    (571578, "Patrick Corbin"),
]


def safe_get(url: str, params: dict = None) -> Optional[dict]:
    """GET with rate limiting."""
    time.sleep(RATE_LIMIT_DELAY)
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        warn(f"api_fetch_failed url={url} error={e}")
        return None


def fetch_pitcher_splits(pitcher_id: int, season: int) -> Optional[dict]:
    """
    Pull split stats for a pitcher in a specific season.
    The statSplits endpoint requires passing the specific split groups
    as a query parameter. We'll try the 'timesFacingOrder' group which
    is what we care about for Phase 2.
    """
    url = f"{BASE_URL}/people/{pitcher_id}/stats"
    params = {
        "stats": "statSplits",
        "group": "pitching",
        "season": season,
        "sitCodes": "vtfo",  # vs times facing order — key split for our use
    }
    log(f"GET {url}")
    log(f"    params={params}")

    t0 = time.time()
    data = safe_get(url, params=params)
    if data is None:
        return None
    elapsed_ms = int((time.time() - t0) * 1000)
    health(f"splits_fetch_latency ms={elapsed_ms}")
    return data


def fetch_available_splits(pitcher_id: int, season: int) -> Optional[dict]:
    """
    Alternative: pull all statSplits without specifying sitCodes.
    Some versions of the API return a list of available split categories
    this way. Useful for discovering what's actually queryable.
    """
    url = f"{BASE_URL}/people/{pitcher_id}/stats"
    params = {
        "stats": "statSplits",
        "group": "pitching",
        "season": season,
    }
    log(f"GET {url} (no sitCodes)")
    return safe_get(url, params=params)


def analyze_splits_response(data: dict, pitcher_name: str) -> dict:
    """Walk the response and report what's present."""
    findings = {
        "top_level_keys": list(data.keys()) if data else [],
        "num_stat_groups": 0,
        "num_splits": 0,
        "split_descriptions": [],
        "first_split_fields": [],
        "first_stat_fields": [],
        "has_tfo_first": False,
        "has_tfo_second": False,
        "has_tfo_third_plus": False,
        "tfo_stats_by_order": {},
    }

    stats = data.get("stats", [])
    findings["num_stat_groups"] = len(stats)
    if not stats:
        return findings

    splits = stats[0].get("splits", [])
    findings["num_splits"] = len(splits)
    if not splits:
        return findings

    first = splits[0]
    findings["first_split_fields"] = sorted(first.keys())
    findings["first_stat_fields"] = sorted(first.get("stat", {}).keys())

    # Try to detect the "times through order" split descriptions.
    # MLB Stats API has used various labels: "1st", "Pitcher's 1st PA vs
    # Batter", "TFO: 1st", etc. We scan descriptions for "1st", "2nd", "3rd".
    for split in splits:
        desc = split.get("split", {}).get("description", "") or split.get("stat", {}).get("description", "")
        code = split.get("split", {}).get("code", "")
        findings["split_descriptions"].append({
            "code": code,
            "description": desc,
            "ab": split.get("stat", {}).get("atBats"),
            "hits": split.get("stat", {}).get("hits"),
            "avg": split.get("stat", {}).get("avg"),
            "ops": split.get("stat", {}).get("ops"),
        })

        desc_lower = (desc or "").lower()
        code_lower = (code or "").lower()

        if "1st" in desc_lower or "first" in desc_lower or code_lower in ("vtfo1", "1"):
            findings["has_tfo_first"] = True
            findings["tfo_stats_by_order"]["1st"] = {
                "ab": split.get("stat", {}).get("atBats"),
                "hits": split.get("stat", {}).get("hits"),
                "avg": split.get("stat", {}).get("avg"),
                "ops": split.get("stat", {}).get("ops"),
            }
        elif "2nd" in desc_lower or "second" in desc_lower or code_lower in ("vtfo2", "2"):
            findings["has_tfo_second"] = True
            findings["tfo_stats_by_order"]["2nd"] = {
                "ab": split.get("stat", {}).get("atBats"),
                "hits": split.get("stat", {}).get("hits"),
                "avg": split.get("stat", {}).get("avg"),
                "ops": split.get("stat", {}).get("ops"),
            }
        elif ("3rd" in desc_lower or "third" in desc_lower
              or code_lower in ("vtfo3", "3")):
            findings["has_tfo_third_plus"] = True
            findings["tfo_stats_by_order"]["3rd+"] = {
                "ab": split.get("stat", {}).get("atBats"),
                "hits": split.get("stat", {}).get("hits"),
                "avg": split.get("stat", {}).get("avg"),
                "ops": split.get("stat", {}).get("ops"),
            }

    return findings


def probe_pitcher(pitcher_id: int, pitcher_name: str, season: int,
                  verbose: bool = False) -> dict:
    """Run both probe attempts against a single pitcher, return findings."""
    log("")
    log("-" * 70)
    log(f"PROBE: {pitcher_name} (ID {pitcher_id}, {season})")
    log("-" * 70)

    # First attempt — with sitCodes=vtfo
    log("")
    log("Attempt 1: statSplits with sitCodes=vtfo")
    data = fetch_pitcher_splits(pitcher_id, season)

    if data and data.get("stats") and data["stats"][0].get("splits"):
        findings = analyze_splits_response(data, pitcher_name)
        log(f"  Returned {findings['num_splits']} split(s)")
    else:
        log("  Empty or failed — trying without sitCodes")
        data = fetch_available_splits(pitcher_id, season)
        if not data:
            warn(f"both_probes_failed pitcher={pitcher_name}")
            return {"pitcher_name": pitcher_name, "success": False}
        findings = analyze_splits_response(data, pitcher_name)
        log(f"  Returned {findings['num_splits']} split(s) "
            f"(may include non-TFO categories)")

    log("")
    log(f"  First-time-through detected:   {findings['has_tfo_first']}")
    log(f"  Second-time-through detected:  {findings['has_tfo_second']}")
    log(f"  Third+-time-through detected:  {findings['has_tfo_third_plus']}")

    if findings["tfo_stats_by_order"]:
        log("")
        log(f"  Stats by order:")
        log(f"  {'Order':<8} {'AB':>5} {'Hits':>5} {'AVG':>7} {'OPS':>7}")
        for order, stats in findings["tfo_stats_by_order"].items():
            ab = stats.get("ab", "-")
            hits = stats.get("hits", "-")
            avg = stats.get("avg", "-")
            ops = stats.get("ops", "-")
            log(f"  {order:<8} {str(ab):>5} {str(hits):>5} {str(avg):>7} {str(ops):>7}")

    if verbose:
        log("")
        log("  First 3 split descriptions returned:")
        for d in findings["split_descriptions"][:3]:
            log(f"    code={d['code']!r} desc={d['description']!r} "
                f"ab={d['ab']} ops={d['ops']}")

    success = (findings["has_tfo_first"] and findings["has_tfo_second"]
               and findings["has_tfo_third_plus"])
    health(f"probe_result pitcher=\"{pitcher_name}\" season={season} "
           f"first={findings['has_tfo_first']} second={findings['has_tfo_second']} "
           f"third_plus={findings['has_tfo_third_plus']} total_splits={findings['num_splits']}")

    findings["pitcher_name"] = pitcher_name
    findings["season"] = season
    findings["success"] = success
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitcher-id", type=int, default=None,
                        help="Single pitcher to probe (overrides default set)")
    parser.add_argument("--pitcher-name", type=str, default="Unknown",
                        help="Display name if using --pitcher-id")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--verbose", action="store_true",
                        help="Dump first 3 split descriptions")
    args = parser.parse_args()

    run_start = time.time()
    log("=" * 70)
    log("PITCHER SPLITS PROBE — Phase 2 times-through-order validation")
    log("=" * 70)

    probes = DEFAULT_PROBES if args.pitcher_id is None else [
        (args.pitcher_id, args.pitcher_name)
    ]
    log(f"Probing {len(probes)} pitcher(s), season {args.season}")
    log("")

    all_findings = []
    for pid, name in probes:
        findings = probe_pitcher(pid, name, args.season, verbose=args.verbose)
        all_findings.append(findings)

    elapsed = time.time() - run_start

    # Verdict
    log("")
    log("=" * 70)
    log("VERDICT")
    log("=" * 70)

    successes = [f for f in all_findings if f.get("success")]
    partial = [f for f in all_findings
               if not f.get("success") and any([f.get("has_tfo_first"),
                                                 f.get("has_tfo_second"),
                                                 f.get("has_tfo_third_plus")])]
    failures = [f for f in all_findings
                if not f.get("success") and not any([f.get("has_tfo_first"),
                                                      f.get("has_tfo_second"),
                                                      f.get("has_tfo_third_plus")])]

    log(f"Probed: {len(all_findings)} pitchers")
    log(f"Full success (all 3 TFO splits): {len(successes)}")
    log(f"Partial success (some TFO data): {len(partial)}")
    log(f"Failures (no TFO data found): {len(failures)}")
    log("")

    if len(successes) == len(all_findings):
        log("GREEN — Splits endpoint returns all times-through-order data cleanly.")
        log("Phase 2 feature `pitcher_first_time_through_advantage` is feasible.")
        log("")
        log("Recommended feature formula:")
        log("  pitcher_first_time_through_advantage =")
        log("      (OPS_allowed_1st_time) - (OPS_allowed_2nd_and_3rd_times)")
        log("")
        log("Negative values indicate the pitcher is BETTER first time through")
        log("(typical pattern — batters learn the pitcher's stuff). Large negative")
        log("values are 'flash starter' pitchers. Values near zero indicate")
        log("consistent effectiveness across the lineup's times at bat.")
        verdict = "GREEN"
    elif len(partial) > 0 and len(failures) == 0:
        log("YELLOW — Partial data returned. Some TFO splits missing for some")
        log("pitchers. May need to handle missing data at the feature engineering")
        log("layer (imputation with league average). Still feasible but less clean.")
        verdict = "YELLOW"
    else:
        log("RED — TFO splits not consistently available through this endpoint.")
        log("Alternatives:")
        log("  1. Try a different stats endpoint (statsSingleSeasonAdvanced?)")
        log("  2. Compute TFO splits ourselves from play-by-play data (Phase 2.5)")
        log("  3. Use Retrosheet event files (one-time 2GB download)")
        verdict = "RED"

    health(f"probe_complete verdict={verdict} probed={len(all_findings)} "
           f"success={len(successes)} partial={len(partial)} "
           f"failed={len(failures)} duration_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()