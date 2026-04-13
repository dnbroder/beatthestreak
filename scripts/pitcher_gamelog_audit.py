"""
scripts/pitcher_gamelog_audit.py — Phase 2 data layer validation.

Pulls a single pitcher's gameLog from the MLB Stats API using
group=pitching (not hitting) and inspects the response structure.

Answers these questions BEFORE we commit to the full Phase 2 data layer:

  1. Does the pitching gameLog endpoint actually return per-game data?
  2. Does it include the fields we need (atBats faced, hits allowed, IP)?
  3. What does a "start" vs "relief appearance" look like — can we filter
     starters cleanly?
  4. Is the API rate limit tolerable at the ~500-pitchers-per-season scale?

Usage:
    python3 scripts/pitcher_gamelog_audit.py
    python3 scripts/pitcher_gamelog_audit.py --pitcher-id 605483 --season 2024
    python3 scripts/pitcher_gamelog_audit.py --verbose

Default pitcher: Brandon Woodruff (id 605483) — right-handed starter with
a long career, so his 2024 gameLog should show ~25-30 starts with plenty
of variation in hits-allowed per game. Good sanity check target.

Logging conventions (matches the rest of the codebase):
  - [PHASE2_AUDIT]        prefix on all routine output
  - [PHASE2_AUDIT] [HEALTH]       single-line operational metrics (grepable)
  - [PHASE2_AUDIT] [HEALTH] WARNING anomalies that would matter in production
"""

import argparse
import json
import logging
import sys
import time
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Logging setup — matches the convention used across mlb_predictor/.
# Uses the same logger name ("beatthestreak") and [MODULE] prefix pattern.
# ---------------------------------------------------------------------------
logger = logging.getLogger("beatthestreak")
logger.setLevel(logging.INFO)

# Stream to stdout with plain format so CLI output stays readable AND
# CloudWatch-compatible if this ever runs inside a container.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


def log(msg: str):
    """Routine audit output."""
    logger.info(f"[PHASE2_AUDIT] {msg}")


def health(msg: str):
    """
    Single-line operational metric. Matches the [HEALTH] convention from
    v1.5.2+ across check_actuals.py, daily_predictions.py, predictions_log.py.
    Grepable from CloudWatch if this ever wraps in a Lambda.
    """
    logger.info(f"[PHASE2_AUDIT]   [HEALTH] {msg}")


def warn(msg: str):
    """
    HEALTH warning. The v2.3 CloudWatch alarm counts '[HEALTH] WARNING'
    occurrences, so this line format triggers production alerting if this
    code ever lands in a scheduled job.
    """
    logger.warning(f"[PHASE2_AUDIT]   [HEALTH] WARNING: {msg}")


BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------
def fetch_pitcher_gamelog(pitcher_id: int, season: int) -> Optional[dict]:
    """Pull one pitcher's per-game pitching stats for a season."""
    url = f"{BASE_URL}/people/{pitcher_id}/stats"
    params = {
        "stats": "gameLog",
        "group": "pitching",  # key difference from batter gameLog
        "season": season,
    }
    log(f"GET {url}")
    log(f"    params={params}")

    t0 = time.time()
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        warn(f"api_fetch_failed pitcher_id={pitcher_id} season={season} error={e}")
        return None

    elapsed_ms = int((time.time() - t0) * 1000)
    payload_kb = len(resp.content) // 1024
    log(f"    {resp.status_code} in {elapsed_ms}ms ({len(resp.content):,} bytes)")
    health(f"api_fetch status={resp.status_code} latency_ms={elapsed_ms} "
           f"payload_kb={payload_kb}")

    if elapsed_ms > 5000:
        warn(f"api_latency_high latency_ms={elapsed_ms} "
             f"(>5s — production pipeline should cache aggressively)")

    return resp.json()


def fetch_pitcher_bio(pitcher_id: int) -> Optional[dict]:
    """Pull basic bio — used for display and handedness confirmation."""
    url = f"{BASE_URL}/people/{pitcher_id}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        people = resp.json().get("people", [])
        return people[0] if people else None
    except requests.RequestException as e:
        warn(f"bio_fetch_failed pitcher_id={pitcher_id} error={e}")
        return None


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def inspect_structure(data: dict) -> dict:
    """Walk the response and report what's present."""
    findings = {
        "top_level_keys": list(data.keys()),
        "num_stat_groups": 0,
        "num_game_splits": 0,
        "first_split_fields": [],
        "first_stat_fields": [],
        "has_at_bats": False,
        "has_hits": False,
        "has_innings_pitched": False,
        "has_batters_faced": False,
        "has_game_pk": False,
        "has_date": False,
        "has_home_away": False,
        "appearances_as_starter": 0,
        "appearances_as_reliever": 0,
        "total_appearances": 0,
    }

    stats = data.get("stats", [])
    findings["num_stat_groups"] = len(stats)
    if not stats:
        return findings

    splits = stats[0].get("splits", [])
    findings["num_game_splits"] = len(splits)
    if not splits:
        return findings

    first = splits[0]
    findings["first_split_fields"] = sorted(first.keys())
    stat = first.get("stat", {})
    findings["first_stat_fields"] = sorted(stat.keys())

    findings["has_at_bats"] = "atBats" in stat
    findings["has_hits"] = "hits" in stat
    findings["has_innings_pitched"] = "inningsPitched" in stat
    findings["has_batters_faced"] = "battersFaced" in stat
    findings["has_game_pk"] = "gamePk" in first.get("game", {}) or "gamePk" in first
    findings["has_date"] = "date" in first
    findings["has_home_away"] = "isHome" in first

    for split in splits:
        s = split.get("stat", {})
        findings["total_appearances"] += 1
        if s.get("gamesStarted", 0) == 1:
            findings["appearances_as_starter"] += 1
        else:
            findings["appearances_as_reliever"] += 1

    return findings


def sample_rows(data: dict, n: int = 3) -> list:
    """Extract N sample games in a shape close to what Phase 2 will consume."""
    stats = data.get("stats", [])
    if not stats:
        return []

    rows = []
    for split in stats[0].get("splits", [])[:n]:
        stat = split.get("stat", {})
        game = split.get("game", {})
        opponent = split.get("opponent", {})
        team = split.get("team", {})

        rows.append({
            "date": split.get("date"),
            "game_pk": game.get("gamePk") or split.get("gamePk"),
            "is_home": split.get("isHome"),
            "team": team.get("name"),
            "opponent": opponent.get("name"),
            "games_started": stat.get("gamesStarted"),
            "innings_pitched": stat.get("inningsPitched"),
            "batters_faced": stat.get("battersFaced"),
            "at_bats_faced": stat.get("atBats"),
            "hits_allowed": stat.get("hits"),
            "earned_runs": stat.get("earnedRuns"),
            "strikeouts": stat.get("strikeOuts"),
            "walks": stat.get("baseOnBalls"),
        })
    return rows


def compute_season_aabbh(data: dict) -> dict:
    """
    Compute season-level AABBH (ABs faced / hits allowed) as a sanity check.
    This is the simplest possible pitcher AABBH — career version of this
    will be the core Phase 2 feature.
    """
    stats = data.get("stats", [])
    if not stats:
        return {}

    total_abs = 0
    total_hits = 0
    total_ip = 0.0
    starts = 0

    for split in stats[0].get("splits", []):
        s = split.get("stat", {})
        total_abs += s.get("atBats", 0)
        total_hits += s.get("hits", 0)
        # inningsPitched comes as "6.2" meaning 6 full + 2 outs, not 6.2
        try:
            ip_str = str(s.get("inningsPitched", "0.0"))
            total_ip += float(ip_str.replace(".1", ".333").replace(".2", ".667"))
        except (ValueError, TypeError):
            pass
        if s.get("gamesStarted", 0) == 1:
            starts += 1

    season_aabbh = total_abs / total_hits if total_hits > 0 else None
    return {
        "total_at_bats_faced": total_abs,
        "total_hits_allowed": total_hits,
        "total_innings_pitched": round(total_ip, 1),
        "starts": starts,
        "season_aabbh": round(season_aabbh, 3) if season_aabbh else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitcher-id", type=int, default=605483,
                        help="MLB Stats API player ID (default: Brandon Woodruff)")
    parser.add_argument("--season", type=int, default=2024,
                        help="Season year (default: 2024)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full JSON response at the end")
    args = parser.parse_args()

    audit_start = time.time()
    log("=" * 70)
    log("PITCHER GAMELOG AUDIT — Phase 2 data layer validation")
    log("=" * 70)

    # Bio lookup for display
    bio = fetch_pitcher_bio(args.pitcher_id)
    if bio:
        name = bio.get("fullName", "Unknown")
        hand = bio.get("pitchHand", {}).get("code", "?")
        pos = bio.get("primaryPosition", {}).get("abbreviation", "?")
        log(f"Pitcher: {name} (ID {args.pitcher_id}) — throws {hand}, position {pos}")
        health(f"pitcher_identified name=\"{name}\" id={args.pitcher_id} "
               f"throws={hand} position={pos}")
    else:
        log(f"Pitcher ID {args.pitcher_id}: bio lookup failed (continuing anyway)")
    log(f"Season:  {args.season}")
    log("")

    # TEST 1
    log("-" * 70)
    log("TEST 1 — Fetch pitching gameLog")
    log("-" * 70)
    data = fetch_pitcher_gamelog(args.pitcher_id, args.season)
    if data is None:
        warn("phase2_feasibility=RED reason=api_unreachable")
        log("")
        log("FAIL: API request errored out. Phase 2 data layer is not feasible")
        log("as designed. Check MLB Stats API availability and rate limits.")
        sys.exit(1)
    log("")

    # TEST 2
    log("-" * 70)
    log("TEST 2 — Inspect response structure")
    log("-" * 70)
    findings = inspect_structure(data)
    log(f"Top-level keys: {findings['top_level_keys']}")
    log(f"Stat groups: {findings['num_stat_groups']}")
    log(f"Game splits: {findings['num_game_splits']}")
    health(f"response_structure stat_groups={findings['num_stat_groups']} "
           f"game_splits={findings['num_game_splits']}")

    if findings["num_game_splits"] == 0:
        warn("phase2_feasibility=RED reason=no_game_splits")
        log("")
        log("FAIL: No game splits in response. Either the pitcher didn't")
        log("appear in that season or the API contract has changed.")
        sys.exit(1)

    log("")
    log(f"First split top-level fields: {findings['first_split_fields']}")
    log(f"First split stat fields:      {findings['first_stat_fields']}")
    log("")

    # TEST 3
    log("-" * 70)
    log("TEST 3 — Phase 2 required field presence")
    log("-" * 70)
    checks = [
        ("atBats (ABs faced)",    findings["has_at_bats"]),
        ("hits (hits allowed)",   findings["has_hits"]),
        ("inningsPitched",        findings["has_innings_pitched"]),
        ("battersFaced",          findings["has_batters_faced"]),
        ("gamePk (game ID)",      findings["has_game_pk"]),
        ("date",                  findings["has_date"]),
        ("isHome (home/away)",    findings["has_home_away"]),
    ]
    for name, ok in checks:
        marker = "OK " if ok else "MISSING"
        log(f"  [{marker}] {name}")

    missing = [name for name, ok in checks if not ok]
    fields_present = len(checks) - len(missing)
    health(f"field_check present={fields_present}/{len(checks)} "
           f"missing={[n for n, ok in checks if not ok]}")

    if missing:
        warn(f"field_check incomplete missing_count={len(missing)} — "
             f"feature engineering will need workarounds")

    # TEST 4
    log("")
    log("-" * 70)
    log("TEST 4 — Appearance type breakdown")
    log("-" * 70)
    log(f"Total appearances: {findings['total_appearances']}")
    log(f"  As starter:   {findings['appearances_as_starter']}")
    log(f"  As reliever:  {findings['appearances_as_reliever']}")
    health(f"appearance_breakdown starter={findings['appearances_as_starter']} "
           f"reliever={findings['appearances_as_reliever']} "
           f"total={findings['total_appearances']}")
    if findings["appearances_as_reliever"] > 0:
        log("")
        log("NOTE: Pitchers with relief appearances will need filtering at")
        log("the feature engineering layer. For Phase 2 we likely only want")
        log("starts when computing 'recent form' features (pitcher_aabbh_recent)")
        log("since relief outings have wildly different workload profiles.")

    # TEST 5
    log("")
    log("-" * 70)
    log("TEST 5 — First 3 games in Phase-2-ready shape")
    log("-" * 70)
    samples = sample_rows(data, n=3)
    for i, row in enumerate(samples, 1):
        log(f"\nGame {i}:")
        for k, v in row.items():
            log(f"  {k:22s} {v}")

    # TEST 6
    log("")
    log("-" * 70)
    log("TEST 6 — Season-level AABBH sanity check")
    log("-" * 70)
    agg = compute_season_aabbh(data)
    for k, v in agg.items():
        log(f"  {k:28s} {v}")

    if agg.get("season_aabbh"):
        aabbh = agg["season_aabbh"]
        log("")
        log(f"Interpretation: This pitcher allowed one hit every {aabbh:.2f}")
        log(f"at-bats faced in {args.season}. League-average pitcher AABBH")
        log(f"is roughly 3.7-4.0. Lower = harder to hit against.")
        health(f"season_aabbh value={aabbh} "
               f"at_bats_faced={agg['total_at_bats_faced']} "
               f"hits_allowed={agg['total_hits_allowed']} starts={agg['starts']}")

        # Sanity guard: starter AABBH should be 2.0-8.0. Elite ~3.0-3.3,
        # league avg ~3.7-4.0, replacement level ~4.5-5.0.
        if aabbh < 2.0 or aabbh > 8.0:
            warn(f"season_aabbh_out_of_range value={aabbh} — "
                 f"expected 2.0-8.0 for a MLB starter, check field mapping")

    # Verdict
    log("")
    log("=" * 70)
    log("VERDICT")
    log("=" * 70)
    required_present = all(ok for _, ok in checks)
    has_games = findings["num_game_splits"] > 0
    audit_elapsed = time.time() - audit_start

    if required_present and has_games:
        verdict = "GREEN"
        log("GREEN — Phase 2 data layer is feasible as designed.")
        log("Next step: zone discovery analysis against 2020-2024 data to find")
        log("the pitcher-side equivalent of the batter 0.5-1.0 optimal zone.")
    elif has_games and not required_present:
        verdict = "YELLOW"
        log("YELLOW — Data layer is mostly feasible but some fields need")
        log("workarounds. Phase 2 can proceed with adjustments. See missing")
        log("fields above.")
    else:
        verdict = "RED"
        log("RED — Data layer is NOT feasible as designed. Alternative data")
        log("source needed. Consider Retrosheet event files or FanGraphs.")

    health(f"audit_complete verdict={verdict} duration_sec={audit_elapsed:.1f} "
           f"fields_ok={fields_present}/{len(checks)} "
           f"game_splits={findings['num_game_splits']}")

    if args.verbose:
        log("")
        log("-" * 70)
        log("FULL RAW RESPONSE")
        log("-" * 70)
        log(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()