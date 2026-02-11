"""
historical_data.py – Pull historical batter game logs from MLB Stats API.

Covers 2000-2024 seasons. Retrosheet is used as a fallback/supplement
for older seasons where the Stats API has gaps.

The Stats API endpoint:
  /api/v1/people/{playerId}/stats?stats=gameLog&group=hitting&season={year}

We first get all players per season, then pull their game logs.
"""

import requests
import pandas as pd
import logging
import time
from typing import Optional

logger = logging.getLogger("beatthestreak")

BASE_URL = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.25  # seconds between API calls to be polite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_get(url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    """GET with retry logic and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {url} - {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Get all team rosters for a season
# ---------------------------------------------------------------------------
def get_season_players(season: int) -> pd.DataFrame:
    """
    Get all MLB players who appeared in a given season.
    Returns DataFrame with player_id, name, team, position, bat_side.
    """
    logger.info(f"Fetching players for {season} season...")

    teams_data = _safe_get(f"{BASE_URL}/teams", params={"sportId": 1, "season": season})
    if not teams_data:
        return pd.DataFrame()

    players = []
    for team in teams_data.get("teams", []):
        team_id = team["id"]
        team_name = team["name"]

        roster_data = _safe_get(
            f"{BASE_URL}/teams/{team_id}/roster",
            params={"season": season, "rosterType": "fullSeason"}
        )
        if not roster_data:
            continue

        for entry in roster_data.get("roster", []):
            person = entry.get("person", {})
            position = entry.get("position", {})

            # Skip pitchers — we only care about batters
            if position.get("abbreviation") == "P":
                continue

            players.append({
                "player_id": person.get("id"),
                "player_name": person.get("fullName"),
                "team": team_name,
                "team_id": team_id,
                "position": position.get("abbreviation"),
                "season": season,
            })

    df = pd.DataFrame(players).drop_duplicates(subset=["player_id", "season"])
    logger.info(f"Found {len(df)} position players for {season}")
    return df


# ---------------------------------------------------------------------------
# Get batter game logs for a player/season
# ---------------------------------------------------------------------------
def get_player_game_log(player_id: int, season: int) -> pd.DataFrame:
    """
    Pull a single player's hitting game log for a season.
    Returns one row per game with all batting stats.
    """
    data = _safe_get(
        f"{BASE_URL}/people/{player_id}/stats",
        params={"stats": "gameLog", "group": "hitting", "season": season}
    )
    if not data:
        return pd.DataFrame()

    splits = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            stat = split.get("stat", {})
            game = split.get("game", {})
            team = split.get("team", {})
            opponent = split.get("opponent", {})
            date_str = split.get("date", "")

            splits.append({
                "player_id": player_id,
                "season": season,
                "date": date_str,
                "game_pk": game.get("gamePk"),
                "team": team.get("name"),
                "opponent": opponent.get("name"),
                "is_home": split.get("isHome", False),

                # Core batting stats
                "at_bats": stat.get("atBats", 0),
                "hits": stat.get("hits", 0),
                "doubles": stat.get("doubles", 0),
                "triples": stat.get("triples", 0),
                "home_runs": stat.get("homeRuns", 0),
                "rbi": stat.get("rbi", 0),
                "walks": stat.get("baseOnBalls", 0),
                "strikeouts": stat.get("strikeOuts", 0),
                "plate_appearances": stat.get("plateAppearances", 0),

                # Derived (from API if available)
                "avg": stat.get("avg", None),
                "obp": stat.get("obp", None),
                "slg": stat.get("slg", None),
                "ops": stat.get("ops", None),
            })

    return pd.DataFrame(splits)


# ---------------------------------------------------------------------------
# Get player bio info (bat side, etc.)
# ---------------------------------------------------------------------------
def get_player_info(player_id: int) -> dict:
    """Get player bio details — bat side, throw side, etc."""
    data = _safe_get(f"{BASE_URL}/people/{player_id}")
    if not data:
        return {}

    people = data.get("people", [])
    if not people:
        return {}

    person = people[0]
    return {
        "player_id": player_id,
        "bat_side": person.get("batSide", {}).get("code", "U"),  # R, L, S (switch)
        "throw_side": person.get("pitchHand", {}).get("code", "U"),
        "height": person.get("height", ""),
        "weight": person.get("weight", 0),
        "birth_date": person.get("birthDate", ""),
        "position": person.get("primaryPosition", {}).get("abbreviation", ""),
    }


# ---------------------------------------------------------------------------
# Get pitcher info for a specific game
# ---------------------------------------------------------------------------
def get_game_pitchers(game_pk: int) -> dict:
    """
    Get starting pitcher info for a game.
    Returns dict with home/away pitcher IDs, names, handedness.
    """
    data = _safe_get(f"{BASE_URL}/game/{game_pk}/boxscore")
    if not data:
        return {}

    result = {}
    for side in ["home", "away"]:
        team_data = data.get("teams", {}).get(side, {})
        pitchers = team_data.get("pitchers", [])

        if pitchers:
            pid = pitchers[0]
            player_key = f"ID{pid}"
            player_data = team_data.get("players", {}).get(player_key, {})
            person = player_data.get("person", {})

            result[f"{side}_pitcher_id"] = pid
            result[f"{side}_pitcher_name"] = person.get("fullName", "Unknown")

            # Get pitcher handedness
            pitcher_info = _safe_get(f"{BASE_URL}/people/{pid}")
            if pitcher_info and pitcher_info.get("people"):
                p = pitcher_info["people"][0]
                result[f"{side}_pitcher_hand"] = p.get("pitchHand", {}).get("code", "U")
            else:
                result[f"{side}_pitcher_hand"] = "U"
        else:
            result[f"{side}_pitcher_id"] = None
            result[f"{side}_pitcher_name"] = "Unknown"
            result[f"{side}_pitcher_hand"] = "U"

    return result


# ---------------------------------------------------------------------------
# Get pitcher season stats
# ---------------------------------------------------------------------------
def get_pitcher_season_stats(pitcher_id: int, season: int) -> dict:
    """Get a pitcher's season-level stats (ERA, WHIP, etc.)."""
    data = _safe_get(
        f"{BASE_URL}/people/{pitcher_id}/stats",
        params={"stats": "season", "group": "pitching", "season": season}
    )
    if not data:
        return {}

    for stat_group in data.get("stats", []):
        splits = stat_group.get("splits", [])
        if splits:
            stat = splits[0].get("stat", {})
            return {
                "pitcher_era": _safe_float(stat.get("era")),
                "pitcher_whip": _safe_float(stat.get("whip")),
                "pitcher_k_per_9": _safe_float(stat.get("strikeoutsPer9Inn")),
                "pitcher_bb_per_9": _safe_float(stat.get("walksPer9Inn")),
                "pitcher_hits_per_9": _safe_float(stat.get("hitsPer9Inn")),
                "pitcher_innings": _safe_float(stat.get("inningsPitched")),
            }
    return {}


def _safe_float(val, default=None):
    """Safely convert a value to float."""
    if val is None or val == "" or val == "-":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Get ballpark info
# ---------------------------------------------------------------------------
def get_venue_info() -> pd.DataFrame:
    """
    Get all MLB venues with basic info.
    Park factors will be added as a static lookup since the API
    doesn't provide them directly.
    """
    data = _safe_get(f"{BASE_URL}/venues", params={"sportId": 1})
    if not data:
        return pd.DataFrame()

    venues = []
    for v in data.get("venues", []):
        venues.append({
            "venue_id": v.get("id"),
            "venue_name": v.get("name"),
            "city": v.get("location", {}).get("city", ""),
            "state": v.get("location", {}).get("state", ""),
            "roof_type": v.get("fieldInfo", {}).get("roofType", "Open"),
            "elevation_ft": v.get("location", {}).get("defaultCoordinates", {}).get("elevation", 0),
        })

    return pd.DataFrame(venues)


# ---------------------------------------------------------------------------
# Static park factors (runs per game relative to league avg)
# 1.0 = neutral, >1.0 = hitter friendly, <1.0 = pitcher friendly
# Based on recent multi-year averages
# ---------------------------------------------------------------------------
PARK_FACTORS = {
    "Coors Field": 1.38,
    "Fenway Park": 1.12,
    "Great American Ball Park": 1.11,
    "Globe Life Field": 1.08,
    "Yankee Stadium": 1.07,
    "Guaranteed Rate Field": 1.05,
    "Citizens Bank Park": 1.04,
    "Wrigley Field": 1.02,
    "Dodger Stadium": 1.01,
    "Target Field": 1.00,
    "Minute Maid Park": 0.99,
    "Truist Park": 0.99,
    "Angel Stadium": 0.98,
    "Busch Stadium": 0.98,
    "Chase Field": 0.98,
    "Kauffman Stadium": 0.97,
    "PNC Park": 0.97,
    "American Family Field": 0.97,
    "Nationals Park": 0.96,
    "Progressive Field": 0.96,
    "Comerica Park": 0.95,
    "Rogers Centre": 0.95,
    "loanDepot park": 0.94,
    "T-Mobile Park": 0.93,
    "Tropicana Field": 0.93,
    "Oracle Park": 0.92,
    "Petco Park": 0.92,
    "Oakland Coliseum": 0.91,
    "Oriole Park at Camden Yards": 1.03,
    "Citi Field": 0.94,
}


def get_park_factor(venue_name: str) -> float:
    """Look up park factor for a venue. Returns 1.0 if unknown."""
    return PARK_FACTORS.get(venue_name, 1.0)


# ---------------------------------------------------------------------------
# Pull season schedule with venues + starting pitchers (efficient bulk pull)
# ---------------------------------------------------------------------------
def get_season_schedule(season: int) -> dict:
    """
    Pull the full season schedule to get venue and starting pitcher info
    for every game. Returns dict keyed by game_pk.

    Much more efficient than calling boxscore for each game individually.
    """
    logger.info(f"Fetching season schedule for {season} (venues + pitchers)...")
    game_info = {}

    # Pull month by month to keep responses manageable
    for month in range(3, 12):  # March through November
        start = f"{season}-{month:02d}-01"
        end_month = month + 1 if month < 12 else 12
        end_year = season if month < 12 else season
        end = f"{end_year}-{end_month:02d}-01"

        data = _safe_get(
            f"{BASE_URL}/schedule",
            params={
                "sportId": 1,
                "startDate": start,
                "endDate": end,
                "gameType": "R",  # Regular season only
                "hydrate": "probablePitcher,venue",
            }
        )
        if not data:
            continue

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                gpk = game.get("gamePk")
                if not gpk:
                    continue

                venue = game.get("venue", {})
                venue_name = venue.get("name", "")

                # Get probable/starting pitchers
                away_team = game.get("teams", {}).get("away", {})
                home_team = game.get("teams", {}).get("home", {})

                away_pitcher = away_team.get("probablePitcher", {})
                home_pitcher = home_team.get("probablePitcher", {})

                game_info[gpk] = {
                    "venue_name": venue_name,
                    "park_factor": get_park_factor(venue_name),
                    "home_pitcher_id": home_pitcher.get("id"),
                    "home_pitcher_name": home_pitcher.get("fullName", "Unknown"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "away_pitcher_name": away_pitcher.get("fullName", "Unknown"),
                    "home_team_name": home_team.get("team", {}).get("name", ""),
                    "away_team_name": away_team.get("team", {}).get("name", ""),
                }

    logger.info(f"Fetched schedule info for {len(game_info)} games in {season}")
    return game_info


# ---------------------------------------------------------------------------
# Master: Build full season dataset
# ---------------------------------------------------------------------------
def build_season_game_logs(season: int, max_players: int = None) -> pd.DataFrame:
    """
    Build complete game log dataset for a season.
    Each row = one player's performance in one game.
    Now enriched with venue/park factors and opposing pitcher data.

    Args:
        season: Year to pull
        max_players: Cap player count (for testing). None = all players.
    """
    logger.info(f"=== Building game logs for {season} ===")

    players = get_season_players(season)
    if players.empty:
        logger.warning(f"No players found for {season}")
        return pd.DataFrame()

    if max_players:
        players = players.head(max_players)
        logger.info(f"Limited to {max_players} players for testing")

    # Pull schedule data (venues + pitchers) in bulk
    schedule = get_season_schedule(season)

    # Cache pitcher season stats to avoid redundant API calls
    pitcher_stats_cache = {}

    all_logs = []
    player_info_cache = {}

    for idx, row in players.iterrows():
        pid = row["player_id"]

        # Get game logs
        logs = get_player_game_log(pid, season)
        if logs.empty:
            continue

        # Get player bio (cached)
        if pid not in player_info_cache:
            player_info_cache[pid] = get_player_info(pid)
        bio = player_info_cache[pid]

        # Merge bio into logs
        logs["bat_side"] = bio.get("bat_side", "U")
        logs["player_name"] = row["player_name"]
        logs["position"] = row.get("position", bio.get("position", ""))

        # ----- Enrich each game with pitcher + venue data -----
        park_factors = []
        pitcher_eras = []
        pitcher_whips = []
        pitcher_k9s = []
        pitcher_hands = []
        platoon_advantages = []

        for _, game_row in logs.iterrows():
            gpk = game_row.get("game_pk")
            game_data = schedule.get(gpk, {})

            # Park factor
            park_factors.append(game_data.get("park_factor", 1.0))

            # Determine opposing pitcher
            is_home = game_row.get("is_home", False)
            if is_home:
                opp_pitcher_id = game_data.get("away_pitcher_id")
            else:
                opp_pitcher_id = game_data.get("home_pitcher_id")

            # Get opposing pitcher stats (cached)
            if opp_pitcher_id and opp_pitcher_id not in pitcher_stats_cache:
                pitcher_stats_cache[opp_pitcher_id] = get_pitcher_season_stats(
                    opp_pitcher_id, season
                )

            p_stats = pitcher_stats_cache.get(opp_pitcher_id, {})
            pitcher_eras.append(p_stats.get("pitcher_era"))
            pitcher_whips.append(p_stats.get("pitcher_whip"))
            pitcher_k9s.append(p_stats.get("pitcher_k_per_9"))

            # Pitcher handedness (from bio cache or quick lookup)
            opp_hand = "U"
            if opp_pitcher_id:
                if opp_pitcher_id not in player_info_cache:
                    player_info_cache[opp_pitcher_id] = get_player_info(opp_pitcher_id)
                opp_bio = player_info_cache[opp_pitcher_id]
                opp_hand = opp_bio.get("throw_side", "U")
            pitcher_hands.append(opp_hand)

            # Platoon advantage
            bat_side = bio.get("bat_side", "U")
            platoon = 1 if bat_side == "S" else (
                1 if (bat_side == "R" and opp_hand == "L") else (
                    1 if (bat_side == "L" and opp_hand == "R") else 0
                )
            )
            platoon_advantages.append(platoon)

        logs["park_factor"] = park_factors
        logs["pitcher_era"] = pitcher_eras
        logs["pitcher_whip"] = pitcher_whips
        logs["pitcher_k_per_9"] = pitcher_k9s
        logs["pitcher_hand"] = pitcher_hands
        logs["platoon_advantage"] = platoon_advantages

        all_logs.append(logs)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx + 1}/{len(players)} players for {season} "
                        f"({len(pitcher_stats_cache)} pitchers cached)")

    if not all_logs:
        return pd.DataFrame()

    result = pd.concat(all_logs, ignore_index=True)
    logger.info(f"Built {len(result)} game log rows for {season} "
                f"(with pitcher + park data)")
    return result


def build_multi_season_dataset(
    start_year: int = 2000,
    end_year: int = 2024,
    max_players_per_season: int = None,
) -> pd.DataFrame:
    """
    Build the full historical dataset across multiple seasons.
    """
    logger.info(f"Building dataset from {start_year} to {end_year}")
    all_seasons = []

    for year in range(start_year, end_year + 1):
        season_data = build_season_game_logs(year, max_players=max_players_per_season)
        if not season_data.empty:
            all_seasons.append(season_data)

    if not all_seasons:
        return pd.DataFrame()

    full_dataset = pd.concat(all_seasons, ignore_index=True)
    logger.info(f"Complete dataset: {len(full_dataset)} rows across {end_year - start_year + 1} seasons")
    return full_dataset