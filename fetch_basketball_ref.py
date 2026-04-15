"""Fetch NBA game data from basketball-reference.com."""
from __future__ import annotations

import os
import time
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting: seconds to wait between requests
REQUEST_DELAY = 3.0

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# NBA team codes
NBA_TEAMS = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def _cache_key(team: str, year: int) -> str:
    """Generate a cache key for a team/year combination."""
    key = f"{team}_{year}"
    return os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.json")


def _read_cache(team: str, year: int) -> Optional[list[dict]]:
    """Read cached data if it exists and is less than 6 hours old."""
    cache_path = _cache_key(team, year)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r") as f:
            cached = json.load(f)

        # Check if cache is less than 6 hours old
        cached_time = datetime.fromisoformat(cached["_cached_at"])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        if age_hours > 6:
            logger.info(f"Cache expired for {team}/{year} ({age_hours:.1f}h old)")
            return None

        logger.info(f"Using cached data for {team}/{year}")
        return cached.get("games", [])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _write_cache(team: str, year: int, games: list[dict]) -> None:
    """Write games data to cache."""
    cache_path = _cache_key(team, year)
    cache_data = {
        "team": team,
        "year": year,
        "_cached_at": datetime.now().isoformat(),
        "games": games,
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)


def fetch_team_season(
    team_code: str,
    season_year: int,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch advanced game logs for a team and season from basketball-reference.com.

    Args:
        team_code: NBA team abbreviation (e.g., 'BOS', 'LAL', 'GSW')
        season_year: The year the season started (e.g., 2025 for 2025-26 season)
        force_refresh: If True, bypass cache

    Returns:
        List of game dictionaries with fields:
        - date: str (YYYY-MM-DD)
        - location: 'home' or 'away'
        - opponent: str
        - result: 'W' or 'L'
        - pace: float
        - ftr: float (Free Throw Rate)
        - efg_pct: float (Effective FG%)
        - tov_pct: float (Turnover %)
        - orb_pct: float (Offensive Rebound %)
    """
    if team_code not in NBA_TEAMS:
        raise ValueError(f"Unknown team code: {team_code}. Valid codes: {list(NBA_TEAMS.keys())}")

    # Check cache first (unless force_refresh)
    if not force_refresh:
        cached = _read_cache(team_code, season_year)
        if cached is not None:
            return cached

    # Build URL
    url = f"https://www.basketball-reference.com/teams/{team_code}/{season_year}/gamelog-advanced/"
    logger.info(f"Fetching {url}")

    # Fetch with rate limiting
    time.sleep(REQUEST_DELAY)

    # Rotate User-Agent to avoid IP-level rate limits
    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]
    import random
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    response = requests.get(url, headers=headers, timeout=30)
    # Retry up to 3 times on server errors with exponential backoff
    retries = 0
    max_retries = 3
    while response.status_code >= 500 and retries < max_retries:
        retries += 1
        wait_time = (2 ** retries) + REQUEST_DELAY
        logger.warning(f"Server error {response.status_code} for {team_code}/{season_year}, retry {retries}/{max_retries} in {wait_time:.1f}s")
        time.sleep(wait_time)
        response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    games = []

    # Find ALL tables with "team_stats" in their id
    all_tables = soup.find_all("table")
    logger.info(f"Found {len(all_tables)} tables on the page")

    # The table ID is now "team_game_log_adv_reg" for advanced stats
    target_table = None
    for table in all_tables:
        table_id = table.get("id", "")
        if "team_game_log_adv" in table_id or table_id == "team_game_log_adv_reg":
            thead = table.find("thead")
            if thead:
                target_table = table
                logger.info(f"Found advanced stats table with id: {table_id}")
                break

    if target_table:
        games = _parse_advanced_table(target_table, team_code)
    else:
        # Fallback: find table with ORtg in second header row
        for table in all_tables:
            thead = table.find("thead")
            if not thead:
                continue
            header_rows = thead.find_all("tr")
            if len(header_rows) >= 2:
                # Check second row (index 1) for ORtg
                second_row_headers = [th.get_text(strip=True) for th in header_rows[1].find_all("th")]
                if "ORtg" in second_row_headers:
                    logger.info(f"Found advanced table via fallback with headers: {second_row_headers[:10]}...")
                    games = _parse_advanced_table(table, team_code)
                    break

    if not games:
        # Fallback: try generic table parsing
        for table in all_tables:
            tbody = table.find("tbody")
            if not tbody:
                continue
            rows = tbody.find_all("tr")
            if len(rows) > 10:
                games = _parse_generic_table(table, team_code)
                if games:
                    break

    if not games:
        raise ValueError(f"Could not find game data table for {team_code}/{season_year}")

    # Write to cache
    _write_cache(team_code, season_year, games)
    logger.info(f"Fetched {len(games)} games for {team_code}/{season_year}")

    return games


def _parse_advanced_table(table: BeautifulSoup, team_code: str) -> list[dict]:
    """Parse the advanced stats table to extract game data."""
    games = []

    tbody = table.find("tbody")
    if not tbody:
        return games

    thead = table.find("thead")
    if not thead:
        return games

    # Headers are spread across TWO rows:
    # Row 0: ['', 'Score', 'Advanced', 'Offensive Four Factors', 'Defensive Four Factors'] (section labels)
    # Row 1: ['Rk', 'Gtm', 'Date', '', 'Opp', 'Rslt', 'Tm', 'Opp', 'OT', 'ORtg', 'DRtg', 'Pace', 'FTr', '3PAr', 'TS%']... (actual column names)
    header_rows = thead.find_all("tr")
    if len(header_rows) < 2:
        logger.warning("Expected 2 header rows, found %d", len(header_rows))
        return games

    # Use the second row (index 1) for actual column names
    actual_header_row = header_rows[1]
    header_ths = actual_header_row.find_all("th")
    headers = [th.get_text(strip=True) for th in header_ths]

    # Build column index map from data-stat attributes
    col_indices = {}
    stat_to_col = {}

    for idx, th in enumerate(header_ths):
        header_text = th.get_text(strip=True)
        data_stat = th.get("data-stat", "")

        if header_text:
            col_indices[header_text] = idx
        if data_stat:
            stat_to_col[data_stat] = idx

    # Key data-stat names (updated per issue):
    # pace, ft_rate (was ftr), efg_pct, team_tov_pct (was tov_pct), team_orb_pct (was orb_pct)
    date_idx = stat_to_col.get("date", col_indices.get("Date"))
    opp_idx = stat_to_col.get("opp_name_abbr", col_indices.get("Opp"))
    rslt_idx = stat_to_col.get("team_game_result", col_indices.get("Rslt"))
    location_idx = stat_to_col.get("game_location")
    pace_idx = stat_to_col.get("pace", col_indices.get("Pace"))
    ftr_idx = stat_to_col.get("ft_rate", col_indices.get("FTr"))
    efg_idx = stat_to_col.get("efg_pct", col_indices.get("eFG%"))
    tov_idx = stat_to_col.get("team_tov_pct", col_indices.get("TOV%"))
    orb_idx = stat_to_col.get("team_orb_pct", col_indices.get("ORB%"))

    def get_cell_text(row_cells: list, idx: int) -> str:
        if idx is not None and idx < len(row_cells):
            return row_cells[idx].get_text(strip=True)
        return ""

    def get_cell_float(row_cells: list, idx: int) -> float:
        if idx is None:
            return 0.0
        text = get_cell_text(row_cells, idx)
        try:
            # Handle percentage
            if "%" in text:
                return float(text.replace("%", "")) / 100
            # Handle decimal without leading 0 (e.g., .253)
            if text.startswith("."):
                text = "0" + text
            return float(text)
        except (ValueError, TypeError):
            return 0.0

    rows = tbody.find_all("tr")
    for row in rows:
        # Skip header/section rows
        if row.get("class") and ("thead" in row.get("class", []) or "stat_column" in row.get("class", [])):
            continue
        if row.get("data-row"):
            pass  # Skip spacer rows

        cells = row.find_all(["th", "td"])
        if len(cells) < 5:
            continue

        # Get data using data-stat attributes directly
        data_stats = {}
        for cell in cells:
            stat = cell.get("data-stat", "")
            if stat:
                data_stats[stat] = cell.get_text(strip=True)

        # Extract values from data-stat map
        date_str = data_stats.get("date", "")
        opponent = data_stats.get("opp_name_abbr", "").lstrip("@")
        result_str = data_stats.get("team_game_result", "")
        location_str = data_stats.get("game_location", "")

        # Skip invalid rows
        if not date_str or not opponent or not result_str:
            continue
        if date_str in ("Date", "Rk", ""):
            continue

        # Parse date (format: YYYY-MM-DD)
        try:
            game_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            try:
                game_date = datetime.strptime(date_str, "%b %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                game_date = date_str

        # Determine location: @ prefix means away, game_location data-stat is "@" for away
        location = "away" if location_str == "@" or opponent.startswith("@") else "home"
        opponent = opponent.lstrip("@")

        # Parse result
        result = "W" if result_str.startswith("W") else "L"

        # Extract advanced stats using correct data-stat names
        # FTr values are stored as .253 (decimal without leading 0)
        def safe_float_stat(stat_name: str, divisor: float = 1.0) -> float:
            val = data_stats.get(stat_name, "")
            try:
                if not val:
                    return 0.0
                # Handle decimal without leading 0
                if val.startswith("."):
                    val = "0" + val
                return float(val) / divisor
            except (ValueError, TypeError):
                return 0.0

        pace = safe_float_stat("pace")
        ftr = safe_float_stat("ft_rate")  # ft_rate stored as proportion (.253), no divisor
        efg_pct = safe_float_stat("efg_pct")  # stored as proportion (.511), no divisor
        tov_pct = safe_float_stat("team_tov_pct", 100)  # stored as percentage (9.7%), needs divisor
        orb_pct = safe_float_stat("team_orb_pct", 100)  # stored as percentage (25.0%), needs divisor

        game = {
            "date": game_date,
            "location": location,
            "opponent": opponent,
            "result": result,
            "pace": round(pace, 1),
            "ftr": round(ftr, 3),
            "efg_pct": round(efg_pct, 3),
            "tov_pct": round(tov_pct, 3),
            "orb_pct": round(orb_pct, 3),
        }

        games.append(game)

    return games


def _parse_generic_table(table: BeautifulSoup, team_code: str) -> list[dict]:
    """Fallback parser for generic tables using data-stat attributes."""
    games = []

    tbody = table.find("tbody")
    if not tbody:
        return games

    rows = tbody.find_all("tr")
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) < 8:
            continue

        # Get data using data-stat attributes
        data = {}
        for cell in cells:
            stat = cell.get("data-stat", "")
            val = cell.get_text(strip=True)
            if stat:
                data[stat] = val

        # Check if this looks like a game row
        if not data.get("date") or not data.get("opp_name_abbr"):
            continue

        # Get basic info - use corrected stat names
        date_str = data.get("date", "")
        opponent = data.get("opp_name_abbr", "").lstrip("@")
        result_str = data.get("team_game_result", "")
        location_str = data.get("game_location", "")

        if not date_str or not opponent:
            continue

        # Parse date
        try:
            game_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            try:
                game_date = datetime.strptime(date_str, "%b %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                game_date = date_str

        location = "away" if location_str == "@" else "home"
        result_char = "W" if result_str.startswith("W") else "L"

        # Get advanced stats - use corrected stat names
        def safe_float(val: str, divisor: float = 1.0) -> float:
            try:
                if not val:
                    return 0.0
                if val.startswith("."):
                    val = "0" + val
                if "%" in val:
                    return float(val.replace("%", "")) / 100
                return float(val) / divisor
            except (ValueError, TypeError):
                return 0.0

        game = {
            "date": game_date,
            "location": location,
            "opponent": opponent,
            "result": result_char,
            "pace": safe_float(data.get("pace", "0")),
            "ftr": safe_float(data.get("ft_rate", "0")),  # ft_rate, stored as proportion (.253)
            "efg_pct": safe_float(data.get("efg_pct", "0")),  # stored as proportion (.511), no divisor
            "tov_pct": safe_float(data.get("team_tov_pct", "0"), 100),  # stored as percentage (9.7%)
            "orb_pct": safe_float(data.get("team_orb_pct", "0"), 100),  # stored as percentage (25.0%)
        }

        games.append(game)

    return games


def fetch_multiple_years(
    team_code: str,
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch game data for multiple seasons.

    Args:
        team_code: NBA team abbreviation
        start_year: First season year (e.g., 2017 for 2017-18 season)
        end_year: Last season year (e.g., 2026 for 2025-26 season)
        force_refresh: If True, bypass cache

    Returns:
        Combined list of all games from all seasons
    """
    all_games = []

    for year in range(start_year, end_year + 1):
        try:
            games = fetch_team_season(team_code, year, force_refresh=force_refresh)
            all_games.extend(games)
            logger.info(f"Fetched {len(games)} games for {team_code}/{year}")
        except Exception as e:
            logger.error(f"Error fetching {team_code}/{year}: {e}")

    return all_games


if __name__ == "__main__":
    # Test fetching for Celtics
    import sys

    team = sys.argv[1] if len(sys.argv) > 1 else "BOS"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

    print(f"Fetching {team} season {year}...")
    games = fetch_team_season(team, year)

    print(f"\nFetched {len(games)} games:")
    for g in games[:5]:
        print(f"  {g['date']}: {g['location']} vs {g['opponent']} - {g['result']}")
        print(f"    Pace: {g['pace']}, FTr: {g['ftr']}, eFG%: {g['efg_pct']:.3f}")