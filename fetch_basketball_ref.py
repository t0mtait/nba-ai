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

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Find the advanced stats table - it has an id like "tfooter_BOS" for the advanced section
    # The page structure: regular stats table + advanced stats table
    # We need to find the row that corresponds to the "Advanced" statistics
    # Actually looking at the CSV, the columns are: ORtg, DRtg, Pace, FTr, 3PAr, TS%, etc.
    # This is in the "Advanced" section of the table

    games = []

    # Find ALL tables with "team_stats" in their id
    all_tables = soup.find_all("table")
    logger.info(f"Found {len(all_tables)} tables on the page")

    # The regular gamelog table has id "tfooter_{TEAM}" but it's actually for basic stats
    # We need to parse the table and extract the columns we need

    # Look for the table - it typically has the game-by-game data
    # Structure: <table id="tfooter_{TEAM}"> contains the advanced stats
    # But actually the table with advanced stats has headers like: ORtg, DRtg, Pace, FTr, etc.

    # Find the correct table by looking for the Pace column header
    for table in all_tables:
        thead = table.find("thead")
        if not thead:
            continue

        # Get all header text
        headers_text = [th.get_text(strip=True) for th in thead.find_all("th")]
        headers_str = " ".join(headers_text)

        # Look for advanced stats columns
        if "ORtg" in headers_str and "DRtg" in headers_str and "Pace" in headers_str:
            logger.info(f"Found advanced stats table with headers: {headers_text[:10]}...")
            games = _parse_advanced_table(table, team_code)
            break

    if not games:
        # Fallback: try to find any table with game data
        for table in all_tables:
            tbody = table.find("tbody")
            if not tbody:
                continue
            rows = tbody.find_all("tr")
            if len(rows) > 10:  # Likely a game log
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

    # Find header row to get column indices
    thead = table.find("thead")
    if not thead:
        return games

    header_rows = thead.find_all("tr")
    # The last header row contains the actual column names
    last_header_row = header_rows[-1] if header_rows else None
    if not last_header_row:
        return games

    header_ths = last_header_row.find_all("th")
    headers = [th.get_text(strip=True) for th in header_ths]

    # Map column names to indices
    col_map = {}
    for idx, h in enumerate(headers):
        if h in ("ORtg", "DRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%"):
            col_map[h] = idx

    # Also need basic columns: Date, Opp, Rslt
    # These might be in earlier th elements
    date_idx = None
    opp_idx = None
    rslt_idx = None

    # Find by looking at the second header row (first is usually row labels)
    if len(header_rows) >= 2:
        second_row = header_rows[-2] if len(header_rows) >= 2 else header_rows[0]
        second_headers = [th.get_text(strip=True) for th in second_row.find_all("th")]
        for idx, h in enumerate(second_headers):
            if h == "Date":
                date_idx = idx
            elif h == "Opp":
                opp_idx = idx
            elif h == "Rslt":
                rslt_idx = idx

    rows = tbody.find_all("tr")
    for row in rows:
        # Skip header/section rows
        if "class" in row.attrs and ("thead" in row["class"] or "stat_column" in row["class"]):
            continue

        cells = row.find_all(["th", "td"])
        if len(cells) < 5:
            continue

        # Extract data using indices
        def get_cell_text(idx: int) -> str:
            if idx is not None and idx < len(cells):
                return cells[idx].get_text(strip=True)
            return ""

        def get_cell_float(idx: int) -> float:
            text = get_cell_text(idx)
            try:
                return float(text.replace("%", "")) / 100 if "%" in text else float(text)
            except (ValueError, TypeError):
                return 0.0

        date_str = get_cell_text(date_idx) if date_idx is not None else ""
        opp_str = get_cell_text(opp_idx) if opp_idx is not None else ""
        rslt_str = get_cell_text(rslt_idx) if rslt_idx is not None else ""

        # Skip if no valid data
        if not date_str or not opp_str or not rslt_str:
            continue
        if date_str in ("Date", "Rk", ""):
            continue

        # Parse date (format: YYYY-MM-DD)
        try:
            # The date might be in different formats, try to parse
            game_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            # Try other formats
            try:
                game_date = datetime.strptime(date_str, "%b %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                game_date = date_str

        # Determine location: @ prefix means away
        location = "away" if opp_str.startswith("@") else "home"
        opponent = opp_str.lstrip("@")

        # Parse result
        result = "W" if rslt_str.startswith("W") else "L"

        # Extract advanced stats using column mappings
        # Headers we need: Pace, FTr, eFG%, TOV%, ORB%
        pace = 0.0
        ftr = 0.0
        efg_pct = 0.0
        tov_pct = 0.0
        orb_pct = 0.0

        # Try to get from col_map (column index based)
        if "Pace" in col_map:
            pace = get_cell_float(col_map["Pace"])
        if "FTr" in col_map:
            ftr = get_cell_float(col_map["FTr"])
        if "eFG%" in col_map:
            efg_pct = get_cell_float(col_map["eFG%"])
        if "TOV%" in col_map:
            tov_pct = get_cell_float(col_map["TOV%"])
        if "ORB%" in col_map:
            orb_pct = get_cell_float(col_map["ORB%"])

        # If col_map didn't work, try by finding cells with header text
        if pace == 0.0:
            # Find cells by their header (stored in data-stat attribute)
            for cell in cells:
                stat = cell.get("data-stat", "")
                val = cell.get_text(strip=True)

                try:
                    if stat == "pace":
                        pace = float(val)
                    elif stat == "ftr":
                        ftr = float(val)
                    elif stat == "efg_pct":
                        efg_pct = float(val) / 100 if val else 0.0
                    elif stat == "tov_pct":
                        tov_pct = float(val) / 100 if val else 0.0
                    elif stat == "orb_pct":
                        orb_pct = float(val) / 100 if val else 0.0
                except (ValueError, TypeError):
                    pass

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
    """Fallback parser for generic tables."""
    games = []

    tbody = table.find("tbody")
    if not tbody:
        return games

    rows = tbody.find_all("tr")
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) < 8:
            continue

        # Try to extract using data-stat attributes
        data = {}
        for cell in cells:
            stat = cell.get("data-stat", "")
            val = cell.get_text(strip=True)
            data[stat] = val

        # Check if this looks like a game row
        if not data.get("date_game") or not data.get("opp_id"):
            continue

        # Get basic info
        date_str = data.get("date_game", "")
        opponent = data.get("opp_id", "").lstrip("@")
        result = data.get("game_result", "")

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

        location = "away" if data.get("opp_id", "").startswith("@") else "home"
        result_char = "W" if result.startswith("W") else "L"

        # Get advanced stats
        def safe_float(val: str) -> float:
            try:
                if "%" in val:
                    return float(val.replace("%", "")) / 100
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        game = {
            "date": game_date,
            "location": location,
            "opponent": opponent,
            "result": result_char,
            "pace": safe_float(data.get("pace", "0")),
            "ftr": safe_float(data.get("ftr", "0")),
            "efg_pct": safe_float(data.get("efg_pct", "0")),
            "tov_pct": safe_float(data.get("tov_pct", "0")),
            "orb_pct": safe_float(data.get("orb_pct", "0")),
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
