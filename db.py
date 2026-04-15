"""SQLite database for NBA game data."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Optional

# Database path
DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "nba_games.db"))


def get_db_path() -> str:
    """Get the database path."""
    return DB_PATH


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Initialize the database with required tables.

    Args:
        db_path: Optional custom database path

    Returns:
        SQLite connection
    """
    if db_path is None:
        db_path = DB_PATH

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create games table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            date TEXT NOT NULL,
            location TEXT NOT NULL CHECK (location IN ('home', 'away')),
            opponent TEXT NOT NULL,
            result TEXT NOT NULL CHECK (result IN ('W', 'L')),
            pace REAL,
            ftr REAL,
            efg_pct REAL,
            tov_pct REAL,
            orb_pct REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE(team, season_year, date, location, opponent)
        )
    """)

    # Create teams table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            fetched_games_count INTEGER DEFAULT 0,
            last_fetched TEXT
        )
    """)

    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_team_season
        ON games(team, season_year)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_date
        ON games(date)
    """)

    conn.commit()
    return conn


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a database connection."""
    if db_path is None:
        db_path = DB_PATH
    return sqlite3.connect(db_path)


def save_games(games: list[dict], team: str, season_year: int, db_path: Optional[str] = None) -> int:
    """
    Save games to the database.

    Args:
        games: List of game dictionaries
        team: Team code (e.g., 'BOS')
        season_year: Season year (e.g., 2025)
        db_path: Optional custom database path

    Returns:
        Number of games saved
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    saved_count = 0

    for game in games:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO games
                (team, season_year, date, location, opponent, result,
                 pace, ftr, efg_pct, tov_pct, orb_pct, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team,
                season_year,
                game.get("date"),
                game.get("location"),
                game.get("opponent"),
                game.get("result"),
                game.get("pace"),
                game.get("ftr"),
                game.get("efg_pct"),
                game.get("tov_pct"),
                game.get("orb_pct"),
                now,
            ))
            saved_count += 1
        except sqlite3.Error as e:
            # Skip duplicates or invalid data
            pass

    conn.commit()
    conn.close()

    # Update teams table
    _update_team_stats(team, saved_count, now, db_path)

    return saved_count


def _update_team_stats(team: str, games_count: int, last_fetched: str, db_path: Optional[str] = None) -> None:
    """Update team statistics in the teams table."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Get team name from the fetch module
    from fetch_basketball_ref import NBA_TEAMS
    team_name = NBA_TEAMS.get(team, team)

    # Update or insert team record
    cursor.execute("""
        INSERT INTO teams (code, name, fetched_games_count, last_fetched)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            name = excluded.name,
            fetched_games_count = teams.fetched_games_count + excluded.fetched_games_count,
            last_fetched = excluded.last_fetched
    """, (team, team_name, games_count, last_fetched))

    conn.commit()
    conn.close()


def get_games(
    team: Optional[str] = None,
    season_year: Optional[int] = None,
    limit: Optional[int] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """
    Get games from the database.

    Args:
        team: Optional team code filter
        season_year: Optional season year filter
        limit: Optional limit on number of results
        db_path: Optional custom database path

    Returns:
        List of game dictionaries
    """
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM games WHERE 1=1"
    params = []

    if team:
        query += " AND team = ?"
        params.append(team.upper())

    if season_year:
        query += " AND season_year = ?"
        params.append(season_year)

    query += " ORDER BY date DESC"

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_teams(db_path: Optional[str] = None) -> list[dict]:
    """
    Get all teams in the database.

    Args:
        db_path: Optional custom database path

    Returns:
        List of team dictionaries with stats
    """
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            t.code,
            t.name,
            t.fetched_games_count,
            t.last_fetched,
            COUNT(g.id) as games_in_db,
            SUM(CASE WHEN g.result = 'W' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN g.result = 'L' THEN 1 ELSE 0 END) as losses
        FROM teams t
        LEFT JOIN games g ON t.code = g.team
        GROUP BY t.code, t.name, t.fetched_games_count, t.last_fetched
        ORDER BY t.name
    """)

    rows = cursor.fetchall()
    conn.close()

    teams = []
    for row in rows:
        d = dict(row)
        d["win_pct"] = (
            round(d["wins"] / (d["wins"] + d["losses"]) * 100, 1)
            if (d["wins"] + d["losses"]) > 0 else 0.0
        )
        teams.append(d)

    return teams


def get_team_info(team_code: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get info for a specific team."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            t.code,
            t.name,
            t.fetched_games_count,
            t.last_fetched,
            COUNT(g.id) as games_in_db,
            SUM(CASE WHEN g.result = 'W' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN g.result = 'L' THEN 1 ELSE 0 END) as losses
        FROM teams t
        LEFT JOIN games g ON t.code = g.team
        WHERE t.code = ?
        GROUP BY t.code, t.name, t.fetched_games_count, t.last_fetched
    """, (team_code.upper(),))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    d = dict(row)
    d["win_pct"] = (
        round(d["wins"] / (d["wins"] + d["losses"]) * 100, 1)
        if (d["wins"] + d["losses"]) > 0 else 0.0
    )
    return d


def fetch_team_games(
    team_code: str,
    start_year: int = 2017,
    end_year: int = 2026,
    force_refresh: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """
    Fetch games from basketball-reference and save to database.

    Args:
        team_code: NBA team code (e.g., 'BOS')
        start_year: First season to fetch
        end_year: Last season to fetch
        force_refresh: Bypass cache
        db_path: Optional custom database path

    Returns:
        Summary dict with fetch results
    """
    from fetch_basketball_ref import fetch_multiple_years, NBA_TEAMS

    if team_code not in NBA_TEAMS:
        raise ValueError(f"Unknown team code: {team_code}")

    # Initialize DB if needed
    if db_path is None:
        db_path = DB_PATH
    init_db(db_path)

    # Fetch games
    all_games, failed_years = fetch_multiple_years(team_code, start_year, end_year, force_refresh=force_refresh)

    # Group by season year and save
    by_season: dict[int, list[dict]] = {}
    for game in all_games:
        # Try to determine season year from date
        date_str = game.get("date", "")
        if date_str:
            try:
                year = int(date_str[:4])
                # NBA seasons span two years - if month is Jan-Jun, it's the second year
                month = int(date_str[5:7]) if len(date_str) >= 7 else 1
                if month >= 1 and month <= 6:
                    season_year = year
                else:
                    season_year = year
                by_season.setdefault(season_year, []).append(game)
            except (ValueError, IndexError):
                pass

    # Save each season
    total_saved = 0
    for season_year, games in sorted(by_season.items()):
        saved = save_games(games, team_code, season_year, db_path)
        total_saved += saved

    return {
        "team": team_code,
        "team_name": NBA_TEAMS.get(team_code, team_code),
        "total_games_fetched": len(all_games),
        "total_games_saved": total_saved,
        "seasons_fetched": len(by_season),
        "seasons": sorted(by_season.keys()),
        "failed_years": sorted(failed_years),
    }


def clear_team_games(team_code: str, db_path: Optional[str] = None) -> int:
    """Clear all games for a team (for re-fetch)."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM games WHERE team = ?", (team_code.upper(),))
    cursor.execute("DELETE FROM teams WHERE code = ?", (team_code.upper(),))

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    return deleted


# Initialize database on module import
def _ensure_db():
    """Ensure database is initialized."""
    init_db()


_ensure_db()
