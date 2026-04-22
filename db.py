"""SQLite database for NBA game data."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Optional

# Database path
DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "nba.db"))


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

    # Games table - stores per-team game records with matchup info
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            date TEXT NOT NULL,
            location TEXT NOT NULL CHECK (location IN ('home', 'away')),
            opponent TEXT NOT NULL,
            result TEXT NOT NULL CHECK (result IN ('W', 'L')),
            team_score INTEGER,
            opponent_score INTEGER,
            pace REAL,
            ftr REAL,
            efg_pct REAL,
            tov_pct REAL,
            orb_pct REAL,
            ortg REAL,
            drtg REAL,
            line_closed_at TEXT,
            home_ml INTEGER,
            away_ml INTEGER,
            home_spread REAL,
            away_spread REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE(team, season_year, date, location, opponent)
        )
    """)

    # Team stats per season (from basketball-reference)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_season_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            pace REAL,
            ortg REAL,
            drtg REAL,
            net_rtg REAL,
            ftr REAL,
            threepar REAL,
            ts_pct REAL,
            efg_pct REAL,
            tov_pct REAL,
            orb_pct REAL,
            drb_pct REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE(team, season_year)
        )
    """)

    # Injuries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS injuries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            player_name TEXT NOT NULL,
            injury TEXT,
            status TEXT,
            date_reported TEXT,
            game_date TEXT,
            season_year INTEGER,
            fetched_at TEXT NOT NULL
        )
    """)

    # Teams metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            fetched_games_count INTEGER DEFAULT 0,
            last_fetched TEXT
        )
    """)

    # Indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_team_season
        ON games(team, season_year)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_date
        ON games(date)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_team_season_stats
        ON team_season_stats(team, season_year)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_injuries_team_date
        ON injuries(team, date_reported)
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
                 team_score, opponent_score,
                 pace, ftr, efg_pct, tov_pct, orb_pct, ortg, drtg,
                 home_ml, away_ml, home_spread, away_spread,
                 fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team,
                season_year,
                game.get("date"),
                game.get("location"),
                game.get("opponent"),
                game.get("result"),
                game.get("team_score"),
                game.get("opponent_score"),
                game.get("pace"),
                game.get("ftr"),
                game.get("efg_pct"),
                game.get("tov_pct"),
                game.get("orb_pct"),
                game.get("ortg"),
                game.get("drtg"),
                game.get("home_ml"),
                game.get("away_ml"),
                game.get("home_spread"),
                game.get("away_spread"),
                now,
            ))
            saved_count += 1
        except sqlite3.Error:
            pass

    conn.commit()
    conn.close()
    _update_team_stats(team, saved_count, now, db_path)
    return saved_count


def save_team_season_stats(stats: list[dict], team: str, season_year: int, db_path: Optional[str] = None) -> int:
    """Save team season-level stats."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    saved = 0
    for s in stats:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO team_season_stats
                (team, season_year, pace, ortg, drtg, net_rtg, ftr, threepar,
                 ts_pct, efg_pct, tov_pct, orb_pct, drb_pct, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team, season_year,
                s.get("pace"), s.get("ortg"), s.get("drtg"), s.get("net_rtg"),
                s.get("ftr"), s.get("threepar"), s.get("ts_pct"),
                s.get("efg_pct"), s.get("tov_pct"), s.get("orb_pct"), s.get("drb_pct"),
                now,
            ))
            saved += 1
        except sqlite3.Error:
            pass
    conn.commit()
    conn.close()
    return saved


def save_injuries(injuries: list[dict], team: str, db_path: Optional[str] = None) -> int:
    """Save injury reports."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    saved = 0
    for inj in injuries:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO injuries
                (team, player_name, injury, status, date_reported, game_date, season_year, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team,
                inj.get("player_name"),
                inj.get("injury"),
                inj.get("status"),
                inj.get("date_reported"),
                inj.get("game_date"),
                inj.get("season_year"),
                now,
            ))
            saved += 1
        except sqlite3.Error:
            pass
    conn.commit()
    conn.close()
    return saved


def _update_team_stats(team: str, games_count: int, last_fetched: str, db_path: Optional[str] = None) -> None:
    """Update team statistics in the teams table."""
    from fetch_basketball_ref import NBA_TEAMS
    conn = get_connection(db_path)
    cursor = conn.cursor()
    team_name = NBA_TEAMS.get(team, team)
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
    """Get games from the database."""
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


def get_matchups(
    team: Optional[str] = None,
    opponent: Optional[str] = None,
    season_year: Optional[int] = None,
    limit: Optional[int] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """
    Get matchup records between two teams.
    Each record represents one team's perspective — so for BOS vs LAL you'll see
    two rows per game (one from each team's perspective).
    """
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    query = "SELECT * FROM games WHERE 1=1"
    params = []
    if team:
        query += " AND team = ?"
        params.append(team.upper())
    if opponent:
        query += " AND opponent = ?"
        params.append(opponent.upper())
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


def get_recent_matchup(
    team: str,
    opponent: str,
    n: int = 10,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Get the last n head-to-head games between two teams."""
    return get_matchups(team=team, opponent=opponent, limit=n, db_path=db_path)


def get_team_stats(team: str, season_year: int, db_path: Optional[str] = None) -> Optional[dict]:
    """Get season-level stats for a team."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM team_season_stats WHERE team = ? AND season_year = ?",
        (team.upper(), season_year)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_injuries(team: Optional[str] = None, db_path: Optional[str] = None) -> list[dict]:
    """Get current injury reports, optionally filtered by team."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if team:
        cursor.execute(
            "SELECT * FROM injuries WHERE team = ? ORDER BY date_reported DESC",
            (team.upper(),)
        )
    else:
        cursor.execute("SELECT * FROM injuries ORDER BY date_reported DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_teams(db_path: Optional[str] = None) -> list[dict]:
    """Get all teams in the database."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            t.code, t.name, t.fetched_games_count, t.last_fetched,
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
        total = d["wins"] + d["losses"]
        d["win_pct"] = round(d["wins"] / total * 100, 1) if total > 0 else 0.0
        teams.append(d)
    return teams


def get_team_info(team_code: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get info for a specific team."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            t.code, t.name, t.fetched_games_count, t.last_fetched,
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
    total = d["wins"] + d["losses"]
    d["win_pct"] = round(d["wins"] / total * 100, 1) if total > 0 else 0.0
    return d


def fetch_team_games(
    team_code: str,
    start_year: int = 2017,
    end_year: int = 2026,
    force_refresh: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Fetch games from basketball-reference and save to database."""
    from fetch_basketball_ref import fetch_multiple_years, NBA_TEAMS
    if team_code not in NBA_TEAMS:
        raise ValueError(f"Unknown team code: {team_code}")
    if db_path is None:
        db_path = DB_PATH
    init_db(db_path)
    all_games = fetch_multiple_years(team_code, start_year, end_year, force_refresh=force_refresh)
    by_season: dict[int, list[dict]] = {}
    for game in all_games:
        date_str = game.get("date", "")
        if date_str:
            try:
                year = int(date_str[:4])
                by_season.setdefault(year, []).append(game)
            except (ValueError, IndexError):
                pass
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
    }


def clear_team_games(team_code: str, db_path: Optional[str] = None) -> int:
    """Clear all games for a team."""
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
    init_db()
_ensure_db()