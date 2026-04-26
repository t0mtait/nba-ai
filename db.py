"""SQLite database for NBA game data."""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "nba.db"))

NBA_TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Reverse mapping: full name -> code
TEAM_NAME_TO_CODE = {v: k for k, v in NBA_TEAMS.items()}

# ESPN team name variations that need mapping
ESPN_TEAM_MAPPING = {
    "Los Angeles Lakers": "LAL",
    "Los Angeles Clippers": "LAC",
    "New Orleans Pelicans": "NOP",
    "New Orleans": "NOP",
    "Portland Trail Blazers": "POR",
    "San Antonio Spurs": "SAS",
    "Golden State Warriors": "GSW",
    "Golden State": "GSW",
    "Utah Jazz": "UTA",
    "Indiana Pacers": "IND",
    "Philadelphia 76ers": "PHI",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Miami Heat": "MIA",
    "Toronto Raptors": "TOR",
    "Phoenix Suns": "PHO",
    "Atlanta Hawks": "ATL",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Houston Rockets": "HOU",
    "Memphis Grizzlies": "MEM",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Sacramento Kings": "SAC",
    "Washington Wizards": "WAS",
}


def get_db_path() -> str:
    return DB_PATH


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = DB_PATH
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Games table
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
            net_rtg REAL,
            ts_pct REAL,
            threepar REAL,
            drb_pct REAL,
            line_closed_at TEXT,
            home_ml INTEGER,
            away_ml INTEGER,
            home_spread REAL,
            away_spread REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE(team, season_year, date, location, opponent)
        )
    """)

    # Team season stats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_season_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            pace REAL, ortg REAL, drtg REAL, net_rtg REAL,
            ftr REAL, threepar REAL, ts_pct REAL,
            efg_pct REAL, tov_pct REAL, orb_pct REAL, drb_pct REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE(team, season_year)
        )
    """)

    # Injuries
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
            will_play INTEGER DEFAULT 1,
            player_min REAL DEFAULT 0,
            player_pts REAL DEFAULT 0,
            player_reb REAL DEFAULT 0,
            player_ast REAL DEFAULT 0,
            player_impact_score REAL DEFAULT 0,
            fetched_at TEXT NOT NULL
        )
    """)

    # Teams metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            fetched_games_count INTEGER DEFAULT 0,
            last_fetched TEXT
        )
    """)

    # Power rankings (user-submitted)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS power_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season_year INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS power_ranking_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ranking_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            rank INTEGER NOT NULL,
            power_rating REAL DEFAULT 0.0,
            net_rtg REAL DEFAULT 0.0,
            ortg_override REAL,
            drtg_override REAL,
            notes TEXT,
            FOREIGN KEY (ranking_id) REFERENCES power_rankings(id),
            UNIQUE(ranking_id, team)
        )
    """)

    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_team_season ON games(team, season_year)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_team_season_stats ON team_season_stats(team, season_year)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_injuries_team_date ON injuries(team, date_reported)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_power_rankings_season ON power_rankings(season_year)")

    conn.commit()

    # Migration: add new columns to injuries table if they don't exist
    try:
        for col, col_type in [
            ("will_play", "INTEGER DEFAULT 1"),
            ("player_min", "REAL DEFAULT 0"),
            ("player_pts", "REAL DEFAULT 0"),
            ("player_reb", "REAL DEFAULT 0"),
            ("player_ast", "REAL DEFAULT 0"),
            ("player_impact_score", "REAL DEFAULT 0"),
        ]:
            cursor.execute(f"ALTER TABLE injuries ADD COLUMN {col} {col_type}")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    return conn


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = DB_PATH
    return sqlite3.connect(db_path)


# ---- Games ----

def save_games(games: list[dict], team: str, season_year: int, db_path: Optional[str] = None) -> int:
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
                 pace, ftr, efg_pct, tov_pct, orb_pct, ortg, drtg, net_rtg,
                 home_ml, away_ml, home_spread, away_spread,
                 fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team, season_year,
                game.get("date"), game.get("location"), game.get("opponent"), game.get("result"),
                game.get("team_score"), game.get("opponent_score"),
                game.get("pace"), game.get("ftr"), game.get("efg_pct"), game.get("tov_pct"), game.get("orb_pct"),
                game.get("ortg"), game.get("drtg"), game.get("net_rtg"),
                game.get("home_ml"), game.get("away_ml"), game.get("home_spread"), game.get("away_spread"),
                now,
            ))
            saved_count += 1
        except sqlite3.Error:
            pass
    conn.commit()
    conn.close()
    _upsert_team(team)
    return saved_count


def _upsert_team(team: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    name = NBA_TEAMS.get(team, team)
    cursor.execute("""
        INSERT INTO teams (code, name) VALUES (?, ?)
        ON CONFLICT(code) DO UPDATE SET name = excluded.name
    """, (team, name))
    conn.commit()
    conn.close()


def get_games(team: Optional[str] = None, season_year: Optional[int] = None,
              limit: Optional[int] = None, db_path: Optional[str] = None) -> list[dict]:
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
    if limit and limit > 0:
        query += " LIMIT ?"
        params.append(int(limit))
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_matchups(team: Optional[str] = None, opponent: Optional[str] = None,
                 season_year: Optional[int] = None, limit: Optional[int] = None,
                 db_path: Optional[str] = None) -> list[dict]:
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
    if limit and limit > 0:
        query += " LIMIT ?"
        params.append(int(limit))
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_matchup(team: str, opponent: str, n: int = 10, db_path: Optional[str] = None) -> list[dict]:
    return get_matchups(team=team, opponent=opponent, limit=n, db_path=db_path)


def get_team_stats(team: str, season_year: int, db_path: Optional[str] = None) -> Optional[dict]:
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


def save_team_season_stats(stats: list[dict], team: str, season_year: int, db_path: Optional[str] = None) -> int:
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


# ---- Injuries ----

def save_injuries(injuries: list[dict], team: str, db_path: Optional[str] = None) -> int:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    saved = 0
    for inj in injuries:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO injuries
                (team, player_name, injury, status, date_reported, game_date, season_year,
                 will_play, player_min, player_pts, player_reb, player_ast, player_impact_score, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team,
                inj.get("player_name"),
                inj.get("injury"),
                inj.get("status"),
                inj.get("date_reported"),
                inj.get("game_date"),
                inj.get("season_year"),
                inj.get("will_play", 1),
                inj.get("player_min", 0),
                inj.get("player_pts", 0),
                inj.get("player_reb", 0),
                inj.get("player_ast", 0),
                inj.get("player_impact_score", 0),
                now,
            ))
            saved += 1
        except sqlite3.Error:
            pass
    conn.commit()
    conn.close()
    return saved


def get_injuries(team: Optional[str] = None, db_path: Optional[str] = None) -> list[dict]:
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


def fetch_player_stats() -> dict:
    """
    Fetch current season player stats from nba_api.
    Returns dict mapping (cleaned) player name -> {min, pts, reb, ast}
    """
    try:
        from nba_api.stats.endpoints import LeagueDashPlayerStats
        stats = LeagueDashPlayerStats(season='2025-26', per_mode_detailed='PerGame')
        dfs = stats.get_data_frames()
        if not dfs:
            return {}
        df = dfs[0]
        player_stats = {}
        for _, row in df.iterrows():
            name = str(row.get('PLAYER_NAME', '')).strip()
            if not name:
                continue
            min_val = float(row.get('MIN', 0) or 0)
            pts_val = float(row.get('PTS', 0) or 0)
            reb_val = float(row.get('REB', 0) or 0)
            ast_val = float(row.get('AST', 0) or 0)
            # Only include players with meaningful minutes (5+ per game)
            if min_val >= 5:
                player_stats[name.lower()] = {
                    'min': min_val,
                    'pts': pts_val,
                    'reb': reb_val,
                    'ast': ast_val,
                }
        logger.info(f"Fetched stats for {len(player_stats)} players")
        return player_stats
    except Exception as e:
        logger.error(f"Error fetching player stats: {e}")
        return {}


def _compute_impact_score(stats: dict) -> float:
    """
    Compute a player's impact score based on per-game production.
    Formula: min * (pts + reb*1.2 + ast*1.5) / 36
    Scaled to roughly -0.5 to -5 range (out for a game vs bench player)
    """
    if not stats:
        return 0.0
    impact = stats['min'] * (stats['pts'] + stats['reb'] * 1.2 + stats['ast'] * 1.5) / 36.0
    return round(impact, 2)


def fetch_injuries_from_espn() -> list[dict]:
    """
    Scrape current NBA injury reports from ESPN.
    Fetches player stats and computes per-game impact scores.
    Returns list of injury dicts with full stats and impact data.
    """
    import requests
    from bs4 import BeautifulSoup

    injuries = []
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}

    # Fetch player stats for impact calculation
    player_stats = fetch_player_stats()

    try:
        url = 'https://www.espn.com/nba/injuries'
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            logger.warning(f"ESPN injuries returned status {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table', {'class': 'Table'})

        today = datetime.now().strftime('%Y-%m-%d')

        for table in tables:
            # Find team name by walking up to the ResponsiveTable div (Level 3 in the DOM)
            team_name = None
            elem = table
            for _ in range(5):
                if not elem or not elem.parent:
                    break
                elem = elem.parent
                if elem and elem.name == 'div' and 'ResponsiveTable' in elem.get('class', []):
                    # The team name is the first part of the text before column headers
                    text = elem.text.strip()
                    for team_name_key in ESPN_TEAM_MAPPING.keys():
                        if text.startswith(team_name_key):
                            team_name = team_name_key
                            break
                    break

            team_code = ESPN_TEAM_MAPPING.get(team_name, team_name) if team_name else None

            tbody = table.find('tbody')
            if not tbody:
                continue

            rows = tbody.find_all('tr')
            for row in rows:
                cells = [td.text.strip() for td in row.find_all(['th', 'td'])]
                if len(cells) < 4:
                    continue

                player_name = cells[0]
                status = cells[3] if len(cells) > 3 else ''
                comment = cells[4] if len(cells) > 4 else ''

                # Map status to our enum
                status_lower = status.lower()
                if 'out' in status_lower:
                    mapped_status = 'out'
                elif 'doubtful' in status_lower:
                    mapped_status = 'doubtful'
                elif 'questionable' in status_lower:
                    mapped_status = 'questionable'
                elif 'probable' in status_lower:
                    mapped_status = 'probable'
                elif 'day-to-day' in status_lower or 'day' in status_lower:
                    mapped_status = 'questionable'
                else:
                    mapped_status = status_lower

                # Try to match player name to stats
                name_key = player_name.lower()
                stats = player_stats.get(name_key, {})

                # Try partial match if exact doesn't work
                if not stats:
                    for stat_name, stat_data in player_stats.items():
                        # Try matching last name
                        stat_last = stat_name.split()[-1]
                        player_last = name_key.split()[-1]
                        if len(stat_last) > 3 and stat_last == player_last:
                            stats = stat_data
                            break

                impact_score = _compute_impact_score(stats)

                injuries.append({
                    'player_name': player_name,
                    'team': team_code or team_name or 'UNK',
                    'injury': comment.split(':')[1].strip() if ':' in comment else comment[:100],
                    'status': mapped_status,
                    'date_reported': today,
                    'game_date': '',
                    'will_play': 1,  # User will update this manually
                    'player_min': stats.get('min', 0),
                    'player_pts': stats.get('pts', 0),
                    'player_reb': stats.get('reb', 0),
                    'player_ast': stats.get('ast', 0),
                    'player_impact_score': impact_score,
                })

        logger.info(f"Fetched {len(injuries)} injury records from ESPN")
    except Exception as e:
        logger.error(f"Error fetching injuries from ESPN: {e}")

    return injuries


# ---- Teams ----

def get_teams(db_path: Optional[str] = None) -> list[dict]:
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


# ---- Power Rankings ----

def save_power_ranking(season_year: int, entries: list[dict]) -> int:
    """
    Save a complete power ranking set for a season.
    entries: [{team, rank, power_rating, net_rtg, ortg_override, drtg_override, notes}, ...]
    Returns ranking_id.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    # Create ranking
    cursor.execute("""
        INSERT INTO power_rankings (season_year, created_at, updated_at)
        VALUES (?, ?, ?)
    """, (season_year, now, now))
    ranking_id = cursor.lastrowid

    for entry in entries:
        cursor.execute("""
            INSERT INTO power_ranking_entries
            (ranking_id, team, rank, power_rating, net_rtg, ortg_override, drtg_override, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ranking_id,
            entry.get("team", "").upper(),
            entry.get("rank", 0),
            entry.get("power_rating", 0.0),
            entry.get("net_rtg", 0.0),
            entry.get("ortg_override"),
            entry.get("drtg_override"),
            entry.get("notes", ""),
        ))

    conn.commit()
    conn.close()
    return ranking_id


def get_latest_power_ranking(season_year: int, db_path: Optional[str] = None) -> Optional[dict]:
    """Get the most recent power ranking for a season."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, season_year, created_at, updated_at
        FROM power_rankings
        WHERE season_year = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (season_year,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    ranking = dict(row)
    cursor.execute("""
        SELECT team, rank, power_rating, net_rtg, ortg_override, drtg_override, notes
        FROM power_ranking_entries
        WHERE ranking_id = ?
        ORDER BY rank ASC
    """, (ranking["id"],))
    entries = [dict(r) for r in cursor.fetchall()]
    conn.close()

    ranking["entries"] = entries
    return ranking


def get_power_ranking_for_team(season_year: int, team: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get a specific team's power ranking entry for a season."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pr.id, pr.season_year, pr.created_at,
               pre.team, pre.rank, pre.power_rating, pre.net_rtg,
               pre.ortg_override, pre.drtg_override, pre.notes
        FROM power_rankings pr
        JOIN power_ranking_entries pre ON pre.ranking_id = pr.id
        WHERE pr.season_year = ? AND pre.team = ?
        ORDER BY pr.created_at DESC
        LIMIT 1
    """, (season_year, team.upper()))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_power_rankings(db_path: Optional[str] = None) -> list[dict]:
    """Get all power rankings summaries."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, season_year, created_at, updated_at
        FROM power_rankings
        ORDER BY season_year DESC, created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Tonight's Games ----

NBA_TEAM_ID_TO_CODE = {
    1610612738: "BOS", 1610612751: "BKN", 1610612755: "PHI", 1610612761: "TOR",
    1610612737: "ATL", 1610612750: "MIN", 1610612746: "CLE", 1610612752: "NYK",
    1610612765: "DET", 1610612749: "MIL", 1610612764: "WAS", 1610612753: "ORL",
    1610612744: "MIA", 1610612747: "LAL", 1610612748: "LAC", 1610612759: "SAS",
    1610612758: "PHO", 1610612760: "OKC", 1610612763: "UTA", 1610612742: "DAL",
    1610612743: "DEN", 1610612754: "IND", 1610612766: "CHA", 1610612756: "SAC",
    1610612741: "CHI", 1610612740: "NOH", 1610612762: "GSW", 1610612757: "POR",
    1610612745: "HOU", 1610612767: "MEM",
}


def get_tonight_games() -> list[dict]:
    try:
        from nba_api.stats.endpoints import ScoreboardV2
        from datetime import date
        today = date.today().strftime("%m/%d/%Y")
        s = ScoreboardV2(game_date=today, proxy=None, timeout=30)

        games = []
        for ds in s.data_sets:
            if ds.key == "GameHeader":
                gh_df = ds.get_data_frame()
                break
        else:
            gh_df = s.game_header.get_data_frame()

        for _, row in gh_df.iterrows():
            game_id = row["GAME_ID"]
            game_date = str(row["GAME_DATE_EST"])[:10] if row["GAME_DATE_EST"] else ""
            home_id = row["HOME_TEAM_ID"]
            away_id = row["VISITOR_TEAM_ID"]

            games.append({
                "game_id": game_id,
                "game_date": game_date,
                "home_team": NBA_TEAM_ID_TO_CODE.get(home_id, f"UNK_{home_id}"),
                "away_team": NBA_TEAM_ID_TO_CODE.get(away_id, f"UNK_{away_id}"),
                "game_time": row.get("GAME_STATUS_TEXT", "7:00 pm ET"),
                "status_id": row.get("GAME_STATUS_ID", 1),
            })
        return games
    except Exception as e:
        logger.error(f"Error fetching tonight's games: {e}")
        return []


# ---- All NBA team codes ----
ALL_TEAM_CODES = list(NBA_TEAMS.keys())


# Initialize database on module import
def _ensure_db():
    init_db()
_ensure_db()