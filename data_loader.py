"""Load NBA game data from SQLite database."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

try:
    import db as _db_module
except ImportError:
    _db_module = None


# Feature columns for prediction models
FEATURE_COLUMNS = [
    "home_net_rtg_diff",
    "home_ortg_diff",
    "home_drtg_diff",
    "home_pace_diff",
    "home_efg_diff",
    "home_tov_diff",
    "home_ftr_diff",
    "h2h_win_pct_diff",
    "h2h_avg_margin",
    "rest_diff",
    "injury_impact",
    "home_court_adv",
]


def load_games_from_db(
    team_code: Optional[str] = None,
    season_year: Optional[int] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load games from the SQLite database.

    Args:
        team_code: Optional team code filter
        season_year: Optional season year filter
        limit: Optional limit on results

    Returns:
        DataFrame with game data
    """
    if _db_module is None:
        raise ImportError("db module not available")

    games = _db_module.get_games(team=team_code, season_year=season_year, limit=limit)

    if not games:
        return pd.DataFrame()

    df = pd.DataFrame(games)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["result_w"] = (df["result"] == "W").astype(int)

    numeric_cols = [
        "pace", "ftr", "efg_pct", "tov_pct", "orb_pct",
        "ortg", "drtg", "net_rtg", "ts_pct", "threepar", "drb_pct",
        "home_net_rtg_diff", "home_ortg_diff", "home_drtg_diff",
        "home_pace_diff", "home_efg_diff", "home_tov_diff", "home_ftr_diff",
        "h2h_win_pct_diff", "h2h_avg_margin", "rest_diff", "injury_impact", "home_court_adv",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_matchup_history(
    team: str,
    opponent: str,
    limit: int = 20,
) -> pd.DataFrame:
    """Get head-to-head history between two teams."""
    if _db_module is None:
        raise ImportError("db module not available")
    games = _db_module.get_recent_matchup(team, opponent, n=limit)
    return pd.DataFrame(games) if games else pd.DataFrame()


def get_team_season_stats(team: str, season_year: int) -> Optional[dict]:
    """Get season-level stats for a team."""
    if _db_module is None:
        return None
    return _db_module.get_team_stats(team, season_year)


def get_injuries(team: Optional[str] = None) -> list[dict]:
    """Get injury reports, optionally filtered by team."""
    if _db_module is None:
        return []
    return _db_module.get_injuries(team=team)


def get_teams() -> list[dict]:
    """Get list of all teams in the database."""
    if _db_module is None:
        return []
    return _db_module.get_teams()