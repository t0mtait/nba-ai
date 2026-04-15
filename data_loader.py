"""Load NBA game data from local /data CSV files or SQLite database.

This module provides functions to load Celtics (or other team) game data
from either legacy CSV files or the new SQLite database.

DEPRECATION NOTICE:
    The CSV-based loading functions are deprecated. Please use
    load_games_from_db() instead for new code.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import pandas as pd

# Import DB module
try:
    import db as _db_module
except ImportError:
    _db_module = None

FEATURE_COLUMNS = [
    "pace",
    "ftr",
    "efg_pct",
    "tov_pct",
    "orb_pct",
]

_COLUMN_RENAME = {
    "ORtg": "ortg",
    "DRtg": "drtg",
    "Pace": "pace",
    "FTr": "ftr",
    "3PAr": "threepar",
    "TS%": "ts_pct",
    "eFG%": "efg_pct",
    "TOV%": "tov_pct",
    "ORB%": "orb_pct",
}


def load_games_from_db(
    team_code: Optional[str] = None,
    season_year: Optional[int] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load games from the SQLite database.

    Args:
        team_code: Optional team code filter (e.g., 'BOS', 'LAL'). If None, loads all teams.
        season_year: Optional season year filter
        limit: Optional limit on number of results

    Returns:
        DataFrame with game data including columns:
        - team, season_year, date, location, opponent, result
        - pace, ftr, efg_pct, tov_pct, orb_pct (features)
        - celtics_win: boolean (derived from result)
    """
    if _db_module is None:
        raise ImportError("db module not available. Install required dependencies.")

    games = _db_module.get_games(team=team_code, season_year=season_year, limit=limit)

    if not games:
        return pd.DataFrame()

    df = pd.DataFrame(games)

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Ensure date is properly formatted
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Create celtics_win column (for backward compatibility)
    # This assumes we're looking at Celtics games - adjust for other teams
    df["celtics_win"] = df["result"] == "W"

    # Ensure numeric columns are properly typed
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_games_for_prediction(
    team_code: str = "BOS",
    season_year: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get home and away games split for prediction model training.

    Args:
        team_code: Team code to filter by
        season_year: Optional season year filter

    Returns:
        Tuple of (home_games_df, away_games_df)
    """
    df = load_games_from_db(team_code=team_code, season_year=season_year)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    home_games = df[df["location"] == "home"].copy()
    away_games = df[df["location"] == "away"].copy()

    return home_games, away_games


# ----- DEPRECATED FUNCTIONS BELOW -----
# These are kept for backward compatibility but may be removed in future versions


def _load_single_csv(file_path: str) -> pd.DataFrame:
    """Load a single CSV file (internal use)."""
    df = pd.read_csv(file_path, header=1)
    cols = list(df.columns)
    rename_map = {}

    if len(cols) > 2:
        rename_map[cols[2]] = "Date"
    if len(cols) > 3:
        rename_map[cols[3]] = "Location"
    if len(cols) > 4:
        rename_map[cols[4]] = "Opponent"
    if len(cols) > 5:
        rename_map[cols[5]] = "Result"
    if len(cols) > 6:
        rename_map[cols[6]] = "Team_PTS"
    if len(cols) > 7:
        rename_map[cols[7]] = "Opponent_PTS"

    df = df.rename(columns=rename_map)
    df = df.rename(columns=_COLUMN_RENAME)

    return df


def load_celtics_games(data_dir: str = "data") -> pd.DataFrame:
    """
    Load Celtics game data from CSV files.

    DEPRECATED: Use load_games_from_db() instead.

    Args:
        data_dir: Path to directory containing CSV files

    Returns:
        DataFrame with game data
    """
    warnings.warn(
        "load_celtics_games() is deprecated. Use load_games_from_db() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    import glob

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = [_load_single_csv(path) for path in csv_files]
    data = pd.concat(frames, ignore_index=True)

    data = data[data["Date"].notna()].copy()

    data["location"] = (
        data["Location"]
        .astype(str)
        .str.strip()
        .map({"@": "away"})
        .fillna("home")
    )
    data.loc[data["location"] != "away", "location"] = "home"

    data["celtics_win"] = (
        data["Result"].astype(str).str.upper().str.startswith("W")
    )

    data["date"] = pd.to_datetime(data["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["opponent"] = data["Opponent"].astype(str)

    # Normalize percentage columns (stored as 0-100 scale in CSV, should be 0-1 for model)
    percent_cols = ["tov_pct", "orb_pct"]
    for col in percent_cols:
        if col in data.columns:
            data[col] = (pd.to_numeric(data[col], errors="coerce") / 100.0).round(3)

    # Ensure other numeric columns are properly typed
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            raise KeyError(f"Missing required column '{col}' in {data_dir} CSVs")
        if col not in percent_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def get_teams_from_db() -> list[dict]:
    """Get list of all teams in the database."""
    if _db_module is None:
        return []
    return _db_module.get_teams()
