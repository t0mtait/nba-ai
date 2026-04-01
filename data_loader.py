"""Load Celtics game data from local /data CSV files."""
from __future__ import annotations

import glob
import os
from typing import List

import pandas as pd

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


def _load_single_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=1)
    cols: List[str] = list(df.columns)
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
            data[col] = pd.to_numeric(data[col], errors="coerce") / 100.0

    # Ensure other numeric columns are properly typed
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            raise KeyError(f"Missing required column '{col}' in {data_dir} CSVs")
        if col not in percent_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data
