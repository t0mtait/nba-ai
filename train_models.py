#!/usr/bin/env python3
"""
train_models.py - Nightly training pipeline for NBA Win Predictor
Downloads latest data, trains models, saves artifacts and pushes to GitHub.
Run via: python3 train_models.py [--commit-message "..."]
"""

import argparse
import joblib
import json
import os
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import kagglehub
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ────────────────────────────────────────────────────────────────────
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(REPO_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_HANDLE = "eoinamoore/historical-nba-data-and-player-box-scores"
DATASET_FILE = "TeamStatistics.csv"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ymd}"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"

NBA_TEAM_IDS = {
    "Hawks": 1610612737,
    "Celtics": 1610612738,
    "Cavaliers": 1610612739,
    "Pelicans": 1610612740,
    "Bulls": 1610612741,
    "Mavericks": 1610612742,
    "Nuggets": 1610612743,
    "Warriors": 1610612744,
    "Rockets": 1610612745,
    "Clippers": 1610612746,
    "Lakers": 1610612747,
    "Heat": 1610612748,
    "Bucks": 1610612749,
    "Timberwolves": 1610612750,
    "Nets": 1610612751,
    "Knicks": 1610612752,
    "Magic": 1610612753,
    "Pacers": 1610612754,
    "76ers": 1610612755,
    "Suns": 1610612756,
    "Trail Blazers": 1610612757,
    "Kings": 1610612758,
    "Spurs": 1610612759,
    "Thunder": 1610612760,
    "Raptors": 1610612761,
    "Jazz": 1610612762,
    "Grizzlies": 1610612763,
    "Wizards": 1610612764,
    "Pistons": 1610612765,
    "Hornets": 1610612766,
}

# ── features ─────────────────────────────────────────────────────────────────
FEAT_COLS = [
    "home", "assists", "reboundsTotal", "blocks", "steals",
    "turnovers", "foulsPersonal", "q1Points", "q2Points",
    "fieldGoalsAttempted", "threePointersAttempted", "freeThrowsAttempted",
]

DISPLAY_COLS = [
    "assists", "reboundsTotal", "blocks", "steals", "turnovers",
    "foulsPersonal", "q1Points", "q2Points",
    "fieldGoalsAttempted", "threePointersAttempted", "freeThrowsAttempted",
]
DISPLAY_LABELS = ["A", "R", "B", "S", "TO", "F", "Q1", "Q2", "FGA", "3FA", "FTA"]

# ── training ─────────────────────────────────────────────────────────────────

def _parse_made_attempted(val: str) -> float:
    parts = str(val).split("-")
    return float(parts[1]) if len(parts) == 2 else float(parts[0])


def _append_recent_espn_games(df: pd.DataFrame, date_col: str = "gameDate") -> pd.DataFrame:
    try:
        team_map = dict(NBA_TEAM_IDS)
        if "teamName" in df.columns and "teamId" in df.columns:
            from_df = (
                df[["teamName", "teamId"]]
                .dropna()
                .drop_duplicates(subset=["teamName"], keep="first")
                .set_index("teamName")["teamId"]
                .to_dict()
            )
            team_map.update(from_df)

        home_df = df[df["home"] == 1]
        existing_keys = set(
            zip(
                home_df[date_col].astype(str).str[:10],
                home_df["teamName"].astype(str),
                home_df["opponentTeamName"].astype(str),
            )
        )

        max_date_str = home_df[date_col].dropna().astype(str).str[:10].max()
        ny_tz = ZoneInfo("America/New_York")
        today_est = datetime.now(ny_tz).date()
        start_date = (
            datetime.strptime(max_date_str, "%Y-%m-%d").date()
            if isinstance(max_date_str, str) and len(max_date_str) == 10
            else today_est - timedelta(days=7)
        )
        if (today_est - start_date).days > 14:
            start_date = today_est - timedelta(days=14)

        new_rows = []
        cur = start_date
        headers = {"User-Agent": "Mozilla/5.0"}
        while cur <= today_est:
            ymd = cur.strftime("%Y%m%d")
            cur += timedelta(days=1)
            try:
                req = urllib.request.Request(ESPN_SCOREBOARD_URL.format(ymd=ymd), headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    events = json.loads(resp.read().decode()).get("events", [])
            except Exception as exc:
                print(f"[{timestamp()}] Warning: ESPN scoreboard fetch failed for {ymd}: {exc}")
                continue

            for ev in events:
                try:
                    if not ev.get("status", {}).get("type", {}).get("completed"):
                        continue
                    comp = (ev.get("competitions") or [{}])[0]
                    competitors = comp.get("competitors", [])
                    home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                    away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                    if not home_c or not away_c:
                        continue

                    tname = home_c.get("team", {}).get("name")
                    oname = away_c.get("team", {}).get("name")
                    if not tname or not oname or tname not in team_map:
                        continue

                    dt_utc = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                    dt_est = dt_utc.astimezone(ny_tz)
                    gdate_full = dt_est.strftime("%Y-%m-%d %H:%M:%S")
                    gdate_short = gdate_full[:10]

                    if (gdate_short, tname, oname) in existing_keys:
                        continue

                    s_req = urllib.request.Request(ESPN_SUMMARY_URL.format(event_id=ev["id"]), headers=headers)
                    with urllib.request.urlopen(s_req, timeout=10) as s_resp:
                        s_data = json.loads(s_resp.read().decode())

                    b_teams = s_data.get("boxscore", {}).get("teams", [])
                    b_home = next((t for t in b_teams if t.get("team", {}).get("name") == tname), None)
                    if not b_home:
                        continue

                    stats = {s["name"]: s["displayValue"] for s in b_home.get("statistics", []) if "name" in s and "displayValue" in s}
                    linescores = home_c.get("linescores", [])
                    if len(linescores) < 2:
                        s_comp = (s_data.get("header", {}).get("competitions") or [{}])[0]
                        s_home = next((c for c in s_comp.get("competitors", []) if c.get("homeAway") == "home"), {})
                        linescores = s_home.get("linescores", [])
                    if len(linescores) < 2:
                        continue

                    q1 = float(linescores[0].get("displayValue", linescores[0].get("value", 0)))
                    q2 = float(linescores[1].get("displayValue", linescores[1].get("value", 0)))
                    fga = _parse_made_attempted(stats["fieldGoalsMade-fieldGoalsAttempted"])
                    tpa = _parse_made_attempted(stats["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
                    fta = _parse_made_attempted(stats["freeThrowsMade-freeThrowsAttempted"])

                    new_rows.append({
                        date_col: gdate_full,
                        "gameDate": gdate_full,
                        "gameDateTimeEst": gdate_full,
                        "teamId": team_map[tname],
                        "teamName": tname,
                        "opponentTeamName": oname,
                        "home": 1,
                        "win": 1.0 if home_c.get("winner") else 0.0,
                        "assists": float(stats["assists"]),
                        "reboundsTotal": float(stats["totalRebounds"]),
                        "blocks": float(stats["blocks"]),
                        "steals": float(stats["steals"]),
                        "turnovers": float(stats.get("turnovers", stats.get("totalTurnovers", 0))),
                        "foulsPersonal": float(stats.get("fouls", stats.get("foulsPersonal", 0))),
                        "q1Points": q1,
                        "q2Points": q2,
                        "fieldGoalsAttempted": fga,
                        "threePointersAttempted": tpa,
                        "freeThrowsAttempted": fta,
                    })
                    existing_keys.add((gdate_short, tname, oname))
                except Exception as exc:
                    print(f"[{timestamp()}] Warning: Failed to parse ESPN event {ev.get('id')}: {exc}")

        if new_rows:
            new_rows.sort(key=lambda r: r[date_col], reverse=True)
            df = pd.concat([pd.DataFrame(new_rows), df], ignore_index=True)
            print(f"[{timestamp()}] Supplemented with {len(new_rows)} recent ESPN NBA game(s)")
    except Exception as exc:
        print(f"[{timestamp()}] Warning: Failed to append recent ESPN games: {exc}")
    return df


def load_data():
    print(f"[{timestamp()}] Downloading dataset...")
    try:
        path = kagglehub.dataset_download(DATASET_HANDLE, path=DATASET_FILE, force_download=True)
    except Exception:
        path = kagglehub.dataset_download(DATASET_HANDLE, path=DATASET_FILE, force_download=False)
    csv_path = os.path.join(path, DATASET_FILE) if os.path.isdir(path) else path
    df = pd.read_csv(csv_path, low_memory=False)
    df = _append_recent_espn_games(df)
    print(f"[{timestamp()}] Loaded {len(df):,} rows, date range: {df['gameDate'].min()} → {df['gameDate'].max()}")
    return df

def prepare_features(df):
    df = df[df["home"] == 1]
    all_game_dates = df["gameDate"].astype(str)
    all_opponent   = df["opponentTeamName"]
    all_team       = df["teamName"]

    df = df.dropna(subset=FEAT_COLS + ["win"])
    print(f"[{timestamp()}] After dropna: {len(df):,} games")

    X = df[FEAT_COLS]
    y = df["win"]
    return X, y, all_game_dates, all_opponent, all_team

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, model.predict(X_test))
    return model, X_train, X_test, y_train, y_test, train_acc, test_acc

def extract_predictions(model, X, y, game_dates, opponents, teams):
    preds   = model.predict(X)
    correct = (preds == y.values)
    df = X.copy()
    df["actual"]  = y.values
    df["pred"]    = preds
    df["correct"] = correct
    df["gameDate"]       = game_dates.loc[X.index].str[:10].values
    df["opponentTeamName"] = opponents.loc[X.index].values
    df["teamName"]       = teams.loc[X.index].values
    df["home"]           = X["home"].values
    return df.sort_values("gameDate", ascending=False)

# ── save artifacts ────────────────────────────────────────────────────────────

def save_artifacts(model, predictions_df, train_acc, test_acc, total_games, train_games, test_games):
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    pred_path  = os.path.join(MODEL_DIR, "predictions.csv")
    meta_path  = os.path.join(MODEL_DIR, "metadata.json")

    joblib.dump(model, model_path)
    predictions_df.to_csv(pred_path, index=False)

    meta = {
        "trained_at":        timestamp(),
        "train_accuracy":    round(train_acc, 4),
        "test_accuracy":     round(test_acc, 4),
        "total_games":       total_games,
        "train_games":       train_games,
        "test_games":        test_games,
        "feature_cols":      FEAT_COLS,
        "display_cols":      DISPLAY_COLS,
        "display_labels":    DISPLAY_LABELS,
        "model_class":       "LogisticRegression",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{timestamp()}] Saved: {model_path}")
    print(f"[{timestamp()}] Saved: {pred_path}")
    print(f"[{timestamp()}] Saved: {meta_path}")

    # Commit if on main and pushed
    if os.environ.get("CI") != "true":
        commit_and_push(model_path, pred_path, meta_path)

    return meta

def commit_and_push(*paths):
    try:
        subprocess.run(["git", "add"] + list(paths), check=True, cwd=REPO_DIR)
        msg = f"nightly: retrain {timestamp()}"
        result = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True, cwd=REPO_DIR)
        if result.stdout.strip():
            subprocess.run(["git", "commit", "-m", msg], check=True, cwd=REPO_DIR)
            print(f"[{timestamp()}] Committed: {msg}")
            subprocess.run(["git", "push", "nba-ai", "main"], check=True, cwd=REPO_DIR)
            print(f"[{timestamp()}] Pushed to nba-ai/main")
    except subprocess.CalledProcessError as e:
        print(f"[{timestamp()}] Git error: {e}")

# ── main ─────────────────────────────────────────────────────────────────────

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-message")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"NBA AI Nightly Training — {timestamp()}")
    print(f"{'='*60}\n")

    try:
        df = load_data()
        X, y, game_dates, opponents, teams = prepare_features(df)
        model, X_train, X_test, y_train, y_test, train_acc, test_acc = train(X, y)

        all_X = pd.concat([X_train, X_test])
        all_y = pd.concat([y_train, y_test])
        preds_df = extract_predictions(
            model, all_X, all_y,
            game_dates, opponents, teams
        )

        total_games = len(all_X)
        train_games = len(X_train)
        test_games  = len(X_test)

        print(f"\n[{timestamp()}] Accuracy — Train: {train_acc:.4f}  Test: {test_acc:.4f}")
        meta = save_artifacts(
            model, preds_df, train_acc, test_acc,
            total_games, train_games, test_games
        )

        print(f"\n[{timestamp()}] ✓ Done! Test accuracy: {test_acc*100:.1f}%")
        return 0

    except Exception as e:
        print(f"\n[{timestamp()}] ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())