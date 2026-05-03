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
from datetime import datetime

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

def load_data():
    print(f"[{timestamp()}] Downloading dataset...")
    path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
    df = pd.read_csv(f"{path}/TeamStatistics.csv")
    print(f"[{timestamp()}] Loaded {len(df):,} rows, date range: {df['gameDate'].min()} → {df['gameDate'].max()}")
    return df

def prepare_features(df):
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