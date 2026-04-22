"""Train NBA moneyline and spread prediction models.

Predicts moneyline odds and point spreads for any NBA matchup using:
- Recent head-to-head matchup history
- Team season-level statistics
- Home/away performance splits
- Injury report adjustments
"""
from __future__ import annotations

import os
import pickle
import json
import argparse
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

from data_loader import load_games_from_db


# ----- Feature engineering -----

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


def compute_h2h_features(team: str, opponent: str, cutoff_date: str, db_path: Optional[str] = None) -> dict:
    """
    Compute head-to-head features between two teams.
    Uses all games before cutoff_date.
    """
    import db as _db

    games = _db.get_matchups(team=team, opponent=opponent, limit=50, db_path=db_path)

    # Filter to games before cutoff
    games = [g for g in games if g["date"] < cutoff_date]

    if not games:
        return {
            "h2h_win_pct": 0.5,
            "h2h_avg_margin": 0.0,
            "h2h_games": 0,
        }

    wins = sum(1 for g in games if g["result"] == "W")
    total = len(games)

    # Margin from team perspective
    margins = []
    for g in games:
        if g["team_score"] and g["opponent_score"]:
            margin = g["team_score"] - g["opponent_score"]
            # Flip sign since games are from team perspective
            if g["location"] == "away":
                margin = -margin
            margins.append(margin)

    avg_margin = np.mean(margins) if margins else 0.0

    return {
        "h2h_win_pct": wins / total if total > 0 else 0.5,
        "h2h_avg_margin": avg_margin,
        "h2h_games": total,
    }


def compute_injury_impact(team: str, game_date: str, db_path: Optional[str] = None) -> float:
    """
    Estimate injury impact for a team on a given date.
    Returns a float: positive = team is undermanned (disadvantage).
    """
    import db as _db

    injuries = _db.get_injuries(team=team, db_path=db_path)

    impact = 0.0
    for inj in injuries:
        # Only count injuries reported before game date
        if inj.get("date_reported", "") >= game_date:
            continue
        status = (inj.get("status") or "").lower()
        # Questionable / doubtful = moderate impact; out = full impact
        if status in ("out", "doubtful"):
            impact -= 2.0
        elif status in ("questionable", "probable"):
            impact -= 0.5

    return impact


def compute_rest_diff(team: str, opponent: str, game_date: str, db_path: Optional[str] = None) -> float:
    """
    Compute rest advantage (days since last game).
    Positive = this team is more rested.
    """
    import db as _db

    def last_game_date(team_code: str, before_date: str) -> Optional[str]:
        games = _db.get_games(team=team_code, limit=10, db_path=db_path)
        before = [g["date"] for g in games if g["date"] < before_date]
        return max(before) if before else None

    team_last = last_game_date(team, game_date)
    opp_last = last_game_date(opponent, game_date)

    if team_last is None or opp_last is None:
        return 0.0

    try:
        team_days = (pd.to_datetime(game_date) - pd.to_datetime(team_last)).days
        opp_days = (pd.to_datetime(game_date) - pd.to_datetime(opp_last)).days
        return team_days - opp_days
    except Exception:
        return 0.0


def build_features_for_game(
    team: str,
    opponent: str,
    game_date: str,
    location: str,
    db_path: Optional[str] = None,
) -> dict:
    """
    Build a feature dict for a single game prediction.
    All features are from `team`'s perspective.
    """
    import db as _db

    # Get current season year
    try:
        season_year = int(game_date[:4])
        if int(game_date[5:7]) <= 6:
            season_year -= 1  # Jan-Jun is part of previous season
    except Exception:
        season_year = 2024

    team_stats = _db.get_team_stats(team, season_year, db_path) or {}
    opp_stats = _db.get_team_stats(opponent, season_year, db_path) or {}

    # Net ratings
    team_net = team_stats.get("net_rtg", 0) or 0
    opp_net = opp_stats.get("net_rtg", 0) or 0
    home_net_rtg_diff = team_net - opp_net if location == "home" else opp_net - team_net

    team_ortg = team_stats.get("ortg", 0) or 0
    opp_ortg = opp_stats.get("ortg", 0) or 0
    home_ortg_diff = team_ortg - opp_ortg if location == "home" else opp_ortg - team_ortg

    team_drtg = team_stats.get("drtg", 0) or 0
    opp_drtg = opp_stats.get("drtg", 0) or 0
    home_drtg_diff = team_drtg - opp_drtg if location == "home" else opp_drtg - team_drtg

    team_pace = team_stats.get("pace", 0) or 0
    opp_pace = opp_stats.get("pace", 0) or 0
    home_pace_diff = team_pace - opp_pace if location == "home" else opp_pace - team_pace

    team_efg = team_stats.get("efg_pct", 0) or 0
    opp_efg = opp_stats.get("efg_pct", 0) or 0
    home_efg_diff = team_efg - opp_efg if location == "home" else opp_efg - team_efg

    team_tov = team_stats.get("tov_pct", 0) or 0
    opp_tov = opp_stats.get("tov_pct", 0) or 0
    home_tov_diff = team_tov - opp_tov if location == "home" else opp_tov - team_tov

    team_ftr = team_stats.get("ftr", 0) or 0
    opp_ftr = opp_stats.get("ftr", 0) or 0
    home_ftr_diff = team_ftr - opp_ftr if location == "home" else opp_ftr - team_ftr

    # H2H features
    h2h = compute_h2h_features(team, opponent, game_date, db_path)
    # If team is home, h2h_win_pct is already from team perspective
    h2h_win_pct_diff = h2h["h2h_win_pct"] - 0.5  # Center around 0.5
    h2h_avg_margin = h2h["h2h_avg_margin"] if location == "home" else -h2h["h2h_avg_margin"]

    # Rest advantage
    rest_diff = compute_rest_diff(team, opponent, game_date, db_path)

    # Injury impact
    team_inj = compute_injury_impact(team, game_date, db_path)
    opp_inj = compute_injury_impact(opponent, game_date, db_path)
    injury_impact = team_inj - opp_inj  # Positive = advantage

    # Home court advantage (historical home win%)
    home_court_adv = 0.58 if location == "home" else -0.58

    return {
        "home_net_rtg_diff": home_net_rtg_diff,
        "home_ortg_diff": home_ortg_diff,
        "home_drtg_diff": home_drtg_diff,
        "home_pace_diff": home_pace_diff,
        "home_efg_diff": home_efg_diff,
        "home_tov_diff": home_tov_diff,
        "home_ftr_diff": home_ftr_diff,
        "h2h_win_pct_diff": h2h_win_pct_diff,
        "h2h_avg_margin": h2h_avg_margin,
        "rest_diff": rest_diff,
        "injury_impact": injury_impact,
        "home_court_adv": home_court_adv,
    }


def build_dataset(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build a full dataset of games with features and targets.
    Each row = one team's perspective on a game.
    """
    import db as _db

    games = _db.get_games(limit=None, db_path=db_path)
    if not games:
        return pd.DataFrame()

    rows = []
    for game in games:
        team = game["team"]
        opponent = game["opponent"]
        location = game["location"]
        game_date = game["date"]
        result_w = 1 if game["result"] == "W" else 0

        # Moneyline target: did team win?
        # Spread target: did team cover? (team_score - opponent_score vs line)
        # We'll compute spread cover from score data
        team_score = game.get("team_score")
        opp_score = game.get("opponent_score")
        home_spread = game.get("home_spread")

        features = build_features_for_game(team, opponent, game_date, location, db_path)
        features["team"] = team
        features["opponent"] = opponent
        features["location"] = location
        features["game_date"] = game_date
        features["result_w"] = result_w

        # Spread cover: team covers if margin > spread (home team -spread)
        if team_score and opp_score is not None and home_spread is not None:
            margin = team_score - opp_score
            if location == "home":
                covered = margin > home_spread  # home team is -spread
            else:
                covered = -margin > home_spread  # away team is +spread
            features["covered_spread"] = 1 if covered else 0
        else:
            features["covered_spread"] = -1  # unknown

        rows.append(features)

    return pd.DataFrame(rows)


def train_moneyline_model(df: pd.DataFrame, output_dir: str = "models") -> dict:
    """Train moneyline (win/loss) model."""
    if df.empty:
        return {"error": "No data for moneyline model"}

    # Filter rows with known outcome
    train_df = df[df["result_w"].isin([0, 1])].copy()

    if len(train_df) < 30:
        return {"error": "Not enough training data for moneyline model"}

    # Sort by date for chronological split
    train_df = train_df.sort_values("game_date")

    # Last 20% = test set
    split_idx = int(len(train_df) * 0.8)
    train_set = train_df.iloc[:split_idx]
    test_set = train_df.iloc[split_idx:]

    X_train = train_set[FEATURE_COLUMNS].fillna(0)
    y_train = train_set["result_w"]
    X_test = test_set[FEATURE_COLUMNS].fillna(0)
    y_test = test_set["result_w"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    model.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ml_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLUMNS}, f)

    return {
        "type": "moneyline",
        "train_accuracy": round(train_acc * 100, 1),
        "test_accuracy": round(test_acc * 100, 1),
        "train_games": len(train_set),
        "test_games": len(test_set),
    }


def train_spread_model(df: pd.DataFrame, output_dir: str = "models") -> dict:
    """Train spread (cover/don't cover) model."""
    if df.empty:
        return {"error": "No data for spread model"}

    # Filter rows with known spread outcome
    train_df = df[df["covered_spread"].isin([0, 1])].copy()

    if len(train_df) < 30:
        return {"error": "Not enough training data for spread model"}

    train_df = train_df.sort_values("game_date")

    split_idx = int(len(train_df) * 0.8)
    train_set = train_df.iloc[:split_idx]
    test_set = train_df.iloc[split_idx:]

    X_train = train_set[FEATURE_COLUMNS].fillna(0)
    y_train = train_set["covered_spread"]
    X_test = test_set[FEATURE_COLUMNS].fillna(0)
    y_test = test_set["covered_spread"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    model.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    # Also train a regression model to predict margin
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_scaled, y_train)
    mae = mean_absolute_error(y_test, reg.predict(X_test_scaled))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "spread_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_cols": FEATURE_COLUMNS}, f)
    with open(os.path.join(output_dir, "spread_reg.pkl"), "wb") as f:
        pickle.dump({"model": reg, "scaler": scaler, "feature_cols": FEATURE_COLUMNS}, f)

    return {
        "type": "spread",
        "train_accuracy": round(train_acc * 100, 1),
        "test_accuracy": round(test_acc * 100, 1),
        "margin_mae": round(mae, 2),
        "train_games": len(train_set),
        "test_games": len(test_set),
    }


def extract_model_insights(model_data: dict, feature_names: list[str]) -> list[dict]:
    """Extract readable insights from a trained model."""
    model = model_data["model"]
    scaler = model_data["scaler"]

    insights = []
    coef = dict(zip(feature_names, model.coef_[0]))
    abs_coef = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)

    for feat, coeff in abs_coef[:6]:
        if abs(coeff) < 0.01:
            continue
        direction = "increases" if coeff > 0 else "decreases"
        insights.append({
            "feature": feat,
            "direction": direction,
            "coefficient": round(float(coeff), 3),
            "description": f"{feat.replace('_', ' ').title()} {direction} win probability",
        })

    return insights


def train_models(output_dir: str = "models", db_path: Optional[str] = None) -> dict:
    """Build dataset and train both moneyline and spread models."""
    print(f"\n{'='*60}")
    print("Training NBA prediction models")
    print(f"{'='*60}\n")

    print("Building feature dataset...")
    df = build_dataset(db_path)

    if df.empty:
        print("ERROR: No games found in database.")
        print("Hint: Run fetch_basketball_ref.py to fetch game data first.")
        return {"error": "No games in database"}

    print(f"Total game records: {len(df)}")
    valid_ml = df[df["result_w"].isin([0, 1])]
    valid_spread = df[df["covered_spread"].isin([0, 1])]
    print(f"Games with win outcome: {len(valid_ml)}")
    print(f"Games with spread outcome: {len(valid_spread)}")

    results = {}

    print("\nTraining moneyline model...")
    ml_result = train_moneyline_model(df, output_dir)
    if "error" not in ml_result:
        print(f"  Moneyline accuracy — train: {ml_result['train_accuracy']}%, test: {ml_result['test_accuracy']}%")

        # Load and extract insights
        with open(os.path.join(output_dir, "ml_model.pkl"), "rb") as f:
            ml_data = pickle.load(f)
        ml_insights = extract_model_insights(ml_data, FEATURE_COLUMNS)
        with open(os.path.join(output_dir, "ml_insights.json"), "w") as f:
            json.dump({"insights": ml_insights}, f, indent=2)
        ml_result["insights"] = ml_insights
        results["moneyline"] = ml_result

    print("\nTraining spread model...")
    sp_result = train_spread_model(df, output_dir)
    if "error" not in sp_result:
        print(f"  Spread accuracy — train: {sp_result['train_accuracy']}%, test: {sp_result['test_accuracy']}%")
        print(f"  Margin MAE: {sp_result['margin_mae']} points")

        with open(os.path.join(output_dir, "spread_model.pkl"), "rb") as f:
            sp_data = pickle.load(f)
        sp_insights = extract_model_insights(sp_data, FEATURE_COLUMNS)
        with open(os.path.join(output_dir, "spread_insights.json"), "w") as f:
            json.dump({"insights": sp_insights}, f, indent=2)
        sp_result["insights"] = sp_insights
        results["spread"] = sp_result

    print(f"\n{'='*60}")
    print("Training complete")
    print(f"{'='*60}")
    print(f"Models saved to: {output_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train NBA moneyline and spread models")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--db-path", type=str, default=None, help="Path to SQLite database")
    args = parser.parse_args()

    result = train_models(output_dir=args.output, db_path=args.db_path)
    if "error" in result:
        print(f"\nError: {result['error']}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())