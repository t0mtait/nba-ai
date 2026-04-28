"""Train NBA moneyline and spread prediction models.

Uses:
1. Power rankings (user-submitted per-team ratings)
2. Home court advantage (historical ~58% home win rate)
3. Past matchups (this season + last season H2H)
4. Injury reports (impact per game date)
"""
from __future__ import annotations

import os
import pickle
import json
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

import db

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "power_diff",
    "home_court_adv",
    "h2h_win_pct_diff",
    "h2h_avg_margin",
    "rest_diff",
    "injury_impact",
    "season_net_rtg_diff",
]


def get_power_ranking(season_year: int, team: str, db_path: Optional[str] = None) -> dict:
    """Get a team's latest power ranking entry."""
    entry = db.get_power_ranking_for_team(season_year, team, db_path)
    if entry:
        return {
            "rank": entry.get("rank", 15),
            "power_rating": entry.get("power_rating", 0.0),
            "net_rtg": entry.get("net_rtg", 0.0),
            "ortg_override": entry.get("ortg_override"),
            "drtg_override": entry.get("drtg_override"),
        }
    return {"rank": 15, "power_rating": 0.0, "net_rtg": 0.0, "ortg_override": None, "drtg_override": None}


def compute_h2h_features(
    team: str,
    opponent: str,
    cutoff_date: str,
    db_path: Optional[str] = None,
) -> dict:
    """Compute H2H features from this season + last season."""
    cur_season = int(cutoff_date[:4])
    if int(cutoff_date[5:7]) <= 6:
        cur_season -= 1
    last_season = cur_season - 1

    games_cur = db.get_matchups(team=team, opponent=opponent, season_year=cur_season, db_path=db_path)
    games_last = db.get_matchups(team=team, opponent=opponent, season_year=last_season, db_path=db_path)

    # Filter to before cutoff date
    games_cur = [g for g in games_cur if g["date"] < cutoff_date]
    games_last = [g for g in games_last if g["date"] < cutoff_date]

    all_games = games_cur + games_last

    if not all_games:
        return {"h2h_win_pct": 0.5, "h2h_avg_margin": 0.0, "h2h_games": 0}

    wins = sum(1 for g in all_games if g["result"] == "W")
    total = len(all_games)

    margins = []
    for g in all_games:
        if g["team_score"] and g["opponent_score"]:
            margin = g["team_score"] - g["opponent_score"]
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
    Uses will_play (user's prediction) and player_impact_score (per-game production value).
    Returns: negative = team is undermanned (disadvantage).

    Impact logic:
    - will_play=1 (user says player will play) → 0 impact
    - will_play=0 (user says player will miss) → -player_impact_score impact
    - will_play not set (null) → fallback to status-based estimate
    """
    injuries = db.get_injuries(team=team, db_path=db_path)

    impact = 0.0
    for inj in injuries:
        if inj.get("date_reported", "") >= game_date:
            continue

        will_play = inj.get("will_play")
        impact_score = inj.get("player_impact_score", 0) or 0
        status = (inj.get("status") or "").lower()

        if will_play is None:
            # No user prediction yet — use status-based estimate
            if status in ("out", "doubtful"):
                impact -= max(impact_score, 2.0)  # At minimum 2.0 if we have no player data
            elif status in ("questionable", "probable"):
                impact -= max(impact_score * 0.3, 0.5)
            # If will_play=1 (user says they play), no impact
        elif will_play == 0:
            # User says player will miss — use full impact score
            impact -= impact_score

    return impact


def compute_rest_diff(
    team: str,
    opponent: str,
    game_date: str,
    db_path: Optional[str] = None,
) -> float:
    """Rest advantage: days since last game. Positive = more rested."""
    def last_game_date(team_code: str, before_date: str) -> Optional[str]:
        games = db.get_games(team=team_code, db_path=db_path)
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
    Build feature dict for a single game prediction.
    All features from team's perspective.
    """
    season_year = int(game_date[:4])
    if int(game_date[5:7]) <= 6:
        season_year -= 1

    # 1. Power ranking diff
    team_pr = get_power_ranking(season_year, team, db_path)
    opp_pr = get_power_ranking(season_year, opponent, db_path)

    team_rank = team_pr["rank"]
    opp_rank = opp_pr["rank"]
    power_diff = (opp_rank - team_rank) / 29.0  # -1 to 1 scale

    # Use power ranking net_rtg if available, else fall back to season stats
    if team_pr["net_rtg"] != 0:
        team_net = team_pr["net_rtg"]
    else:
        team_stats = db.get_team_stats(team, season_year, db_path)
        team_net = (team_stats.get("net_rtg", 0) or 0) if team_stats else 0.0

    if opp_pr["net_rtg"] != 0:
        opp_net = opp_pr["net_rtg"]
    else:
        opp_stats = db.get_team_stats(opponent, season_year, db_path)
        opp_net = (opp_stats.get("net_rtg", 0) or 0) if opp_stats else 0.0

    season_net_rtg_diff = team_net - opp_net if location == "home" else opp_net - team_net

    # 2. Home court advantage
    home_court_adv = 0.58 if location == "home" else -0.58

    # 3. H2H features
    h2h = compute_h2h_features(team, opponent, game_date, db_path)
    h2h_win_pct_diff = h2h["h2h_win_pct"] - 0.5
    h2h_avg_margin = h2h["h2h_avg_margin"] if location == "home" else -h2h["h2h_avg_margin"]

    # 4. Rest diff
    rest_diff = compute_rest_diff(team, opponent, game_date, db_path)

    # 5. Injury impact
    team_inj = compute_injury_impact(team, game_date, db_path)
    opp_inj = compute_injury_impact(opponent, game_date, db_path)
    injury_impact = team_inj - opp_inj

    return {
        "power_diff": power_diff,
        "home_court_adv": home_court_adv,
        "h2h_win_pct_diff": h2h_win_pct_diff,
        "h2h_avg_margin": h2h_avg_margin,
        "rest_diff": rest_diff,
        "injury_impact": injury_impact,
        "season_net_rtg_diff": season_net_rtg_diff,
    }


def build_dataset(db_path: Optional[str] = None) -> pd.DataFrame:
    """Build dataset of games with computed features and targets."""
    games = db.get_games(db_path=db_path)
    if not games:
        return pd.DataFrame()

    rows = []
    for game in games:
        team = game["team"]
        opponent = game["opponent"]
        location = game["location"]
        game_date = game["date"]
        result_w = 1 if game["result"] == "W" else 0

        team_score = game.get("team_score")
        opp_score = game.get("opponent_score")
        home_spread = game.get("home_spread")

        features = build_features_for_game(team, opponent, game_date, location, db_path)
        features["team"] = team
        features["opponent"] = opponent
        features["location"] = location
        features["game_date"] = game_date
        features["result_w"] = result_w

        # Spread cover: team covers if margin > spread
        if team_score is not None and opp_score is not None and home_spread is not None:
            margin = team_score - opp_score
            if location == "home":
                covered = margin > home_spread
            else:
                covered = -margin > home_spread
            features["covered_spread"] = 1 if covered else 0
        else:
            features["covered_spread"] = -1

        rows.append(features)

    return pd.DataFrame(rows)


def train_moneyline_model(df: pd.DataFrame, output_dir: str = "models") -> dict:
    if df.empty:
        return {"error": "No data for moneyline model"}

    train_df = df[df["result_w"].isin([0, 1])].copy()
    if len(train_df) < 30:
        return {"error": "Not enough training data for moneyline model"}

    train_df = train_df.sort_values("game_date")
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
    if df.empty:
        return {"error": "No data for spread model"}

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


def backtest(db_path: Optional[str] = None, models_dir: str = "models") -> dict:
    """
    Backtest the moneyline model on historical games.
    """
    df = build_dataset(db_path)
    if df.empty:
        return {"error": "No data to backtest"}

    model_path = os.path.join(models_dir, "ml_model.pkl")
    if not os.path.exists(model_path):
        return {"error": "Model not trained yet"}

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]

    # Use all games with known outcomes
    test_df = df[df["result_w"].isin([0, 1])].copy()
    test_df = test_df.sort_values("game_date")

    if len(test_df) < 30:
        return {"error": "Not enough games to backtest"}

    # Split: last 20% as test
    split_idx = int(len(test_df) * 0.8)
    out_of_sample = test_df.iloc[split_idx:]

    X = out_of_sample[FEATURE_COLUMNS].fillna(0)
    y_true = out_of_sample["result_w"].values

    X_scaled = scaler.transform(X)
    win_probs = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)

    results = []
    units = 0.0  # ROI tracking (1 unit per bet)

    for i, (_, row) in enumerate(out_of_sample.iterrows()):
        prob = win_probs[i]
        pred = predictions[i]
        actual = y_true[i]

        # Only bet when confidence >55% or <45%
        if prob > 0.55:
            bet_home = True
            bet_on = row["team"] if row["location"] == "home" else row["opponent"]
        elif prob < 0.45:
            bet_home = False
            bet_on = row["opponent"] if row["location"] == "home" else row["team"]
        else:
            results.append({"date": row["game_date"], "matchup": f"{row['team']} vs {row['opponent']}",
                            "bet": "PASS", "result": "PUSH", "prob": round(float(prob), 3), "correct": None})
            continue

        correct = (pred == actual)
        # Simplified: bet 1 unit, win 0.91 units on ML bet, lose 1 unit
        if correct:
            units += 0.91
            roi = "WIN"
        else:
            units -= 1.0
            roi = "LOSS"

        results.append({
            "date": row["game_date"],
            "matchup": f"{row['team']} {'home' if row['location'] == 'home' else 'away'} vs {row['opponent']}",
            "bet": f"BET {bet_on}",
            "result": roi,
            "prob": round(float(prob), 3),
            "correct": bool(correct),
        })

    # Aggregate stats
    bet_results = [r for r in results if r["bet"] != "PASS"]
    correct_bets = sum(1 for r in bet_results if r["correct"])
    total_bets = len(bet_results)
    accuracy = round(correct_bets / total_bets * 100, 1) if total_bets > 0 else 0.0
    total_units = round(units, 2)
    roi_pct = round(total_units / total_bets * 100, 1) if total_bets > 0 else 0.0

    return {
        "total_games": len(out_of_sample),
        "games_bet": total_bets,
        "games_passed": len(results) - total_bets,
        "correct": correct_bets,
        "accuracy": accuracy,
        "roi_units": total_units,
        "roi_pct": roi_pct,
        "details": results[-50:],  # last 50 for display
    }


def train_models(output_dir: str = "models", db_path: Optional[str] = None) -> dict:
    logger.info("Training NBA prediction models...")

    df = build_dataset(db_path)
    if df.empty:
        logger.error("No games found in database.")
        return {"error": "No games in database"}

    logger.info(f"Total game records: {len(df)}")
    valid_ml = df[df["result_w"].isin([0, 1])]
    valid_spread = df[df["covered_spread"].isin([0, 1])]
    logger.info(f"Games with win outcome: {len(valid_ml)}")
    logger.info(f"Games with spread outcome: {len(valid_spread)}")

    results = {}

    ml_result = train_moneyline_model(df, output_dir)
    if "error" not in ml_result:
        logger.info(f"  Moneyline accuracy — train: {ml_result['train_accuracy']}%, test: {ml_result['test_accuracy']}%")
        results["moneyline"] = ml_result

    sp_result = train_spread_model(df, output_dir)
    if "error" not in sp_result:
        logger.info(f"  Spread accuracy — train: {sp_result['train_accuracy']}%, test: {sp_result['test_accuracy']}%")
        results["spread"] = sp_result

    logger.info("Training complete")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--db-path", type=str, default=None, help="Path to SQLite database")
    args = parser.parse_args()

    result = train_models(output_dir=args.output, db_path=args.db_path)
    if "error" in result:
        logger.error(f"Error: {result['error']}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())