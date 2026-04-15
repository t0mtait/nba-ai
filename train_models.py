"""Train and save the Celtics (or other team) prediction models."""
from __future__ import annotations

import os
import pickle
import argparse
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data_loader import (
    load_games_from_db,
    FEATURE_COLUMNS,
    get_teams_from_db,
)

# Model output directory
MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def train_models(
    team_code: str = "BOS",
    season_year: Optional[int] = None,
    output_dir: str = MODELS_DIR,
) -> dict:
    """
    Train prediction models for a team using chronological train/test split.

    The 20 most recent games are held out as the TEST set.
    All older games are used for TRAINING.

    Args:
        team_code: Team code to train on (e.g., 'BOS', 'LAL')
        season_year: Optional specific season to train on
        output_dir: Directory to save models

    Returns:
        Dict with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training prediction model for {team_code}")
    print(f"{'='*60}\n")

    # Load games from database
    print(f"Loading games from database...")
    games = load_games_from_db(team_code=team_code, season_year=season_year)

    if games.empty:
        print(f"ERROR: No games found in database for {team_code}")
        print(f"Hint: Run 'python -c \"import db; db.fetch_team_games('{team_code}')\"' to fetch data")
        return {"error": "No games found"}

    print(f"Total games loaded: {len(games)}")

    # Sort by date - oldest first
    games = games.sort_values("date")
    print(f"Date range: {games['date'].min()} to {games['date'].max()}")

    # Split by location
    home_games = games[games["location"] == "home"].copy()
    away_games = games[games["location"] == "away"].copy()

    print(f"Home games: {len(home_games)}")
    print(f"Away games: {len(away_games)}")

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Train models for each location
    for location, df in [("home", home_games), ("away", away_games)]:
        if df.empty:
            print(f"\n⚠️  No {location} games found, skipping {location} model")
            continue

        print(f"\n{'-'*40}")
        print(f"Training {location.upper()} model")
        print(f"{'-'*40}")

        # Prepare features and target
        subset = df[FEATURE_COLUMNS].copy()
        y = df["celtics_win"].values
        dates = df["date"].values
        opponents = df["opponent"].values
        paces = df["pace"].values if "pace" in df.columns else [0] * len(subset)
        ftrs = df["ftr"].values if "ftr" in df.columns else [0] * len(subset)
        efg_pcts = df["efg_pct"].values if "efg_pct" in df.columns else [0] * len(subset)
        tovs = df["tov_pct"].values if "tov_pct" in df.columns else [0] * len(subset)
        orbs = df["orb_pct"].values if "orb_pct" in df.columns else [0] * len(subset)

        # Drop rows with missing features
        mask = subset.notna().all(axis=1)
        valid_indices = mask[mask].index
        subset = subset.loc[valid_indices]
        y = y[valid_indices.get_indexer(range(len(y)))]
        dates = dates[valid_indices.get_indexer(range(len(dates)))]
        opponents = opponents[valid_indices.get_indexer(range(len(opponents)))]
        paces = paces[valid_indices.get_indexer(range(len(paces)))]
        ftrs = ftrs[valid_indices.get_indexer(range(len(ftrs)))]
        efg_pcts = efg_pcts[valid_indices.get_indexer(range(len(efg_pcts)))]
        tovs = tovs[valid_indices.get_indexer(range(len(tovs)))]
        orbs = orbs[valid_indices.get_indexer(range(len(orbs)))]

        # Rebuild with proper alignment
        df_valid = df.loc[valid_indices].copy()
        subset = df_valid[FEATURE_COLUMNS].copy()
        y = df_valid["celtics_win"].values
        dates = df_valid["date"].values
        opponents = df_valid["opponent"].values

        if len(subset) < 10:
            print(f"⚠️  Not enough valid games for {location} model ({len(subset)} games)")
            continue

        # Chronological split: last 20 games are test set
        test_df = df_valid.tail(20)
        train_df = df_valid.iloc[:-20]

        # Only use test set if we have enough training data
        if len(train_df) < 30:
            print(f"⚠️  Not enough training games ({len(train_df)}), need at least 30")
            return {"error": "Not enough training data"}

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["celtics_win"]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["celtics_win"]

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train model
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(X_train, y_train)

        # Evaluate on TRAINING set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"\n{location.capitalize()} TRAINING accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")

        # Evaluate on TEST set (held-out 20 most recent games)
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"{location.capitalize()} TEST accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")

        print("\nClassification Report (TEST set):")
        print(classification_report(y_test, y_test_pred))

        # Save model with team-specific name (fall back to default for BOS)
        model_path = os.path.join(output_dir, f"model_{location}_{team_code}.pkl")
        if team_code == "BOS":
            model_path = os.path.join(output_dir, f"model_{location}.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✓ Saved {location} model to {model_path}")

        # Generate predictions for ALL games (train + test)
        all_predictions = model.predict(subset)
        all_probs = model.predict_proba(subset)

        # Build predictions DataFrame with required columns
        pred_df = pd.DataFrame({
            "date": dates,
            "opponent": opponents,
            "location": location,
            "pace": paces,
            "ftr": ftrs,
            "efg_pct": efg_pcts,
            "tov_pct": tovs,
            "orb_pct": orbs,
            "prediction": all_predictions,
            "actual": y,
            "win_prob": all_probs[:, 1],
            "correct": all_predictions == y,
        })

        # Split into train/test predictions for accuracy tracking
        train_preds = pred_df.iloc[:-20]
        test_preds = pred_df.tail(20)

        results[location] = {
            "model": model,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "predictions": pred_df,
            "train_preds": train_preds,
            "test_preds": test_preds,
        }

    # Save feature columns
    feature_cols_path = os.path.join(output_dir, "feature_cols.pkl")
    with open(feature_cols_path, "wb") as f:
        pickle.dump(
            {"home": FEATURE_COLUMNS, "away": FEATURE_COLUMNS},
            f,
        )
    print(f"\n✓ Saved feature columns to {feature_cols_path}")

    # Combine and save team-specific predictions CSV
    all_preds = []
    for location, result in results.items():
        all_preds.append(result["predictions"])

    if all_preds:
        combined_preds = pd.concat(all_preds, ignore_index=True)
        combined_preds = combined_preds.sort_values("date", ascending=False)

        # Save team-specific predictions CSV
        preds_path = os.path.join(output_dir, f"{team_code}_predictions.csv")
        combined_preds.to_csv(preds_path, index=False)
        print(f"✓ Saved {len(combined_preds)} game predictions to {preds_path}")

        # Also save as default for BOS
        if team_code == "BOS":
            default_preds_path = os.path.join(output_dir, "game_predictions.csv")
            combined_preds.to_csv(default_preds_path, index=False)
            print(f"✓ Saved default predictions to {default_preds_path}")

        # Calculate overall training accuracy
        train_correct = sum(r["train_accuracy"] * len(r["train_preds"]) for r in results.values())
        train_total = sum(len(r["train_preds"]) for r in results.values())
        overall_train_accuracy = train_correct / train_total * 100 if train_total > 0 else 0

        # Calculate overall test accuracy
        test_correct = sum(r["test_accuracy"] * len(r["test_preds"]) for r in results.values())
        test_total = sum(len(r["test_preds"]) for r in results.values())
        overall_test_accuracy = test_correct / test_total * 100 if test_total > 0 else 0

        total_games = train_total + test_total

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE FOR {team_code}")
        print(f"{'='*60}")
        print(f"Total games: {total_games}")
        print(f"Training accuracy (older games): {overall_train_accuracy:.1f}%")
        print(f"Test accuracy (20 most recent): {overall_test_accuracy:.1f}%")
        print(f"Models saved to: {output_dir}/")

        return {
            "team": team_code,
            "total_games": total_games,
            "training_accuracy": overall_train_accuracy,
            "test_accuracy": overall_test_accuracy,
            "models_saved": list(results.keys()),
        }
    else:
        print("\n❌ ERROR: No models were trained successfully")
        return {"error": "Training failed"}


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train NBA prediction models")
    parser.add_argument(
        "--team",
        type=str,
        default="BOS",
        help="Team code to train on (default: BOS)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Specific season year to train on (e.g., 2025 for 2024-25 season)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=MODELS_DIR,
        help=f"Output directory for models (default: {MODELS_DIR})",
    )

    args = parser.parse_args()

    result = train_models(
        team_code=args.team,
        season_year=args.season,
        output_dir=args.output,
    )

    if "error" in result:
        print(f"\nError: {result['error']}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
