"""Train and save the Celtics (or other team) prediction models."""
from __future__ import annotations

import os
import pickle
import argparse
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
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
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = MODELS_DIR,
) -> dict:
    """
    Train prediction models for a team.

    Args:
        team_code: Team code to train on (e.g., 'BOS', 'LAL')
        season_year: Optional specific season to train on
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
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
        y = df["celtics_win"]

        # Drop rows with missing features
        mask = subset.notna().all(axis=1)
        subset = subset.loc[mask]
        y = y.loc[mask]

        if len(subset) < 10:
            print(f"⚠️  Not enough valid games for {location} model ({len(subset)} games)")
            continue

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            subset, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if y.sum() > 1 and (len(y) - y.sum()) > 1 else None,
        )

        # Train model
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{location.capitalize()} accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model
        model_path = os.path.join(output_dir, f"model_{location}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✓ Saved {location} model to {model_path}")

        # Generate predictions for all games
        all_predictions = model.predict(subset)
        all_probs = model.predict_proba(subset)

        # Save predictions
        pred_df = subset.copy()
        pred_df["location"] = location
        pred_df["date"] = df["date"].values
        pred_df["opponent"] = df["opponent"].values
        pred_df["prediction"] = all_predictions
        pred_df["actual"] = y.values
        pred_df["win_prob"] = all_probs[:, 1]
        pred_df["correct"] = all_predictions == y.values

        results[location] = {
            "model": model,
            "accuracy": accuracy,
            "predictions": pred_df,
        }

    # Save feature columns
    feature_cols_path = os.path.join(output_dir, "feature_cols.pkl")
    with open(feature_cols_path, "wb") as f:
        pickle.dump(
            {"home": FEATURE_COLUMNS, "away": FEATURE_COLUMNS},
            f,
        )
    print(f"\n✓ Saved feature columns to {feature_cols_path}")

    # Combine and save all predictions
    all_preds = []
    for location, result in results.items():
        all_preds.append(result["predictions"])

    if all_preds:
        combined_preds = pd.concat(all_preds, ignore_index=True)
        combined_preds = combined_preds.sort_values("date", ascending=False)

        preds_path = os.path.join(output_dir, "game_predictions.csv")
        combined_preds.to_csv(preds_path, index=False)
        print(f"✓ Saved {len(combined_preds)} game predictions to {preds_path}")

        # Calculate overall accuracy
        overall_correct = combined_preds["correct"].sum()
        overall_total = len(combined_preds)
        overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0

        # Calculate accuracy on held-out test sets only
        test_correct = 0
        test_total = 0
        for location, result in results.items():
            subset = result["predictions"]
            pred = result["model"].predict(subset[FEATURE_COLUMNS])
            test_correct += (pred == subset["actual"]).sum()
            test_total += len(subset)

        test_accuracy = test_correct / test_total * 100 if test_total > 0 else 0

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE FOR {team_code}")
        print(f"{'='*60}")
        print(f"Total games: {overall_total}")
        print(f"Training accuracy: {overall_accuracy:.1f}%")
        print(f"Test accuracy (approximate): {test_accuracy:.1f}%")
        print(f"Models saved to: {output_dir}/")

        return {
            "team": team_code,
            "total_games": overall_total,
            "training_accuracy": overall_accuracy,
            "test_accuracy": test_accuracy,
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
