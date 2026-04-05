"""Train and save the Celtics prediction models."""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_celtics_games, FEATURE_COLUMNS

games = load_celtics_games("data")

home_games = games[games["location"] == "home"].copy()
away_games = games[games["location"] == "away"].copy()

y1 = home_games["celtics_win"]
y2 = away_games["celtics_win"]

feature_cols1 = FEATURE_COLUMNS
feature_cols2 = FEATURE_COLUMNS

subset1 = home_games[feature_cols1]
subset2 = away_games[feature_cols2]

mask1 = subset1.notna().all(axis=1)
mask2 = subset2.notna().all(axis=1)

home_games = home_games.loc[mask1].copy()
away_games = away_games.loc[mask2].copy()

subset1 = home_games[feature_cols1]
y1 = home_games["celtics_win"]

subset2 = away_games[feature_cols2]
y2 = away_games["celtics_win"]

print("Home:", subset1.shape, y1.shape)
print("Away:", subset2.shape, y2.shape)

# Train away model
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    subset2, y2,
    test_size=0.2,
    random_state=42,
    stratify=y2
)

model_away = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
model_away.fit(X_train2, y_train2)

y_pred2 = model_away.predict(X_test2)
print("\nAway accuracy:", accuracy_score(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))

# Train home model
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    subset1, y1,
    test_size=0.2,
    random_state=42,
    stratify=y1
)

model_home = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
model_home.fit(X_train1, y_train1)

y_pred1 = model_home.predict(X_test1)
print("\nHome accuracy:", accuracy_score(y_test1, y_pred1))
print(classification_report(y_test1, y_pred1))

# Save models
os.makedirs("models", exist_ok=True)

with open("models/model_home.pkl", "wb") as f:
    pickle.dump(model_home, f)

with open("models/model_away.pkl", "wb") as f:
    pickle.dump(model_away, f)

# Save feature columns
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump({"home": feature_cols1, "away": feature_cols2}, f)

home_predictions = model_home.predict(subset1)
home_probs = model_home.predict_proba(subset1)
away_predictions = model_away.predict(subset2)
away_probs = model_away.predict_proba(subset2)

home_games_pred = subset1.copy()
home_games_pred["location"] = "home"
home_games_pred["date"] = home_games["date"].values
home_games_pred["opponent"] = home_games["opponent"].values
home_games_pred["prediction"] = home_predictions
home_games_pred["actual"] = y1.values
home_games_pred["win_prob"] = home_probs[:, 1]
home_games_pred["correct"] = home_predictions == y1.values

away_games_pred = subset2.copy()
away_games_pred["location"] = "away"
away_games_pred["date"] = away_games["date"].values
away_games_pred["opponent"] = away_games["opponent"].values
away_games_pred["prediction"] = away_predictions
away_games_pred["actual"] = y2.values
away_games_pred["win_prob"] = away_probs[:, 1]
away_games_pred["correct"] = away_predictions == y2.values

# Combine and save
all_games = pd.concat([home_games_pred, away_games_pred], ignore_index=True)
all_games.to_csv("models/game_predictions.csv", index=False)

print("\nModels saved to models/")
print(f"Game predictions saved: {len(all_games)} games")
print(f"Overall accuracy on test set: {(all_games['correct'].sum() / len(all_games) * 100):.2f}%")
