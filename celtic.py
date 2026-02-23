import pandas as pd
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

subset1 = subset1[mask1]
y1 = y1[mask1]

subset2 = subset2[mask2]
y2 = y2[mask2]

print("Home:", subset1.shape, y1.shape)
print("Away:", subset2.shape, y2.shape)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    subset2, y2,
    test_size=0.2,
    random_state=42,
    stratify=y2
)

model_away = LogisticRegression(max_iter=1000)
model_away.fit(X_train2, y_train2)

y_pred2 = model_away.predict(X_test2)
print("Away accuracy:", accuracy_score(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))


X_train1, X_test1, y_train1, y_test1 = train_test_split(
    subset1, y1,
    test_size=0.2,
    random_state=42,
    stratify=y1
)

model_home = LogisticRegression(max_iter=1000)
model_home.fit(X_train1, y_train1)

y_pred1 = model_home.predict(X_test1)
print("Home accuracy:", accuracy_score(y_test1, y_pred1))
print(classification_report(y_test1, y_pred1))


import numpy as np

coef_df_away = pd.DataFrame({
    "feature": feature_cols2,
    "coef": model_away.coef_[0],
    "odds_ratio": np.exp(model_away.coef_[0])
}).sort_values("coef", ascending=False)

coef_df_home = pd.DataFrame({
    "feature": feature_cols1,
    "coef": model_home.coef_[0],
    "odds_ratio": np.exp(model_home.coef_[0])
}).sort_values("coef", ascending=False)

coef_df_home
