"""NBA Win Predictor - Display model data and performance results."""
import os
import shutil
import threading
import time
import kagglehub
import pandas as pd
from flask import Flask, jsonify, make_response, request, render_template_string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

DATASET_HANDLE = "eoinamoore/historical-nba-data-and-player-box-scores"
DATASET_FILE = "TeamStatistics.csv"
REFRESH_INTERVAL_SECONDS = int(os.environ.get("REFRESH_INTERVAL_SECONDS", "3600"))

featCols = ['teamId', 'home', 'assists', 'reboundsTotal', 'blocks', 'steals',
            'turnovers', 'foulsPersonal', 'q1Points', 'q2Points',
            'fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted']
# Columns to display in table (excludes teamId and home, shown separately)
displayCols = ['assists', 'reboundsTotal', 'blocks', 'steals', 'turnovers',
               'foulsPersonal', 'q1Points', 'q2Points',
               'fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted']
# Display labels for table headers (matches displayCols order)
displayLabels = ['A', 'R', 'B', 'S', 'TO', 'F', 'Q1', 'Q2', 'FGA', '3FA', 'FTA']

_refresh_lock = threading.Lock()
_state = {
    "model": None,
    "predictions_df": pd.DataFrame(),
    "train_acc": 0.0,
    "test_acc": 0.0,
    "total_games": 0,
    "train_games": 0,
    "test_games": 0,
    "last_updated": 0.0,
}


def _cleanup_old_cache_versions(csv_path: str) -> None:
    """Remove older cached dataset versions so container disk usage stays bounded."""
    try:
        current_version_dir = os.path.dirname(os.path.abspath(csv_path))
        versions_dir = os.path.dirname(current_version_dir)
        if os.path.basename(versions_dir) != "versions" or not os.path.isdir(versions_dir):
            return
        for entry in os.listdir(versions_dir):
            candidate = os.path.join(versions_dir, entry)
            if os.path.abspath(candidate) != current_version_dir and os.path.isdir(candidate):
                shutil.rmtree(candidate, ignore_errors=True)
    except Exception:
        pass


def refresh_data(force_download: bool = True) -> None:
    """Download the latest Kaggle dataset and retrain the model."""
    with _refresh_lock:
        path = kagglehub.dataset_download(
            DATASET_HANDLE,
            path=DATASET_FILE,
            force_download=force_download,
        )
        csv_path = os.path.join(path, DATASET_FILE) if os.path.isdir(path) else path
        _cleanup_old_cache_versions(csv_path)

        df = pd.read_csv(csv_path, low_memory=False)
        # Preserve columns before dropna for display
        date_col = 'gameDate' if 'gameDate' in df.columns else 'gameDateTimeEst'
        all_game_dates = df[date_col].astype(str)
        all_opponent_names = df['opponentTeamName']
        all_team_names = df['teamName']

        df = df.dropna(subset=featCols + ['win'])
        X = df[featCols]
        y = df['win']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        # Build prediction data for ALL games (train + test), sorted by date descending
        all_X = pd.concat([X_train, X_test])
        all_y = pd.concat([y_train, y_test])
        all_preds = list(train_preds) + list(test_preds)

        predictions_df = all_X.copy()
        predictions_df['actual'] = all_y.values
        predictions_df['pred'] = all_preds
        predictions_df['correct'] = predictions_df['pred'] == predictions_df['actual']
        predictions_df['_sortDate'] = all_game_dates.loc[all_X.index].values
        predictions_df['gameDate'] = all_game_dates.loc[all_X.index].str[:10].values
        predictions_df['opponentTeamName'] = all_opponent_names.loc[all_X.index].values
        predictions_df['teamName'] = all_team_names.loc[all_X.index].values
        predictions_df['home'] = all_X['home'].values
        predictions_df = (
            predictions_df.sort_index()
            .sort_values('_sortDate', ascending=False, kind='mergesort')
            .drop(columns=['_sortDate'])
        )

        _state.update({
            "model": model,
            "predictions_df": predictions_df,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "total_games": len(all_X),
            "train_games": len(X_train),
            "test_games": len(X_test),
            "last_updated": time.time(),
        })


def ensure_fresh_data() -> None:
    """Refresh data if TTL has expired."""
    if time.time() - _state["last_updated"] >= REFRESH_INTERVAL_SECONDS:
        try:
            refresh_data(force_download=True)
        except Exception as exc:
            app.logger.warning("Failed to refresh dataset: %s", exc)
            if _state["predictions_df"].empty:
                raise


def _background_refresh_loop() -> None:
    """Periodically refresh dataset in the background."""
    while True:
        time.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            refresh_data(force_download=True)
        except Exception as exc:
            app.logger.warning("Background dataset refresh failed: %s", exc)


# Load initial data at startup and launch periodic refresh thread
refresh_data(force_download=True)
_refresh_thread = threading.Thread(target=_background_refresh_loop, daemon=True)
_refresh_thread.start()


HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Win Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0a0a0f; color: #f0f0f5; min-height: 100vh; padding: 24px; }
        .container { max-width: 900px; margin: 0 auto; }
        .header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 4px; }
        h1 { font-size: 1.5rem; font-weight: 800; }
        h1 span { color: #3ecf6a; }
        .tagline { color: #888899; font-size: 0.85rem; margin-bottom: 32px; }
        .dataset-btn { display: inline-block; padding: 8px 16px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; color: #888899; text-decoration: none; font-size: 0.8rem; transition: all 0.2s; }
        .dataset-btn:hover { background: rgba(255,255,255,0.1); color: #f0f0f5; }
        .card { background: #111118; border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; margin-bottom: 20px; overflow-x: auto; }
        .section-title { font-size: 1rem; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
        .stat-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px; text-align: center; }
        .stat-value { font-size: 1.75rem; font-weight: 800; color: #3ecf6a; }
        .stat-label { font-size: 0.75rem; color: #888899; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.75rem; min-width: 800px; }
        th { text-align: left; padding: 8px 10px; background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.08); border-right: 1px solid rgba(255,255,255,0.08); color: #888899; font-weight: 600; text-transform: uppercase; font-size: 0.65rem; letter-spacing: 0.5px; white-space: normal; }
        td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); }
        tr:hover td { background: rgba(255,255,255,0.02); }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
        .badge.win { background: rgba(62,207,106,0.15); color: #3ecf6a; }
        .badge.loss { background: rgba(239,68,68,0.15); color: #ef4444; }
        .correct { color: #3ecf6a; }
        .incorrect { color: #ef4444; }
        .correct-row td { background: rgba(62,207,106,0.08); }
        .incorrect-row td { background: rgba(239,68,68,0.08); }
        .section { margin-top: 24px; }
        .features { display: flex; flex-wrap: wrap; gap: 8px; }
        .feature-tag { padding: 4px 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; font-size: 0.8rem; color: #888899; }
        .pagination { display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 16px; }
        .pagination a, .pagination span { padding: 6px 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; color: #888899; text-decoration: none; font-size: 0.85rem; }
        .pagination a:hover { background: rgba(255,255,255,0.1); color: #f0f0f5; }
        .pagination .current { background: rgba(62,207,106,0.15); border-color: rgba(62,207,106,0.3); color: #3ecf6a; }
        .pagination .disabled { opacity: 0.4; pointer-events: none; }
        .pagination .page-info { padding: 6px 12px; color: #888899; font-size: 0.85rem; }
        .legend { display: flex; flex-wrap: wrap; gap: 12px 20px; margin-bottom: 16px; font-size: 0.7rem; color: #888899; }
        .legend span { white-space: nowrap; }
        .legend strong { color: #f0f0f5; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🏀 NBA <span>Win Predictor</span></h1>
                <p class="tagline">Logistic Regression on Team Statistics</p>
            </div>
            <a class="dataset-btn" href="https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=TeamStatistics.csv" target="_blank">📂 View Dataset</a>
        </div>

        <div class="card">
            <div class="section-title">📊 Model Performance</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ "%.1f"|format(test_acc * 100) }}%</div>
                    <div class="stat-label">Test Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ train_games }}</div>
                    <div class="stat-label">Training Games</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ test_games }}</div>
                    <div class="stat-label">Testing Games</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ total_games }}</div>
                    <div class="stat-label">Total Games</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-title">🔧 Feature Columns</div>
            <div class="features">
                {% for f in featCols %}
                <span class="feature-tag">{{ f }}</span>
                {% endfor %}
            </div>
        </div>

        <div class="card">
            <div class="section-title">📋 Predictions</div>
            <div class="legend">
                <span><strong>P</strong> = Predicted Result</span>
                <span><strong>A</strong> = Actual Result</span>
                <span><strong>H/A</strong> = Home/Away</span>
                <span><strong>A</strong> = Assists</span>
                <span><strong>R</strong> = Rebounds</span>
                <span><strong>B</strong> = Blocks</span>
                <span><strong>S</strong> = Steals</span>
                <span><strong>TO</strong> = Turnovers</span>
                <span><strong>F</strong> = Fouls</span>
                <span><strong>Q1/Q2</strong> = Quarter Points</span>
                <span><strong>FGA</strong> = Field Goals Attempted</span>
                <span><strong>3FA</strong> = 3-Pointers Attempted</span>
                <span><strong>FTA</strong> = Free Throws Attempted</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Team</th>
                        <th>Opp</th>
                        <th>H/A</th>
                        {% for f in displayLabels %}
                        <th>{{ f }}</th>
                        {% endfor %}
                        <th>P</th>
                        <th>A</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, row in page_items.iterrows() %}
                    <tr class="{{ 'correct-row' if row.get('correct') else 'incorrect-row' }}">
                        <td>{{ row.get('gameDate', 'N/A')[:10] if row.get('gameDate') else 'N/A' }}</td>
                        <td>{{ row.get('teamName', 'N/A') }}</td>
                        <td>{{ row.get('opponentTeamName', 'N/A') }}</td>
                        <td><span class="badge">{{ 'Home' if row.get('home') == 1 else 'Away' }}</span></td>
                        {% for f in displayCols %}
                        <td>{{ "{:.0f}".format(row.get(f, 0)) }}</td>
                        {% endfor %}
                        <td><span class="badge {{ 'win' if row.get('pred') == 1 else 'loss' }}">{{ 'W' if row.get('pred') == 1 else 'L' }}</span></td>
                        <td><span class="badge {{ 'win' if row.get('actual') == 1 else 'loss' }}">{{ 'W' if row.get('actual') == 1 else 'L' }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <div class="pagination">
                {% if page > 1 %}
                <a href="?page={{ page - 1 }}">&laquo; Prev</a>
                {% else %}
                <span class="disabled">&laquo; Prev</span>
                {% endif %}
                <span class="page-info">{{ page }} / {{ total_pages }}</span>
                {% if page < total_pages %}
                <a href="?page={{ page + 1 }}">Next &raquo;</a>
                {% else %}
                <span class="disabled">Next &raquo;</span>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
'''

PER_PAGE = 20


@app.route('/')
def index():
    ensure_fresh_data()
    predictions_df = _state["predictions_df"]
    page = request.args.get('page', 1, type=int)
    total_pages = max(1, (len(predictions_df) + PER_PAGE - 1) // PER_PAGE)
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    page_items = predictions_df.iloc[start:end]
    rendered = render_template_string(
        HTML,
        page_items=page_items,
        featCols=featCols,
        displayCols=displayCols,
        displayLabels=displayLabels,
        test_acc=_state["test_acc"],
        train_acc=_state["train_acc"],
        total_games=_state["total_games"],
        train_games=_state["train_games"],
        test_games=_state["test_games"],
        page=page,
        total_pages=total_pages,
        range=range,
    )
    response = make_response(rendered)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
