"""NBA Win Predictor - Display model data and performance results."""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(REPO_DIR, "models")

app = Flask(__name__)

# ── Load saved model + metadata ────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "model.joblib")
meta_path  = os.path.join(MODEL_DIR, "metadata.json")
pred_path  = os.path.join(MODEL_DIR, "predictions.csv")   # was saved as .csv (no pyarrow)

model   = joblib.load(model_path)
meta    = pd.read_json(meta_path, orient="records").iloc[0].to_dict()

feat_cols     = meta["feature_cols"]
display_cols  = meta["display_cols"]
display_labels= meta["display_labels"]
test_acc      = meta["test_accuracy"]
train_acc     = meta["train_accuracy"]
total_games   = meta["total_games"]
train_games   = meta["train_games"]
test_games    = meta["test_games"]

# Predictions (newest first)
predictions_df = pd.read_csv(pred_path)

PER_PAGE = 20

# ── HTML template ─────────────────────────────────────────────────────────────
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
        .tagline { color: #888899; font-size: 0.85rem; margin-bottom: 8px; }
        .trained-tag { color: #888899; font-size: 0.75rem; }
        .trained-tag span { color: #aaa; }
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
        .correct-row td { background: rgba(62,207,106,0.08); }
        .incorrect-row td { background: rgba(239,68,68,0.08); }
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
                <p class="trained-tag">Trained <span>{{ trained_at }}</span> &nbsp;·&nbsp; {{ total_games|default(0)|int|commatize }} total games</p>
            </div>
            <a class="dataset-btn" href="https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=TeamStatistics.csv" target="_blank">📂 View Dataset</a>
        </div>

        <div class="card">
            <div class="section-title">📊 Model Performance</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ "%.1f"|format((test_accuracy or 0) * 100) }}%</div>
                    <div class="stat-label">Test Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ (train_games or 0)|int|commatize }}</div>
                    <div class="stat-label">Training Games</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ (test_games or 0)|int|commatize }}</div>
                    <div class="stat-label">Testing Games</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ (total_games or 0)|int|commatize }}</div>
                    <div class="stat-label">Total Games</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-title">🔧 Feature Columns</div>
            <div class="features">
                {% for f in feat_cols %}
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
                        {% for f in display_labels %}
                        <th>{{ f }}</th>
                        {% endfor %}
                        <th>P</th>
                        <th>A</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, row in page_items.iterrows() %}
                    <tr class="{{ 'correct-row' if row.get('correct', False) else 'incorrect-row' }}">
                        <td>{{ str(row.get('gameDate', 'N/A'))[:10] }}</td>
                        <td>{{ row.get('teamName', 'N/A') }}</td>
                        <td>{{ row.get('opponentTeamName', 'N/A') }}</td>
                        <td><span class="badge">{{ 'Home' if row.get('home') == 1 else 'Away' }}</span></td>
                        {% for f in display_cols %}
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

# ── custom jinja filter ────────────────────────────────────────────────────
def commatize(n):
    return f"{int(n):,}"

app.jinja_env.filters["commatize"] = commatize

# ── routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    from flask import request
    page = request.args.get('page', 1, type=int)
    total_pages = (len(predictions_df) + PER_PAGE - 1) // PER_PAGE
    start = (page - 1) * PER_PAGE
    end   = start + PER_PAGE
    page_items = predictions_df.iloc[start:end]

    return render_template_string(
        HTML,
        page_items     = page_items,
        feat_cols      = feat_cols,
        display_cols   = display_cols,
        display_labels = display_labels,
        test_accuracy  = test_acc,
        train_accuracy = train_acc,
        total_games    = total_games,
        train_games    = train_games,
        test_games     = test_games,
        trained_at     = meta.get("trained_at", "unknown"),
        page           = page,
        total_pages    = total_pages,
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)