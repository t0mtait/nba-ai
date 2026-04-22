# NBA Prediction Model

ML-powered NBA moneyline and point spread predictions using team stats, matchup history, and injury data.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000 in your browser.

## What It Predicts

| Bet Type | Description |
|---|---|
| **Moneyline** | Win/loss probability for either team |
| **Point Spread** | Predicted margin + cover probability |

## Input Features

- **Team season stats**: Pace, offensive/defensive rating, effective FG%, turnover rate, free throw rate
- **Head-to-head history**: Win%, average margin, game count between two teams
- **Home court advantage**: Historical ~58% home win rate
- **Rest differential**: Days since last game for each team
- **Injury impact**: Estimated penalty based on out/doubtful players

## Project Structure

```
├── app.py              # FastAPI backend
├── db.py               # SQLite schema (games, team stats, injuries)
├── train_models.py     # Trains moneyline + spread models
├── data_loader.py      # Database access utilities
├── fetch_basketball_ref.py  # Scrapes games from basketball-reference
├── index.html          # Web UI
├── models/             # Trained models
│   ├── ml_model.pkl    # Moneyline model
│   ├── spread_model.pkl # Spread cover model
│   ├── spread_reg.pkl  # Margin regression model
│   └── *_insights.json # Model feature insights
└── nba.db              # SQLite database
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict moneyline + spread for a matchup |
| `/teams` | GET | List all teams in database |
| `/teams/{code}/stats/{year}` | GET | Team season stats |
| `/teams/{code}/injuries` | GET | Current injuries |
| `/matchups/{team}/vs/{opp}` | GET | H2H history |
| `/api/train` | POST | Retrain models |
| `/model-stats` | GET | Model performance insights |

### `/predict` Request Body

```json
{
  "team": "BOS",
  "opponent": "LAL",
  "game_date": "2025-04-10",
  "location": "home"
}
```

### `/predict` Response

```json
{
  "matchup": "BOS (home) vs LAL",
  "moneyline": {
    "win_probability": 0.58,
    "predicted_win": true,
    "recommendation": "BET BOS",
    "confidence": "medium"
  },
  "spread": {
    "predicted_margin": -4.2,
    "predicted_spread": 4.2,
    "cover_probability": 0.62,
    "predicted_cover": true,
    "recommendation": "BET BOS -4",
    "confidence": "medium"
  },
  "raw_features": { ... }
}
```

## Training

```bash
python train_models.py
```

Uses chronological 80/20 train/test split. Last 20% of games held out as test set.

## Data Fetching

```bash
python -c "import db; db.fetch_team_games('BOS', 2020, 2025)"
```

Fetches from basketball-reference.com. Run for all 30 teams to build full dataset.

## Docker

```bash
docker-compose up -d
```