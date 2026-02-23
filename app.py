"""FastAPI backend for Celtics win prediction."""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load models
with open("models/model_home.pkl", "rb") as f:
    model_home = pickle.load(f)

with open("models/model_away.pkl", "rb") as f:
    model_away = pickle.load(f)

with open("models/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Load game predictions
game_predictions = pd.read_csv("models/game_predictions.csv")


class PredictionRequest(BaseModel):
    location: str  # "home" or "away"
    pace: float
    ftr: float
    efg_pct: float
    tov_pct: float
    orb_pct: float


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict Celtics win based on game stats."""
    if request.location.lower() == "home":
        model = model_home
        cols = feature_cols["home"]
    else:
        model = model_away
        cols = feature_cols["away"]

    features = [getattr(request, col) for col in cols]
    
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    return {
        "location": request.location,
        "win_prediction": bool(prediction),
        "win_probability": float(probability[1]),
        "loss_probability": float(probability[0])
    }


@app.get("/game-stats")
async def game_stats():
    """Get game-by-game prediction statistics."""
    # Calculate overall stats
    home_games = game_predictions[game_predictions["location"] == "home"]
    away_games = game_predictions[game_predictions["location"] == "away"]
    
    overall_accuracy = (game_predictions["correct"].sum() / len(game_predictions) * 100)
    home_accuracy = (home_games["correct"].sum() / len(home_games) * 100) if len(home_games) > 0 else 0
    away_accuracy = (away_games["correct"].sum() / len(away_games) * 100) if len(away_games) > 0 else 0
    
    # Get recent games sorted by date (most recent first)
    recent_games = game_predictions.copy()
    recent_games["date"] = pd.to_datetime(recent_games["date"], errors="coerce")
    recent_games = recent_games.sort_values("date", ascending=False).head(20)
    recent_games["date"] = recent_games["date"].dt.strftime("%Y-%m-%d")
    recent_games = recent_games.where(pd.notna(recent_games), None)
    recent_games = recent_games.to_dict("records")
    
    return {
        "overall_accuracy": float(overall_accuracy),
        "home_accuracy": float(home_accuracy),
        "away_accuracy": float(away_accuracy),
        "total_games": len(game_predictions),
        "home_games_count": len(home_games),
        "away_games_count": len(away_games),
        "recent_games": recent_games
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
