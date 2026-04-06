"""FastAPI backend for Celtics win prediction."""
import os
from contextlib import suppress

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
import pandas as pd

import pickle

app = FastAPI()

# Model and data paths
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

# Global model containers (loaded lazily to allow health checks even if models are missing)
_model_home = None
_model_away = None
_feature_cols = None
_game_predictions = None
_load_error = None


def _load_models():
    """Load models and game predictions from disk."""
    global _model_home, _model_away, _feature_cols, _game_predictions, _load_error
    if _load_error is not None:
        raise _load_error

    try:
        with open(os.path.join(MODELS_DIR, "model_home.pkl"), "rb") as f:
            _model_home = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "model_away.pkl"), "rb") as f:
            _model_away = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
            _feature_cols = pickle.load(f)
        _game_predictions = pd.read_csv(os.path.join(MODELS_DIR, "game_predictions.csv"))
        _load_error = None
    except FileNotFoundError as e:
        _load_error = RuntimeError(f"Model file not found: {e}. Run train_models.py first.")
        raise _load_error from e
    except Exception as e:
        _load_error = RuntimeError(f"Failed to load models: {e}")
        raise _load_error from e


def _get_model_home():
    if _model_home is None:
        _load_models()
    return _model_home


def _get_model_away():
    if _model_away is None:
        _load_models()
    return _model_away


def _get_feature_cols():
    if _feature_cols is None:
        _load_models()
    return _feature_cols


def _get_game_predictions():
    global _game_predictions
    if _game_predictions is None:
        _load_models()
    return _game_predictions


class PredictionRequest(BaseModel):
    location: str  # "home" or "away"
    pace: float = Field(..., ge=0, le=200, description="Game pace (typical NBA range: 85-105)")
    ftr: float = Field(..., ge=0, le=1, description="Free Throw Rate (0-1)")
    efg_pct: float = Field(..., ge=0, le=1, description="Effective FG% (0-1)")
    tov_pct: float = Field(..., ge=0, le=1, description="Turnover % (0-1)")
    orb_pct: float = Field(..., ge=0, le=1, description="Offensive Rebound % (0-1)")

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        if v.lower() not in ("home", "away"):
            raise ValueError("location must be 'home' or 'away'")
        return v.lower()


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    """Health check endpoint - reports whether models are loaded."""
    try:
        _load_models()
        return {"status": "healthy", "models_loaded": True}
    except RuntimeError as e:
        return {"status": "unhealthy", "models_loaded": False, "error": str(e)}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict Celtics win based on game stats."""
    try:
        feature_cols = _get_feature_cols()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if request.location.lower() == "home":
        model = _get_model_home()
        cols = feature_cols["home"]
    else:
        model = _get_model_away()
        cols = feature_cols["away"]

    # Validate that request has all required feature columns
    missing = [col for col in cols if not hasattr(request, col) or getattr(request, col) is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields for {request.location} model: {missing}"
        )

    features = pd.DataFrame([[getattr(request, col) for col in cols]], columns=cols)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        "location": request.location,
        "win_prediction": bool(prediction),
        "win_probability": float(probability[1]),
        "loss_probability": float(probability[0])
    }


@app.get("/game-stats")
async def game_stats():
    """Get game-by-game prediction statistics."""
    try:
        game_predictions = _get_game_predictions()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if game_predictions.empty:
        raise HTTPException(status_code=404, detail="No game predictions found")

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

    # Round float columns to avoid floating-point precision artifacts
    float_cols = ["pace", "ftr", "efg_pct", "tov_pct", "orb_pct", "win_prob"]
    for col in float_cols:
        if col in recent_games.columns:
            recent_games[col] = recent_games[col].round(3)

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
