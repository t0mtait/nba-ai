"""FastAPI backend for NBA moneyline and spread predictions."""
from __future__ import annotations

import os
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import pickle

import db
import train_models as tm
from data_loader import load_games_from_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NBA Prediction API",
    description="Predict NBA moneyline and point spread outcomes for any matchup",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "nba.db"))


def _load_model(name: str) -> dict:
    """Load a model + scaler bundle from disk."""
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Run train_models.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_or_compute_features(
    team: str,
    opponent: str,
    game_date: str,
    location: str,
    db_path: Optional[str] = None,
) -> dict:
    """Get cached features or build them on the fly."""
    return tm.build_features_for_game(team, opponent, game_date, location, db_path)


# ----- Pydantic models -----

class PredictionRequest(BaseModel):
    team: str = Field(..., description="Team code (e.g. BOS, LAL)")
    opponent: str = Field(..., description="Opponent team code")
    game_date: str = Field(..., description="Game date (YYYY-MM-DD)")
    location: str = Field(..., description="'home' or 'away'")

    @field_validator("team", "opponent")
    @classmethod
    def validate_team_code(cls, v: str) -> str:
        return v.upper()

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        if v.lower() not in ("home", "away"):
            raise ValueError("location must be 'home' or 'away'")
        return v.lower()


class FetchRequest(BaseModel):
    start_year: int = Field(default=2017, ge=2000, le=2030)
    end_year: int = Field(default=2026, ge=2000, le=2030)
    force_refresh: bool = Field(default=False)


# ----- Routes -----

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    ml_exists = os.path.exists(os.path.join(MODELS_DIR, "ml_model.pkl"))
    sp_exists = os.path.exists(os.path.join(MODELS_DIR, "spread_model.pkl"))
    return {
        "status": "healthy" if (ml_exists and sp_exists) else "unhealthy",
        "moneyline_model": ml_exists,
        "spread_model": sp_exists,
        "database": os.path.exists(DATABASE_PATH),
    }


@app.get("/teams")
async def list_teams():
    try:
        return db.get_teams()
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams/{team_code}")
async def get_team(team_code: str):
    team = db.get_team_info(team_code.upper())
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_code} not found")
    return team


@app.get("/teams/{team_code}/stats/{season_year}")
async def get_team_stats(team_code: str, season_year: int):
    stats = db.get_team_stats(team_code.upper(), season_year)
    if not stats:
        raise HTTPException(status_code=404, detail=f"No stats found for {team_code} in {season_year}")
    return stats


@app.get("/teams/{team_code}/injuries")
async def get_team_injuries(team_code: str):
    injuries = db.get_injuries(team=team_code.upper())
    return {"team": team_code.upper(), "injuries": injuries}


@app.get("/matchups/{team}/vs/{opponent}")
async def get_matchup_history(team: str, opponent: str, limit: int = Query(default=20)):
    team = team.upper()
    opponent = opponent.upper()
    games = db.get_recent_matchup(team, opponent, n=limit)
    return {
        "matchup": f"{team} vs {opponent}",
        "team": team,
        "opponent": opponent,
        "count": len(games),
        "games": games,
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict moneyline and spread for a given matchup.
    Returns win probability, predicted margin, and recommended bet.
    """
    team = request.team.upper()
    opponent = request.opponent.upper()
    location = request.location.lower()

    # Load models
    try:
        ml_data = _load_model("ml_model.pkl")
        sp_data = _load_model("spread_model.pkl")
        reg_data = _load_model("spread_reg.pkl")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Models not loaded: {e}. Run train_models.py first.",
        )

    # Build features
    features = _get_or_compute_features(team, opponent, request.game_date, location)
    feature_vec = pd.DataFrame([features])[ml_data["feature_cols"]].fillna(0)
    feature_vec_scaled = ml_data["scaler"].transform(feature_vec)

    # Moneyline prediction
    ml_model = ml_data["model"]
    win_prob = float(ml_model.predict_proba(feature_vec_scaled)[0][1])
    predicted_win = bool(ml_model.predict(feature_vec_scaled)[0])

    # Spread regression (predicted margin in points)
    reg = reg_data["model"]
    predicted_margin = float(reg.predict(feature_vec_scaled)[0])

    # Spread classification (will team cover?)
    sp_model = sp_data["model"]
    cover_prob = float(sp_model.predict_proba(feature_vec_scaled)[0][1])
    predicted_cover = bool(sp_model.predict(feature_vec_scaled)[0])

    # Adjust for home/away
    if location == "home":
        predicted_spread = round(-predicted_margin, 1)
    else:
        predicted_spread = round(predicted_margin, 1)

    # Clamp spread to typical NBA range (-15 to +15)
    predicted_spread = max(-15, min(15, predicted_spread))

    # Moneyline recommendation
    if win_prob > 0.55:
        ml_rec = f"BET {team}"
        ml_conf = "high" if win_prob > 0.65 else "medium"
    elif win_prob < 0.45:
        ml_rec = f"BET {opponent}"
        ml_conf = "high" if win_prob < 0.35 else "medium"
    else:
        ml_rec = "PASS"
        ml_conf = None

    # Spread recommendation
    if cover_prob > 0.55:
        if location == "home":
            sp_rec = f"BET {team} -{abs(predicted_spread)}"
        else:
            sp_rec = f"BET {team} +{abs(predicted_spread)}"
        sp_conf = "high" if cover_prob > 0.65 else "medium"
    elif cover_prob < 0.45:
        if location == "home":
            sp_rec = f"BET {opponent} +{abs(predicted_spread)}"
        else:
            sp_rec = f"BET {opponent} -{abs(predicted_spread)}"
        sp_conf = "high" if cover_prob < 0.35 else "medium"
    else:
        sp_rec = "PASS"
        sp_conf = None

    return {
        "matchup": f"{team} ({'home' if location == 'home' else 'away'}) vs {opponent}",
        "game_date": request.game_date,
        "location": location,
        "moneyline": {
            "win_probability": win_prob,
            "predicted_win": predicted_win,
            "recommendation": ml_rec,
            "confidence": ml_conf,
        },
        "spread": {
            "predicted_margin": round(predicted_margin, 1),
            "predicted_spread": predicted_spread,
            "cover_probability": cover_prob,
            "predicted_cover": predicted_cover,
            "recommendation": sp_rec,
            "confidence": sp_conf,
        },
        "features_used": list(ml_data["feature_cols"]),
        "raw_features": {k: round(v, 3) if isinstance(v, float) else v for k, v in features.items()},
    }


@app.get("/model-stats")
async def model_stats():
    """Get trained model performance metrics."""
    results = {}
    for name, label in [("ml_model.pkl", "moneyline"), ("spread_model.pkl", "spread")]:
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            continue
        try:
            with open(os.path.join(MODELS_DIR, f"{label}_insights.json")) as f:
                insights = json.load(f).get("insights", [])
            results[label] = {"insights": insights}
        except Exception:
            results[label] = {}
    return results


@app.post("/teams/{team_code}/fetch")
async def fetch_team_games(team_code: str, request: FetchRequest, background_tasks: BackgroundTasks):
    """Fetch games for a team from basketball-reference.com."""
    from fetch_basketball_ref import NBA_TEAMS

    team_code = team_code.upper()
    if team_code not in NBA_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown team code: {team_code}. Valid: {list(NBA_TEAMS.keys())}",
        )

    try:
        result = db.fetch_team_games(
            team_code=team_code,
            start_year=request.start_year,
            end_year=request.end_year,
            force_refresh=request.force_refresh,
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching games for {team_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams/{team_code}/games")
async def get_team_games(
    team_code: str,
    season_year: Optional[int] = None,
    limit: Optional[int] = None,
):
    team_code = team_code.upper()
    try:
        games = db.get_games(team=team_code, season_year=season_year, limit=limit)
        return {"team": team_code, "count": len(games), "games": games}
    except Exception as e:
        logger.error(f"Error fetching games for {team_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/db-stats")
async def db_stats():
    try:
        teams = db.get_teams()
        total_games = sum(t.get("games_in_db", 0) for t in teams)
        return {"total_teams": len(teams), "total_games": total_games, "teams": teams}
    except Exception as e:
        logger.error(f"Error fetching DB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TrainRequest(BaseModel):
    pass


@app.post("/api/train")
async def train_models():
    """
    Retrain both moneyline and spread models.
    Uses all available game data in the database.
    """
    logger.info("Training prediction models...")

    try:
        result = tm.train_models(output_dir=MODELS_DIR, db_path=DATABASE_PATH)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(f"Training complete: {result}")
        return {"status": "complete", "results": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)