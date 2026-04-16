"""FastAPI backend for Celtics win prediction."""
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
from data_loader import load_games_from_db, FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NBA Win Prediction API",
    description="Predict NBA game outcomes using advanced statistics",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and data paths
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "nba_games.db"))


def _get_model_paths(team_code: str):
    """Get model paths for a team, with fallback to default models."""
    model_home_path = os.path.join(MODELS_DIR, f"model_home_{team_code}.pkl")
    model_away_path = os.path.join(MODELS_DIR, f"model_away_{team_code}.pkl")

    # Fall back to default if team-specific doesn't exist
    if not os.path.exists(model_home_path):
        model_home_path = os.path.join(MODELS_DIR, "model_home.pkl")
    if not os.path.exists(model_away_path):
        model_away_path = os.path.join(MODELS_DIR, "model_away.pkl")

    return model_home_path, model_away_path


def _load_models_for_team(team_code: str):
    """Load models for a specific team."""
    model_home_path, model_away_path = _get_model_paths(team_code)

    try:
        with open(model_home_path, "rb") as f:
            model_home = pickle.load(f)
        with open(model_away_path, "rb") as f:
            model_away = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)
        return model_home, model_away, feature_cols
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found: {e}. Run train_models.py first.")
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")


class PredictionRequest(BaseModel):
    team_code: str = Field(default="BOS", description="Team code (e.g., BOS, LAL)")
    location: str  # "home" or "away"
    pace: float = Field(..., ge=0, le=200, description="Game pace (typical NBA range: 85-105)")
    ftr: float = Field(..., ge=0, le=1, description="Free Throw Rate (0-1)")
    efg_pct: float = Field(..., ge=0, le=1, description="Effective FG% (0-1)")
    tov_pct: float = Field(..., ge=0, le=1, description="Turnover % (0-1)")
    orb_pct: float = Field(..., ge=0, le=1, description="Offensive Rebound % (0-1)")

    @field_validator("team_code")
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


class TeamInfo(BaseModel):
    code: str
    name: str
    fetched_games_count: int
    games_in_db: int
    wins: int
    losses: int
    win_pct: float
    last_fetched: Optional[str]


# ----- ROUTES -----

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    """Health check endpoint - reports whether models are loaded."""
    try:
        # Check that model files exist
        model_home_path = os.path.join(MODELS_DIR, "model_home.pkl")
        model_away_path = os.path.join(MODELS_DIR, "model_away.pkl")
        models_exist = os.path.exists(model_home_path) and os.path.exists(model_away_path)
        return {
            "status": "healthy" if models_exist else "unhealthy",
            "models_loaded": models_exist,
            "database": os.path.exists(DATABASE_PATH),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "models_loaded": False,
            "error": str(e),
            "database": os.path.exists(DATABASE_PATH),
        }


@app.get("/teams", response_model=list[dict])
async def list_teams():
    """List all teams in the database with their stats."""
    try:
        teams = db.get_teams()
        return teams
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams/{team_code}", response_model=dict)
async def get_team(team_code: str):
    """Get info for a specific team."""
    team = db.get_team_info(team_code.upper())
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_code} not found in database")
    return team


@app.post("/teams/{team_code}/fetch")
async def fetch_team_games(
    team_code: str,
    request: FetchRequest,
    background_tasks: BackgroundTasks,
):
    """
    Fetch games from basketball-reference.com for a team.

    This runs in the background and returns immediately with a task ID.
    """
    from fetch_basketball_ref import NBA_TEAMS

    team_code = team_code.upper()
    if team_code not in NBA_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown team code: {team_code}. Valid codes: {list(NBA_TEAMS.keys())}",
        )

    # Check if already being fetched (simple check)
    team_info = db.get_team_info(team_code)
    if team_info and team_info.get("last_fetched"):
        # Already fetched, but allow re-fetch
        pass

    try:
        # Fetch synchronously for now (could be made async)
        result = db.fetch_team_games(
            team_code=team_code,
            start_year=request.start_year,
            end_year=request.end_year,
            force_refresh=request.force_refresh,
        )

        return {
            "status": "complete",
            "team": result["team"],
            "team_name": result["team_name"],
            "total_games_fetched": result["total_games_fetched"],
            "total_games_saved": result["total_games_saved"],
            "seasons": result["seasons"],
        }
    except Exception as e:
        logger.error(f"Error fetching games for {team_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams/{team_code}/games")
async def get_team_games(
    team_code: str,
    season_year: Optional[int] = None,
    limit: Optional[int] = None,
):
    """Get games for a specific team."""
    team_code = team_code.upper()

    try:
        games = db.get_games(team=team_code, season_year=season_year, limit=limit)
        return {
            "team": team_code,
            "count": len(games),
            "games": games,
        }
    except Exception as e:
        logger.error(f"Error fetching games for {team_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict game outcome based on game stats."""
    try:
        model_home, model_away, feature_cols = _load_models_for_team(request.team_code)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    location = request.location.lower()
    if location == "home":
        model = model_home
        cols = feature_cols.get("home", FEATURE_COLUMNS)
    else:
        model = model_away
        cols = feature_cols.get("away", FEATURE_COLUMNS)

    # Build feature vector
    features = pd.DataFrame([[
        getattr(request, col, 0) for col in cols
    ]], columns=cols)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        "team_code": request.team_code,
        "location": request.location,
        "win_prediction": bool(prediction),
        "win_probability": float(probability[1]),
        "loss_probability": float(probability[0]),
    }


@app.get("/game-stats")
async def game_stats(team_code: str = Query(default="BOS")):
    """Get game-by-game prediction statistics for a team."""
    try:
        team_preds_path = os.path.join(MODELS_DIR, f"{team_code}_predictions.csv")
        default_preds_path = os.path.join(MODELS_DIR, "game_predictions.csv")

        if os.path.exists(team_preds_path):
            preds_path = team_preds_path
            used_fallback = False
        elif os.path.exists(default_preds_path):
            preds_path = default_preds_path
            used_fallback = True
        else:
            raise HTTPException(
                status_code=404,
                detail="No game predictions found. Train models first."
            )

        game_predictions = pd.read_csv(preds_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    # Round float columns
    float_cols = ["pace", "ftr", "efg_pct", "tov_pct", "orb_pct", "win_prob"]
    for col in float_cols:
        if col in recent_games.columns:
            recent_games[col] = recent_games[col].round(3)

    recent_games = recent_games.where(pd.notna(recent_games), None)
    recent_games = recent_games.to_dict("records")

    # Load saved insights for this team
    insights = []
    insights_path = os.path.join(MODELS_DIR, f"{team_code}_insights.json")
    if os.path.exists(insights_path):
        try:
            with open(insights_path) as f:
                insights_data = json.load(f)
                insights = insights_data.get("insights", [])
        except Exception:
            pass

    return {
        "overall_accuracy": float(overall_accuracy),
        "home_accuracy": float(home_accuracy),
        "away_accuracy": float(away_accuracy),
        "total_games": len(game_predictions),
        "home_games_count": len(home_games),
        "away_games_count": len(away_games),
        "recent_games": recent_games,
        "used_fallback": used_fallback,
        "team_code": team_code,
        "insights": insights,
    }


@app.get("/db-stats")
async def db_stats():
    """Get statistics from the database."""
    try:
        teams = db.get_teams()
        total_games = sum(t.get("games_in_db", 0) for t in teams)

        return {
            "total_teams": len(teams),
            "total_games": total_games,
            "teams": teams,
        }
    except Exception as e:
        logger.error(f"Error fetching DB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TrainRequest(BaseModel):
    team_code: str = Field(default="BOS", max_length=10)
    season_year: Optional[int] = None


@app.post("/api/train")
async def train_team(body: TrainRequest):
    """
    Train prediction models for a team.

    Uses chronological split: last 20 games are held out as test,
    all older games are used for training.

    Returns training and test accuracy metrics.
    """
    team_code = body.team_code.upper()
    logger.info(f"Training models for {team_code} (season={body.season_year})")

    try:
        # Check if we have games for this team
        games = load_games_from_db(team_code=team_code, season_year=body.season_year)
        if games.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No games found in database for {team_code}. Fetch data first."
            )

        # Run training (CPU-bound, runs in-process)
        result = tm.train_models(
            team_code=team_code,
            season_year=body.season_year,
            output_dir=MODELS_DIR,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(f"Training complete for {team_code}: {result}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training {team_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
