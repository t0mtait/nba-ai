"""FastAPI backend for NBA tonight's games predictions."""
from __future__ import annotations

import os
import logging
import pickle
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

import db
import train_models as tm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NBA Tonight's Games", description="ML-powered predictions")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "nba.db"))


def _load_model(name: str) -> dict:
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---- Pydantic models ----

class PowerRankingEntry(BaseModel):
    team: str
    rank: int
    power_rating: float = 0.0
    net_rtg: float = 0.0
    ortg_override: Optional[float] = None
    drtg_override: Optional[float] = None
    notes: str = ""


class PowerRankingRequest(BaseModel):
    season_year: int
    entries: list[PowerRankingEntry]


class InjuryRequest(BaseModel):
    team: str
    player_name: str
    injury: str = ""
    status: str = "questionable"
    date_reported: str = ""
    game_date: str = ""
    season_year: Optional[int] = None


# ---- Routes ----

@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    ml = os.path.exists(os.path.join(MODELS_DIR, "ml_model.pkl"))
    sp = os.path.exists(os.path.join(MODELS_DIR, "spread_model.pkl"))
    return {"status": "healthy" if (ml and sp) else "unhealthy",
             "moneyline_model": ml, "spread_model": sp,
             "database": os.path.exists(DATABASE_PATH)}


# --- Tonight's Games ---

@app.get("/tonight")
async def get_tonight():
    try:
        games = db.get_tonight_games()
        return {"games": games, "count": len(games)}
    except Exception as e:
        logger.error(f"Error fetching tonight's games: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
async def get_predictions():
    try:
        games = db.get_tonight_games()
    except Exception as e:
        logger.error(f"Error fetching today's games: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not games:
        return {"predictions": [], "count": 0}

    try:
        ml_data = _load_model("ml_model.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not trained. POST /api/train first.")

    sp_data, reg_data = None, None
    try:
        sp_data = _load_model("spread_model.pkl")
        reg_data = _load_model("spread_reg.pkl")
    except FileNotFoundError:
        pass

    predictions = []
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["game_date"]
        game_id = game["game_id"]
        game_time = game.get("game_time", "")
        status_id = game.get("status_id", 1)

        if status_id == 3:
            predictions.append({
                "game_id": game_id, "game_date": game_date, "game_time": game_time,
                "home_team": home_team, "away_team": away_team, "status": "completed",
                "home_score": game.get("home_score"), "away_score": game.get("away_score"),
            })
            continue

        # Home perspective
        home_feats = tm.build_features_for_game(home_team, away_team, game_date, "home")
        home_vec = pd.DataFrame([home_feats])[ml_data["feature_cols"]].fillna(0)
        home_vec_s = ml_data["scaler"].transform(home_vec)
        home_win_prob = float(ml_data["model"].predict_proba(home_vec_s)[0][1])
        home_pred_margin = float(reg_data["model"].predict(home_vec_s)[0]) if reg_data else 0.0
        home_cover_prob = float(sp_data["model"].predict_proba(home_vec_s)[0][1]) if sp_data else 0.5

        # Away perspective
        away_feats = tm.build_features_for_game(away_team, home_team, game_date, "away")
        away_vec = pd.DataFrame([away_feats])[ml_data["feature_cols"]].fillna(0)
        away_vec_s = ml_data["scaler"].transform(away_vec)
        away_win_prob = float(ml_data["model"].predict_proba(away_vec_s)[0][1])

        pred_spread = round(-home_pred_margin, 1)
        pred_spread = max(-15, min(15, pred_spread))

        def ml_rec(team, prob):
            if prob > 0.55:
                return f"BET {team}", "medium" if prob < 0.65 else "high"
            elif prob < 0.45:
                return f"BET {team}", "medium" if prob > 0.35 else "high"
            return "PASS", None

        home_ml_rec, home_ml_conf = ml_rec(home_team, home_win_prob)
        away_ml_rec, away_ml_conf = ml_rec(away_team, away_win_prob)

        if sp_data:
            def spread_rec(home_prob):
                if home_prob > 0.55:
                    return f"BET {home_team} -{abs(pred_spread)}", "medium" if home_prob < 0.65 else "high"
                elif home_prob < 0.45:
                    return f"BET {away_team} +{abs(pred_spread)}", "medium" if home_prob > 0.35 else "high"
                return "PASS", None
            sp_rec_str, sp_conf = spread_rec(home_cover_prob)
        else:
            sp_rec_str, sp_conf = "N/A", None

        predictions.append({
            "game_id": game_id, "game_date": game_date, "game_time": game_time,
            "status": "scheduled", "home_team": home_team, "away_team": away_team,
            "moneyline": {
                "home": {"win_probability": round(home_win_prob, 3), "predicted_win": home_win_prob > 0.5,
                         "recommendation": home_ml_rec, "confidence": home_ml_conf},
                "away": {"win_probability": round(away_win_prob, 3), "predicted_win": away_win_prob > 0.5,
                         "recommendation": away_ml_rec, "confidence": away_ml_conf},
            },
            "spread": {
                "predicted_spread": pred_spread,
                "home_cover_probability": round(home_cover_prob, 3),
                "recommendation": sp_rec_str,
                "confidence": sp_conf,
            } if sp_data else None,
        })

    return {"predictions": predictions, "count": len(predictions)}


# --- Training ---

@app.post("/api/train")
async def train_models():
    logger.info("Training models...")
    try:
        result = tm.train_models(output_dir=MODELS_DIR, db_path=DATABASE_PATH)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"status": "complete", "results": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest")
async def backtest():
    """Backtest the model on historical games."""
    try:
        result = tm.backtest(db_path=DATABASE_PATH, models_dir=MODELS_DIR)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Power Rankings ---

@app.get("/api/power-rankings")
async def get_power_rankings():
    """Get all power rankings."""
    rankings = db.get_all_power_rankings(db_path=DATABASE_PATH)
    return {"rankings": rankings, "count": len(rankings)}


@app.get("/api/power-rankings/{season_year}")
async def get_power_ranking(season_year: int):
    """Get latest power ranking for a season."""
    ranking = db.get_latest_power_ranking(season_year, db_path=DATABASE_PATH)
    if not ranking:
        return {"season_year": season_year, "entries": [], "message": "No power ranking found"}
    return ranking


@app.post("/api/power-rankings")
async def save_power_ranking(req: PowerRankingRequest):
    """Save a power ranking for a season."""
    try:
        entries = [e.model_dump() for e in req.entries]
        ranking_id = db.save_power_ranking(req.season_year, entries)
        return {"status": "saved", "ranking_id": ranking_id, "season_year": req.season_year,
                "count": len(entries)}
    except Exception as e:
        logger.error(f"Error saving power ranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/power-rankings/{season_year}")
async def delete_power_ranking(season_year: int):
    """Delete the latest power ranking for a season."""
    conn = db.get_connection(db_path=DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM power_ranking_entries
        WHERE ranking_id IN (SELECT id FROM power_rankings WHERE season_year = ?)
    """, (season_year,))
    cur.execute("DELETE FROM power_rankings WHERE season_year = ?", (season_year,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "season_year": season_year}


# --- Injuries ---

@app.get("/api/injuries")
async def get_injuries(team: Optional[str] = None):
    """Get injury reports, optionally filtered by team."""
    injuries = db.get_injuries(team=team, db_path=DATABASE_PATH)
    return {"injuries": injuries, "count": len(injuries)}


@app.post("/api/injuries")
async def add_injury(req: InjuryRequest):
    """Add an injury report."""
    try:
        year = req.season_year or int(req.date_reported[:4]) if req.date_reported else None
        saved = db.save_injuries([req.model_dump()], req.team, db_path=DATABASE_PATH)
        return {"status": "saved", "team": req.team, "saved": saved}
    except Exception as e:
        logger.error(f"Error saving injury: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/injuries/{injury_id}")
async def delete_injury(injury_id: int):
    """Delete an injury record."""
    conn = db.get_connection(db_path=DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM injuries WHERE id = ?", (injury_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "id": injury_id}


@app.post("/api/injuries/fetch")
async def fetch_injuries():
    """
    Fetch current NBA injury reports from ESPN and save to database.
    Auto-maps team names to codes and stores all active injuries.
    """
    try:
        injuries = db.fetch_injuries_from_espn()
        if not injuries:
            return {"status": "no_data", "saved": 0, "message": "No injuries found or fetch failed"}

        # Group by team and save
        from collections import defaultdict
        by_team = defaultdict(list)
        for inj in injuries:
            by_team[inj['team']].append(inj)

        total_saved = 0
        for team_code, team_injuries in by_team.items():
            saved = db.save_injuries(team_injuries, team_code, db_path=DATABASE_PATH)
            total_saved += saved

        return {
            "status": "complete",
            "fetched": len(injuries),
            "saved": total_saved,
            "teams_updated": list(by_team.keys()),
        }
    except Exception as e:
        logger.error(f"Error fetching injuries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/injuries/{injury_id}")
async def update_injury_will_play(injury_id: int, will_play: int = 1):
    """
    Update the will_play flag for an injury record.
    will_play=1: user predicts player WILL play despite injury (no impact penalty)
    will_play=0: user predicts player WILL miss (full impact penalty applies)
    """
    conn = db.get_connection(db_path=DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE injuries SET will_play = ? WHERE id = ?", (will_play, injury_id))
    conn.commit()
    rows_affected = cur.rowcount
    conn.close()
    if rows_affected == 0:
        raise HTTPException(status_code=404, detail="Injury record not found")
    return {"status": "updated", "id": injury_id, "will_play": will_play}


# --- H2H ---

@app.get("/api/h2h/{team}/vs/{opponent}")
async def get_h2h(team: str, opponent: str, limit: int = 20):
    """Get head-to-head history."""
    cur_season = 2026  # this will be dynamic in real usage
    games = db.get_recent_matchup(team.upper(), opponent.upper(), n=limit, db_path=DATABASE_PATH)
    # Split into this season and last season
    this_s = [g for g in games if g.get("season_year") == cur_season]
    last_s = [g for g in games if g.get("season_year") == cur_season - 1]
    return {
        "matchup": f"{team.upper()} vs {opponent.upper()}",
        "total_games": len(games),
        "this_season": this_s, "last_season": last_s,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)