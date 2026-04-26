"""Fetch all NBA teams data and train models."""
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NBA_TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

import db
from fetch_basketball_ref import fetch_team_season

def fetch_all_teams(start_year=2017, end_year=2026):
    """Fetch all 30 teams across all seasons."""
    db.init_db()

    total_saved = 0
    for team_code in NBA_TEAMS:
        logger.info(f"Fetching {team_code} ({NBA_TEAMS[team_code]})...")
        for year in range(start_year, end_year + 1):
            try:
                games = fetch_team_season(team_code, year, force_refresh=False)
                if games:
                    saved = db.save_games(games, team_code, year)
                    total_saved += saved
                    logger.info(f"  {team_code}/{year}: {saved} games saved")
            except Exception as e:
                logger.error(f"  Error fetching {team_code}/{year}: {e}")
    return total_saved

if __name__ == "__main__":
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2017
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    print(f"Fetching seasons {start_year}-{end_year} for all {len(NBA_TEAMS)} teams...")
    total = fetch_all_teams(start_year, end_year)
    print(f"\nTotal games saved: {total}")
    if total > 0:
        print("\nTraining models...")
        import train_models as tm
        result = tm.train_models(output_dir="models", db_path=db.DB_PATH)
        print(f"Training complete: {result}")