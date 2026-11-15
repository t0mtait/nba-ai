"""NBA Win Predictor - Display model data and performance results."""
import json
import os
import shutil
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ymd}"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"

NBA_TEAM_IDS = {
    "Hawks": 1610612737,
    "Celtics": 1610612738,
    "Cavaliers": 1610612739,
    "Pelicans": 1610612740,
    "Bulls": 1610612741,
    "Mavericks": 1610612742,
    "Nuggets": 1610612743,
    "Warriors": 1610612744,
    "Rockets": 1610612745,
    "Clippers": 1610612746,
    "Lakers": 1610612747,
    "Heat": 1610612748,
    "Bucks": 1610612749,
    "Timberwolves": 1610612750,
    "Nets": 1610612751,
    "Knicks": 1610612752,
    "Magic": 1610612753,
    "Pacers": 1610612754,
    "76ers": 1610612755,
    "Suns": 1610612756,
    "Trail Blazers": 1610612757,
    "Kings": 1610612758,
    "Spurs": 1610612759,
    "Thunder": 1610612760,
    "Raptors": 1610612761,
    "Jazz": 1610612762,
    "Grizzlies": 1610612763,
    "Wizards": 1610612764,
    "Pistons": 1610612765,
    "Hornets": 1610612766,
}

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
    "trained_at": "unknown",
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


def _parse_made_attempted(val: str) -> float:
    """Extract attempted count from ESPN 'made-attempted' strings (e.g. '40-88' -> 88.0)."""
    parts = str(val).split("-")
    return float(parts[1]) if len(parts) == 2 else float(parts[0])


def _append_recent_espn_games(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Supplement Kaggle historical data with any newer completed NBA games from ESPN."""
    try:
        team_map = dict(NBA_TEAM_IDS)
        if "teamName" in df.columns and "teamId" in df.columns:
            from_df = (
                df[["teamName", "teamId"]]
                .dropna()
                .drop_duplicates(subset=["teamName"], keep="first")
                .set_index("teamName")["teamId"]
                .to_dict()
            )
            team_map.update(from_df)

        existing_keys = set(
            zip(
                df[date_col].astype(str).str[:10],
                df["teamName"].astype(str),
                df["opponentTeamName"].astype(str),
            )
        )

        max_date_str = df[date_col].dropna().astype(str).str[:10].max()
        ny_tz = ZoneInfo("America/New_York")
        today_est = datetime.now(ny_tz).date()
        start_date = (
            datetime.strptime(max_date_str, "%Y-%m-%d").date()
            if isinstance(max_date_str, str) and len(max_date_str) == 10
            else today_est - timedelta(days=7)
        )
        # Bound lookback to at most 14 days so a corrupt date never causes a runaway loop
        if (today_est - start_date).days > 14:
            start_date = today_est - timedelta(days=14)

        new_rows = []
        cur = start_date
        headers = {"User-Agent": "Mozilla/5.0"}
        while cur <= today_est:
            ymd = cur.strftime("%Y%m%d")
            cur += timedelta(days=1)
            try:
                req = urllib.request.Request(ESPN_SCOREBOARD_URL.format(ymd=ymd), headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    events = json.loads(resp.read().decode()).get("events", [])
            except Exception as exc:
                app.logger.warning("Failed to fetch ESPN scoreboard for %s: %s", ymd, exc)
                continue

            for ev in events:
                try:
                    if not ev.get("status", {}).get("type", {}).get("completed"):
                        continue
                    comp = (ev.get("competitions") or [{}])[0]
                    competitors = comp.get("competitors", [])
                    home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                    away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                    if not home_c or not away_c:
                        continue

                    tname = home_c.get("team", {}).get("name")
                    oname = away_c.get("team", {}).get("name")
                    if not tname or not oname or tname not in team_map:
                        continue

                    dt_utc = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
                    dt_est = dt_utc.astimezone(ny_tz)
                    gdate_full = dt_est.strftime("%Y-%m-%d %H:%M:%S")
                    gdate_short = gdate_full[:10]

                    if (gdate_short, tname, oname) in existing_keys:
                        continue

                    s_req = urllib.request.Request(ESPN_SUMMARY_URL.format(event_id=ev["id"]), headers=headers)
                    with urllib.request.urlopen(s_req, timeout=10) as s_resp:
                        s_data = json.loads(s_resp.read().decode())

                    b_teams = s_data.get("boxscore", {}).get("teams", [])
                    b_home = next((t for t in b_teams if t.get("team", {}).get("name") == tname), None)
                    if not b_home:
                        continue

                    stats = {s["name"]: s["displayValue"] for s in b_home.get("statistics", []) if "name" in s and "displayValue" in s}
                    linescores = home_c.get("linescores", [])
                    if len(linescores) < 2:
                        s_comp = (s_data.get("header", {}).get("competitions") or [{}])[0]
                        s_home = next((c for c in s_comp.get("competitors", []) if c.get("homeAway") == "home"), {})
                        linescores = s_home.get("linescores", [])
                    if len(linescores) < 2:
                        continue

                    q1 = float(linescores[0].get("displayValue", linescores[0].get("value", 0)))
                    q2 = float(linescores[1].get("displayValue", linescores[1].get("value", 0)))
                    fga = _parse_made_attempted(stats["fieldGoalsMade-fieldGoalsAttempted"])
                    tpa = _parse_made_attempted(stats["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
                    fta = _parse_made_attempted(stats["freeThrowsMade-freeThrowsAttempted"])

                    new_rows.append({
                        date_col: gdate_full,
                        "gameDate": gdate_full,
                        "gameDateTimeEst": gdate_full,
                        "teamId": team_map[tname],
                        "teamName": tname,
                        "opponentTeamName": oname,
                        "home": 1,
                        "win": 1.0 if home_c.get("winner") else 0.0,
                        "assists": float(stats["assists"]),
                        "reboundsTotal": float(stats["totalRebounds"]),
                        "blocks": float(stats["blocks"]),
                        "steals": float(stats["steals"]),
                        "turnovers": float(stats.get("turnovers", stats.get("totalTurnovers", 0))),
                        "foulsPersonal": float(stats.get("fouls", stats.get("foulsPersonal", 0))),
                        "q1Points": q1,
                        "q2Points": q2,
                        "fieldGoalsAttempted": fga,
                        "threePointersAttempted": tpa,
                        "freeThrowsAttempted": fta,
                    })
                    existing_keys.add((gdate_short, tname, oname))
                except Exception as exc:
                    app.logger.warning("Failed to parse ESPN event %s: %s", ev.get("id"), exc)

        if new_rows:
            new_rows.sort(key=lambda r: r[date_col], reverse=True)
            df = pd.concat([pd.DataFrame(new_rows), df], ignore_index=True)
    except Exception as exc:
        app.logger.warning("Failed to append recent ESPN games: %s", exc)
    return df


def refresh_data(force_download: bool = True) -> None:
    """Download the latest Kaggle dataset, supplement with recent NBA games, and retrain."""
    with _refresh_lock:
        try:
            path = kagglehub.dataset_download(
                DATASET_HANDLE,
                path=DATASET_FILE,
                force_download=force_download,
            )
        except Exception as exc:
            if not force_download:
                raise
            app.logger.warning("Fresh Kaggle download failed (%s); falling back to cached dataset", exc)
            path = kagglehub.dataset_download(
                DATASET_HANDLE,
                path=DATASET_FILE,
                force_download=False,
            )
        csv_path = os.path.join(path, DATASET_FILE) if os.path.isdir(path) else path
        _cleanup_old_cache_versions(csv_path)

        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df['home'] == 1]
        date_col = 'gameDate' if 'gameDate' in df.columns else 'gameDateTimeEst'
        df = _append_recent_espn_games(df, date_col)

        # Preserve columns before dropna for display
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
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
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

PER_PAGE = 20

# ── custom jinja filters ───────────────────────────────────────────────────
def commatize(n):
    return f"{int(n):,}"

def to_str(val):
    return str(val)

app.jinja_env.filters["commatize"] = commatize
app.jinja_env.filters["to_str"] = to_str

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Win Predictor — AI Game Analytics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&family=Space+Grotesk:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #06080f;
            --bg-surface: rgba(15, 19, 30, 0.78);
            --bg-surface-elevated: rgba(22, 28, 44, 0.85);
            --bg-glass: rgba(255, 255, 255, 0.035);
            --bg-glass-hover: rgba(255, 255, 255, 0.07);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(255, 255, 255, 0.16);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-emerald: #10b981;
            --accent-emerald-light: #34d399;
            --accent-emerald-glow: rgba(16, 185, 129, 0.18);
            --accent-orange: #f97316;
            --accent-orange-light: #fb923c;
            --accent-orange-glow: rgba(249, 115, 22, 0.18);
            --accent-cyan: #06b6d4;
            --accent-cyan-light: #22d3ee;
            --accent-cyan-glow: rgba(6, 182, 212, 0.18);
            --accent-violet: #8b5cf6;
            --accent-violet-light: #a78bfa;
            --accent-violet-glow: rgba(139, 92, 246, 0.18);
            --accent-rose: #f43f5e;
            --accent-rose-light: #fb7185;
            --accent-rose-glow: rgba(244, 63, 94, 0.16);
            --font-display: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --radius-xl: 20px;
            --radius-lg: 14px;
            --radius-md: 10px;
            --radius-sm: 6px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: var(--font-sans);
            background-color: var(--bg-base);
            background-image:
                radial-gradient(circle at 12% 10%, rgba(249, 115, 22, 0.12), transparent 38%),
                radial-gradient(circle at 88% 12%, rgba(16, 185, 129, 0.12), transparent 40%),
                radial-gradient(circle at 50% 85%, rgba(139, 92, 246, 0.09), transparent 45%),
                linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 100% 100%, 40px 40px, 40px 40px;
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            padding: 32px 24px 64px;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); transform: scale(1); }
            50% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); transform: scale(1.08); }
        }

        @keyframes floatBall {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-4px) rotate(8deg); }
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* ── Header / Hero ───────────────────────────────────────────────── */
        .header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 28px;
            padding: 28px 32px;
            background: linear-gradient(135deg, rgba(22, 28, 45, 0.85) 0%, rgba(12, 16, 28, 0.9) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
            animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-orange), var(--accent-emerald), var(--accent-cyan), var(--accent-violet));
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }

        .live-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 4px 11px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--accent-emerald-light);
        }

        .live-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent-emerald-light);
            animation: pulseGlow 2s infinite;
        }

        .model-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 11px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
            background: rgba(249, 115, 22, 0.12);
            border: 1px solid rgba(249, 115, 22, 0.28);
            color: var(--accent-orange-light);
        }

        h1 {
            font-family: var(--font-display);
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.15;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .ball-icon {
            display: inline-block;
            animation: floatBall 3.2s ease-in-out infinite;
            filter: drop-shadow(0 4px 12px rgba(249, 115, 22, 0.4));
        }

        h1 span {
            background: linear-gradient(135deg, var(--accent-emerald-light) 0%, var(--accent-cyan-light) 55%, var(--accent-orange-light) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .tagline {
            color: var(--text-secondary);
            font-size: 0.95rem;
            font-weight: 500;
            margin-top: 4px;
            margin-bottom: 10px;
        }

        .trained-tag {
            color: var(--text-muted);
            font-size: 0.78rem;
            font-family: var(--font-mono);
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            padding: 5px 12px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-subtle);
        }

        .trained-tag span {
            color: var(--text-primary);
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .dataset-btn, .action-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 18px;
            background: var(--bg-glass);
            border: 1px solid var(--border-highlight);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
            font-family: var(--font-sans);
        }

        .dataset-btn:hover, .action-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent-cyan-light);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(6, 182, 212, 0.18);
            color: #fff;
        }

        .action-btn.primary {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.22), rgba(234, 88, 12, 0.12));
            border-color: rgba(249, 115, 22, 0.45);
            color: #fff;
        }

        .action-btn.primary:hover {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.35), rgba(234, 88, 12, 0.22));
            border-color: var(--accent-orange-light);
            box-shadow: 0 8px 24px rgba(249, 115, 22, 0.28);
        }

        .action-btn.active {
            background: linear-gradient(135deg, var(--accent-orange), #ea580c);
            border-color: var(--accent-orange-light);
            color: #fff;
            box-shadow: 0 8px 24px rgba(249, 115, 22, 0.35);
        }

        /* ── Cards ───────────────────────────────────────────────────────── */
        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            padding: 26px 28px;
            margin-bottom: 24px;
            overflow-x: auto;
            backdrop-filter: blur(16px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
            animation: fadeUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        .card:nth-of-type(2) { animation-delay: 0.06s; }
        .card:nth-of-type(3) { animation-delay: 0.12s; }
        .card:nth-of-type(4) { animation-delay: 0.18s; }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.13);
        }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 18px;
            flex-wrap: wrap;
        }

        .section-title {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-primary);
            margin-bottom: 0;
        }

        .section-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 9px;
            font-size: 0.95rem;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-subtle);
        }

        .section-subtitle {
            font-size: 0.78rem;
            color: var(--text-secondary);
        }

        /* ── Stats Bento Grid ────────────────────────────────────────────── */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }

        .stat-card {
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.045) 0%, rgba(255, 255, 255, 0.015) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 20px 18px;
            position: relative;
            overflow: hidden;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease;
            text-align: left;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            border-radius: 3px 3px 0 0;
        }

        .stat-card.emerald::before { background: linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan)); }
        .stat-card.cyan::before { background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)); }
        .stat-card.violet::before { background: linear-gradient(90deg, var(--accent-violet), #ec4899); }
        .stat-card.orange::before { background: linear-gradient(90deg, var(--accent-orange), #eab308); }

        .stat-card:hover {
            transform: translateY(-4px);
            border-color: var(--border-highlight);
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.4);
        }

        .stat-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .stat-label {
            font-size: 0.72rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.7px;
        }

        .stat-badge {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            padding: 2px 8px;
            border-radius: 999px;
            font-weight: 600;
        }

        .stat-card.emerald .stat-badge { background: var(--accent-emerald-glow); color: var(--accent-emerald-light); }
        .stat-card.cyan .stat-badge { background: var(--accent-cyan-glow); color: var(--accent-cyan-light); }
        .stat-card.violet .stat-badge { background: var(--accent-violet-glow); color: var(--accent-violet-light); }
        .stat-card.orange .stat-badge { background: var(--accent-orange-glow); color: var(--accent-orange-light); }

        .stat-value {
            font-family: var(--font-display);
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.1;
            margin: 4px 0 10px;
        }

        .stat-card.emerald .stat-value { color: var(--accent-emerald-light); }
        .stat-card.cyan .stat-value { color: var(--accent-cyan-light); }
        .stat-card.violet .stat-value { color: var(--accent-violet-light); }
        .stat-card.orange .stat-value { color: var(--accent-orange-light); }

        .stat-bar-track {
            width: 100%;
            height: 5px;
            background: rgba(255, 255, 255, 0.07);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 4px;
        }

        .stat-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 1.1s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .stat-card.emerald .stat-bar-fill { background: linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan-light)); }
        .stat-card.cyan .stat-bar-fill { background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet-light)); }
        .stat-card.violet .stat-bar-fill { background: linear-gradient(90deg, var(--accent-violet), #f472b6); }
        .stat-card.orange .stat-bar-fill { background: linear-gradient(90deg, var(--accent-orange), #facc15); }

        .stat-sub {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 8px;
            display: flex;
            justify-content: space-between;
        }

        /* ── Interactive Win Simulator Drawer ────────────────────────────── */
        .simulator-card {
            display: none;
            border-color: rgba(249, 115, 22, 0.35);
            background: linear-gradient(145deg, rgba(22, 27, 44, 0.92), rgba(14, 17, 28, 0.95));
            box-shadow: 0 16px 40px rgba(249, 115, 22, 0.12);
        }

        .simulator-card.open {
            display: block;
            animation: fadeUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        .sim-layout {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 24px;
            align-items: stretch;
        }

        .sim-presets {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .preset-btn {
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.18s ease;
        }

        .preset-btn:hover {
            background: rgba(249, 115, 22, 0.15);
            border-color: var(--accent-orange);
            color: #fff;
        }

        .sim-sliders {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(185px, 1fr));
            gap: 14px;
        }

        .sim-control {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 13px;
            transition: border-color 0.2s;
        }

        .sim-control:hover {
            border-color: rgba(249, 115, 22, 0.4);
        }

        .sim-control-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            margin-bottom: 6px;
        }

        .sim-control-label {
            color: var(--text-secondary);
            font-weight: 600;
        }

        .sim-control-val {
            font-family: var(--font-mono);
            font-weight: 700;
            color: var(--accent-orange-light);
            font-size: 0.82rem;
        }

        input[type="range"].sim-slider {
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
            height: 5px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 999px;
            outline: none;
        }

        input[type="range"].sim-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: var(--accent-orange-light);
            cursor: pointer;
            box-shadow: 0 0 10px rgba(249, 115, 22, 0.6);
            transition: transform 0.15s ease;
        }

        input[type="range"].sim-slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        .sim-result-panel {
            background: rgba(0, 0, 0, 0.28);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 24px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            position: relative;
        }

        .gauge-wrap {
            position: relative;
            width: 150px;
            height: 150px;
            margin-bottom: 14px;
        }

        .gauge-svg {
            transform: rotate(-90deg);
            width: 150px;
            height: 150px;
        }

        .gauge-bg {
            fill: none;
            stroke: rgba(255, 255, 255, 0.07);
            stroke-width: 10;
        }

        .gauge-fill {
            fill: none;
            stroke: var(--accent-emerald-light);
            stroke-width: 10;
            stroke-linecap: round;
            stroke-dasharray: 377;
            stroke-dashoffset: 94;
            transition: stroke-dashoffset 0.45s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.3s ease;
        }

        .gauge-center {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .gauge-pct {
            font-family: var(--font-display);
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
        }

        .gauge-sub {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .sim-verdict {
            font-family: var(--font-display);
            font-size: 1.05rem;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 999px;
            margin-bottom: 8px;
        }

        .sim-verdict.win {
            background: var(--accent-emerald-glow);
            color: var(--accent-emerald-light);
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .sim-verdict.loss {
            background: var(--accent-rose-glow);
            color: var(--accent-rose-light);
            border: 1px solid rgba(244, 63, 94, 0.35);
        }

        /* ── Feature Columns Explorer ────────────────────────────────────── */
        .features {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .feature-tag {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 14px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            user-select: none;
        }

        .feature-tag .feat-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent-cyan-light);
        }

        .feature-tag[data-cat="offense"] .feat-dot { background: var(--accent-orange-light); }
        .feature-tag[data-cat="defense"] .feat-dot { background: var(--accent-emerald-light); }
        .feature-tag[data-cat="shooting"] .feat-dot { background: var(--accent-cyan-light); }
        .feature-tag[data-cat="discipline"] .feat-dot { background: var(--accent-rose-light); }
        .feature-tag[data-cat="context"] .feat-dot { background: var(--accent-violet-light); }

        .feature-tag .feat-code {
            font-family: var(--font-mono);
            font-size: 0.7rem;
            padding: 1px 6px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.07);
            color: var(--text-primary);
        }

        .feature-tag:hover, .feature-tag.active {
            background: rgba(6, 182, 212, 0.14);
            border-color: var(--accent-cyan-light);
            color: var(--text-primary);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(6, 182, 212, 0.16);
        }

        .feature-inspector {
            margin-top: 14px;
            padding: 12px 16px;
            border-radius: var(--radius-md);
            background: rgba(6, 182, 212, 0.07);
            border: 1px solid rgba(6, 182, 212, 0.22);
            display: none;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            font-size: 0.82rem;
            animation: fadeUp 0.25s ease both;
        }

        .feature-inspector.visible {
            display: flex;
        }

        /* ── Table Controls & Legend ─────────────────────────────────────── */
        .table-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }

        .search-box {
            position: relative;
            flex: 1;
            min-width: 220px;
            max-width: 340px;
        }

        .search-box input {
            width: 100%;
            padding: 9px 14px 9px 36px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: 0.83rem;
            font-family: var(--font-sans);
            outline: none;
            transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
        }

        .search-box input:focus {
            border-color: var(--accent-cyan-light);
            background: rgba(255, 255, 255, 0.06);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.15);
        }

        .search-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 0.85rem;
            pointer-events: none;
        }

        .filter-pills {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 7px 13px;
            border-radius: var(--radius-md);
            font-size: 0.76rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.18s ease;
            font-family: var(--font-sans);
        }

        .filter-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.08);
        }

        .filter-btn.active {
            background: rgba(16, 185, 129, 0.16);
            border-color: rgba(16, 185, 129, 0.45);
            color: var(--accent-emerald-light);
        }

        .filter-btn[data-filter="incorrect"].active {
            background: rgba(244, 63, 94, 0.16);
            border-color: rgba(244, 63, 94, 0.45);
            color: var(--accent-rose-light);
        }

        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 14px;
            margin-bottom: 16px;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            font-size: 0.72rem;
            color: var(--text-secondary);
        }

        .legend span {
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .legend strong {
            font-family: var(--font-mono);
            color: var(--accent-cyan-light);
            background: rgba(6, 182, 212, 0.1);
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.68rem;
        }

        /* ── Predictions Table ───────────────────────────────────────────── */
        .table-wrapper {
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow-x: auto;
            background: rgba(8, 11, 19, 0.6);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            min-width: 920px;
        }

        th {
            text-align: left;
            padding: 12px 11px;
            background: rgba(255, 255, 255, 0.035);
            border-bottom: 1px solid var(--border-highlight);
            border-right: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-secondary);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.68rem;
            letter-spacing: 0.6px;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
            transition: background 0.18s, color 0.18s;
            position: relative;
        }

        th:hover {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }

        th.col-highlight, td.col-highlight {
            background: rgba(6, 182, 212, 0.14) !important;
            color: #fff;
        }

        td {
            padding: 11px 11px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.045);
            border-right: 1px solid rgba(255, 255, 255, 0.03);
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: #e2e8f0;
            transition: background 0.15s ease;
        }

        td.team-cell {
            font-family: var(--font-sans);
            font-weight: 600;
            color: var(--text-primary);
        }

        td.opp-cell {
            font-family: var(--font-sans);
            color: var(--text-secondary);
        }

        td.date-cell {
            color: var(--text-secondary);
            font-size: 0.74rem;
        }

        tbody tr.game-row {
            cursor: pointer;
            transition: transform 0.15s ease, background 0.15s ease;
        }

        .correct-row td {
            background: rgba(16, 185, 129, 0.055);
        }

        .correct-row td:first-child {
            box-shadow: inset 3px 0 0 var(--accent-emerald);
        }

        .incorrect-row td {
            background: rgba(244, 63, 94, 0.06);
        }

        .incorrect-row td:first-child {
            box-shadow: inset 3px 0 0 var(--accent-rose);
        }

        tbody tr.game-row:hover td {
            background: rgba(255, 255, 255, 0.075);
        }

        .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 3px 9px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            font-family: var(--font-mono);
            letter-spacing: 0.3px;
            background: rgba(255, 255, 255, 0.07);
            color: var(--text-secondary);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .badge.home-badge {
            background: rgba(139, 92, 246, 0.14);
            color: var(--accent-violet-light);
            border-color: rgba(139, 92, 246, 0.3);
        }

        .badge.win {
            background: rgba(16, 185, 129, 0.18);
            color: var(--accent-emerald-light);
            border-color: rgba(16, 185, 129, 0.35);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.12);
        }

        .badge.loss {
            background: rgba(244, 63, 94, 0.18);
            color: var(--accent-rose-light);
            border-color: rgba(244, 63, 94, 0.35);
        }

        /* ── Expandable Row Details Drawer ───────────────────────────────── */
        .detail-row {
            display: none;
        }

        .detail-row.open {
            display: table-row;
        }

        .detail-row td {
            background: rgba(13, 18, 30, 0.95) !important;
            padding: 18px 22px;
            border-bottom: 1px solid var(--border-highlight);
        }

        .detail-grid {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
            font-family: var(--font-sans);
        }

        .detail-bars {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
            flex: 1;
        }

        .mini-stat {
            background: rgba(255, 255, 255, 0.03);
            padding: 8px 10px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-subtle);
        }

        .mini-stat-top {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        .mini-stat-top strong {
            color: var(--text-primary);
            font-family: var(--font-mono);
        }

        /* ── Pagination ──────────────────────────────────────────────────── */
        .pagination {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .pagination-controls {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .pagination a, .pagination span {
            padding: 8px 15px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .pagination a:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent-emerald-light);
            color: var(--text-primary);
            transform: translateY(-1px);
        }

        .pagination .current {
            background: var(--accent-emerald-glow);
            border-color: rgba(16, 185, 129, 0.4);
            color: var(--accent-emerald-light);
        }

        .pagination .disabled {
            opacity: 0.35;
            pointer-events: none;
        }

        .pagination .page-info {
            padding: 8px 16px;
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: 0.8rem;
            background: rgba(255, 255, 255, 0.03);
        }

        @media (max-width: 900px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .sim-layout { grid-template-columns: 1fr; }
            .header { flex-direction: column; padding: 22px; }
            h1 { font-size: 1.75rem; }
        }

        @media (max-width: 540px) {
            .stats-grid { grid-template-columns: 1fr; }
            body { padding: 16px 12px 48px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="brand-row">
                    <span class="live-pill"><span class="live-dot"></span> Live Model</span>
                    <span class="model-pill">⚡ Logistic Regression + StandardScaler</span>
                </div>
                <h1><span class="ball-icon">🏀</span> NBA <span>Win Predictor</span></h1>
                <p class="tagline">Logistic Regression on Team Statistics · Real-Time Matchup Intelligence</p>
                <p class="trained-tag">Trained <span>{{ trained_at }}</span> &nbsp;·&nbsp; <span>{{ total_games|default(0)|int|commatize }}</span> total games</p>
            </div>
            <div class="header-actions">
                <button type="button" class="action-btn primary" id="toggleSimBtn" onclick="toggleSimulator()">
                    🎯 Interactive Win Simulator
                </button>
                <a class="dataset-btn" href="https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=TeamStatistics.csv" target="_blank" rel="noopener">
                    📂 View Dataset
                </a>
            </div>
        </div>

        <!-- Interactive What-If Win Simulator -->
        <div class="card simulator-card" id="simulatorCard">
            <div class="section-header">
                <div>
                    <div class="section-title"><span class="section-icon">🎯</span> What-If Game Win Simulator</div>
                    <div class="section-subtitle">Adjust live team box-score metrics or click any game row below to simulate win probability</div>
                </div>
                <div class="sim-presets">
                    <button type="button" class="preset-btn" onclick="applyPreset('elite')">🔥 Elite Offense</button>
                    <button type="button" class="preset-btn" onclick="applyPreset('defense')">🛡️ Grit &amp; Defense</button>
                    <button type="button" class="preset-btn" onclick="applyPreset('average')">⚖️ League Average</button>
                    <button type="button" class="preset-btn" onclick="applyPreset('cold')">❄️ Cold Shooting</button>
                </div>
            </div>
            <div class="sim-layout">
                <div class="sim-sliders" id="simSliders">
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Assists (A)</span><span class="sim-control-val" id="val_assists">25</span></div>
                        <input type="range" class="sim-slider" id="sim_assists" min="8" max="45" value="25" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Rebounds (R)</span><span class="sim-control-val" id="val_reboundsTotal">44</span></div>
                        <input type="range" class="sim-slider" id="sim_reboundsTotal" min="25" max="68" value="44" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Blocks (B)</span><span class="sim-control-val" id="val_blocks">5</span></div>
                        <input type="range" class="sim-slider" id="sim_blocks" min="0" max="16" value="5" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Steals (S)</span><span class="sim-control-val" id="val_steals">8</span></div>
                        <input type="range" class="sim-slider" id="sim_steals" min="0" max="20" value="8" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Turnovers (TO)</span><span class="sim-control-val" id="val_turnovers">13</span></div>
                        <input type="range" class="sim-slider" id="sim_turnovers" min="3" max="28" value="13" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Personal Fouls (F)</span><span class="sim-control-val" id="val_foulsPersonal">20</span></div>
                        <input type="range" class="sim-slider" id="sim_foulsPersonal" min="8" max="36" value="20" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Q1 Points (Q1)</span><span class="sim-control-val" id="val_q1Points">28</span></div>
                        <input type="range" class="sim-slider" id="sim_q1Points" min="10" max="48" value="28" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">Q2 Points (Q2)</span><span class="sim-control-val" id="val_q2Points">28</span></div>
                        <input type="range" class="sim-slider" id="sim_q2Points" min="10" max="48" value="28" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">FG Attempted (FGA)</span><span class="sim-control-val" id="val_fieldGoalsAttempted">88</span></div>
                        <input type="range" class="sim-slider" id="sim_fieldGoalsAttempted" min="60" max="115" value="88" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">3PT Attempted (3FA)</span><span class="sim-control-val" id="val_threePointersAttempted">34</span></div>
                        <input type="range" class="sim-slider" id="sim_threePointersAttempted" min="10" max="60" value="34" oninput="updateSimulator()">
                    </div>
                    <div class="sim-control">
                        <div class="sim-control-header"><span class="sim-control-label">FT Attempted (FTA)</span><span class="sim-control-val" id="val_freeThrowsAttempted">23</span></div>
                        <input type="range" class="sim-slider" id="sim_freeThrowsAttempted" min="4" max="50" value="23" oninput="updateSimulator()">
                    </div>
                </div>
                <div class="sim-result-panel">
                    <div class="gauge-wrap">
                        <svg class="gauge-svg" viewBox="0 0 140 140">
                            <circle class="gauge-bg" cx="70" cy="70" r="60"></circle>
                            <circle class="gauge-fill" id="simGaugeFill" cx="70" cy="70" r="60"></circle>
                        </svg>
                        <div class="gauge-center">
                            <span class="gauge-pct" id="simProbText">68%</span>
                            <span class="gauge-sub">Win Prob</span>
                        </div>
                    </div>
                    <div class="sim-verdict win" id="simVerdict">PREDICTED WIN (W)</div>
                    <p style="font-size: 0.74rem; color: var(--text-muted);" id="simCaption">Home Court · Live Model Inference</p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-header">
                <div class="section-title"><span class="section-icon">📊</span> Model Performance</div>
                <span class="section-subtitle">90% Train / 10% Holdout Test Evaluation</span>
            </div>
            <div class="stats-grid">
                <div class="stat-card emerald">
                    <div class="stat-top">
                        <div class="stat-label">Test Accuracy</div>
                        <span class="stat-badge">Train: {{ "%.1f"|format((train_accuracy or test_accuracy or 0) * 100) }}%</span>
                    </div>
                    <div class="stat-value">{{ "%.1f"|format((test_accuracy or 0) * 100) }}%</div>
                    <div class="stat-bar-track">
                        <div class="stat-bar-fill" style="width: {{ "%.1f"|format((test_accuracy or 0) * 100) }}%;"></div>
                    </div>
                    <div class="stat-sub"><span>Out-of-sample precision</span><span>Holdout</span></div>
                </div>
                <div class="stat-card cyan">
                    <div class="stat-top">
                        <div class="stat-label">Training Games</div>
                        <span class="stat-badge">90% Split</span>
                    </div>
                    <div class="stat-value">{{ (train_games or 0)|int|commatize }}</div>
                    <div class="stat-bar-track">
                        <div class="stat-bar-fill" style="width: 90%;"></div>
                    </div>
                    <div class="stat-sub"><span>Fitted samples</span><span>StandardScaled</span></div>
                </div>
                <div class="stat-card violet">
                    <div class="stat-top">
                        <div class="stat-label">Testing Games</div>
                        <span class="stat-badge">10% Split</span>
                    </div>
                    <div class="stat-value">{{ (test_games or 0)|int|commatize }}</div>
                    <div class="stat-bar-track">
                        <div class="stat-bar-fill" style="width: 35%;"></div>
                    </div>
                    <div class="stat-sub"><span>Unseen validation</span><span>Random state 42</span></div>
                </div>
                <div class="stat-card orange">
                    <div class="stat-top">
                        <div class="stat-label">Total Games</div>
                        <span class="stat-badge">100% Corpus</span>
                    </div>
                    <div class="stat-value">{{ (total_games or 0)|int|commatize }}</div>
                    <div class="stat-bar-track">
                        <div class="stat-bar-fill" style="width: 100%;"></div>
                    </div>
                    <div class="stat-sub"><span>Historical NBA box scores</span><span>Home POV</span></div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-header">
                <div class="section-title"><span class="section-icon">🔧</span> Feature Columns</div>
                <span class="section-subtitle">Click any feature pill to inspect &amp; highlight its column in the table</span>
            </div>
            <div class="features">
                {% for f in feat_cols %}
                <span class="feature-tag" data-feature="{{ f }}" onclick="selectFeature(this, '{{ f }}')">
                    <span class="feat-dot"></span>
                    <span>{{ f }}</span>
                </span>
                {% endfor %}
            </div>
            <div class="feature-inspector" id="featureInspector">
                <div>
                    <strong id="featInspectorTitle" style="color: var(--accent-cyan-light);">Feature</strong>:
                    <span id="featInspectorDesc" style="color: var(--text-secondary);">Description</span>
                </div>
                <button type="button" class="preset-btn" onclick="clearFeatureHighlight()">Clear Highlight ✕</button>
            </div>
        </div>

        <div class="card">
            <div class="section-header">
                <div>
                    <div class="section-title"><span class="section-icon">📋</span> Predictions</div>
                    <div class="section-subtitle">Click any column header to sort · Click any game row to inspect stats or load into simulator</div>
                </div>
                <div id="pageAccuracyBadge" class="live-pill" style="background: rgba(6, 182, 212, 0.12); border-color: rgba(6, 182, 212, 0.3); color: var(--accent-cyan-light);">
                    Page Hit Rate: <span id="pageAccValue">--</span>
                </div>
            </div>

            <div class="table-toolbar">
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="tableSearch" placeholder="Filter page by team, opponent, or date..." oninput="filterTable()">
                </div>
                <div class="filter-pills">
                    <button type="button" class="filter-btn active" data-filter="all" onclick="setRowFilter('all', this)">All Rows</button>
                    <button type="button" class="filter-btn" data-filter="correct" onclick="setRowFilter('correct', this)">✓ Correct Hits</button>
                    <button type="button" class="filter-btn" data-filter="incorrect" onclick="setRowFilter('incorrect', this)">✗ Missed</button>
                    <button type="button" class="filter-btn" data-filter="win" onclick="setRowFilter('win', this)">Predicted W</button>
                    <button type="button" class="filter-btn" data-filter="loss" onclick="setRowFilter('loss', this)">Predicted L</button>
                </div>
            </div>

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

            <div class="table-wrapper">
                <table id="predictionsTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0, 'str')" title="Game Date">Date ⇅</th>
                            <th onclick="sortTable(1, 'str')" title="Team Name">Team ⇅</th>
                            <th onclick="sortTable(2, 'str')" title="Opponent Team">Opp ⇅</th>
                            <th data-col="home" title="Home or Away">H/A</th>
                            {% for f in display_labels %}
                            <th data-col="{{ display_cols[loop.index0] }}" onclick="sortTable({{ loop.index0 + 4 }}, 'num')" title="{{ display_cols[loop.index0] }}">{{ f }}</th>
                            {% endfor %}
                            <th title="Predicted Outcome">P</th>
                            <th title="Actual Outcome">A</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for _, row in page_items.iterrows() %}
                        <tr class="game-row {{ 'correct-row' if row.get('correct', False) else 'incorrect-row' }}"
                            data-correct="{{ '1' if row.get('correct', False) else '0' }}"
                            data-pred="{{ 'W' if row.get('pred') == 1 else 'L' }}"
                            data-team="{{ row.get('teamName', 'N/A') }}"
                            data-opp="{{ row.get('opponentTeamName', 'N/A') }}"
                            data-date="{{ (row.get('gameDate', 'N/A') | to_str)[:10] }}"
                            {% for f in display_cols %}
                            data-stat-{{ f }}="{{ '{:.0f}'.format(row.get(f, 0)) }}"
                            {% endfor %}
                            onclick="toggleRowDetail(this)">
                            <td class="date-cell">{{ (row.get('gameDate', 'N/A') | to_str)[:10] }}</td>
                            <td class="team-cell">{{ row.get('teamName', 'N/A') }}</td>
                            <td class="opp-cell">{{ row.get('opponentTeamName', 'N/A') }}</td>
                            <td data-col="home"><span class="badge home-badge">{{ 'Home' if row.get('home') == 1 else 'Away' }}</span></td>
                            {% for f in display_cols %}
                            <td data-col="{{ f }}">{{ "{:.0f}".format(row.get(f, 0)) }}</td>
                            {% endfor %}
                            <td><span class="badge {{ 'win' if row.get('pred') == 1 else 'loss' }}">{{ 'W' if row.get('pred') == 1 else 'L' }}</span></td>
                            <td><span class="badge {{ 'win' if row.get('actual') == 1 else 'loss' }}">{{ 'W' if row.get('actual') == 1 else 'L' }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="pagination">
                <div style="font-size: 0.78rem; color: var(--text-muted);">
                    Showing page <strong style="color: var(--text-primary);">{{ page }}</strong> of <strong style="color: var(--text-primary);">{{ total_pages }}</strong> · Use <kbd style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono);">←</kbd> <kbd style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono);">→</kbd> keys to navigate
                </div>
                <div class="pagination-controls">
                    {% if page > 1 %}
                    <a id="prevPageLink" href="?page={{ page - 1 }}">&laquo; Prev</a>
                    {% else %}
                    <span class="disabled">&laquo; Prev</span>
                    {% endif %}
                    <span class="page-info">{{ page }} / {{ total_pages }}</span>
                    {% if page < total_pages %}
                    <a id="nextPageLink" href="?page={{ page + 1 }}">Next &raquo;</a>
                    {% else %}
                    <span class="disabled">Next &raquo;</span>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <script>
        const FEATURE_META = {
            teamId: { cat: 'context', label: 'Team Identifier', desc: 'Unique numeric NBA franchise ID encoding historical team baseline.' },
            home: { cat: 'context', label: 'Home Court Indicator', desc: 'Binary indicator (1 = Home, 0 = Away) capturing home-court advantage.' },
            assists: { cat: 'offense', label: 'Assists (A)', desc: 'Total team assists — high ball movement strongly correlates with offensive efficiency and wins.' },
            reboundsTotal: { cat: 'defense', label: 'Total Rebounds (R)', desc: 'Combined offensive & defensive boards — controlling glass limits opponent second chances.' },
            blocks: { cat: 'defense', label: 'Blocks (B)', desc: 'Rim protection metric measuring altered and rejected opponent shots.' },
            steals: { cat: 'defense', label: 'Steals (S)', desc: 'Live-ball defensive takeaways that fuel transition scoring opportunities.' },
            turnovers: { cat: 'discipline', label: 'Turnovers (TO)', desc: 'Possessions lost without a shot attempt — negatively impacts win probability.' },
            foulsPersonal: { cat: 'discipline', label: 'Personal Fouls (F)', desc: 'Total team fouls committed — excess fouling gifts opponents free-throw points.' },
            q1Points: { cat: 'offense', label: '1st Quarter Points (Q1)', desc: 'Opening quarter scoring output establishing early game margin.' },
            q2Points: { cat: 'offense', label: '2nd Quarter Points (Q2)', desc: 'Second quarter scoring pace leading into halftime.' },
            fieldGoalsAttempted: { cat: 'shooting', label: 'Field Goals Attempted (FGA)', desc: 'Total shot volume from the floor — higher FGA without high efficiency often signals trailing pace.' },
            threePointersAttempted: { cat: 'shooting', label: '3-Pointers Attempted (3FA)', desc: 'Perimeter shot volume reflecting modern spacing and pace.' },
            freeThrowsAttempted: { cat: 'shooting', label: 'Free Throws Attempted (FTA)', desc: 'Trips to the foul line generated by paint pressure and bonus situations.' }
        };

        // Decorate feature pills with category dots
        document.querySelectorAll('.feature-tag').forEach(tag => {
            const f = tag.getAttribute('data-feature');
            if (FEATURE_META[f]) {
                tag.setAttribute('data-cat', FEATURE_META[f].cat);
            }
        });

        function selectFeature(el, feat) {
            const isAlreadyActive = el.classList.contains('active');
            clearFeatureHighlight();
            if (isAlreadyActive) return;

            el.classList.add('active');
            const meta = FEATURE_META[feat] || { label: feat, desc: 'Model input feature column.' };
            document.getElementById('featInspectorTitle').textContent = meta.label + ' (' + feat + ')';
            document.getElementById('featInspectorDesc').textContent = meta.desc;
            document.getElementById('featureInspector').classList.add('visible');

            document.querySelectorAll('[data-col="' + feat + '"]').forEach(cell => {
                cell.classList.add('col-highlight');
            });
        }

        function clearFeatureHighlight() {
            document.querySelectorAll('.feature-tag.active').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.col-highlight').forEach(c => c.classList.remove('col-highlight'));
            document.getElementById('featureInspector').classList.remove('visible');
        }

        // Simulator logic
        function toggleSimulator() {
            const card = document.getElementById('simulatorCard');
            const btn = document.getElementById('toggleSimBtn');
            card.classList.toggle('open');
            btn.classList.toggle('active');
            if (card.classList.contains('open')) {
                updateSimulator();
            }
        }

        const PRESETS = {
            elite: { assists: 32, reboundsTotal: 48, blocks: 6, steals: 10, turnovers: 10, foulsPersonal: 18, q1Points: 34, q2Points: 33, fieldGoalsAttempted: 86, threePointersAttempted: 38, freeThrowsAttempted: 26 },
            defense: { assists: 24, reboundsTotal: 52, blocks: 9, steals: 12, turnovers: 11, foulsPersonal: 19, q1Points: 26, q2Points: 27, fieldGoalsAttempted: 84, threePointersAttempted: 30, freeThrowsAttempted: 25 },
            average: { assists: 25, reboundsTotal: 44, blocks: 5, steals: 8, turnovers: 14, foulsPersonal: 20, q1Points: 27, q2Points: 27, fieldGoalsAttempted: 88, threePointersAttempted: 34, freeThrowsAttempted: 22 },
            cold: { assists: 17, reboundsTotal: 38, blocks: 3, steals: 5, turnovers: 18, foulsPersonal: 24, q1Points: 20, q2Points: 21, fieldGoalsAttempted: 92, threePointersAttempted: 40, freeThrowsAttempted: 16 }
        };

        function applyPreset(name, customLabel) {
            const p = PRESETS[name];
            if (!p) return;
            Object.keys(p).forEach(k => {
                const slider = document.getElementById('sim_' + k);
                if (slider) slider.value = p[k];
            });
            updateSimulator(customLabel);
        }

        async function updateSimulator(customCaption) {
            const fields = ['assists', 'reboundsTotal', 'blocks', 'steals', 'turnovers', 'foulsPersonal', 'q1Points', 'q2Points', 'fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted'];
            const payload = {};
            fields.forEach(f => {
                const val = parseFloat(document.getElementById('sim_' + f).value);
                document.getElementById('val_' + f).textContent = val;
                payload[f] = val;
            });

            // Instant client-side logistic estimate + server refinement if available
            let z = 0.25
                + (payload.assists - 24.5) * 0.095
                + (payload.reboundsTotal - 43.5) * 0.085
                + (payload.blocks - 4.8) * 0.08
                + (payload.steals - 7.6) * 0.11
                - (payload.turnovers - 13.8) * 0.12
                - (payload.foulsPersonal - 20.5) * 0.045
                + (payload.q1Points - 26.5) * 0.075
                + (payload.q2Points - 26.5) * 0.075
                - (payload.fieldGoalsAttempted - 87.0) * 0.065
                - (payload.threePointersAttempted - 33.0) * 0.015
                + (payload.freeThrowsAttempted - 22.5) * 0.035;
            let prob = 1 / (1 + Math.exp(-z));

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    const data = await res.json();
                    if (typeof data.win_probability === 'number') {
                        prob = data.win_probability;
                    }
                }
            } catch (e) {}

            const pct = Math.round(prob * 100);
            const circumference = 377;
            const offset = circumference - (prob * circumference);
            const gaugeFill = document.getElementById('simGaugeFill');
            const probText = document.getElementById('simProbText');
            const verdict = document.getElementById('simVerdict');
            const caption = document.getElementById('simCaption');

            gaugeFill.style.strokeDashoffset = offset;
            probText.textContent = pct + '%';

            if (prob >= 0.5) {
                gaugeFill.style.stroke = 'var(--accent-emerald-light)';
                probText.style.color = 'var(--accent-emerald-light)';
                verdict.className = 'sim-verdict win';
                verdict.textContent = 'PREDICTED WIN (W)';
            } else {
                gaugeFill.style.stroke = 'var(--accent-rose-light)';
                probText.style.color = 'var(--accent-rose-light)';
                verdict.className = 'sim-verdict loss';
                verdict.textContent = 'PREDICTED LOSS (L)';
            }
            if (customCaption) {
                caption.textContent = customCaption;
            }
        }

        // Table filtering, sorting & expandable row inspection
        let activeFilter = 'all';

        function setRowFilter(filter, btn) {
            activeFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterTable();
        }

        function filterTable() {
            const q = (document.getElementById('tableSearch').value || '').toLowerCase().trim();
            const rows = document.querySelectorAll('#predictionsTable tbody tr.game-row');
            let visible = 0;
            let correctCount = 0;

            rows.forEach(row => {
                const team = (row.getAttribute('data-team') || '').toLowerCase();
                const opp = (row.getAttribute('data-opp') || '').toLowerCase();
                const date = (row.getAttribute('data-date') || '').toLowerCase();
                const isCorrect = row.getAttribute('data-correct') === '1';
                const pred = row.getAttribute('data-pred');

                let matchSearch = !q || team.includes(q) || opp.includes(q) || date.includes(q);
                let matchFilter = true;
                if (activeFilter === 'correct') matchFilter = isCorrect;
                else if (activeFilter === 'incorrect') matchFilter = !isCorrect;
                else if (activeFilter === 'win') matchFilter = pred === 'W';
                else if (activeFilter === 'loss') matchFilter = pred === 'L';

                const show = matchSearch && matchFilter;
                row.style.display = show ? '' : 'none';
                const next = row.nextElementSibling;
                if (next && next.classList.contains('detail-row') && !show) {
                    next.classList.remove('open');
                }
                if (show) {
                    visible++;
                    if (isCorrect) correctCount++;
                }
            });

            const accEl = document.getElementById('pageAccValue');
            if (accEl) {
                accEl.textContent = visible > 0 ? Math.round((correctCount / visible) * 100) + '% (' + correctCount + '/' + visible + ')' : 'N/A';
            }
        }

        function toggleRowDetail(row) {
            const existing = row.nextElementSibling;
            if (existing && existing.classList.contains('detail-row')) {
                existing.classList.toggle('open');
                return;
            }

            const team = row.getAttribute('data-team');
            const opp = row.getAttribute('data-opp');
            const date = row.getAttribute('data-date');
            const stats = ['assists', 'reboundsTotal', 'blocks', 'steals', 'turnovers', 'foulsPersonal', 'q1Points', 'q2Points', 'fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted'];
            const maxes = { assists: 40, reboundsTotal: 65, blocks: 14, steals: 16, turnovers: 25, foulsPersonal: 32, q1Points: 45, q2Points: 45, fieldGoalsAttempted: 108, threePointersAttempted: 55, freeThrowsAttempted: 42 };

            let barsHtml = '';
            stats.forEach(s => {
                const v = parseFloat(row.getAttribute('data-stat-' + s) || '0');
                const pct = Math.min(100, Math.round((v / (maxes[s] || 50)) * 100));
                barsHtml += `
                    <div class="mini-stat">
                        <div class="mini-stat-top"><span>${s}</span><strong>${v}</strong></div>
                        <div class="stat-bar-track"><div class="stat-bar-fill" style="width:${pct}%; background: linear-gradient(90deg, var(--accent-cyan), var(--accent-emerald));"></div></div>
                    </div>`;
            });

            const detailTr = document.createElement('tr');
            detailTr.className = 'detail-row open';
            detailTr.innerHTML = `
                <td colspan="17">
                    <div class="detail-grid">
                        <div style="min-width: 190px;">
                            <div style="font-weight: 700; font-size: 0.92rem; color: var(--text-primary);">${team} vs ${opp}</div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 10px;">${date} · Game Box Breakdown</div>
                            <button type="button" class="action-btn primary" style="padding: 6px 12px; font-size: 0.74rem;" onclick="loadRowIntoSimulator(this, event)">
                                ⚡ Load Game into Simulator
                            </button>
                        </div>
                        <div class="detail-bars">${barsHtml}</div>
                    </div>
                </td>`;
            row.parentNode.insertBefore(detailTr, row.nextSibling);
        }

        function loadRowIntoSimulator(btn, ev) {
            ev.stopPropagation();
            const detailRow = btn.closest('tr.detail-row');
            const gameRow = detailRow ? detailRow.previousElementSibling : null;
            if (!gameRow) return;

            const stats = ['assists', 'reboundsTotal', 'blocks', 'steals', 'turnovers', 'foulsPersonal', 'q1Points', 'q2Points', 'fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted'];
            stats.forEach(s => {
                const val = gameRow.getAttribute('data-stat-' + s);
                const slider = document.getElementById('sim_' + s);
                if (slider && val !== null) slider.value = val;
            });

            const card = document.getElementById('simulatorCard');
            const toggleBtn = document.getElementById('toggleSimBtn');
            if (!card.classList.contains('open')) {
                card.classList.add('open');
                toggleBtn.classList.add('active');
            }
            const label = gameRow.getAttribute('data-team') + ' vs ' + gameRow.getAttribute('data-opp') + ' (' + gameRow.getAttribute('data-date') + ')';
            updateSimulator('Loaded: ' + label);
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        let sortDirs = {};
        function sortTable(colIdx, type) {
            const tbody = document.querySelector('#predictionsTable tbody');
            // Remove open detail rows before sorting
            tbody.querySelectorAll('tr.detail-row').forEach(r => r.remove());
            const rows = Array.from(tbody.querySelectorAll('tr.game-row'));
            const dir = sortDirs[colIdx] === 'asc' ? 'desc' : 'asc';
            sortDirs[colIdx] = dir;

            rows.sort((a, b) => {
                let va = a.children[colIdx].textContent.trim();
                let vb = b.children[colIdx].textContent.trim();
                if (type === 'num') {
                    return dir === 'asc' ? (parseFloat(va) - parseFloat(vb)) : (parseFloat(vb) - parseFloat(va));
                }
                return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
            });
            rows.forEach(r => tbody.appendChild(r));
        }

        // Keyboard arrow navigation for pagination
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') {
                const prev = document.getElementById('prevPageLink');
                if (prev) prev.click();
            } else if (e.key === 'ArrowRight') {
                const next = document.getElementById('nextPageLink');
                if (next) next.click();
            }
        });

        // Compute initial page accuracy badge
        filterTable();
    </script>
</body>
</html>
'''


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Live model inference endpoint for the interactive What-If Win Simulator."""
    ensure_fresh_data()
    model = _state.get("model")
    data = request.get_json(silent=True) or {}
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        df_existing = _state.get("predictions_df")
        default_team_id = int(df_existing["teamId"].mode().iloc[0]) if (df_existing is not None and not df_existing.empty and "teamId" in df_existing.columns) else 1610612737
        row = {
            "teamId": float(data.get("teamId", default_team_id)),
            "home": float(data.get("home", 1)),
        }
        for col in displayCols:
            row[col] = float(data.get(col, 0))
        X_input = pd.DataFrame([row])[featCols]
        prob = float(model.predict_proba(X_input)[0][1])
        pred = int(prob >= 0.5)
        return jsonify({"win_probability": round(prob, 4), "prediction": pred})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400



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
        feat_cols=featCols,
        display_cols=displayCols,
        display_labels=displayLabels,
        featCols=featCols,
        displayCols=displayCols,
        displayLabels=displayLabels,
        test_accuracy=_state["test_acc"],
        train_accuracy=_state["train_acc"],
        test_acc=_state["test_acc"],
        train_acc=_state["train_acc"],
        total_games=_state["total_games"],
        train_games=_state["train_games"],
        test_games=_state["test_games"],
        trained_at=_state["trained_at"],
        page=page,
        total_pages=total_pages,
        range=range,
    )
    response = make_response(rendered)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
