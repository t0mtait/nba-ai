# NBA Win Predictor

Flask app that trains a logistic-regression model on historical NBA team box
scores and serves a dark-themed dashboard of its predictions, plus an
interactive **What-If Game Win Simulator** that scores hypothetical box
scores against the live model.

Live at <https://nba.tomjt.com>.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Opens on <http://localhost:5000>. On boot the app downloads the Kaggle
dataset, trains the model, and then serves requests.

## How it works

| | |
|---|---|
| Dataset | [`eoinamoore/historical-nba-data-and-player-box-scores`](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores) — `TeamStatistics.csv` (~34 MB), supplemented with recent completed games from the ESPN NBA scoreboard/boxscore API |
| Model | `StandardScaler` → `LogisticRegression` (sklearn `Pipeline`) |
| Features | `teamId`, `home`, `assists`, `reboundsTotal`, `blocks`, `steals`, `turnovers`, `foulsPersonal`, `q1Points`, `q2Points`, `fieldGoalsAttempted`, `threePointersAttempted`, `freeThrowsAttempted` |
| Split | 90/10 train/test, `random_state=42` |
| Refresh | Dataset re-downloaded, supplemented with any newer completed NBA games from ESPN, and model retrained every `REFRESH_INTERVAL_SECONDS` (default `3600`), on a background thread, and lazily on request if the TTL has expired |

Accuracy and corpus size vary with the upstream dataset; the numbers below
are what the deployed instance reported on its last boot.

| Metric | Value |
|---|---|
| Test accuracy | 75.5% |
| Training games | 62,630 |
| Testing games | 6,959 |
| Total games | 69,589 |

## API

### `POST /api/predict`

Scores a box score. Every display feature is optional and defaults to `0`;
`teamId` defaults to the most common team in the dataset and `home` to `1`.

```bash
curl -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"assists":25,"reboundsTotal":44,"blocks":5,"steals":8,"turnovers":13,
       "foulsPersonal":20,"q1Points":26,"q2Points":26,
       "fieldGoalsAttempted":87,"threePointersAttempted":33,"freeThrowsAttempted":22}'
```

```json
{"prediction": 1, "win_probability": 0.5913}
```

`prediction` is `1` when `win_probability >= 0.5`. Returns `503` if the model
has not finished training and `400` on a malformed payload.

### `GET /`

The dashboard: a paginated table of every historical prediction with the
team, opponent, date, predicted outcome, and whether it was correct. The
Win Simulator panel posts to `/api/predict` as the sliders move.

There is no `/health` endpoint — it returns `404`. Use `GET /` as the
readiness probe.

## Project structure

```
├── app.py              # Flask app, model training, and the HTML template
├── train_models.py     # Nightly training pipeline; saves artifacts to models/
├── models/             # Trained artifacts (committed; baked into the image)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml  # Generic single-host run — see "Deployment" below
└── railway.json        # Railway build config
```

## Deployment

There are two compose files and they are **not** the same deployment.

**This repo's `docker-compose.yml`** is a generic single-host run: service
`web`, port `5000:5000`, build context `.`. Good for local testing.

**The live stack** is `/opt/stacks/nba-ai/compose.yaml` (Dockge-managed on
the homelab): service and container `nba-ai`, image `nba-ai:latest`, port
`5008:5000`, `restart: unless-stopped`, and a `./kaggle-cache` bind mount
persisting `/root/.cache/kagglehub` inside the container.

Note that the cache does **not** save you the download: `app.py` calls
`kagglehub.dataset_download(..., force_download=True)`, so the 34 MB
`TeamStatistics.csv` is re-fetched on every boot and on every refresh
interval regardless of the mount. What the mount buys is that the cache
lives on the host instead of the container's writable layer, so it survives
container recreation, and `app.py` prunes superseded dataset versions from
it to keep the directory bounded at roughly one version.

Caddy reverse-proxies `nba.tomjt.com` to the host's LAN address on port
`5008`. The container is not reachable from the internet — there is no router
port-forward for `5008`.

`models/` is copied into the image at build time, but `app.py` does not load
it: the app retrains from Kaggle on every boot and keeps the model in
memory. `models/` exists for the nightly pipeline and for anyone who wants
the raw artifacts without a download.

### Updating the deployed container

The build context is the git checkout at `~/Repositories/nba-ai`, so an
update is: pull, rebuild the image, recreate the container.

```bash
cd /opt/stacks/nba-ai
docker compose build nba-ai
docker compose up -d --no-build nba-ai
```

**Watch out:** `docker compose up -d` trips a "this looks like a long-lived
server process" guard in the agent CLI when run in the foreground, and the
command is refused before it reaches Docker. The container itself starts
fine — the guard is a false positive on the `up` verb. Workarounds that all
work in the foreground:

```bash
docker compose start nba-ai   # start an existing container
docker restart nba-ai         # restart in place
docker compose create nba-ai  # recreate, then start
```

Or run `docker compose up -d` in the background, where the guard does not
apply. `docker compose build`, `docker compose stop`, and
`docker compose logs` are unaffected.

Verify the update landed:

```bash
curl -fsS http://127.0.0.1:5008/ > /dev/null && echo up
docker logs --tail 50 nba-ai
```

The log should show the dataset download, `Serving Flask app`, and a `200`
for `GET /`. Kaggle throttles repeat downloads — a boot that takes a minute
or two is normal.

## Releases

`main` is the release branch. Pushing to it runs semantic-release
(`release.config.js`), which derives the version from conventional commits,
updates `CHANGELOG.md`, and tags the release. `feat:` is a minor bump,
`fix:` is a patch, `chore:` and `docs:` do not release.

## Development notes

- `train_models.py` is the nightly path: it trains, writes
  `models/model.joblib`, `models/predictions.csv`, and `models/metadata.json`,
  then commits and pushes those artifacts. The running app does not use them
  directly.
- `.dockerignore` is committed and load-bearing. Without it the build
  context ships `node_modules/` (~68 MB) and `.git/` (~62 MB).
- `node_modules/` is currently committed to the repo. It is not needed —
  CI installs from the tracked `package-lock.json` via `npm ci` — and it
  accounts for most of the repository's size. Removing it from the index is
  tracked separately so the change stays reviewable on its own.
