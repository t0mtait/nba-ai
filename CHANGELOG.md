## [2.1.1](https://github.com/t0mtait/nba-ai/compare/v2.1.0...v2.1.1) (2026-08-10)


### Bug Fixes

* dockerfile change for 500 fix ([f0f22b6](https://github.com/t0mtait/nba-ai/commit/f0f22b64bde85c140fa68aa2d95c468ee7f9c272))

# [2.1.0](https://github.com/t0mtait/nba-ai/compare/v2.0.0...v2.1.0) (2026-08-08)


### Features

* add nightly training pipeline + cron job ([f5cd4ea](https://github.com/t0mtait/nba-ai/commit/f5cd4eaa386b9fc2320098d2b699f3eae76a49d3))
* railway config setup ([198ca77](https://github.com/t0mtait/nba-ai/commit/198ca77162b4191d8f375e002710eae04a47876a))

# [2.0.0](https://github.com/t0mtait/nba-ai/compare/v1.0.0...v2.0.0) (2026-07-29)


### Bug Fixes

* add class_weight='balanced' to LogisticRegression models ([34a3617](https://github.com/t0mtait/nba-ai/commit/34a36170e9fc2a4f4cea41a6e29b4289f15fefa9))
* add missing List import in data_loader.py ([1b136c6](https://github.com/t0mtait/nba-ai/commit/1b136c61275584aebe9d4f64e44739faa066f3e5))
* add missing semantic-release plugins ([3897510](https://github.com/t0mtait/nba-ai/commit/3897510c9d525bbd1ece792583c4ce3aaa5f0c1a))
* add semrel ([ffbffeb](https://github.com/t0mtait/nba-ai/commit/ffbffeb899c60423152a3263e51172a5db67a25f))
* call extract_insights in train_models and save insights JSON ([ae84e16](https://github.com/t0mtait/nba-ai/commit/ae84e167a246b215b5e689a2ebb5e8ce20f20f43))
* compute overall accuracy on held-out test sets in train_models.py ([35941c5](https://github.com/t0mtait/nba-ai/commit/35941c531a51145418e91a1e63035ae425e4d44a))
* correct [object Event] error in loadGameStats, add box-sizing and width to input fields, shorten labels ([87f6f71](https://github.com/t0mtait/nba-ai/commit/87f6f7196a54c16fd2ca10faec1cf0d283b3ad36))
* correct stat parsing and redesign UI ([33f5b05](https://github.com/t0mtait/nba-ai/commit/33f5b05c7d49989ad1ffcf27d9c67b0d905c7c62))
* CSS layout, chronological train/test split, team-aware game-stats ([ab54188](https://github.com/t0mtait/nba-ai/commit/ab54188f851498ae457fbe2d268b4e7c4c792d73))
* overflow containment, team-stats fallback detection, post-fetch refresh ([c63172d](https://github.com/t0mtait/nba-ai/commit/c63172d374915c7584966166eee12eeac1370720))
* parameterize LIMIT clause and handle missing spread models gracefully ([d5039a4](https://github.com/t0mtait/nba-ai/commit/d5039a49c7aa401445ecc8494466f86bb71d92d1))
* remove not needed stuff ([7cd3a5f](https://github.com/t0mtait/nba-ai/commit/7cd3a5fc0738835286a81030dd97d490ed8875b5))
* semrel fix: ([539b447](https://github.com/t0mtait/nba-ai/commit/539b447e46cf2f7edb8d57b760f48078c40c1f5c))
* update prediction form placeholder values to match dataset averages ([93da615](https://github.com/t0mtait/nba-ai/commit/93da615bd24488d515371845b6c81be94f5d6832))
* workflows dir ([6649f02](https://github.com/t0mtait/nba-ai/commit/6649f026aed51695a2413c816199703791439670))


### Code Refactoring

* pivot from Celtics win-predictor to general NBA moneyline/spread predictor ([798dfd4](https://github.com/t0mtait/nba-ai/commit/798dfd4791e728cb13b230a6dfcf3f04060cc4f6))


### Features

* add model insights section + fix input overflow ([6f14f22](https://github.com/t0mtait/nba-ai/commit/6f14f22be0b34ef193849b5a29031d21e545d6b9))
* add Train Model button and /api/train endpoint ([b34479d](https://github.com/t0mtait/nba-ai/commit/b34479dbe3aa17cfe035eb85038111d6e4b0d7a8))


### BREAKING CHANGES

* - new prediction model:
- Predicts moneyline win probability AND point spread outcomes for any NBA matchup
- No longer Celtics-specific; works for all 30 teams

New data sources:
- Head-to-head matchup history (win%, avg margin, game count)
- Team season stats (pace, ortg, drtg, net_rtg, efg%, tov%, ftr%)
- Home court advantage (~58% historical home win rate)
- Rest differential (days since last game for each team)
- Injury impact (out/doubtful player penalty)

New models:
- ml_model.pkl: Logistic regression for moneyline (win/loss)
- spread_model.pkl: Logistic regression for spread cover classification
- spread_reg.pkl: Ridge regression for predicted margin in points

Database schema changes (db.py):
- games table: added team_score, opponent_score, home_ml, away_ml, home_spread, away_spread
- New team_season_stats table: season-level per-team stats
- New injuries table: player injury tracking
- Renamed nba_games.db -> nba.db

New API endpoints:
- POST /predict: returns moneyline + spread predictions with recommendations
- GET /teams/{code}/stats/{year}: team season stats
- GET /teams/{code}/injuries: current injury reports
- GET /matchups/{team}/vs/{opp}: head-to-head history
- GET /model-stats: model feature insights

Feature columns (12 total):
home_net_rtg_diff, home_ortg_diff, home_drtg_diff, home_pace_diff,
home_efg_diff, home_tov_diff, home_ftr_diff, h2h_win_pct_diff,
h2h_avg_margin, rest_diff, injury_impact, home_court_adv
