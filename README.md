# TD Edge — NFL Touchdown Value Analysis

A free starter dashboard for NFL touchdown market research. It downloads historical NFL player statistics through `nflreadpy`, accepts bookmaker odds via CSV/Excel, provides an initial time-based anytime-touchdown model test, compares best bookmaker prices, and offers experimental anytime-TD probability/expected-value estimates.

## Important status note

This is a research prototype, not a live odds service and not a proven profitable betting model. The app does not automatically ingest sportsbook odds or extract screenshots, and it does not produce first-touchdown / multi-touchdown probabilities. The anytime-TD probability and expected-value tool is experimental, not calibrated or proven profitable. It restricts training and player history to games dated before each odds event to reduce future-data leakage.

## Run locally

Requires Python 3.11 or 3.12 and an internet connection for data downloads.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit. Use the sidebar to download public player stats, then use **Odds scanner** to upload odds.

## Data

The public data loader uses `nflreadpy` from the nflverse project. The app uses weekly player statistics. The exact seasons available depend on the upstream dataset's publication status.

## Odds CSV format

Required template columns:

- `game_date`, `season`, `week`, `game_id`
- `player`, `team`, `opponent`, `market`
- `bookmaker`, `decimal_odds`, `captured_at_utc`

Use consistent player names and market labels across bookmakers. Example market labels: `Anytime TD`, `First TD`, `2+ TDs`, `Rushing TD`, `Receiving TD`.

## Model and validation

The initial anytime-TD baseline uses lagged player-game opportunity features and a season-based holdout. It deliberately excludes same-game outcomes from its predictors. Brier score is compared with a simple baseline. This is only an initial sanity check; before any betting use, add calibration plots, log loss, rolling-origin validation across several seasons, timestamped historical odds, realistic price/market matching, and uncertainty estimates.

The first-TD, 2+ TD, rushing-TD and receiving-TD markets are listed in the odds template but are not yet predicted by this model. First touchdown requires modelling scoring order and team/game scoring conditions; multi-TD requires a distinct count/probability model.

## Free deployment

1. Create a GitHub repository and upload these files.
2. Visit https://share.streamlit.io/ and connect GitHub.
3. Select the repository and `app.py` as the entrypoint.
4. Deploy. See https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy.

Do not add API keys or passwords to this repository. No API key is required for this starter version.

## Suggested next development steps

1. Add historical play-by-play-derived red-zone and team scoring features.
2. Add a proper calibration curve and season-by-season backtest.
3. Add odds history and closing-line value analysis.
4. Add specialist first-TD and 2+ TD models.
5. Add an automated odds provider only after checking its market coverage, terms and cost.
