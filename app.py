from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TD Edge | NFL Touchdown Analytics", page_icon="🏈", layout="wide")

ODDS_COLUMNS = ["game_date", "season", "week", "game_id", "player", "team", "opponent", "market", "bookmaker", "decimal_odds", "captured_at_utc"]
MARKETS = ["Anytime TD", "First TD", "2+ TDs", "Rushing TD", "Receiving TD"]


def to_pandas(obj):
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def load_nfl_stats(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_player_stats(seasons=seasons, summary_level="week")
    return to_pandas(raw)


def clean_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ODDS_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["decimal_odds"] = pd.to_numeric(out["decimal_odds"], errors="coerce")
    out = out[out["decimal_odds"].between(1.01, 1000, inclusive="both")].copy()
    out["market"] = out["market"].astype(str).str.strip()
    out["player"] = out["player"].astype(str).str.strip()
    out["bookmaker"] = out["bookmaker"].astype(str).str.strip()
    out["implied_probability"] = 1 / out["decimal_odds"]
    return out


def market_value_table(odds: pd.DataFrame, estimates: pd.DataFrame | None = None) -> pd.DataFrame:
    if odds.empty:
        return odds
    out = odds.copy()
    key_cols = [c for c in ["game_date", "player", "team", "opponent", "market"] if c in out.columns]
    if not key_cols:
        key_cols = ["player", "market"]
    idx = out.groupby(key_cols, dropna=False)["decimal_odds"].idxmax()
    out = out.loc[idx].copy()
    out["best_book_odds"] = out["decimal_odds"]
    if estimates is not None and not estimates.empty:
        e = estimates.copy()
        if "model_probability" in e.columns:
            join_cols = [c for c in key_cols if c in e.columns]
            if "player" in join_cols and "market" in join_cols:
                e = e.drop_duplicates(join_cols, keep="last")
                out = out.merge(e[join_cols + ["model_probability"]], on=join_cols, how="left")
                out["expected_roi"] = out["model_probability"] * out["decimal_odds"] - 1
                out["edge_vs_raw_implied"] = out["model_probability"] - (1 / out["decimal_odds"])
    return out.sort_values("expected_roi", ascending=False) if "expected_roi" in out.columns else out


def build_player_game_features(stats: pd.DataFrame) -> pd.DataFrame:
    """Create strictly lagged player-game features. This is a baseline, not film-level analysis."""
    df = stats.copy()
    lower = {c.lower(): c for c in df.columns}
    def find(*names):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    player_col = find("player_id", "gsis_id", "player_name")
    season_col = find("season")
    week_col = find("week")
    if not all([player_col, season_col, week_col]):
        raise ValueError("Stats data needs player_id (or player_name), season and week columns.")

    df["_player_key"] = df[player_col].astype(str)
    df["season"] = pd.to_numeric(df[season_col], errors="coerce")
    df["week"] = pd.to_numeric(df[week_col], errors="coerce")
    df = df.dropna(subset=["season", "week"]).sort_values(["_player_key", "season", "week"])

    def numeric_col(name, default=0.0):
        col = find(name)
        if col is None:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    carries = numeric_col("carries")
    targets = numeric_col("targets")
    receptions = numeric_col("receptions")
    rush_tds = numeric_col("rushing_tds")
    rec_tds = numeric_col("receiving_tds")
    rush_yards = numeric_col("rushing_yards")
    rec_yards = numeric_col("receiving_yards")
    df["target_any_td"] = ((rush_tds + rec_tds) > 0).astype(int)
    df["_carries"] = carries
    df["_targets"] = targets
    df["_receptions"] = receptions
    df["_rush_yards"] = rush_yards
    df["_rec_yards"] = rec_yards
    df["_tds"] = rush_tds + rec_tds

    group_keys = ["_player_key", "season"]
    feature_specs = [("_carries", "prior_carries_avg4"), ("_targets", "prior_targets_avg4"),
                     ("_receptions", "prior_receptions_avg4"), ("_rush_yards", "prior_rush_yards_avg4"),
                     ("_rec_yards", "prior_rec_yards_avg4"), ("_tds", "prior_tds_avg8")]
    for source, dest in feature_specs:
        window = 8 if source == "_tds" else 4
        df[dest] = df.groupby(group_keys)[source].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["prior_games"] = df.groupby(group_keys).cumcount()
    df["prior_any_td_rate"] = df.groupby(group_keys)["target_any_td"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["prior_any_td_rate"] = df["prior_any_td_rate"].fillna(df["target_any_td"].mean())
    for c in ["prior_carries_avg4", "prior_targets_avg4", "prior_receptions_avg4", "prior_rush_yards_avg4", "prior_rec_yards_avg4", "prior_tds_avg8"]:
        df[c] = df[c].fillna(0)
    df["player"] = df[find("player_display_name", "player_name", "player_id")].astype(str)
    team_col = find("recent_team", "team")
    opp_col = find("opponent_team", "opponent")
    df["team"] = df[team_col].astype(str) if team_col else ""
    df["opponent"] = df[opp_col].astype(str) if opp_col else ""
    df["game_date"] = df[find("game_date")].astype(str) if find("game_date") else ""
    return df


def train_anytime_model(stats: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    features = ["prior_carries_avg4", "prior_targets_avg4", "prior_receptions_avg4", "prior_rush_yards_avg4", "prior_rec_yards_avg4", "prior_tds_avg8", "prior_any_td_rate", "prior_games"]
    frame = build_player_game_features(stats)
    frame = frame[frame["prior_games"] >= 1].copy()
    if frame["target_any_td"].nunique() < 2 or len(frame) < 100:
        raise ValueError("Not enough usable player-game rows to fit a baseline model. Load multiple seasons.")
    latest_season = int(frame["season"].max())
    train = frame[frame["season"] < latest_season]
    test = frame[frame["season"] == latest_season]
    if train.empty or test.empty or train["target_any_td"].nunique() < 2:
        train, test = frame.iloc[:-max(1, len(frame)//5)], frame.iloc[-max(1, len(frame)//5):]
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=1000, class_weight=None))])
    model.fit(train[features], train["target_any_td"])
    test_prob = model.predict_proba(test[features])[:, 1]
    brier = float(np.mean((test_prob - test["target_any_td"].to_numpy()) ** 2))
    base = float(np.mean((test["target_any_td"].mean() - test["target_any_td"].to_numpy()) ** 2))
    return model, frame, test, brier, base, features



MODEL_FEATURES = [
    "prior_carries_avg4", "prior_targets_avg4", "prior_receptions_avg4",
    "prior_rush_yards_avg4", "prior_rec_yards_avg4", "prior_tds_avg8",
    "prior_any_td_rate", "prior_games",
]


def normalize_player_name(value) -> str:
    return " ".join(str(value).strip().lower().split())


def estimate_anytime_td_value(stats: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Estimate next-game ATD probabilities using only information before each event date.

    This is an experimental research estimate, not a calibrated betting signal. For
    each odds event date, both training labels and player usage features are restricted
    to historical games dated before that event.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    all_frame = build_player_game_features(stats)
    all_frame["_game_date_dt"] = pd.to_datetime(all_frame["game_date"], errors="coerce")
    if all_frame["_game_date_dt"].notna().sum() < 100:
        raise ValueError("The historical stats need valid game_date values to prevent using future game results.")

    out = odds.copy()
    out["player_key"] = out["player"].map(normalize_player_name)
    out["decimal_odds"] = pd.to_numeric(out["decimal_odds"], errors="coerce")
    out["raw_implied_probability_pct"] = (100 / out["decimal_odds"]).round(2)
    out["model_probability"] = np.nan
    out["model_last_season"] = np.nan
    out["model_training_rows"] = np.nan

    atd_mask = out["market"].astype(str).str.strip().str.lower().isin(
        ["anytime td", "anytime touchdown", "anytime touchdown scorer"]
    )
    out["_event_date_dt"] = pd.to_datetime(out["game_date"], errors="coerce")

    for event_date in out.loc[atd_mask, "_event_date_dt"].dropna().drop_duplicates():
        row_mask = atd_mask & (out["_event_date_dt"] == event_date)
        # Strict cutoff: never train on or use stats from the target game date or later.
        history = all_frame[all_frame["_game_date_dt"] < event_date].copy()
        train = history[history["prior_games"] >= 1].copy()
        if len(train) < 100 or train["target_any_td"].nunique() < 2:
            continue

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000, class_weight=None)),
        ])
        model.fit(train[MODEL_FEATURES], train["target_any_td"])

        future_rows = []
        # Use each player's latest eligible season and only games before event_date.
        for player, player_history in history.groupby("player", sort=False):
            player_history = player_history.sort_values(["season", "week"])
            latest_season = player_history["season"].max()
            season_history = player_history[player_history["season"] == latest_season]
            if season_history.empty:
                continue
            row = {"player_key": normalize_player_name(player), "last_season": int(latest_season)}
            row["prior_carries_avg4"] = float(season_history["_carries"].tail(4).mean())
            row["prior_targets_avg4"] = float(season_history["_targets"].tail(4).mean())
            row["prior_receptions_avg4"] = float(season_history["_receptions"].tail(4).mean())
            row["prior_rush_yards_avg4"] = float(season_history["_rush_yards"].tail(4).mean())
            row["prior_rec_yards_avg4"] = float(season_history["_rec_yards"].tail(4).mean())
            row["prior_tds_avg8"] = float(season_history["_tds"].tail(8).mean())
            row["prior_any_td_rate"] = float(season_history["target_any_td"].mean())
            row["prior_games"] = int(len(season_history))
            future_rows.append(row)

        if not future_rows:
            continue
        future = pd.DataFrame(future_rows).drop_duplicates("player_key", keep="last")
        future["model_probability"] = model.predict_proba(future[MODEL_FEATURES])[:, 1]
        lookup = future.set_index("player_key")
        keys = out.loc[row_mask, "player_key"]
        matched = keys.isin(lookup.index)
        matched_index = keys.index[matched]
        if len(matched_index):
            matched_keys = out.loc[matched_index, "player_key"]
            out.loc[matched_index, "model_probability"] = matched_keys.map(lookup["model_probability"]).to_numpy()
            out.loc[matched_index, "model_last_season"] = matched_keys.map(lookup["last_season"]).to_numpy()
            out.loc[matched_index, "model_training_rows"] = len(train)

    out["model_probability_pct"] = (100 * out["model_probability"]).round(2)
    out["edge_percentage_points"] = (100 * out["model_probability"] - 100 / out["decimal_odds"]).round(2)
    out["expected_roi_pct"] = (100 * (out["model_probability"] * out["decimal_odds"] - 1)).round(2)
    out["expected_profit_per_100"] = (100 * (out["model_probability"] * out["decimal_odds"] - 1)).round(2)
    latest_season = pd.to_numeric(stats.get("season"), errors="coerce")
    latest_season = int(latest_season.max()) if latest_season.notna().any() else 0
    return out.drop(columns=["player_key", "_event_date_dt"]), latest_season


st.title("🏈 TD Edge")
st.caption("NFL touchdown value-analysis · Free starter build · Research mode")
st.warning("This is an early research prototype, not a live betting feed or a proven profitable model. Predictions are disabled until sufficient historical data is loaded and tested.")

with st.sidebar:
    st.header("Data setup")
    seasons_text = st.text_input("NFL seasons to download", "2018,2019,2020,2021,2022,2023,2024,2025,2026")
    load_btn = st.button("Download public NFL player stats", type="primary", use_container_width=True)
    st.caption("Requires internet access in the environment running this app. The initial download can take a little while.")

if "stats_df" not in st.session_state:
    st.session_state.stats_df = None
if "odds_df" not in st.session_state:
    st.session_state.odds_df = pd.DataFrame(columns=ODDS_COLUMNS)

if load_btn:
    try:
        seasons = sorted({int(x.strip()) for x in seasons_text.split(",") if x.strip()})
        with st.spinner(f"Downloading player stats for {min(seasons)}–{max(seasons)}…"):
            st.session_state.stats_df = load_nfl_stats(seasons)
        st.success(f"Loaded {len(st.session_state.stats_df):,} player-game rows.")
    except Exception as exc:
        st.error(f"Could not download data: {exc}")
        st.info("You can still use the upload controls below. Check your internet connection and the nflreadpy installation if downloading fails.")

stats_tab, odds_tab, model_tab, guide_tab = st.tabs(["Overview", "Odds scanner", "Model & backtest", "Setup guide"])

with stats_tab:
    stats = st.session_state.stats_df
    if stats is None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Historical player-game rows", "—")
        c2.metric("Odds rows imported", f"{len(st.session_state.odds_df):,}")
        c3.metric("Validated model", "Not yet")
        st.subheader("Start here")
        st.markdown("1. Click **Download public NFL player stats** in the sidebar.\n2. Import odds on the Odds scanner tab.\n3. Review model performance before using any estimates.")
    else:
        st.success("Historical player stats loaded")
        c1, c2, c3 = st.columns(3)
        c1.metric("Player-game rows", f"{len(stats):,}")
        c2.metric("Seasons", f"{stats['season'].min()}–{stats['season'].max()}" if "season" in stats else "Loaded")
        c3.metric("Columns", f"{len(stats.columns):,}")
        st.dataframe(stats.head(30), use_container_width=True)
        st.download_button("Download loaded stats as CSV", stats.to_csv(index=False).encode(), "nfl_player_stats.csv", "text/csv")

with odds_tab:
    st.subheader("Import bookmaker odds")
    st.write("Upload a CSV or Excel file using the template columns. Screenshots can be transcribed into this format; automatic screenshot extraction is a later enhancement.")
    uploaded = st.file_uploader("Upload odds CSV or Excel", type=["csv", "xlsx", "xls"], key="odds_upload")
    if uploaded is not None:
        try:
            imported = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            cleaned = clean_odds(imported)
            st.session_state.odds_df = pd.concat([st.session_state.odds_df, cleaned], ignore_index=True).drop_duplicates()
            st.success(f"Imported {len(cleaned):,} valid odds rows.")
        except Exception as exc:
            st.error(f"Couldn't read that file: {exc}")
    odds = st.session_state.odds_df
    if not odds.empty:
        markets = [m for m in MARKETS if m in odds["market"].astype(str).unique()]
        selected_market = st.selectbox("Market", markets if markets else sorted(odds["market"].dropna().unique()))
        view = odds[odds["market"] == selected_market].copy()
        if "player" in view.columns:
            view = view.sort_values("decimal_odds", ascending=False)
        st.subheader("All imported quotes")
        st.dataframe(view, use_container_width=True)

        # Explicit best-price comparison. Keep event fields in the key so prices
        # from different games are never compared against one another.
        best_keys = [c for c in ["game_date", "game_id", "player", "team", "opponent", "market"] if c in view.columns]
        if "player" in best_keys and "market" in best_keys and not view.empty:
            comparison = view.copy()
            comparison["decimal_odds"] = pd.to_numeric(comparison["decimal_odds"], errors="coerce")
            comparison = comparison.dropna(subset=["decimal_odds"])
            # Treat blank event identifiers consistently; keep the remaining event
            # fields in the group key to avoid merging unrelated fixtures.
            grouped = comparison.groupby(best_keys, dropna=False, sort=False)
            best_indices = grouped["decimal_odds"].idxmax()
            best = comparison.loc[best_indices].copy()
            counts = grouped["decimal_odds"].count().rename("bookmakers_compared").reset_index()
            best = best.merge(counts, on=best_keys, how="left")
            best = best.rename(columns={"bookmaker": "best_bookmaker", "decimal_odds": "best_odds"})
            best["implied_probability_pct"] = (100 / best["best_odds"]).round(2)
            best = best.sort_values(["player", "best_odds"], ascending=[True, False])
            st.subheader("Best available price by player, market and event")
            st.caption("The highest decimal odds in each matching group are shown below. A group with one quote has no competing bookmaker price yet.")
            display_cols = [c for c in ["game_date", "game_id", "player", "team", "opponent", "market", "best_bookmaker", "best_odds", "implied_probability_pct", "bookmakers_compared"] if c in best.columns]
            st.dataframe(best[display_cols], use_container_width=True)
        st.divider()
        st.subheader("Experimental anytime-TD probability & expected value")
        st.caption("This fits a simple logistic model on loaded historical player usage. Estimates are research-only, not calibrated or proven profitable. First-TD, 2+ TD, rushing-TD and receiving-TD probabilities are not calculated by this model.")
        stats_for_value = st.session_state.stats_df
        if stats_for_value is None:
            st.info("Download historical player stats in the sidebar before calculating estimates.")
        else:
            loaded_seasons = pd.to_numeric(stats_for_value.get("season"), errors="coerce")
            latest_loaded_season = int(loaded_seasons.max()) if loaded_seasons is not None and loaded_seasons.notna().any() else None
            if latest_loaded_season is not None and latest_loaded_season < 2026:
                st.warning(f"Your loaded data ends in {latest_loaded_season}. For current-season estimates, reload stats including 2026 in the sidebar; older usage may be stale.")
            if st.button("Estimate anytime-TD probabilities & value", type="primary"):
                try:
                    with st.spinner("Fitting the research model and estimating player probabilities…"):
                        value_rows, latest_stats_season = estimate_anytime_td_value(stats_for_value, odds)
                    st.session_state.value_rows = value_rows
                    st.session_state.value_stats_season = latest_stats_season
                except Exception as exc:
                    st.error(f"Could not calculate estimates: {exc}")
            if "value_rows" in st.session_state:
                value_rows = st.session_state.value_rows
                if st.session_state.get("value_stats_season", 0) < 2026:
                    st.warning("These estimates use historical data ending before the 2026 season. Refresh the data before using them for current games.")
                show_cols = [c for c in ["game_date", "player", "team", "opponent", "market", "bookmaker", "decimal_odds", "raw_implied_probability_pct", "model_probability_pct", "edge_percentage_points", "expected_roi_pct", "model_last_season"] if c in value_rows.columns]
                st.dataframe(value_rows[show_cols], use_container_width=True)
                st.caption("Expected ROI = model probability × decimal odds − 1. It is only as trustworthy as the probability estimate; it does not account for model uncertainty or guarantee future returns. A missing model probability means the player name was not matched or the market is not supported.")
                st.download_button("Download odds with experimental estimates", value_rows.to_csv(index=False).encode(), "td_edge_value_estimates.csv", "text/csv")
        st.download_button("Download cleaned odds", odds.to_csv(index=False).encode(), "td_edge_odds_clean.csv", "text/csv")
        st.caption("Compare only quotes for the same player, market and event. The best price is not necessarily a profitable bet; that requires a reliable probability estimate.")
    else:
        st.info("No odds imported yet. Download the blank template below.")
    template = pd.DataFrame([
        {"game_date":"2026-10-18", "season":2026, "week":7, "game_id":"TEAM1_TEAM2_2026-10-18", "player":"Example Player", "team":"ABC", "opponent":"XYZ", "market":"Anytime TD", "bookmaker":"Example Bookmaker", "decimal_odds":2.50, "captured_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ], columns=ODDS_COLUMNS)
    st.download_button("Download odds CSV template", template.to_csv(index=False).encode(), "td_edge_odds_template.csv", "text/csv")

with model_tab:
    st.subheader("Anytime touchdown baseline")
    st.write("This baseline learns from player usage in previous games only. It does not yet include cornerback assignments, coverage data, odds-based calibration or real-time injury updates.")
    stats = st.session_state.stats_df
    if stats is None:
        st.info("Download historical player stats first.")
    elif st.button("Fit baseline and run time-based holdout test", type="primary"):
        try:
            with st.spinner("Building lagged features and testing the model…"):
                model, frame, test, brier, base_brier, features = train_anytime_model(stats)
            st.session_state.model_results = {"brier": brier, "base_brier": base_brier, "n_test": len(test), "test_season": int(test["season"].max()), "features": features}
            a, b, c = st.columns(3)
            a.metric("Holdout Brier score", f"{brier:.4f}")
            b.metric("Simple baseline Brier", f"{base_brier:.4f}")
            c.metric("Holdout rows", f"{len(test):,}")
            if brier < base_brier:
                st.success("The model beat the simple holdout baseline on Brier score. This is an encouraging first check, not proof of betting value.")
            else:
                st.warning("The model did not beat the simple holdout baseline. We should improve or simplify it before using its probabilities.")
            st.caption(f"Holdout season: {int(test['season'].max())}. Features: {', '.join(features)}")
        except Exception as exc:
            st.error(f"Model could not be fitted: {exc}")
    if "model_results" in st.session_state:
        r = st.session_state.model_results
        st.write(f"Last run: holdout season {r['test_season']}, {r['n_test']:,} player-game rows. Brier score {r['brier']:.4f} vs simple baseline {r['base_brier']:.4f}.")
        st.caption("Next validation steps: calibration curve, log loss, season-by-season results, and a historical odds backtest with timestamps.")
    st.markdown("**Other markets:** first-TD and 2+ TD models are deliberately not estimated yet. They need separate target definitions and more careful testing before their output should be displayed as actionable probabilities.")

with guide_tab:
    st.subheader("What happens next")
    st.markdown("""
    **Phase 1 — Data and odds**
    - Download public weekly player stats.
    - Import odds from multiple bookmakers using the CSV template.
    - Preserve game date, market and capture timestamp.

    **Phase 2 — Reliable probability estimates**
    - Validate anytime-TD probabilities on a later season.
    - Add calibration and uncertainty checks.
    - Add first-TD and 2+ TD models only after the baseline is credible.

    **Phase 3 — Matchups and automation**
    - Add team defence, red-zone and player-role features.
    - Incorporate injury/roster changes only when their timestamps are known.
    - Connect a live odds provider if free/manual collection becomes the bottleneck.

    **Important:** Never use future information in historical features. Do not treat a positive expected ROI from an unvalidated model as a real edge.
    """)
    st.markdown("Data source: [nflverse / nflreadpy](https://github.com/nflverse/nflreadpy). Hosting guide: [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).")
