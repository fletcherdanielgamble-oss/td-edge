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
    """Load weekly player stats and attach real game dates from NFL schedules.

    nflreadpy's weekly player-stat table does not consistently expose a usable
    game_date column. We therefore enrich it from the schedule data, preferring
    game_id joins and falling back to season/week/team joins.
    """
    import nflreadpy as nfl
    stats = to_pandas(nfl.load_player_stats(seasons=seasons, summary_level="week"))

    # If the stats already contain usable dates, preserve them.
    date_col = next((c for c in ["game_date", "gameday", "game_date_dt"] if c in stats.columns), None)
    if date_col is not None and pd.to_datetime(stats[date_col], errors="coerce").notna().sum() >= 100:
        stats["game_date"] = pd.to_datetime(stats[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        return stats

    schedules = to_pandas(nfl.load_schedules(seasons=seasons))
    schedule_date_col = next((c for c in ["gameday", "game_date", "date"] if c in schedules.columns), None)
    if schedule_date_col is None:
        raise ValueError("NFL schedule data did not include a recognised game date column.")
    schedules["game_date"] = pd.to_datetime(schedules[schedule_date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    # First choice: stable game ID, if both datasets expose it.
    if "game_id" in stats.columns and "game_id" in schedules.columns:
        date_map = schedules[["game_id", "game_date"]].dropna(subset=["game_id"]).drop_duplicates("game_id")
        stats = stats.drop(columns=["game_date"], errors="ignore").merge(date_map, on="game_id", how="left")

    # Fallback: weekly stats commonly identify team/season/week rather than game_id.
    if "game_date" not in stats.columns or pd.to_datetime(stats["game_date"], errors="coerce").notna().sum() < 100:
        team_col = next((c for c in ["recent_team", "team"] if c in stats.columns), None)
        required = {"season", "week", "home_team", "away_team", "season", "week", "game_date"}
        if team_col and {"season", "week", "home_team", "away_team", "game_date"}.issubset(schedules.columns):
            home = schedules[["season", "week", "home_team", "game_date"]].rename(columns={"home_team": "_join_team"})
            away = schedules[["season", "week", "away_team", "game_date"]].rename(columns={"away_team": "_join_team"})
            schedule_teams = pd.concat([home, away], ignore_index=True).dropna(subset=["_join_team", "game_date"])
            if "season_type" in stats.columns and "season_type" in schedules.columns:
                # Retain season_type when available to avoid mixing preseason/regular/postseason weeks.
                home = schedules[["season", "week", "season_type", "home_team", "game_date"]].rename(columns={"home_team": "_join_team"})
                away = schedules[["season", "week", "season_type", "away_team", "game_date"]].rename(columns={"away_team": "_join_team"})
                schedule_teams = pd.concat([home, away], ignore_index=True).dropna(subset=["_join_team", "game_date"])
                join_cols = ["season", "week", "season_type"]
            else:
                join_cols = ["season", "week"]
            schedule_teams = schedule_teams.drop_duplicates(join_cols + ["_join_team", "game_date"])
            stats = stats.drop(columns=["game_date"], errors="ignore").merge(
                schedule_teams,
                left_on=join_cols + [team_col],
                right_on=join_cols + ["_join_team"],
                how="left",
            ).drop(columns=["_join_team"], errors="ignore")

    valid_dates = pd.to_datetime(stats.get("game_date"), errors="coerce") if "game_date" in stats.columns else pd.Series(dtype="datetime64[ns]")
    if valid_dates.notna().sum() < 100:
        raise ValueError("Could not match enough player stats to real NFL game dates. Reload the data; the schedule join did not find dates for enough rows.")
    stats["game_date"] = valid_dates.dt.strftime("%Y-%m-%d")
    return stats


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
    df["_rush_tds"] = rush_tds
    df["_rec_tds"] = rec_tds
    df["_carries"] = carries
    df["_targets"] = targets
    df["_receptions"] = receptions
    df["_rush_yards"] = rush_yards
    df["_rec_yards"] = rec_yards
    df["_tds"] = rush_tds + rec_tds

    group_keys = ["_player_key", "season"]
    feature_specs = [("_carries", "prior_carries_avg4"), ("_targets", "prior_targets_avg4"),
                     ("_receptions", "prior_receptions_avg4"), ("_rush_yards", "prior_rush_yards_avg4"),
                     ("_rec_yards", "prior_rec_yards_avg4"), ("_tds", "prior_tds_avg8"),
                     ("_rush_tds", "prior_rush_tds_avg8"), ("_rec_tds", "prior_rec_tds_avg8")]
    for source, dest in feature_specs:
        window = 8 if source in ["_tds", "_rush_tds", "_rec_tds"] else 4
        df[dest] = df.groupby(group_keys)[source].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["prior_games"] = df.groupby(group_keys).cumcount()
    df["prior_any_td_rate"] = df.groupby(group_keys)["target_any_td"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["prior_any_td_rate"] = df["prior_any_td_rate"].fillna(0.0)
    for c in ["prior_carries_avg4", "prior_targets_avg4", "prior_receptions_avg4", "prior_rush_yards_avg4", "prior_rec_yards_avg4", "prior_tds_avg8", "prior_rush_tds_avg8", "prior_rec_tds_avg8"]:
        df[c] = df[c].fillna(0)
    df["player"] = df[find("player_display_name", "player_name", "player_id")].astype(str)
    position_col = find("position", "position_group", "player_position")
    df["position"] = df[position_col].astype(str).str.upper().str.strip() if position_col else "UNKNOWN"
    team_col = find("recent_team", "team")
    opp_col = find("opponent_team", "opponent")
    df["team"] = df[team_col].astype(str) if team_col else ""
    df["opponent"] = df[opp_col].astype(str) if opp_col else ""
    date_col = find("game_date", "gameday", "game_date_dt")
    df["game_date"] = df[date_col].astype(str) if date_col else ""
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
    "prior_rush_tds_avg8", "prior_rec_tds_avg8", "prior_any_td_rate", "prior_games",
]


def normalize_player_name(value) -> str:
    return " ".join(str(value).strip().lower().split())


def estimate_anytime_td_value(stats: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Experimental position-aware next-game ATD estimates with strict date cutoffs."""
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
    out["model_position_group"] = ""
    out["model_type"] = ""
    out["player_historical_td_rate_pct"] = np.nan

    atd_mask = out["market"].astype(str).str.strip().str.lower().isin(
        ["anytime td", "anytime touchdown", "anytime touchdown scorer"]
    )
    out["_event_date_dt"] = pd.to_datetime(out["game_date"], errors="coerce")

    for event_date in out.loc[atd_mask, "_event_date_dt"].dropna().drop_duplicates():
        row_mask = atd_mask & (out["_event_date_dt"] == event_date)
        history = all_frame[all_frame["_game_date_dt"] < event_date].copy()
        train = history[history["prior_games"] >= 1].copy()
        if len(train) < 100 or train["target_any_td"].nunique() < 2:
            continue

        def make_model():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(max_iter=1000, class_weight=None)),
            ])

        global_model = make_model()
        global_model.fit(train[MODEL_FEATURES], train["target_any_td"])
        position_models = {}
        for pos, pos_train in train.groupby("position"):
            if len(pos_train) >= 100 and pos_train["target_any_td"].nunique() >= 2:
                pos_model = make_model()
                pos_model.fit(pos_train[MODEL_FEATURES], pos_train["target_any_td"])
                position_models[pos] = (pos_model, len(pos_train))

        future_rows = []
        for player, player_history in history.groupby("player", sort=False):
            player_history = player_history.sort_values(["season", "week"])
            latest_season = player_history["season"].max()
            season_history = player_history[player_history["season"] == latest_season]
            if season_history.empty:
                continue
            row = {"player_key": normalize_player_name(player), "last_season": int(latest_season),
                   "position": str(season_history["position"].iloc[-1]).upper().strip()}
            row["prior_carries_avg4"] = float(season_history["_carries"].tail(4).mean())
            row["prior_targets_avg4"] = float(season_history["_targets"].tail(4).mean())
            row["prior_receptions_avg4"] = float(season_history["_receptions"].tail(4).mean())
            row["prior_rush_yards_avg4"] = float(season_history["_rush_yards"].tail(4).mean())
            row["prior_rec_yards_avg4"] = float(season_history["_rec_yards"].tail(4).mean())
            row["prior_tds_avg8"] = float(season_history["_tds"].tail(8).mean())
            row["prior_rush_tds_avg8"] = float(season_history["_rush_tds"].tail(8).mean())
            row["prior_rec_tds_avg8"] = float(season_history["_rec_tds"].tail(8).mean())
            row["prior_any_td_rate"] = float(season_history["target_any_td"].mean())
            row["prior_games"] = int(len(season_history))
            row["player_historical_td_rate_pct"] = round(100 * row["prior_any_td_rate"], 2)
            future_rows.append(row)

        if not future_rows:
            continue
        future = pd.DataFrame(future_rows).drop_duplicates("player_key", keep="last")
        future["model_probability"] = np.nan
        future["model_training_rows"] = len(train)
        future["model_type"] = "global_fallback"
        for pos, idx in future.groupby("position").groups.items():
            if pos in position_models:
                chosen_model, n_train = position_models[pos]
                future.loc[idx, "model_probability"] = chosen_model.predict_proba(future.loc[idx, MODEL_FEATURES])[:, 1]
                future.loc[idx, "model_training_rows"] = n_train
                future.loc[idx, "model_type"] = "position_specific"
            else:
                future.loc[idx, "model_probability"] = global_model.predict_proba(future.loc[idx, MODEL_FEATURES])[:, 1]
        lookup = future.set_index("player_key")
        keys = out.loc[row_mask, "player_key"]
        matched = keys.isin(lookup.index)
        matched_index = keys.index[matched]
        if len(matched_index):
            matched_keys = out.loc[matched_index, "player_key"]
            out.loc[matched_index, "model_probability"] = matched_keys.map(lookup["model_probability"]).to_numpy()
            out.loc[matched_index, "model_last_season"] = matched_keys.map(lookup["last_season"]).to_numpy()
            out.loc[matched_index, "model_training_rows"] = matched_keys.map(lookup["model_training_rows"]).to_numpy()
            out.loc[matched_index, "model_position_group"] = matched_keys.map(lookup["position"]).fillna("").to_numpy()
            out.loc[matched_index, "model_type"] = matched_keys.map(lookup["model_type"]).fillna("").to_numpy()
            out.loc[matched_index, "player_historical_td_rate_pct"] = matched_keys.map(lookup["player_historical_td_rate_pct"]).to_numpy()

    out["model_probability_pct"] = (100 * out["model_probability"]).round(2)
    out["edge_percentage_points"] = (100 * out["model_probability"] - 100 / out["decimal_odds"]).round(2)
    out["expected_roi_pct"] = (100 * (out["model_probability"] * out["decimal_odds"] - 1)).round(2)
    out["expected_profit_per_100"] = (100 * (out["model_probability"] * out["decimal_odds"] - 1)).round(2)
    latest_season = pd.to_numeric(stats.get("season"), errors="coerce")
    latest_season = int(latest_season.max()) if latest_season.notna().any() else 0
    return out.drop(columns=["player_key", "_event_date_dt"]), latest_season

def validate_position_models(stats: pd.DataFrame):
    """Walk-forward season tests, always training only on earlier seasons."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import brier_score_loss, log_loss

    frame = build_player_game_features(stats)
    frame = frame[frame["prior_games"] >= 1].copy()
    seasons = sorted(int(x) for x in frame["season"].dropna().unique())
    reports, predictions = [], []
    for season in seasons[1:]:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season].copy()
        if len(train) < 1000 or len(test) < 100 or train["target_any_td"].nunique() < 2:
            continue
        def make_model():
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(max_iter=1000)),
            ])
        global_model = make_model()
        global_model.fit(train[MODEL_FEATURES], train["target_any_td"])
        test["predicted_probability"] = global_model.predict_proba(test[MODEL_FEATURES])[:, 1]
        test["model_type"] = "global_fallback"
        for pos, pos_train in train.groupby("position"):
            idx = test.index[test["position"] == pos]
            if len(idx) == 0 or len(pos_train) < 100 or pos_train["target_any_td"].nunique() < 2:
                continue
            model = make_model()
            model.fit(pos_train[MODEL_FEATURES], pos_train["target_any_td"])
            test.loc[idx, "predicted_probability"] = model.predict_proba(test.loc[idx, MODEL_FEATURES])[:, 1]
            test.loc[idx, "model_type"] = "position_specific"
        y = test["target_any_td"].to_numpy()
        pred = test["predicted_probability"].to_numpy()
        baseline = float(train["target_any_td"].mean())  # Never use test outcomes as baseline.
        reports.append({
            "test_season": season, "test_rows": len(test), "train_rows": len(train),
            "brier": brier_score_loss(y, pred),
            "baseline_brier": brier_score_loss(y, np.full(len(y), baseline)),
            "log_loss": log_loss(y, pred, labels=[0, 1]),
            "baseline_log_loss": log_loss(y, np.full(len(y), baseline), labels=[0, 1]),
            "observed_td_rate_pct": 100 * float(np.mean(y)),
            "mean_prediction_pct": 100 * float(np.mean(pred)),
            "position_model_pct": 100 * float((test["model_type"] == "position_specific").mean()),
        })
        predictions.append(test[["season", "position", "target_any_td", "predicted_probability", "model_type"]].copy())
    if not reports:
        raise ValueError("Need several seasons of player stats for chronological validation. Download 2018–2026.")
    report = pd.DataFrame(reports)
    preds = pd.concat(predictions, ignore_index=True)
    # Fixed-width probability bands make reliability easier to inspect.
    preds["probability_band"] = pd.cut(preds["predicted_probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = preds.groupby("probability_band", observed=False).agg(
        n=("target_any_td", "size"),
        mean_prediction=("predicted_probability", "mean"),
        actual_td_rate=("target_any_td", "mean"),
    ).reset_index()
    calibration["probability_band"] = calibration["probability_band"].astype(str)
    calibration["predicted_pct"] = (100 * calibration["mean_prediction"]).round(2)
    calibration["observed_pct"] = (100 * calibration["actual_td_rate"]).round(2)
    return report, calibration[["probability_band", "n", "predicted_pct", "observed_pct"]], preds


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
                show_cols = [c for c in ["game_date", "player", "team", "opponent", "market", "bookmaker", "decimal_odds", "raw_implied_probability_pct", "model_probability_pct", "edge_percentage_points", "expected_roi_pct", "model_position_group", "model_type", "model_training_rows", "player_historical_td_rate_pct", "model_last_season"] if c in value_rows.columns]
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
    st.subheader("Walk-forward anytime-TD model validation")
    st.write("Each test season is predicted using only earlier seasons. Position-specific models are used where enough training examples exist; otherwise a global fallback is used. No historical betting odds are available, so this tests probability quality, **not betting ROI**.")
    stats = st.session_state.stats_df
    if stats is None:
        st.info("Download historical NFL player stats first (ideally 2018–2026).")
    else:
        if st.button("Run chronological validation", type="primary"):
            try:
                with st.spinner("Testing each season against earlier seasons…"):
                    report, calibration, preds = validate_position_models(stats)
                st.session_state.validation_results = (report, calibration, preds)
            except Exception as exc:
                st.error(f"Validation failed: {exc}")
        if "validation_results" in st.session_state:
            report, calibration, preds = st.session_state.validation_results
            from sklearn.metrics import brier_score_loss, log_loss
            y = preds["target_any_td"].to_numpy()
            p = preds["predicted_probability"].to_numpy()
            a, b, c = st.columns(3)
            a.metric("Pooled Brier score", f"{brier_score_loss(y,p):.4f}")
            b.metric("Pooled log loss", f"{log_loss(y,p,labels=[0,1]):.4f}")
            c.metric("Out-of-sample player-games", f"{len(preds):,}")
            st.caption("Lower Brier and log loss are better. Compare each season to its prior-season-only baseline; the pooled metrics combine multiple seasons.")
            st.subheader("Results by test season")
            st.dataframe(report.round(4), use_container_width=True, hide_index=True)
            st.download_button("Download season validation report", report.to_csv(index=False).encode(), "td_edge_validation_by_season.csv", "text/csv")
            st.subheader("Probability calibration (all tested seasons)")
            st.caption("For well-calibrated probabilities, predicted % should be close to observed %. Small bands can be noisy; check sample counts.")
            st.dataframe(calibration, use_container_width=True, hide_index=True)
            st.line_chart(calibration.set_index("probability_band")[["predicted_pct", "observed_pct"]])
            st.download_button("Download calibration table", calibration.to_csv(index=False).encode(), "td_edge_calibration.csv", "text/csv")
            st.subheader("By player position")
            pos_rows = []
            for pos, grp in preds.groupby("position"):
                if len(grp) < 30:
                    continue
                pos_rows.append({"position": pos, "test_rows": len(grp), "brier": brier_score_loss(grp["target_any_td"], grp["predicted_probability"]), "observed_td_rate_pct": 100 * grp["target_any_td"].mean(), "mean_prediction_pct": 100 * grp["predicted_probability"].mean(), "position_model_pct": 100 * (grp["model_type"] == "position_specific").mean()})
            st.dataframe(pd.DataFrame(pos_rows).round(3), use_container_width=True, hide_index=True)
            st.download_button("Download out-of-sample predictions", preds.to_csv(index=False).encode(), "td_edge_validation_predictions.csv", "text/csv")
    st.warning("This model does not yet include opponent strength, injury/news information, betting-market priors, or historical bookmaker odds. Backtest scores do not establish profitability.")
    st.markdown("**Other markets:** first TD and 2+ TD are not modelled yet.")

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
