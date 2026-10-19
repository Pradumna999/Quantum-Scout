"""Q-Scout dashboard. Reads stored results only; it never trains or calls IBM.

    QSCOUT_DATA=data QSCOUT_RESULTS=results streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qscout.serve import live_summary, load_forecasts, score_forecasts
from qscout.store import load_calibration

DATA = Path(os.environ.get("QSCOUT_DATA", "data"))
RESULTS = Path(os.environ.get("QSCOUT_RESULTS", "results"))
TIER_COLOR = {"ok": "#2E7D5B", "caution": "#B7791F", "unreliable": "#B3261E"}
INK, ACCENT = "#17232B", "#0E5A6B"

st.set_page_config(page_title="Q-Scout", layout="wide")


@st.cache_data(show_spinner=False)
def _calibration(root: str) -> pd.DataFrame | None:
    try:
        return load_calibration(root)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def _forecasts(root: str) -> pd.DataFrame:
    return load_forecasts(root)


@st.cache_data(show_spinner=False)
def _scores(root: str) -> pd.DataFrame:
    cal = _calibration(root)
    fc = _forecasts(root)
    if cal is None or fc.empty:
        return pd.DataFrame()
    return score_forecasts(fc, cal)


def _empty(msg: str, how: str) -> None:
    st.info(f"{msg}\n\n{how}")


def _log_axis(metric: str) -> str:
    return "log" if metric.endswith("_error") else "linear"


def page_fleet() -> None:
    st.header("Fleet")
    fc = _forecasts(str(DATA))
    if fc.empty:
        return _empty("No forecasts have been issued yet.", "Run `qscout train` then `qscout nightly`.")
    latest = fc.sort_values("origin_ts").groupby("backend")["origin_ts"].max().rename("latest_origin")
    h = st.selectbox("Horizon (calibration cycles)", sorted(fc["horizon"].unique()))
    cur = fc[(fc["horizon"] == h) & fc["origin_ts"].isin(latest)]
    share = (
        cur.groupby(["backend", "tier"]).size().unstack(fill_value=0).reindex(columns=list(TIER_COLOR), fill_value=0)
    )
    share = share.div(share.sum(axis=1), axis=0)
    fig = go.Figure()
    for tier, color in TIER_COLOR.items():
        fig.add_bar(x=share.index, y=share[tier], name=tier, marker_color=color)
    fig.update_layout(barmode="stack", yaxis_title="share of series", height=360, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(latest.reset_index(), hide_index=True)
    by_metric = cur.groupby(["metric", "tier"]).size().unstack(fill_value=0)
    st.caption("Series per trust tier, by metric")
    st.dataframe(by_metric)


def page_series() -> None:
    st.header("Series")
    cal = _calibration(str(DATA))
    if cal is None:
        return _empty("No calibration data found.", "Run `qscout synth` or `qscout ingest-ibm` + `qscout build-curated`.")
    c1, c2, c3 = st.columns(3)
    backend = c1.selectbox("Backend", sorted(cal["backend"].unique()))
    metric = c2.selectbox("Metric", sorted(cal.loc[cal["backend"] == backend, "metric"].unique()))
    sub = cal[(cal["backend"] == backend) & (cal["metric"] == metric)]
    entity = c3.selectbox("Entity", sorted(sub["entity_id"].unique(), key=lambda s: [int(x) for x in s.split("-")]))
    s = sub[sub["entity_id"] == entity].sort_values("snapshot_ts_utc")
    fig = go.Figure()
    fig.add_scatter(x=s["snapshot_ts_utc"], y=s["value"], mode="lines+markers", name="calibration",
                    line=dict(color=INK, width=1.5), marker=dict(size=4))
    fc = _forecasts(str(DATA))
    if not fc.empty:
        f = fc[(fc["backend"] == backend) & (fc["metric"] == metric) & (fc["entity_id"] == entity)]
        if not f.empty:
            f = f[f["origin_ts"] == f["origin_ts"].max()].sort_values("horizon")
            gap = s["snapshot_ts_utc"].diff().median()
            x = f["origin_ts"].iloc[0] + f["horizon"] * gap
            fig.add_scatter(x=x, y=f["hi"], mode="lines", line=dict(width=0), showlegend=False)
            fig.add_scatter(x=x, y=f["lo"], mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor="rgba(14,90,107,0.18)", name="90% interval")
            fig.add_scatter(x=x, y=f["point"], mode="lines+markers", name="forecast",
                            line=dict(color=ACCENT, dash="dot"),
                            marker=dict(size=8, color=[TIER_COLOR[t] for t in f["tier"]]))
            st.caption("Forecast markers are coloured by trust tier: green ok, amber caution, red unreliable.")
    sc = _scores(str(DATA))
    if not sc.empty:
        r = sc[(sc["backend"] == backend) & (sc["metric"] == metric) & (sc["entity_id"] == entity)]
        if not r.empty:
            fig.add_scatter(x=r["target_ts"], y=r["realized"], mode="markers", name="scored outcome",
                            marker=dict(symbol=["circle" if c else "x" for c in r["covered"]], size=9,
                                        color=ACCENT, line=dict(width=1, color="white")))
    fig.update_layout(height=460, yaxis_type=_log_axis(metric), yaxis_title=f"{metric}",
                      margin=dict(t=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")


def page_forward() -> None:
    st.header("Forward test")
    sc = _scores(str(DATA))
    if sc.empty:
        return _empty("No forecast has been scored yet.",
                      "Forecasts are scored once the calibration they target has been recorded. Keep "
                      "`qscout nightly` and `qscout ingest-ibm` running, then wait a few cycles.")
    summ = live_summary(sc)
    st.caption("n counts rows, which are strongly correlated within a cycle. Read n_origins as the "
               "real sample size. Coverage should be near 0.90.")
    st.dataframe(summ.round(3), hide_index=True)
    roll = sc.sort_values("target_ts").groupby("target_ts")["covered"].mean().rolling(14, min_periods=3).mean()
    fig = go.Figure(go.Scatter(x=roll.index, y=roll.values, line=dict(color=ACCENT)))
    fig.add_hline(y=0.9, line_dash="dot", line_color=INK)
    fig.update_layout(height=320, yaxis_title="rolling coverage (14 cycles)", yaxis_range=[0, 1], margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")


def page_backtests() -> None:
    st.header("Backtests")
    runs = sorted(p.parent for p in RESULTS.glob("*/run_meta.json")) if RESULTS.exists() else []
    if not runs:
        return _empty("No experiment results found.", "Run `qscout run-dev` (results/dev) first.")
    run = st.selectbox("Run", runs, format_func=lambda p: p.name)
    st.caption("Runs in dev mode never touch the test period; final runs are one-shot.")
    tables = {p.stem: pd.read_csv(p) for p in sorted(run.glob("*.csv"))}
    pt = tables.get("e2_point")
    if pt is not None:
        g = pt[pt["model"] != "persist"].groupby(["horizon", "model", "group"])["skill_mse"].mean().reset_index()
        fig = go.Figure()
        for (m, grp), d in g.groupby(["model", "group"]):
            fig.add_scatter(x=d["horizon"], y=d["skill_mse"], mode="lines+markers", name=f"{m} ({grp})")
        fig.add_hline(y=0, line_dash="dot", line_color=INK)
        fig.update_layout(height=340, xaxis_title="horizon (cycles)", yaxis_title="skill vs persistence (MSE)",
                          margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    name = st.selectbox("Table", sorted(tables))
    st.dataframe(tables[name].round(4), hide_index=True)


PAGES = {"Fleet": page_fleet, "Series": page_series, "Forward test": page_forward, "Backtests": page_backtests}
st.sidebar.title("Q-Scout")
st.sidebar.caption("Quantum calibration drift: forecasts, intervals and trust.")
PAGES[st.sidebar.radio("View", list(PAGES))]()
