"""Live business-activity dashboard.

  +---------------------------------------+----------------------------+
  |  LIVE CAMERA  (tvkur iframe)          |  INSIGHTS                  |
  |                                       |  - latest counts           |
  |                                       |  - 1h avg, peak hour       |
  |                                       |  - anomaly flag            |
  |                                       |  - re-ID stats             |
  +---------------------------------------+----------------------------+
  |  FOOTFALL OVER TIME (line chart, last 200 samples)                 |
  +--------------------------------------------------------------------+
  |  MODEL PREDICTION (latest annotated YOLO frame)  |  TOP REGULARS  |
  +--------------------------------------------------------------------+

Reads `data/footfall.db` (filled by collector.py), `data/reid.db` (re-ID registry),
and `data/frames/latest_{cam_id}.jpg` (annotated YOLO output written by the collector).
Auto-refreshes every REFRESH_SEC seconds.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.cameras import CAMERAS

DB_PATH = ROOT / "data" / "footfall.db"
REID_DB = ROOT / "data" / "reid.db"
FRAMES_DIR = ROOT / "data" / "frames"
REFRESH_SEC = 15

# tvkur iframe URL for each camera (only filled for cameras we've resolved).
LIVE_PLAYER_URL = {
    "konya_hukumet": "https://player.tvkur.com/l/c77i84vbb2nj4i0fr80g",
}

st.set_page_config(page_title="Turkey Business Activity — Live",
                   layout="wide", initial_sidebar_state="collapsed")

st.title("Turkey Business Activity — Live Footfall + Re-ID")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="refresh")
except ImportError:
    st.caption("Tip: `pip install streamlit-autorefresh` for hands-free live updates.")


# ---------------- data loaders ----------------

@st.cache_data(ttl=REFRESH_SEC)
def load_footfall() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query("SELECT * FROM footfall WHERE ok = 1", conn)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df["hour"] = df["ts"].dt.hour
    return df


@st.cache_data(ttl=REFRESH_SEC)
def load_reid_stats(cam_id: str) -> dict:
    if not REID_DB.exists():
        return {}
    with sqlite3.connect(str(REID_DB)) as conn:
        cur = conn.execute(
            "SELECT cls, COUNT(*), SUM(sightings), "
            "       SUM(CASE WHEN sightings>=3 THEN 1 ELSE 0 END) "
            "FROM entities WHERE cam_id=? GROUP BY cls", (cam_id,))
        per_class = {r[0]: {"unique": r[1], "total_sightings": r[2],
                            "regulars": r[3]} for r in cur.fetchall()}
        total = conn.execute("SELECT COUNT(*), SUM(sightings) FROM entities "
                             "WHERE cam_id=?", (cam_id,)).fetchone()
        regulars = conn.execute(
            "SELECT entity_id, cls, sightings, first_seen, last_seen "
            "FROM entities WHERE cam_id=? ORDER BY sightings DESC LIMIT 10",
            (cam_id,)).fetchall()
    return {
        "per_class":       per_class,
        "total_unique":    total[0] or 0,
        "total_sightings": total[1] or 0,
        "top_regulars":    [{"entity_id": r[0], "cls": r[1], "sightings": r[2],
                             "first_seen": r[3], "last_seen": r[4]} for r in regulars],
    }


def rolling_z(s: pd.Series, window: int = 12, thresh: float = 2.5) -> pd.Series:
    mu = s.rolling(window, min_periods=4).mean()
    sd = s.rolling(window, min_periods=4).std().replace(0, np.nan)
    return ((s - mu) / sd).abs() > thresh


# ---------------- camera select ----------------

df = load_footfall()
if df.empty:
    st.warning("No data yet. Start the collector first:\n\n"
               "    python -m app.collector --backend sqlite --interval 15 --only konya_hukumet\n")
    st.stop()

cams = sorted(df["cam_id"].unique())
cam_id = st.selectbox("Camera",
                      cams,
                      format_func=lambda c: CAMERAS.get(c, {}).get("name", c))
cam_name = CAMERAS.get(cam_id, {}).get("name", cam_id)
d = df[df["cam_id"] == cam_id].sort_values("ts")
d["anomaly"] = rolling_z(d["person"])
latest = d.iloc[-1]


# ---------------- row 1: live camera + insights side by side ----------------

cam_col, insights_col = st.columns([3, 2], gap="large")

with cam_col:
    st.subheader(f"Live: {cam_name}")
    embed_url = LIVE_PLAYER_URL.get(cam_id)
    if embed_url:
        st.components.v1.iframe(embed_url, height=400, scrolling=False)
        st.caption(f"Stream source: {embed_url}")
    else:
        st.info("No live-player URL configured for this camera. "
                "Add it to `LIVE_PLAYER_URL` in app/streamlit_app.py to enable the embed.")

with insights_col:
    st.subheader("Insights")
    c1, c2 = st.columns(2)
    c1.metric("People (latest)", int(latest["person"]))
    c2.metric("Vehicles (latest)", int(latest["vehicles"]))

    one_hour = d[d["ts"] >= d["ts"].max() - pd.Timedelta(hours=1)]
    c3, c4 = st.columns(2)
    c3.metric("People — 1h avg", round(one_hour["person"].mean(), 1))
    if not d.groupby("hour")["person"].mean().empty:
        peak_hour = int(d.groupby("hour")["person"].mean().idxmax())
        c4.metric("Peak hour (UTC)", f"{peak_hour:02d}:00")

    anomalies = d[d["anomaly"]]
    if not anomalies.empty:
        last_an = anomalies.iloc[-1]
        st.error(f"Anomaly at {last_an['ts'].strftime('%H:%M:%S')}: "
                 f"{int(last_an['person'])} people (z-score > 2.5). "
                 f"{len(anomalies)} total in window.")
    else:
        st.success("No anomalies in the current window.")

    st.caption(f"Last update: {latest['ts'].strftime('%Y-%m-%d %H:%M:%S')} UTC · "
               f"{len(d)} samples in DB · auto-refresh every {REFRESH_SEC}s")

    # re-ID block
    reid = load_reid_stats(cam_id)
    if reid:
        st.markdown("##### Re-identification")
        r1, r2, r3 = st.columns(3)
        r1.metric("Unique entities", reid["total_unique"])
        r2.metric("Total sightings", int(reid["total_sightings"] or 0))
        regulars = sum(s["regulars"] for s in reid["per_class"].values())
        r3.metric("Regulars (≥3 sightings)", regulars)
        if reid["per_class"]:
            for cls, s in reid["per_class"].items():
                st.caption(f"{cls}: {s['unique']} unique · "
                           f"{s['total_sightings']} sightings · "
                           f"{s['regulars']} regulars")


# ---------------- row 2: footfall time series ----------------

st.subheader("Footfall over time")
chart = d.set_index("ts")[["person", "vehicles"]].tail(200)
st.line_chart(chart, height=240)


# ---------------- row 3: model prediction + top regulars ----------------

pred_col, regs_col = st.columns([3, 2], gap="large")

with pred_col:
    st.subheader("Model prediction — latest YOLO frame")
    annotated = FRAMES_DIR / f"latest_{cam_id}.jpg"
    if annotated.exists():
        st.image(str(annotated),
                 caption=f"YOLO @ conf=0.25 on {cam_name} — boxes show classified detections",
                 use_container_width=True)
    else:
        st.info(f"No annotated frame yet at `{annotated}`. The collector writes one per "
                f"sample (`--frames-dir data/frames`).")

with regs_col:
    st.subheader("Top returning entities")
    if reid and reid["top_regulars"]:
        regulars_df = pd.DataFrame(reid["top_regulars"])
        regulars_df["last_seen"] = pd.to_datetime(regulars_df["last_seen"]).dt.strftime("%H:%M:%S")
        regulars_df["first_seen"] = pd.to_datetime(regulars_df["first_seen"]).dt.strftime("%H:%M:%S")
        st.dataframe(
            regulars_df[["entity_id", "cls", "sightings", "first_seen", "last_seen"]],
            hide_index=True, use_container_width=True,
        )
        st.caption("Entity IDs are persistent across collector restarts (stored in "
                   "`data/reid.db`). Demo-grade HSV embedding — tune the threshold in "
                   "the collector if you see too many merges or splits.")
    else:
        st.info("Re-ID registry empty. The collector populates it as detections come in.")
