"""Live business-activity dashboard — 4 cameras side by side, last 24 hours.

  +---------------------------+---------------------------+
  |  Konya  (live + YOLO)     |  Giresun (live + YOLO)    |
  |  24h metrics + anomaly    |  24h metrics + anomaly    |
  +---------------------------+---------------------------+
  |  Otogar (live + YOLO)     |  Kadikoy (live + YOLO)    |
  |  24h metrics + anomaly    |  24h metrics + anomaly    |
  +---------------------------+---------------------------+
  |  FOOTFALL — last 24h, all four cameras (line chart)   |
  +-------------------------------------------------------+
  |  RE-ID SUMMARY per camera                             |
  +-------------------------------------------------------+

Each tile shows the live player (iframe embed where available) next to the latest
annotated YOLO frame, plus the camera's last-24h counts, peak hour and anomaly flag.

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
from app.cameras import CAMERAS, GRID_CAMERAS

DB_PATH = ROOT / "data" / "footfall.db"
REID_DB = ROOT / "data" / "reid.db"
FRAMES_DIR = ROOT / "data" / "frames"
REFRESH_SEC = 15
WINDOW_HOURS = 24

st.set_page_config(page_title="Turkey Business Activity — Live",
                   layout="wide", initial_sidebar_state="collapsed")

st.title("Turkey Business Activity — Live Footfall (4 cameras · last 24h)")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="refresh")
except ImportError:
    st.caption("Tip: `pip install streamlit-autorefresh` for hands-free live updates.")


# ---------------- data loaders ----------------

@st.cache_data(ttl=REFRESH_SEC)
def load_footfall() -> pd.DataFrame:
    """All decoded footfall rows within the last WINDOW_HOURS."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query("SELECT * FROM footfall WHERE ok = 1", conn)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = df["ts"].max() - pd.Timedelta(hours=WINDOW_HOURS)
    df = df[df["ts"] >= cutoff].copy()
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
    return {
        "per_class":       per_class,
        "total_unique":    total[0] or 0,
        "total_sightings": total[1] or 0,
    }


def rolling_z(s: pd.Series, window: int = 12, thresh: float = 2.5) -> pd.Series:
    mu = s.rolling(window, min_periods=4).mean()
    sd = s.rolling(window, min_periods=4).std().replace(0, np.nan)
    return ((s - mu) / sd).abs() > thresh


# ---------------- per-camera tile ----------------

def render_tile(cam_id: str, df: pd.DataFrame) -> None:
    cam = CAMERAS.get(cam_id, {})
    cam_name = cam.get("name", cam_id)
    st.markdown(f"#### {cam_name}")

    live_col, det_col = st.columns(2, gap="small")

    # left: live player (iframe) where we have an embeddable URL
    with live_col:
        embed = cam.get("embed")
        if embed:
            st.components.v1.iframe(embed, height=220, scrolling=False)
            st.caption("Live stream")
        else:
            st.info("No iframe embed for this camera — the YOLO frame (right) is the live view.")
            if cam.get("page"):
                st.caption(f"Source: {cam['page']}")

    # right: latest annotated YOLO frame = the live detection
    with det_col:
        annotated = FRAMES_DIR / f"latest_{cam_id}.jpg"
        if annotated.exists():
            st.image(str(annotated), caption="Latest YOLO detection",
                     use_container_width=True)
        else:
            st.info("No annotated frame yet (collector writes one per sample).")

    # metrics for this camera over the last 24h
    d = df[df["cam_id"] == cam_id].sort_values("ts")
    if d.empty:
        st.caption("No footfall data in the last 24h for this camera. "
                   f"Run the collector with `--only {cam_id}` (open network required).")
        st.divider()
        return

    d = d.copy()
    d["anomaly"] = rolling_z(d["person"])
    latest = d.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("People (now)", int(latest["person"]))
    m2.metric("Vehicles (now)", int(latest["vehicles"]))
    m3.metric("People — 24h avg", round(d["person"].mean(), 1))
    hourly = d.groupby("hour")["person"].mean()
    if not hourly.empty:
        m4.metric("Peak hour (UTC)", f"{int(hourly.idxmax()):02d}:00")

    anomalies = d[d["anomaly"]]
    if not anomalies.empty:
        last_an = anomalies.iloc[-1]
        st.error(f"⚠ Anomaly at {last_an['ts'].strftime('%H:%M:%S')} UTC: "
                 f"{int(last_an['person'])} people (z>2.5). {len(anomalies)} in 24h.")
    else:
        st.success("No anomalies in the last 24h.")

    st.caption(f"Last update {latest['ts'].strftime('%H:%M:%S')} UTC · "
               f"{len(d)} samples / 24h")
    st.divider()


# ---------------- layout ----------------

df = load_footfall()
if df.empty:
    st.warning(
        "No data yet. Start the collector (open network required) — it samples all four "
        "grid cameras:\n\n"
        "    python -m app.collector --interval 20 "
        "--only konya_hukumet,giresun_gazi,otogar_kavsagi,kadikoy\n")
    st.stop()

st.caption(f"Auto-refresh every {REFRESH_SEC}s · window = last {WINDOW_HOURS}h · "
           f"{df['ts'].min():%Y-%m-%d %H:%M} → {df['ts'].max():%Y-%m-%d %H:%M} UTC")

# 2x2 grid of the four cameras
row1 = st.columns(2, gap="large")
row2 = st.columns(2, gap="large")
cells = [row1[0], row1[1], row2[0], row2[1]]
for cell, cam_id in zip(cells, GRID_CAMERAS):
    with cell:
        render_tile(cam_id, df)

# combined 24h footfall comparison
st.subheader("Footfall over the last 24h — all cameras")
pivot = (df[df["cam_id"].isin(GRID_CAMERAS)]
         .assign(cam=lambda x: x["cam_id"].map(
             lambda c: CAMERAS.get(c, {}).get("name", c)))
         .pivot_table(index="ts", columns="cam", values="person"))
if not pivot.empty:
    st.line_chart(pivot, height=280)
else:
    st.info("No comparable footfall series yet for the grid cameras.")

# re-ID summary across the grid
st.subheader("Re-identification — unique vs returning (per camera)")
reid_rows = []
for cam_id in GRID_CAMERAS:
    s = load_reid_stats(cam_id)
    if not s:
        continue
    regulars = sum(c["regulars"] for c in s["per_class"].values())
    reid_rows.append({"camera": CAMERAS.get(cam_id, {}).get("name", cam_id),
                      "unique entities": s["total_unique"],
                      "total sightings": int(s["total_sightings"] or 0),
                      "regulars (≥3)": regulars})
if reid_rows:
    st.dataframe(pd.DataFrame(reid_rows), hide_index=True, use_container_width=True)
else:
    st.info("Re-ID registry empty. The collector populates it as detections come in.")
