"""Read footfall.db and emit a plot + stats, mirroring the notebook's Section 4 analysis.

If FIREBASE_CREDENTIALS is set, also push every row to Firestore via FirebaseStore.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "footfall.db"
PNG = ROOT / "data" / "footfall.png"

conn = sqlite3.connect(str(DB))
df = pd.read_sql_query(
    "SELECT ts, cam_id, cam_name, person, vehicles, ok FROM footfall ORDER BY ts", conn
)
conn.close()
print(f"rows: {len(df)}")
print(df.tail(15).to_string(index=False))

real = df[df["cam_id"].str.startswith("konya")].copy()
print(f"\nKonya rows: {len(real)}  (ok rows: {(real['ok']==1).sum()})")
if len(real) >= 2:
    real["ts"] = pd.to_datetime(real["ts"])
    p, v = real["person"].dropna(), real["vehicles"].dropna()
    print(f"person  : min={p.min()}  median={p.median():g}  max={p.max()}  total_obs={p.sum()}")
    print(f"vehicles: min={v.min()}  median={v.median():g}  max={v.max()}  total_obs={v.sum()}")

    # rolling z-score anomaly flag (window=4 since the run is short)
    win = max(3, min(8, len(real) // 2))
    mu = real["person"].rolling(win, min_periods=3).mean()
    sd = real["person"].rolling(win, min_periods=3).std().replace(0, np.nan)
    real["anomaly"] = ((real["person"] - mu) / sd).abs() > 2.0

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(real["ts"], real["person"], marker="o", label="person")
    ax[0].plot(real["ts"], real["vehicles"], marker="s", label="vehicles", alpha=0.7)
    an = real[real["anomaly"] == True]
    if len(an):
        ax[0].scatter(an["ts"], an["person"], color="red", zorder=5, s=80, label="anomaly")
    ax[0].set_title(f"Konya Hukumet Meydani — live counts ({len(real)} samples)")
    ax[0].set_ylabel("count per frame"); ax[0].legend()
    ax[0].tick_params(axis="x", rotation=30)

    real["hour"] = real["ts"].dt.hour
    real.groupby("hour")["person"].mean().plot(kind="bar", ax=ax[1])
    ax[1].set_title("Avg person count by UTC hour")
    ax[1].set_xlabel("hour (UTC)")

    plt.tight_layout()
    plt.savefig(PNG, dpi=110)
    print(f"\nplot -> {PNG}")

# Firebase upload if creds present
cred = os.environ.get("FIREBASE_CREDENTIALS")
if cred and os.path.exists(cred):
    print(f"\n--- Pushing {len(df)} rows to Firestore (cred: {cred}) ---")
    sys.path.insert(0, str(ROOT))
    from app.firebase_store import FirebaseStore
    fb = FirebaseStore()
    pushed = 0
    for _, row in df.iterrows():
        record = {
            "ts": row["ts"], "cam_id": row["cam_id"], "cam_name": row["cam_name"],
            "person": int(row["person"]) if pd.notna(row["person"]) else None,
            "vehicles": int(row["vehicles"]) if pd.notna(row["vehicles"]) else None,
            "counts": json.dumps({"person": row["person"], "vehicles": row["vehicles"]}),
            "ok": int(row["ok"]),
        }
        fb.write(record); pushed += 1
    print(f"pushed {pushed} rows to Firestore (collections: footfall, latest)")
else:
    print(f"\nFIREBASE_CREDENTIALS not set or path doesn't exist (cred={cred!r}).")
    print("To push to Firestore:")
    print("  1. Firebase console -> Project settings -> Service accounts -> Generate new private key")
    print("  2. Save the JSON, set:   $env:FIREBASE_CREDENTIALS = 'C:\\path\\to\\serviceAccount.json'")
    print("  3. Re-run:   python scripts/analyze_footfall.py        (to push the existing rows)")
    print("     or:       python -m app.collector --backend firebase --interval 15 --only konya_hukumet")
