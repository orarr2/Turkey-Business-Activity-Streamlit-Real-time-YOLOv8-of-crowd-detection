# Turkey Business Activity - Live YOLOv8 Crowd Detection

Turn public live-stream cameras in Turkey into quantitative time series:

> **live HLS stream → YOLOv8 frame inference → counts + appearance re-ID → Firestore →
> real-time web dashboard + Jupyter analytics.**

The project samples a handful of street / market / square cameras every 20 s, runs
YOLOv8 on each frame, writes the counts and a per-detection appearance signature to
Firestore, and pushes the result to a browser dashboard via `onSnapshot` - no polling,
no refresh, everything updates the moment the collector posts a new sample.


> All source, configs and the notebook live in [`src/`](src/). The repo root only carries this `README.md` and the gitignore so the GitHub landing page stays clean.

---

## What the program does, end to end

```
 ┌───────────────────────┐    ┌────────────────────────┐    ┌────────────────────┐
 │  Live cameras         │    │  Cloud collector       │    │  Firebase          │
 │  (IBB istanbuluseyret,│ ─► │  GCP e2-micro VM       │ ─► │  Firestore (24h TTL)│
 │   webcamera24 tvkur)  │    │  • fallback per slot   │    │   footfall/{auto}   │
 │                       │    │  • YOLOv8n predict     │    │   latest/{slot_id}  │
 │                       │    │  • appearance re-ID    │    │   reid_stats/{slot} │
 │                       │    │  • anomaly gates       │    │   config/grid       │
 │                       │    │  • Storage snapshots   │    │  Storage (24h)      │
 └───────────────────────┘    └────────────────────────┘    └──────────┬─────────┘
                                                                       │ onSnapshot
                                                                       ▼
                                           ┌────────────────────────────────────────┐
                                           │  web/  static HTML dashboard            │
                                           │  • 4-slot grid with active-cam badge   │
                                           │  • per-tile mini chart + anomaly       │
                                           │  • combined 24 h chart                  │
                                           │  • re-ID summary table                  │
                                           └────────────────────────────────────────┘
```

The two halves are decoupled. The collector runs 24/7 on a GCP e2-micro
(Always Free). The dashboard is plain HTML/JS — anyone can serve `web/` and
subscribe to the live data. Because the state lives in Firestore, every visitor
sees the accumulated history, and Firestore's TTL policy prunes the last 24h to
keep the DB small. Anomaly / returning-visitor snapshots go to Firebase Storage
(also 24h lifecycle).

Cameras are grouped into **4 grid slots** (2 Konya, 2 Istanbul). Each slot has a
primary cam and a fallback chain within the same source site
(webcamera24/tvkur for Konya; istanbuluseyret.ibb.gov.tr for Istanbul). If the
primary fails 3 samples in a row the collector switches to the next cam in the
chain and updates `config/grid` — the dashboard re-renders that tile with the
new active cam. Every 15 min it retries the primary.

---

## Quick start

The project ships zero-config for **viewers** — the Firebase Web SDK identifier
is committed, Firestore Rules make the four public collections read-only, the
cloud collector is running, and the dashboard just lights up.

```bash
# Anyone who clones the repo
pip install -r src/requirements.txt
jupyter lab turkey_business_activity.ipynb   # notebook lives at the repo root;
                                             # imports find app/ under src/ automatically
# or just the dashboard (no notebook, no analysis):
cd src && python serve.py                    # opens http://localhost:8000 with live counts
```

Cloud deployment (for the maintainer only, requires a Firebase Admin
service-account key) lives in [`src/deploy/gcp-vm/`](src/deploy/gcp-vm/README.md).

`serve.py` is a small no-cache static server that binds `web/` on port 8000 (override
with `--port`, suppress the browser pop with `--no-browser`, auto-falls-back to the
next free port if 8000 is busy).

Full step-by-step (Python venv, Windows PowerShell variants, troubleshooting): see
[`LOCAL_RUN.md`](src/LOCAL_RUN.md). Firebase project/service-account setup and security
rules: see [`docs/firebase_setup.md`](src/docs/firebase_setup.md).

---

## What the model predicts

**Detector - YOLOv8n (Ultralytics)** loaded once per process with
[`load_model("yolov8n.pt")`](src/app/detect_core.py:30). Nano variant for CPU-friendly
inference; swap to `yolov8s.pt` / `yolov8m.pt` for better small-object recall.

Each call returns boxes + class ids + confidences for the **COCO classes the project
cares about** ([`CLASSES_OF_INTEREST`](src/app/detect_core.py:18)):

| COCO id | name        | role                                       |
|:-------:|-------------|--------------------------------------------|
| 0       | `person`    | the primary footfall signal                |
| 1       | `bicycle`   | vehicle bucket                             |
| 2       | `car`       | vehicle bucket                             |
| 3       | `motorcycle`| vehicle bucket                             |
| 5       | `bus`       | vehicle bucket                             |
| 7       | `truck`     | vehicle bucket                             |

`detect_with_boxes(frame, conf, imgsz)` returns:

```python
counts = {
    "person": 23, "car": 4, "bus": 0, "truck": 1, "bicycle": 0, "motorcycle": 2,
    "vehicles": 7,   # sum of all non-person classes above
}
boxes = [{"x1":…, "y1":…, "x2":…, "y2":…, "cls":"person", "conf":0.71}, …]
```

**Burst-median sampling.** A single frame is a noisy estimator - a pedestrian
occluded for a moment, or a car at the edge of the confidence band, flips the
count between consecutive frames. Each sampling round therefore grabs a short
burst (default 3 frames ~1 s apart), detects on every frame, and keeps the
**median** count per class ([`grab_burst` / `detect_burst`](src/app/detect_core.py)).
Re-ID and snapshots use the frame whose count matches the median, so images and
numbers stay consistent. The raw per-frame series is kept on each doc (`burst`)
for transparency.

**Input size + confidence.** The collector runs at `--imgsz 960` (vs YOLO's 640
default): these wide street shots shrink a distant pedestrian or car to a few
pixels, and at 640 the model undercounts badly. Default confidence is
`--conf 0.30`, and any camera can carry its own calibrated `"conf"` override in
[`cameras.py`](src/app/cameras.py) - notebook section 10 measures MAE/bias per
camera and per input size against your own manual counts and tells you what to set.

Per sampling round the collector writes:

- **`footfall/{auto-id}`** - append-only history doc:
  `{ts, slot, cam_id, cam_name, person, vehicles, counts, burst, is_night, ok,
  is_anomaly, anomaly?, new_entities, seen_entities, expire_at}`.
  Powers the 24 h charts, the anomaly badges and the events table.
  `expire_at` is 24h ahead; Firestore's TTL policy auto-deletes expired docs.
- **`latest/{slot_id}`** - overwritten each sample. Powers the "now" KPI tiles cheaply
  (one doc per slot, not a full history scan). Contains the current `cam_id`
  so the dashboard can label the tile with which cam is active right now.
- **`reid_stats/{slot_id}`** - overwritten each sample with the appearance-registry
  rollup for the currently-active camera in that slot.
- **`config/grid`** - one document, updated whenever a slot switches cameras.
  Lists the active_cam / embed URL / display area for each of the 4 slots.
  The dashboard subscribes to this and re-renders when a fallback happens.
- **`config/profile_{slot_id}`** - the hour-of-week activity baseline (running
  mean/std per `(day-of-week, hour)` bucket, per metric), persisted every
  ~30 min and reloaded on startup.

### Anomaly detection - two layers × two metrics, decided server-side

The collector - not the browser - decides what is anomalous; the dashboard
renders its verdicts verbatim (`is_anomaly` + the `anomaly` map on each doc),
so the badge, the events table and the snapshots always agree. Both **people
and vehicles** are tracked per slot, because "business activity" on these
streets is foot traffic *and* vehicle traffic:

1. **Rolling window** (last 30 samples ≈ 20 min) - robust z-score built on
   **median + MAD** instead of mean/std, so outliers already inside the window
   can't inflate the spread and mask the next event.
   - `spike` - z ≥ 3.5 with an absolute floor (≥ 5 people / ≥ 4 vehicles moved);
   - `drop` - z ≤ -3 while the recent median is itself busy (≥ 8 people /
     ≥ 6 vehicles): "the street just emptied" fires, a quiet street sitting
     at 0 stays silent.
2. **Hour-of-week profile** ([`HourlyProfile`](src/app/collector.py)) - a
   Welford running mean/std per `(day-of-week, hour)` bucket in Turkey local
   time. Once a bucket has ≥ 10 samples, values far outside it flag as
   `contextual_spike` / `contextual_drop` - "this is not what a Wednesday
   14:00 looks like here" - which catches slow build-ups and dead-at-rush-hour
   cases the 20-minute window can't see.

Every verdict carries `observed` vs `expected` (+ z and the hour bucket), each
event saves a raw + annotated snapshot, per-slot cooldowns throttle repeats
(5 min rolling / 30 min contextual), and flagged samples are excluded from the
baselines so an ongoing event can't normalize itself. On startup the collector
reseeds its rolling windows from the last hour of Firestore history and reloads
the persisted profiles - a service restart doesn't re-warm from zero.

### Re-identification ("have I seen this person/car before?")

Implemented in [`app/reid.py`](src/app/reid.py) - deliberately dependency-free
(no torchreid / no OSNet, so the notebook + collector + dashboard all share it):

1. Crop each detection box and resize (`64×128` for persons, `96×96` for vehicles).
2. Convert to HSV, mask out very dark pixels (V<30) so night-time sodium-light
   background doesn't dominate the signature.
3. Build a `8×8×8` HSV color histogram, append `[aspect_ratio, normalized_area]`,
   L2-normalize → **514-dimensional unit vector**.
4. Store in SQLite (`data/reid.db`). On every new detection, cosine-match against
   the same camera × same class entities; if best similarity ≥ `--reid-threshold`
   (default 0.92) bump `sightings`, otherwise insert a fresh entity.

What the dashboard surfaces from this: per-camera **unique entities**, **total
sightings**, and **regulars** (entities seen ≥ 3 times) in the bottom table -
labeled as an **appearance-based estimate**, which is what it is: two people in
similar clothing can merge, the same person can split after a lighting change.
Accurate for trends in daylight, weaker at night. The registry is pruned of
entities not seen for 48 h (on startup + every 6 h) so it stays small and the
counts describe the recent crowd, not everything since the DB was created. Swap
`embed_crop()` for an OSNet/torchreid forward pass when you need production
accuracy; the SQLite registry stays the same.

### Stream resolution

Cameras come in several `kind`s - [`resolve_stream`](src/app/detect_core.py:93) routes
each one through the right resolver:

| kind          | example                          | how it's resolved              |
|---------------|----------------------------------|--------------------------------|
| `hls`         | IBB livestream `.m3u8`           | used as-is                     |
| `youtube`     | YouTube Live page                | `yt-dlp` → HLS                 |
| `skyline`     | skylinewebcams.com page          | scrape rotating HLS token      |
| `webcamera24` | webcamera24.com page             | pull embedded tvkur player id  |

Some HLS hosts (`content.tvkur.com`, `livestream.ibb.gov.tr`, `skylinewebcams.com`)
require `Referer` / `Origin` headers that `cv2.VideoCapture` can't set on Windows;
for those the collector downloads the latest `.ts` segment manually with the right
headers and decodes locally ([`_grab_via_segment`](src/app/detect_core.py:133)).

---

## The dashboard (`web/`)

Pure static page - no build step. Module ES imports, Firebase web SDK v10,
Chart.js 4. Opens with [`python serve.py`](src/serve.py) and renders:

- **2×2 camera grid** - each tile has a live iframe (tvkur player or a
  corsproxy.io-wrapped page for hosts with strict `X-Frame-Options`), four KPIs
  (people now, vehicles now, 24 h average, 24 h peak), an anomaly badge showing
  the collector's latest verdict (▲ spike / ▼ drop, which metric, observed vs
  expected), and a per-tile mini chart of the last 30 samples with anomalous
  points enlarged in red on the series that fired.
- **Combined 24 h chart** stacking all four cameras' people series.
- **Anomaly events table** - every flagged sample of the last 24 h across all
  slots: when, where, spike or drop, people or vehicles, observed vs expected
  (with the hour-of-week norm when the contextual layer fired), and a link to
  the saved snapshot. This is the collector's own log - nothing is recomputed
  client-side.
- **Re-ID summary table** - unique entities / total sightings / regulars per
  cam, tagged as an appearance-based estimate.
- **Status pill** in the header - `live` when every camera reported within 120 s,
  `partial` if some are stale, `down` if Firestore has no recent writes (usually
  means the collector isn't running).

Connection state is `connection refused` when nothing is bound to port 8000 - that
is the role of [`serve.py`](src/serve.py). The dashboard itself doesn't open any port;
it just talks to Firestore from the browser tab.

---

## Notebook - `turkey_business_activity.ipynb`

`jupyter lab turkey_business_activity.ipynb` opens the offline analysis side:

- footfall time series + diurnal pattern + peak-hour bands per camera,
- robust (median + MAD) rolling anomaly markers on the same series - the same
  statistic the cloud collector uses,
- dwell-time and prolonged-stops via ByteTrack on consecutive frames,
- appearance-registry summary (regulars, unique counts) read from `data/reid.db`,
- site-selection composite score combining footfall, dwell time, and consistency,
- **section 10: accuracy calibration** - capture frames from the live grid
  cameras, count people/vehicles yourself, and get MAE + bias per camera and
  per input size (640 vs 960) plus a concrete `conf`/`imgsz` recommendation.

Reuses the exact same `detect_core` + `reid` modules as the collector so the
numbers reconcile.

---

## Camera catalog

[`app/cameras.py`](src/app/cameras.py) is the source of truth. The four cameras shipped
in the dashboard grid (`GRID_CAMERAS`):

| id                       | name                                   | host         |
|--------------------------|----------------------------------------|--------------|
| `konya_hukumet`          | Konya - Hükümet / Sarraflar Yeraltı    | tvkur        |
| `otogar_kavsagi`         | Konya - Otogar Kavşağı                 | tvkur        |
| `konya_kulturpark`       | Konya - Kültürpark                     | tvkur        |
| `konya_millet_caddesi`   | Konya - Millet / Hastane Kavşağı       | tvkur        |

IBB Istanbul streams (`taksim`, `kapali_carsi`, `misir_carsisi`, `sultanahmet_1`,
`kadikoy`, `eyup_sultan`, `uskudar`, `beyazit_meydan`) and `giresun_gazi`
(skylinewebcams) are in the catalog but **geo-restricted** to a Turkey-routed IP.
Run the collector from a Turkey VPN/VPS to populate those tiles too. From any other
network you'll see `MISS` rows for them and the dashboard will leave them blank.

Verifying a stream resolves on your network:

```bash
python -m app.detect_core --resolve konya_hukumet,otogar_kavsagi
```

---

## Operational notes

- **Storage:** Firestore free tier ≈ 20 k writes/day. At one write per camera per
  20 s that is ~4,300 writes/day/camera. Stay modest on free tier, raise
  `--interval`, or batch. For many cameras at high frequency keep only `latest` in
  Firestore and ship `footfall` to BigQuery instead.
- **Privacy by design:** the collector stores **aggregate counts** (and an HSV
  histogram appearance hash for re-ID), never raw frames of people. Crops live in
  memory only and are dropped after embedding.
- **Scope:** only public, intentionally-public cameras (city tourism cams,
  official infrastructure feeds, market broadcasters). Cameras exposed to the
  internet without owner consent are explicitly out of scope.

---

## Repo map

| Path | Purpose |
|------|---------|
| [`serve.py`](src/serve.py) | One-shot launcher for the dashboard (no-cache static server). |
| [`turkey_business_activity.ipynb`](src/turkey_business_activity.ipynb) | Offline analytics notebook. |
| [`app/collector.py`](src/app/collector.py) | 24/7 sampler that writes to Firestore. |
| [`app/detect_core.py`](src/app/detect_core.py) | YOLO loading, stream resolution, frame grabbing, detection. |
| [`app/reid.py`](src/app/reid.py) | Appearance-based re-identification (SQLite + HSV histograms). |
| [`app/cameras.py`](src/app/cameras.py) | Verified camera catalog. |
| [`app/firebase_store.py`](src/app/firebase_store.py) | Firestore writer (`footfall` / `latest` / `reid_stats`). |
| [`web/`](src/web/) | Static HTML/JS dashboard. |
| [`docs/firebase_setup.md`](src/docs/firebase_setup.md) | Firebase project + security rules walkthrough. |
| [`docs/turkey_cameras.md`](src/docs/turkey_cameras.md) | Camera sources and architecture notes. |
| [`LOCAL_RUN.md`](src/LOCAL_RUN.md) | Step-by-step local-machine quickstart. |
