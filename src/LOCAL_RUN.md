# Run locally - Turkey footfall (Firebase)

There are two roles here, and they need different amounts of setup:

- **Viewer** (anyone who clones the repo): open the notebook, Run All, done.
  Live counts stream in from the admin's collector via the shared Firestore
  project. Zero configuration - the Firebase Web SDK config ships in the repo,
  Firestore Rules make it read-only for the public.
- **Admin** (the one machine that also *writes* new samples to Firestore):
  additionally holds the Firebase service-account JSON and runs the collector.

## Viewer path (default)

## Prerequisites
- Python 3.10+ and `pip`
- Open internet (so the IBB / YouTube streams resolve)

## 1. Get the code
```bash
git clone <your-repo-url> turkey-footfall
cd turkey-footfall/src        # all code + configs live in src/; the repo
                              # root only carries README.md
```

## 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Run the notebook
```bash
jupyter lab turkey_business_activity.ipynb
```
Run All. Cells 0-23 do the local YOLO analysis (offline). The last live cell
opens the dashboard at http://localhost:8000 with real-time counts pushed by
the admin's collector. Nothing else to configure.

---

## Admin path (only for the person who runs the collector)

Everything in the viewer path, plus:

## 4. Drop in the service-account key
Put your Firebase service-account JSON at the repo root (any of these names is
auto-detected by the notebook: `serviceAccount.json`,
`*firebase-adminsdk*.json`). Or set the env var yourself:
```bash
export FIREBASE_CREDENTIALS="$PWD/serviceAccount.json"          # Mac/Linux
# Windows PowerShell:  $env:FIREBASE_CREDENTIALS = "$PWD\serviceAccount.json"
```
The JSON is gitignored - it never enters the repo. This is the only real
secret in the project.

## 5. Smoke test (one camera)
```bash
python -m app.collector --interval 20 --only konya_hukumet
```
Expect lines like `konya_hukumet: person=23 vehicles=2  new=… seen_again=…` every 20s.
`Ctrl+C` to stop. (First run downloads the YOLO weights `yolov8n.pt`, ~6 MB.)

## 6. Run for real - two terminals

Anyone who opens the dashboard sees the counts and charts the collector has been
accumulating in Firestore - it does *not* reset per visitor.

**Terminal 1 - collector (the 4 grid cameras, leave running):**
```bash
python -m app.collector --interval 20 \
    --only konya_hukumet,giresun_gazi,otogar_kavsagi,kadikoy
```
The collector pushes three things to Firestore each sample:
- `footfall/{auto-id}` - append-only history (24h chart + anomaly z-score)
- `latest/{cam_id}` - overwritten each sample (the big "now" numbers)
- `reid_stats/{cam_id}` - unique entities + total sightings + regulars

**Terminal 2 - the 4-camera live HTML dashboard:**
```bash
python serve.py             # one-shot launcher: serves web/ on :8000 and opens the browser
# alternatives:
#   python serve.py --port 8765        pick a different port
#   python serve.py --no-browser       skip auto-opening the browser
#   cd web && python -m http.server 8000   plain http.server, same result, no niceties
```
Open **http://localhost:8000**. 2×2 grid: live video iframe + people/vehicle counts +
anomaly badge + mini chart per camera, a combined 24h chart for all four cameras, and
the re-ID summary table. Everything updates via `onSnapshot` - no polling, no refresh.

If the browser shows `ERR_CONNECTION_REFUSED` you just don't have anything bound to
port 8000 yet - run `python serve.py` from the project root and refresh.

## Camera ids (from `app/cameras.py`)
**Dashboard grid (`GRID_CAMERAS`):** `konya_hukumet`, `giresun_gazi`, `otogar_kavsagi`, `kadikoy`.
Others: `taksim`, `beyazit_meydan`, `kapali_carsi` (Grand Bazaar), `misir_carsisi` (Spice Bazaar),
`sultanahmet_1`, `eyup_sultan`, `uskudar`.
Drop `--only` to run all of them.

> `giresun_gazi` (skylinewebcams) and `otogar_kavsagi` (webcamera24) resolve from their public pages
> and rotate tokens - verify once with `python -m app.detect_core --resolve giresun_gazi,otogar_kavsagi`.

## Troubleshooting
- **`MISS (empty frame)`** on every round → that stream is down or your network blocks it. Try another
  camera id; confirm you are on an open network (not a corporate/VPN filter).
- **`FileNotFoundError ... service-account`** → `FIREBASE_CREDENTIALS` is unset or the path is wrong.
- **Dashboard shows "no data"** → the admin's collector isn't running right now, or someone deleted `web/firebase-config.js` from the repo (it should be committed).
- **`firebase write failed ... PERMISSION_DENIED`** → you are not using the service-account key (the
  Admin SDK bypasses the read-only rules; the web apiKey does not).
- **Notebook instead of app** → `jupyter lab turkey_business_activity.ipynb` for the analysis
  (footfall, peak hours, anomalies, dwell-time, site score).

## Local files NOT in git (admin only)
| File | Source |
|------|--------|
| `serviceAccount.json` (or `*firebase-adminsdk*.json`) | Firebase console → Service accounts → Generate new private key |

(`web/firebase-config.js` **is** in the repo - it's the public Web SDK
identifier, not a secret. Firestore Rules are what actually protect the DB.)
