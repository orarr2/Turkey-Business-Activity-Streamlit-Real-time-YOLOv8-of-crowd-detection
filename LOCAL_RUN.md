# Run locally — Turkey footfall (Firebase)

Everything you need to run the collector + live dashboard on your own machine.

## Prerequisites
- Python 3.10+ and `pip`
- Your Firebase **service-account JSON** (downloaded from Firebase console → Settings → Service accounts)
- Open internet (so the IBB / YouTube streams resolve)

## 1. Get the code
```bash
git clone <your-repo-url> turkey-footfall
cd turkey-footfall
```

## 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Provide your own Firebase config

The repo ships two placeholder templates — fill in **your** project values from
**Firebase Console → Project settings → Your apps → Web app → SDK setup**.

```bash
cp .env.example .env                                    # server-side env vars
cp web/firebase-config.example.js web/firebase-config.js   # client-side web SDK
```

Edit each file and paste the values Firebase gave you. Both `.env` and
`web/firebase-config.js` are gitignored, so your keys never reach the repo.

## 4. Point at your service-account key
Put the JSON somewhere (e.g. project root as `serviceAccount.json`), then:
```bash
export FIREBASE_CREDENTIALS="$PWD/serviceAccount.json"          # Mac/Linux
# Windows PowerShell:  $env:FIREBASE_CREDENTIALS = "$PWD\serviceAccount.json"
```

## 5. Smoke test (one camera)
```bash
python -m app.collector --backend firebase --interval 20 --only konya_hukumet
```
Expect lines like `konya_hukumet: person=23 vehicles=2` every 20s. `Ctrl+C` to stop.
(First run downloads the YOLO weights `yolov8n.pt`, ~6 MB.)

## 6. Run for real — two terminals
**Terminal 1 — collector (crowded cameras, leave running):**
```bash
python -m app.collector --backend firebase --interval 20 \
    --only konya_hukumet,giresun_gazi,otogar_kavsagi,kadikoy
```
**Terminal 2 — dashboards:**
```bash
streamlit run app/streamlit_app.py        # 4 cameras side by side, last 24h
cd web && python -m http.server 8000      # or the Firebase web cards
```
Open the Streamlit URL it prints (or **http://localhost:8000** for the web cards) — both update live.
The Streamlit grid shows the four `GRID_CAMERAS` (Konya, Giresun, Otogar Kavsagi, Kadikoy) so collect
those four to fill it.

## Camera ids (from `app/cameras.py`)
**Dashboard grid (`GRID_CAMERAS`):** `konya_hukumet`, `giresun_gazi`, `otogar_kavsagi`, `kadikoy`.
Others: `taksim`, `beyazit_meydan`, `kapali_carsi` (Grand Bazaar), `misir_carsisi` (Spice Bazaar),
`sultanahmet_1`, `eyup_sultan`, `uskudar`.
Drop `--only` to run all of them.

> `giresun_gazi` (skylinewebcams) and `otogar_kavsagi` (webcamera24) resolve from their public pages
> and rotate tokens — verify once with `python -m app.detect_core --resolve giresun_gazi,otogar_kavsagi`.

## Troubleshooting
- **`MISS (empty frame)`** on every round → that stream is down or your network blocks it. Try another
  camera id; confirm you are on an open network (not a corporate/VPN filter).
- **`FileNotFoundError ... service-account`** → `FIREBASE_CREDENTIALS` is unset or the path is wrong.
- **Dashboard shows "no data"** → the collector isn't writing yet, or `web/firebase-config.js` is missing.
- **`firebase write failed ... PERMISSION_DENIED`** → you are not using the service-account key (the
  Admin SDK bypasses the read-only rules; the web apiKey does not).
- **Notebook instead of app** → `jupyter lab notebooks/turkey_business_activity.ipynb` for the analysis
  (footfall, peak hours, anomalies, dwell-time, site score).

## Local files NOT in git (you provide them)
| File | Source |
|------|--------|
| `web/firebase-config.js` | paste from step 3 |
| `serviceAccount.json` | Firebase console → Service accounts → Generate new private key |
