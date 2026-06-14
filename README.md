# YOLO Visual Detection — Live Public Streams

ניצול של עולם ה-**live streaming הציבורי** (שמתעדכן 24/7 וכמעט לא מנוצל ל-data science) כדי לבנות
סדרות זמן כמותיות: מזרם לייב → ספירת אובייקטים עם YOLO → סדרת זמן → זיהוי אנומליות → חיבור לדאטה
חיצוני (GDELT / ACLED).

## מה יש כאן

| קובץ | תיאור |
|------|--------|
| [`LOCAL_RUN.md`](LOCAL_RUN.md) | **Quickstart** — run the collector + live dashboard locally, end to end |
| [`notebooks/turkey_business_activity.ipynb`](notebooks/turkey_business_activity.ipynb) | **Turkey business-activity notebook** — footfall, peak hours, anomalies, dwell-time / prolonged stops, site score |
| [`app/collector.py`](app/collector.py) | **Continuous collector** — samples cameras 24/7, pushes to Firestore (footfall + latest + re-ID stats) |
| [`web/`](web/) | **Real-time HTML dashboard** — 2×2 grid of live video tiles, per-camera anomaly badge + mini chart, combined 24h chart, re-ID summary. Backed by Firestore so a returning visitor sees all accumulated data, not a fresh local file. |
| [`app/cameras.py`](app/cameras.py) · [`app/detect_core.py`](app/detect_core.py) · [`app/firebase_store.py`](app/firebase_store.py) | Camera catalog · shared detection core · Firestore writer |
| [`docs/turkey_cameras.md`](docs/turkey_cameras.md) | Verified Turkey commercial/market camera streams + **live-data architecture** |
| [`docs/firebase_setup.md`](docs/firebase_setup.md) | Firebase project, service account, security rules, cost |

## Quick start

The collector writes to Firestore; the HTML page subscribes via `onSnapshot` and
updates in real time. Any visitor who opens the dashboard sees the full history the
collector has accumulated, not a fresh local file — that's what makes the data
shared and aggregative.

```bash
pip install -r requirements.txt
cp .env.example .env                                       # fill in your Firebase values
cp web/firebase-config.example.js web/firebase-config.js   # same web SDK values
export FIREBASE_CREDENTIALS=/path/to/serviceAccount.json   # Admin SDK key

# terminal 1 — collector pushing the 4 grid cameras into Firestore (leave running)
python -m app.collector --interval 20 \
    --only konya_hukumet,giresun_gazi,otogar_kavsagi,kadikoy

# terminal 2 — serve the HTML dashboard
cd web && python -m http.server 8000                       # open http://localhost:8000
```

The dashboard shows a 2×2 grid (live video iframe + people/vehicle counts + anomaly
badge + mini chart per camera), a combined 24h chart for all four cameras, and a
re-ID summary table. Firebase setup walkthrough (project creation, service account,
security rules, cost): [`docs/firebase_setup.md`](docs/firebase_setup.md).

### Just the analysis

`jupyter lab notebooks/turkey_business_activity.ipynb` — footfall, peak hours, anomaly
z-score, dwell-time tracking, re-identification, and a site-selection score. Run locally
so the streams resolve.

## עקרונות

- **רק זרמים ציבוריים-בכוונה** (מצלמות תיירות/מסחר מכוונות, מצלמות תשתית רשמיות). **לא** נוגעים
  במצלמות שנחשפו לאינטרנט בלי הרשאה — זה מחוץ לתחום הפרויקט, חוקית ומוסרית.
- **דגימה דלילה** (פריים כל כמה שניות) — מספיק לסדרת זמן והוגן לשרת.
- **שמירת ספירות מצרפיות בלבד**, לא פריימים גולמיים של אנשים. פרטיות by design.

פירוט מלא של מצלמות ושיקולים: [`docs/turkey_cameras.md`](docs/turkey_cameras.md).
