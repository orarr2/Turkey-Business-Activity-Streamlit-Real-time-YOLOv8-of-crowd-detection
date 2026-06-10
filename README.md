# YOLO Visual Detection — Live Public Streams

ניצול של עולם ה-**live streaming הציבורי** (שמתעדכן 24/7 וכמעט לא מנוצל ל-data science) כדי לבנות
סדרות זמן כמותיות: מזרם לייב → ספירת אובייקטים עם YOLO → סדרת זמן → זיהוי אנומליות → חיבור לדאטה
חיצוני (GDELT / ACLED).

## מה יש כאן

| קובץ | תיאור |
|------|--------|
| [`LOCAL_RUN.md`](LOCAL_RUN.md) | **Quickstart** — run the collector + live dashboard locally, end to end |
| [`notebooks/turkey_business_activity.ipynb`](notebooks/turkey_business_activity.ipynb) | **Turkey business-activity notebook** — footfall, peak hours, anomalies, dwell-time / prolonged stops, site score |
| [`app/collector.py`](app/collector.py) | **Continuous collector** — samples cameras 24/7 into SQLite or Firestore (makes the data live) |
| [`app/streamlit_app.py`](app/streamlit_app.py) | **Live dashboard (local)** — auto-refreshing footfall/anomaly view from SQLite |
| [`web/`](web/) | **Real-time web dashboard** — Firebase `onSnapshot` live cards + chart |
| [`app/cameras.py`](app/cameras.py) · [`app/detect_core.py`](app/detect_core.py) · [`app/firebase_store.py`](app/firebase_store.py) | Camera catalog · shared detection core · Firestore writer |
| [`docs/turkey_cameras.md`](docs/turkey_cameras.md) | Verified Turkey commercial/market camera streams + **live-data architecture** |
| [`docs/firebase_setup.md`](docs/firebase_setup.md) | Firebase project, service account, security rules, cost |

## Turkey business activity — quick start

```bash
pip install -r requirements.txt

# 1) Explore the analysis (run locally — IBB streams need an open network)
jupyter lab notebooks/turkey_business_activity.ipynb

# 2) Make it LIVE: collector runs 24/7, dashboard auto-refreshes
python -m app.collector --db data/footfall.db --interval 20    # terminal 1
streamlit run app/streamlit_app.py                             # terminal 2
```

**Why two processes?** A notebook cell runs once and stops, so it can't be "live". The collector keeps
sampling into a database; the dashboard reads that database and refreshes — so the numbers are always
fresh without re-running any cell. Details in [`docs/turkey_cameras.md`](docs/turkey_cameras.md).

## Firebase (real-time live app)

For a true live web app, write counts to Firestore and let the browser subscribe with `onSnapshot`
(instant updates, no polling). Crowded cameras are sampled first.

```bash
export FIREBASE_CREDENTIALS=/path/to/firebase-service-account.json
python -m app.collector --backend firebase --interval 20 \
    --only konya_hukumet,kapali_carsi,misir_carsisi,eminonu,istiklal_1   # collector

cp web/firebase-config.example.js web/firebase-config.js   # paste your web config
cd web && python -m http.server 8000                       # open http://localhost:8000
```

Full walkthrough (project creation, service account, security rules, cost): [`docs/firebase_setup.md`](docs/firebase_setup.md).
The Konya Sarraflar Yeraltı Çarşısı crowd cam (`konya_hukumet`, YouTube `pnf9VoLDvFE`) is pre-configured.

## עקרונות

- **רק זרמים ציבוריים-בכוונה** (מצלמות תיירות/מסחר מכוונות, מצלמות תשתית רשמיות). **לא** נוגעים
  במצלמות שנחשפו לאינטרנט בלי הרשאה — זה מחוץ לתחום הפרויקט, חוקית ומוסרית.
- **דגימה דלילה** (פריים כל כמה שניות) — מספיק לסדרת זמן והוגן לשרת.
- **שמירת ספירות מצרפיות בלבד**, לא פריימים גולמיים של אנשים. פרטיות by design.

פירוט מלא של מצלמות ושיקולים: [`docs/turkey_cameras.md`](docs/turkey_cameras.md).
