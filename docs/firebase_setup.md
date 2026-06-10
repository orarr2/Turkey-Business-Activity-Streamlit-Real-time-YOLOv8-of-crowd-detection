# Firebase setup — live updating data

Architecture with Firebase (Firestore pushes updates to the browser in real time, so the
dashboard is truly live — no polling):

```
 live streams → app/collector.py → Firestore ─┬─ web/  (onSnapshot, instant updates)
                (runs 24/7, --backend          │
                 firebase|both)        collections:
                                         footfall  (history → charts)
                                         latest    (one doc/camera → KPIs)
```

## 1. Create the project (one-time)

1. https://console.firebase.google.com → **Add project**.
2. **Build → Firestore Database → Create database** (start in *test mode* for development;
   lock down rules before any public deploy — see §5).

## 2. Backend credentials (for the collector)

1. Project settings (gear) → **Service accounts → Generate new private key** → download the JSON.
2. Save it outside git (it is gitignored if named `firebase-service-account.json`), then:

```bash
export FIREBASE_CREDENTIALS=/path/to/firebase-service-account.json
pip install firebase-admin
```

## 3. Run the collector against Firebase

```bash
# crowded cameras first; firebase backend (or "both" to also keep local SQLite)
python -m app.collector --backend firebase --interval 20 \
    --only konya_hukumet,kapali_carsi,misir_carsisi,eminonu,istiklal_1
```

Each round writes one history doc per camera to `footfall` and overwrites `latest/{cam_id}`.

> Run this on an **open network** (your machine / a VM / Cloud Run). IBB/YouTube hosts are
> blocked from restricted sandboxes. Keep it alive with `systemd` / Docker / `nohup`.

## 4. Web frontend (live dashboard)

1. Firebase console → Project settings → **Your apps → Web app** → copy the SDK config.
2. `cp web/firebase-config.example.js web/firebase-config.js` and paste your values in.
3. Serve the folder (any static server):

```bash
cd web && python -m http.server 8000
# open http://localhost:8000
```

The page subscribes with `onSnapshot` — every collector write appears immediately, no refresh.

## 5. Security rules (before deploying publicly)

Test mode allows anyone to read/write. For a read-only public dashboard, lock writes to the
backend (service account bypasses rules) and allow only reads:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{db}/documents {
    match /{document=**} {
      allow read: if true;     // public dashboard
      allow write: if false;   // only the collector (Admin SDK) writes
    }
  }
}
```

## Cost note

Firestore free tier ≈ 20k writes/day. One write per camera per round: at 20s interval that is
~4,300 writes/day/camera. Keep camera count modest on the free tier, or raise `--interval`, or
batch. For many cameras at high frequency, move history to BigQuery / keep only `latest` in Firestore.
