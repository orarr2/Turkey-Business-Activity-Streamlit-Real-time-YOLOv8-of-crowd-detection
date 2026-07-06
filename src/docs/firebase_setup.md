# Firebase setup - live updating data

Architecture with Firebase (Firestore pushes updates to the browser in real time, so the
dashboard is truly live - no polling):

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
   lock down rules before any public deploy - see §5).

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
2. Create `web/firebase-config.js` with an `export const firebaseConfig = {…}`
   holding your project's `apiKey`, `authDomain`, `projectId`, etc.
3. Serve the folder (any static server):

```bash
cd web && python -m http.server 8000
# open http://localhost:8000
```

The page subscribes with `onSnapshot` - every collector write appears immediately, no refresh.

## 5. Security rules (before deploying publicly) — **this is what protects your DB**

Test mode lets **anyone on the internet read *and write*** your database — they could fill your
quota or run up a bill. Note that the web SDK config (`apiKey`, `projectId`) is **public by
design** and ships in every visitor's browser; it is *not* a secret. The security rules — not the
apiKey — are what actually protect your data.

The locked-down rules live in [`firestore.rules`](../firestore.rules) at the repo root: **public
read on the dashboard collections (`footfall`, `latest`, `reid_stats`, `events`, `config`), all
client writes denied** (the collector uses the Admin SDK, which bypasses rules, so blocking client
writes doesn't affect it), and everything else locked.

> **TTL policies:** add a Firestore TTL policy on `footfall.expire_at` AND another on
> `events.expire_at` (Firestore console → Time-to-live) so both history collections
> self-prune after 24h.

Deploy them with the Firebase CLI:

```bash
npm install -g firebase-tools          # one-time
firebase login
# Create .firebaserc alongside firebase.json:  {"projects":{"default":"<your-project-id>"}}
firebase deploy --only firestore:rules
```

Verify afterwards in **Firebase console → Firestore → Rules** that writes show `if false`.

## 6. App Check (anti-abuse / read-quota protection)

Rules make the data read-only, but a scraper can still hammer **reads** and burn your read quota.
App Check stops that by requiring every request to carry a reCAPTCHA-v3 attestation token proving it
came from *your* web app.

1. Firebase console → **App Check → Apps** → register the web app with the **reCAPTCHA v3** provider.
2. Copy the **site key** into `web/firebase-config.js` as `recaptchaSiteKey`.
   `web/app.js` initializes App Check automatically when it's set.
3. When you're confident the dashboard works, console → **App Check → Firestore → Enforce**.

> Enable enforcement only *after* the site key is live in the page — otherwise enforced reads are
> rejected and the dashboard goes blank. Until you enforce, App Check runs in monitor-only mode.

## 7. Rate limit & cost cap

Firestore free (Spark) tier ≈ **20k writes/day**. Each camera writes ~2 docs/round (footfall +
latest), or ~3 with re-ID on. At a 20s interval that's ~4,300–13,000 writes/day/camera.

- **Writer side (built in):** `app/collector.py` clamps `--interval` to a 5s floor and prints the
  projected daily write count on startup, warning if it would exceed the free tier. Raise
  `--interval` or run fewer cameras to stay under it.
- **Platform side (set this up):** add a **budget alert** in Google Cloud console → *Billing →
  Budgets & alerts*, and if you're on the Blaze plan, an **App Engine daily spending limit**, so a
  runaway process or abuse can't run up an unbounded bill. This is the real hard cap — Firestore has
  no per-user request rate limit of its own.

For many cameras at high frequency, move history to BigQuery / keep only `latest` in Firestore.
