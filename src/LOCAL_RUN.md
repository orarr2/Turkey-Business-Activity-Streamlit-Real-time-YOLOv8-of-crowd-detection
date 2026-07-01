# Run locally — Turkey footfall

Two flows, two roles:

- **Viewer**: analyze locally + watch the live dashboard. Zero configuration.
- **Admin**: diagnose the cloud collector, test new cameras, run a backup
  collector if the VM is down. Needs the service-account JSON.

The cloud collector on the GCP VM does the real 24/7 work — see
[`deploy/gcp-vm/README.md`](deploy/gcp-vm/README.md) for that side. This
document is only about running **locally**.

## Viewer path — zero config

### Prerequisites
- Python 3.10+ and `pip`
- Open internet (so the IBB / YouTube streams resolve locally)

### 1. Get the code
```bash
git clone <your-repo-url> turkey-footfall
cd turkey-footfall/src         # everything lives under src/
```

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Open the viewer notebook
```bash
jupyter lab viewer.ipynb
```
Run All. Cells 1-6 sample a public camera; cells 7-15 run YOLO / anomaly /
re-ID / dwell / activity score locally. Section 7 embeds the live dashboard so
you can compare "my minute of samples" against the cloud's 24-hour history.

Or open the dashboard on its own:
```bash
python serve.py                 # binds :8000 and opens the browser
```

---

## Admin path — VM health + backup collector

### 1. Drop in the service-account key
The Admin SDK JSON stays private on your machine. Put it at the repo root as
`serviceAccount.json`, or set the env var yourself:
```bash
export FIREBASE_CREDENTIALS="$PWD/serviceAccount.json"          # Mac/Linux
# Windows PowerShell:  $env:FIREBASE_CREDENTIALS = "$PWD\serviceAccount.json"
```
The JSON is gitignored — it never enters the repo. This is the only real
secret in the project.

### 2. Open the admin notebook
```bash
jupyter lab admin.ipynb
```
Sections 1-3 are dry diagnostics (no writes). Section 4 is the backup
collector — only run it when the VM is genuinely down; the VM and a local
run competing on the same slot_ids will just overwrite each other's samples.

### 3. Optional — run the collector from a terminal (same effect as Section 4)
```bash
python -m app.collector --interval 20
```
Requires `FIREBASE_CREDENTIALS` and (optionally) `FIREBASE_STORAGE_BUCKET` if
you want anomaly / returning-visitor snapshots to land in Firebase Storage
instead of `web/snapshots/`. On the VM, both env vars are set by the systemd
unit; locally you set them yourself.

## Local files NOT in git (admin only)
| File | Source |
|------|--------|
| `serviceAccount.json` (or `*firebase-adminsdk*.json`) | Firebase console → Service accounts → Generate new private key |

(`web/firebase-config.js` **is** in the repo — it's the public Web SDK
identifier, not a secret. Firestore Rules are what actually protect the DB.)

## Troubleshooting

- **`MISS (empty frame)`** on every round → that stream is down or your
  network blocks it. Try `CAM_ID = 'konya_hukumet'` (tvkur, reachable from
  anywhere). IBB legacy streams may be geo-restricted to Turkey IPs.
- **`FileNotFoundError ... service-account`** in admin.ipynb → set
  `FIREBASE_CREDENTIALS` or copy the JSON next to the notebook.
- **Dashboard shows "no recent writes"** → the VM collector is not running
  (or is wedged). SSH to the VM and `journalctl -u collector -f`. Meanwhile,
  admin.ipynb § 4 runs a bounded backup so the dashboard stays alive.
- **Dashboard shows a `↳ fallback: ...` badge on a tile** → that slot's
  primary camera failed 3+ samples in a row and the collector switched to a
  backup camera in the chain. Every 15 min the primary is retried; the badge
  will disappear when it recovers.
- **`firebase write failed ... PERMISSION_DENIED`** → you are not using the
  Admin SDK service-account (the web apiKey doesn't bypass Firestore rules;
  the Admin SDK does).
