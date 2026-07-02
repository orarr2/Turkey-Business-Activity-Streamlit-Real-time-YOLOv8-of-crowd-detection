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
source .venv/bin/activate                    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r src/requirements.txt
```

### 3. Open the notebook
The notebook lives at the repo root and finds `app/` under `src/` on its own:
```bash
jupyter lab turkey_business_activity.ipynb   # run from the repo root
```
Run All. Cells 1-6 sample a public camera; cells 7-15 run YOLO / anomaly /
re-ID / dwell / activity score locally. Section 7 embeds the live dashboard so
you can compare "my minute of samples" against the cloud's 24-hour history.

Or open the dashboard on its own:
```bash
cd src && python serve.py                    # binds :8000 and opens the browser
```

---

## Troubleshooting

- **`MISS (empty frame)`** on every round → that stream is down or your
  network blocks it. Try `CAM_ID = 'konya_hukumet'` (tvkur, reachable from
  anywhere). IBB legacy streams may be geo-restricted to Turkey IPs.
- **Dashboard shows "no recent writes"** → the cloud collector is not running
  (or is wedged). Only the project maintainer can restart it (SSH to the GCP
  VM: `sudo systemctl restart collector`).
- **Dashboard shows a `↳ fallback: ...` badge on a tile** → that slot's
  primary camera failed 3+ samples in a row and the collector switched to a
  backup camera in the chain. Every 15 min the primary is retried; the badge
  will disappear when it recovers.
- **`firebase write failed ... PERMISSION_DENIED`** → you are not using the
  Admin SDK service-account (the web apiKey doesn't bypass Firestore rules;
  the Admin SDK does).
