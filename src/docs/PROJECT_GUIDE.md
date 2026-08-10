# Project Guide — Business Activity / Live Footfall

Single consolidated operational reference for the whole project. Merges what
used to live in six separate documents (`deploy/gcp-vm/README.md`,
`deploy/REBUILD.md`, `deploy/cloudflare-proxy/README.md`,
`deploy/gcp-billing-killswitch/README.md`, `docs/firebase_setup.md`, and the
pre-existing Hebrew `MODEL_GUIDE_HE.md`). The Hebrew twin of this document is
[`PROJECT_GUIDE_HE.md`](PROJECT_GUIDE_HE.md) — same chapter numbering, so
cross-references work in both directions.

---

## Quick navigation

1. [What the project does](#1-what-the-project-does)
2. [Architecture — where each piece runs](#2-architecture--where-each-piece-runs)
3. [The VM — deep dive](#3-the-vm--deep-dive)
4. [VM commands cheatsheet](#4-vm-commands-cheatsheet)
5. [The 7 live analysis layers](#5-the-7-live-analysis-layers)
6. [The deep-window analysis (`behavior.analyze_window`)](#6-the-deep-window-analysis-behavioranalyze_window)
7. [The notebook — offline analytics](#7-the-notebook--offline-analytics)
8. [Model choice + parameters](#8-model-choice--parameters)
9. [Anomalies + reporting](#9-anomalies--reporting)
10. [Active-learning loop](#10-active-learning-loop)
11. [Firebase project setup](#11-firebase-project-setup)
12. [Cloudflare proxy for IBB](#12-cloudflare-proxy-for-ibb)
13. [GCP billing kill-switch](#13-gcp-billing-kill-switch)
14. [Troubleshooting / FAQ](#14-troubleshooting--faq)
15. [Appendix: design decisions taken](#15-appendix-design-decisions-taken)

---

## 1. What the project does

The system turns 4 public street cameras into a quantitative time series:

> **live HLS stream → YOLOv8 frame inference → counts + appearance re-ID →
> Firestore → real-time web dashboard + Jupyter analytics.**

Every sampling round (40 s in the shipped cloud deployment; `--interval` sets
it), the collector grabs a burst from each active camera, runs YOLO on each
frame, computes gated counts + appearance signatures + anomaly gates, then
writes the result to Firestore. The web dashboard subscribes with `onSnapshot`
— no polling, no refresh; every collector write appears immediately.

The grid is **country-generic**: it always runs 4 cameras from ONE country and
rotates through a priority ladder (**Turkey → Thailand → Japan → USA**),
falling to the next country only when the active one goes fully dark. Turkey
is the project's subject; from GCP the Istanbul IBB feed is geo-blocked, so
the grid usually runs the foreign benches (YouTube-Live-backed street cams)
until a Cloudflare Worker on a non-Google ASN restores IBB (see chapter 12).

Outputs:

- **Footfall** — how many people / vehicles per camera per round.
- **Re-identification** — same-person / same-vehicle recognition over time
  (OSNet embeddings when the ONNX file is present, HSV histogram otherwise).
- **Anomalies** — extreme load, camera obstruction, blackouts, loitering,
  returning visitor, unattended object, fall suspect.
- **Typical vehicle speed** in km/h (published only when the camera has real
  statistical mass: ≥ 5 samples and ≥ 10 % of rounds carrying one — plazas
  without real traffic show "-" rather than a fabricated number).
- **PDF report** by e-mail on demand (dashboard "Send Report" button); a
  scheduled twice-a-day digest goes to the project archive mailbox only.

Everything runs on free tiers: open-source model, GCP `e2-micro` on Always
Free ($0/month), GitHub Actions on a public repo, Firebase Spark plan.

### 1.1 Two runtimes, one code base

| | Cloud VM (production 24/7) | Local notebook (accuracy reference) |
|---|---|---|
| Host | GCP `e2-micro` (1 GB RAM) | Any laptop |
| Detector | `yolov8s.pt` @ `imgsz 640` (memory-fallback: `yolov8n.pt` @ 768) | `yolo26m.pt` @ `imgsz 960` |
| Purpose | Runs forever, feeds the dashboard + reports | Deep analysis; also the ground-truth reference for calibration |
| Data flow | Firestore + Firebase Storage | Local CSV cache pulled from Firestore |
| Notebook file | — | `turkey_business_activity.ipynb` (git-tracked, `MODEL_WEIGHTS = 'yolo26m.pt'`) |
| Twin notebook | — | `turkey_business_activity_yolov8n.ipynb` (local-only, `.gitignore`d; mirrors the VM model to compare apples-to-apples on the same camera) |

The twin notebook is intentionally not in git; it is a hand copy of the main
notebook with `MODEL_WEIGHTS = 'yolov8s.pt'` (or `yolov8n.pt`) so the operator
can see, on their own machine, what the VM would see on the same frame.

### 1.2 The country fallback ladder

The collector never locks onto a fixed set of cameras. A class called
`CountryDirector` manages two nested ladders:

- **Country ladder (priority)**: Turkey → Thailand → Japan → USA. The grid
  runs 4 cameras from ONE country; it moves to the next country only when the
  active one cannot deliver a single live camera. A dead single camera does
  not shift the grid — a bench camera from the SAME country backfills.
- **Camera ladder (per country)**: `CameraPool` walks that country's list and
  keeps the first 4 live cameras assigned each round; a camera that misses 3
  consecutive samples rests 15 min; `tvkur` (Konya) cameras are a fast-fail
  path — one miss is enough to rest them.
- **Host-level breaker** (`HostBreaker`): 4 consecutive 403/429 → 20-min rest
  for every camera on that host, then a single probe re-opens. This means a
  blocking CDN gets ~3 requests/hour instead of ~120.
- **Pre-report recovery**: a few minutes before each scheduled report (12:00
  and 20:00 Israel time) the collector re-probes higher-priority countries so
  Turkey reclaims the grid the moment IBB unblocks.

Day/night gates in the report use each **camera's** timezone (the US bench
alone spans Eastern / Central / Pacific).

---

## 2. Architecture — where each piece runs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GCP e2-micro VM (1 GB RAM, 24/7)                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ collector.py:  loop { grab_frame → YOLO → count → re-ID → events }   │  │
│  │ 4 cameras in parallel, ~40 s per round                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                             ↓  Firestore + Firebase Storage                 │
└─────────────────────────────────────────────────────────────────────────────┘
       ↓                                                              ↓
┌──────────────────┐                                     ┌─────────────────────┐
│ Dashboard in the │                                     │ On-demand report    │
│ browser (any     │                                     │ (dashboard button   │
│ static host or   │                                     │ → workflow_dispatch │
│ localhost:8000)  │                                     │ → PDF to inbox)     │
└──────────────────┘                                     └─────────────────────┘
       ↓ operator labels                                          ↓ push
┌──────────────────┐          ┌───────────────────────┐    ┌─────────────────┐
│ training_sync    │  ──→     │ GitHub Actions        │    │ Inbox           │
│ pushes labels to │          │ train-head (free)     │    │ notification    │
│ Storage          │          │ promotion gate → SA   │    └─────────────────┘
└──────────────────┘          └───────────────────────┘
                                        ↓ hot-load
                              ┌───────────────────────┐
                              │ Collector pulls new   │
                              │ head without restart  │
                              └───────────────────────┘
```

Terms:

- **VM** — Virtual Machine on GCP. The project uses `e2-micro`, the smallest
  GCP box (2 shared vCPUs, 1 GB RAM), which is Always Free.
- **Firestore** — Google's managed NoSQL doc store. The dashboard subscribes
  with `onSnapshot`, so writes appear in the browser without polling.
- **Firebase Storage** — object bucket for JPEG snapshots + JSON heatmap
  exports. Has a 24 h lifecycle rule on `snapshots/`.
- **GitHub Actions** — the free CI runner is where the head fine-tune runs on
  public-repo minutes; the promoted Detect head lands in Storage and the
  collector hot-swaps it in-place.

Key principle: **the dashboard is a pure consumer**. Anyone who clones the
repo can serve `web/` and see the same live grid — because all state lives in
Firestore, and Firestore's TTL prunes each history collection to 24 h.

---

## 3. The VM — deep dive

### 3.1 The machine

```
Provider    : Google Cloud Platform
Machine     : e2-micro (Always Free)
CPU         : 2 vCPU shared (0.25 vCPU guaranteed, ~1 vCPU burst)
RAM         : 1024 MB total (~950 MB usable after kernel)
Disk        : 30 GB SSD (Standard persistent disk)
OS          : Debian 12 (bookworm)
Zone        : us-east1-c (Virginia)
Public IP   : ephemeral (rotates on stop/start; static costs money)
Instance ID : turkey-collector
Project     : turkey-footfall
Cost        : $0/month (Always Free — one e2-micro per account, us-central1 /
              us-east1 / us-west1 only)
```

**Reality checks:**

- **"0.25 vCPU guaranteed"** means the base allocation is a quarter of a core;
  bursts to ~1 vCPU when the host has headroom. Rounds occasionally take ~25 %
  longer at random — that is the shared-tenant tax, not a bug.
- **1 GB RAM is tight** for an 11 M-parameter model + 4 HLS decoders + OSNet.
  A 2 GB `/swapfile` is added by hand (not by `install.sh`) as insurance;
  observed peak use ≈ 273 MB.
- **`us-east1-c`, not close to Turkey** — Always Free is restricted to
  us-central1 / us-east1 / us-west1. RTT to Istanbul is ~150 ms which is
  irrelevant here — sampling is every 40 s, latency has no effect.

### 3.2 Folder layout on disk

```
/opt/turkey-footfall/              ← the repo clone; git pull refreshes it
├── src/
│   ├── app/                       ← all Python modules (collector, detect, tracker, reid, heatmap, faces, pose, gestures, behavior, live_analysis, dashboard_server, alerts, adapters …)
│   ├── tools/                     ← CLI helpers (analyze_window, calibrate_conf, probe_country, daily_digest, train_head, promote_adapter, fetch_training_data …)
│   ├── data/
│   │   ├── reid.db                ← SQLite of OSNet embeddings
│   │   ├── osnet_x0_25_msmt17.onnx ← the re-ID model (~5 MB)
│   │   ├── confidence_boost.json  ← learned per-(cam,cls) gate nudges
│   │   ├── blacklist_auto.json    ← polygons the auto-blacklist learned
│   │   ├── per_camera_conf.json   ← WS4 output: precision-calibrated gates
│   │   └── adapters/              ← head-only fine-tuned artifacts
│   │       ├── current.json       ← pointer to the active head
│   │       ├── history.jsonl      ← promotion / rejection log
│   │       └── head_run<N>.pt     ← the head file itself
│   ├── web/
│   │   ├── snapshots/             ← live snapshot cache
│   │   │   ├── review_frames/     ← frames queued for labelling (LRU 500)
│   │   │   ├── live_samples/      ← crops for visual search (LRU 1000)
│   │   │   ├── entities/          ← per-entity crops (LRU 400)
│   │   │   ├── anomalies/         ← anomaly snapshots (24 h TTL)
│   │   │   └── heatmaps/          ← per-cam heat overlays + <cam>.json
│   │   └── firebase-config.js     ← PUBLIC web SDK keys (safe in git)
│   ├── .venv/                     ← Python virtualenv (~2 GB, dominated by torch)
│   └── deploy/                    ← install.sh, systemd unit templates, worker.js
├── yolov8s.pt                     ← detection weights (ultralytics auto-downloads on first use)

/etc/turkey-footfall/              ← protected config (root:root)
├── serviceAccount.json            ← Firebase Admin SDK key (0400)
├── proxy.env                      ← IBB_PROXY_URL, IBB_PROXY_SECRET (0600)
└── digest.env                     ← GMAIL_USER, GMAIL_APP_PASSWORD (0600)

/etc/systemd/system/
├── collector.service
├── digest.service
└── digest.timer

/var/log/journal/                  ← systemd journal (rotates automatically)
/swapfile                          ← 2 GB swap (manual step; see 3.7)
```

Notes:

- The collector runs as root because it must read `serviceAccount.json`
  (mode 0400 root:root). Nothing else on the VM needs elevated privileges.
- `.venv/` is dominated by `torch` (~1.2 GB) + `ultralytics` (~200 MB). Do
  not try to trim it; both are load-bearing.
- Everything under `web/snapshots/` is regenerated within a round — safe to
  delete for a clean-slate restart.

### 3.3 Installation

**Prerequisites (once, in the GCP Console):**

1. Enable billing on the project (needed even for free-tier VMs).
2. Enable APIs: Compute Engine, Secret Manager, Cloud Storage.
3. Secret Manager → create secret `firebase-sa`, paste the Firebase Admin SDK
   JSON as the value.
4. Grant `<PROJECT_NUMBER>-compute@developer.gserviceaccount.com` the role
   **Secret Manager Secret Accessor** on `firebase-sa`.
5. Firestore console → Time-to-live → add TTL on `footfall.expire_at` and
   `events.expire_at`.
6. Firebase Console → Storage → Get started; then GCP → Cloud Storage → the
   bucket → Lifecycle → delete files under `snapshots/` after 1 day.

**Create the VM (Console → Compute Engine → Create instance):**

- Name: `turkey-collector`
- Region: `us-east1` (or `us-central1` / `us-west1`)
- Machine type: `e2-micro`
- Boot disk: Debian 12, Standard persistent disk, 30 GB
- Firewall: leave HTTP/HTTPS unchecked (the collector listens on nothing)
- Identity & API access: keep default service account

**Install the collector (once the VM is up):**

Click SSH next to the VM in the Console, then:

```bash
curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh \
  | sudo bash
```

The script is idempotent — re-running it is the standard way to refresh code.
Its six steps:

1. `apt-get install` system packages: `git`, `python3-venv`, `ffmpeg`, the
   OpenCV shared libs, `fonts-dejavu-core`.
2. Clone (or fetch + reset) `/opt/turkey-footfall`.
3. Create `.venv`, `pip install -r requirements.txt` with
   `TMPDIR=/var/tmp` (avoids exhausting the RAM-backed `/tmp` mid-install).
4. Pull the Firebase Admin key from Secret Manager to
   `/etc/turkey-footfall/serviceAccount.json` (mode 0400 root:root).
5. Detect the Firebase Storage bucket (`<project>.firebasestorage.app` first,
   then legacy `<project>.appspot.com`); render the systemd unit templates
   with `sed`; install to `/etc/systemd/system/`.
6. `systemctl enable --now collector.service`; install `digest.service` +
   `digest.timer` (the timer only enables if `/etc/turkey-footfall/digest.env`
   exists).

**Post-install one-time steps `install.sh` does NOT cover:**

```bash
# 2 GB swap — the 1 GB VM needs it (observed peak: 273 MB used)
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Firebase-Storage-secret env for the digest
sudo tee /etc/turkey-footfall/digest.env > /dev/null <<'EOF'
GMAIL_USER=<gmail address>
GMAIL_APP_PASSWORD=<16-char app password from myaccount.google.com/apppasswords>
EOF
sudo chmod 600 /etc/turkey-footfall/digest.env
sudo systemctl enable --now digest.timer

# Cloudflare-Worker relay for IBB (see chapter 12 for the wrangler setup)
sudo tee /etc/turkey-footfall/proxy.env > /dev/null <<'EOF'
IBB_PROXY_URL=https://ibb-proxy.<subdomain>.workers.dev
IBB_PROXY_SECRET=<same secret you set on the worker>
EOF
sudo chmod 600 /etc/turkey-footfall/proxy.env
sudo systemctl restart collector

# OSNet re-ID model (~5 MB, optional but recommended)
sudo bash /opt/turkey-footfall/src/tools/setup_reid.sh
```

### 3.4 `collector.service` — the systemd unit

The template lives in `src/deploy/gcp-vm/collector.service`; `install.sh`
renders it with `sed -e 's|__INSTALL_DIR__|/opt/turkey-footfall|g' ...`
before writing to `/etc/systemd/system/`. Key lines and why they exist:

```ini
Environment=OMP_NUM_THREADS=2
# Matches the e2-micro's 2 shared vCPUs. torch's default oversubscribes and
# thrashes context switches on this box.

Environment=MALLOC_ARENA_MAX=2
# glibc grows one malloc arena per thread by default (~50-150 MB of RSS on a
# threaded Python process). Real money on a 1 GB host that was kernel-oom-killed
# at 696 MB peak.

Environment=FIREBASE_CREDENTIALS=/etc/turkey-footfall/serviceAccount.json
Environment=FIREBASE_STORAGE_BUCKET=<detected at install time>
Environment=REID_MODEL=/opt/turkey-footfall/src/data/osnet_x0_25_msmt17.onnx

EnvironmentFile=-/etc/turkey-footfall/proxy.env
# Optional IBB relay + secret; when the file is absent the collector runs
# unchanged (IBB stays 403 from GCP, Turkey grid depends on the YouTube tier).

Environment=EXTRA_CLASSES=bird:0.35,dog:0.35,cat:0.40,backpack:0.35,handbag:0.40,suitcase:0.35,umbrella:0.35
# fix1 opt-in: bags feed the unattended-object watch, animals + parasols get
# their own counts ("other objects" line in the report). Detection-only, never
# the training chain.

Environment=FALL_CHECK=1
# fix1-A11: a person-loiter event triggers ONE pose pass on that person's crop;
# a horizontal torso upgrades the alert to "Possible FALL".

ExecStart=/opt/turkey-footfall/src/.venv/bin/python \
  -m app.collector --weights yolov8s.pt --interval 40 --imgsz 640 \
  --burst 2 --burst-stride 13

MemoryHigh=760M
MemoryMax=900M
# The cgroup guardrails. If the journal shows reclaim throttling or oom-kills,
# fall back to `--weights yolov8n.pt --imgsz 768` in ExecStart before touching
# anything else.
```

**IMPORTANT for code updates**: `git pull` + `systemctl restart` alone
suffices — the installed unit carries **machine-local `Environment=` lines**
(`FIREBASE_CREDENTIALS`, `FIREBASE_STORAGE_BUCKET`, `REID_MODEL`) that are
deliberately NOT in the repo template. Overwriting the installed unit from
the template DROPS those lines and the collector then crash-loops with
`FileNotFoundError: Firebase service-account JSON not found`. To change a
flag, edit the installed unit in place:

```bash
sudo sed -i 's#--weights yolov8s.pt --interval 40 --imgsz 640#--weights yolov8s.pt --interval 40 --imgsz 512#' \
  /etc/systemd/system/collector.service
sudo systemctl daemon-reload && sudo systemctl restart collector
```

### 3.5 `digest.service` + `digest.timer`

The daily digest is now on-demand only (via the dashboard button); the timer
that used to run twice a day now writes to the project archive mailbox only.
Enabling it needs `/etc/turkey-footfall/digest.env` with `GMAIL_USER` and
`GMAIL_APP_PASSWORD`.

### 3.6 The main loop — sample flow

Simplified from `app/collector.py`:

```python
while True:
    round_start = time.time()
    for slot_id, cam_id in director.current_grid():
        try:
            counts, boxes, frames = sample_slot(cam_id)
        except Exception as e:
            record_miss(cam_id, e)
            continue
        write_firestore(slot_id, cam_id, counts)
        heatmap.accumulate(cam_id, boxes, frames[-1].shape)
        run_anomaly_gates(cam_id, counts, boxes, frames)
        maybe_capture_review_frame(cam_id, frames[-1], boxes)
    reload_review_overrides_if_due()
    reload_adapter_if_due()
    time.sleep(max(0, INTERVAL - (time.time() - round_start)))
```

Each `sample_slot` grabs a `burst` (default 2 frames, `--burst-stride 13`
frames apart = ~0.5 s at 25 fps), runs `detect_and_count`, applies ROI +
gate filters + auto-blacklist polygons, then aggregates the burst by median
(single hallucinated boxes cannot drag a bin).

### 3.7 The memory model

The `MemoryHigh=760M / MemoryMax=900M` cgroup guardrails plus the
`MALLOC_ARENA_MAX=2` + `OMP_NUM_THREADS=2` envs plus the 2 GB `/swapfile` are
what make the whole thing fit on 1 GB physical RAM. Observed steady-state
after the 2026-08-05 accuracy overhaul (4 cams, `burst 2`,
`yolov8s @ 640`): ~410 MB RSS, 96-100 % CPU idle, swap at 0-30 MB.

If the journal shows any of the failure modes below, drop to
`yolov8n @ 768`:

- Reclaim throttling (`memory pressure` messages in `journalctl -u collector`)
- Kernel oom-kill loops (`Killed process ... (python)` in `journalctl -k`)
- Rounds stretching past interval (`! round took Ns > interval` in the app
  log; the dashboard's "counts from Ns ago" label turns red)

### 3.8 Firestore layout

| Collection | Rows/day | Doc shape | Client access |
|---|---|---|---|
| `footfall/{auto}` | ~17 k | `{cam_id, ts, person, vehicles, ok, night, expire_at}` (TTL 24 h) | read-only |
| `latest/{cam_id}` | ~2.16 k (upserts) | last sample per camera | read-only |
| `reid_stats/{cam_id}` | daily upsert | unique / seen-again counts | read-only |
| `events/{auto}` | ~0-50 | anomaly events (TTL 24 h) | read-only |
| `config/grid` | 1 doc | current 4 cams + active country | read-only |
| `training_events/{auto}` | 1/promotion | mAP + labels_total for the AL curve | read-only |

Spark plan quota: 20,000 writes/day. At `--interval 40`, the collector uses
~17 k (4 slots × 2 writes/round × 2160 samples/day). Raising `--interval` to
60 s brings it strictly under the quota.

### 3.9 Firebase Storage layout

```
snapshots/
├── review_frames/<cam>/<ts>_uNN.jpg   ← review queue (LRU on VM disk, mirrored to Storage in batches)
├── live_samples/<cam>/<ts>_<cls>.jpg  ← visual-search crops
├── entities/<eid>/<ts>.jpg            ← per-entity crops (up to 6 per entity)
├── anomalies/<cam>/<ts>.jpg           ← anomaly evidence (24 h lifecycle)
├── heatmaps/<cam>.jpg                 ← per-cam overlay (last refresh)
└── heatmaps/<cam>.json                ← full per-daypart × per-layer grid
training/
├── labels/                             ← operator verdicts uploaded by training_sync
├── frames/                             ← frames referenced by the labels
└── adapters/<run>/head.pt              ← promoted head + metadata
```

### 3.10 Hot-swap of the Detect head

Every 30 rounds the collector polls `data/adapters/current.json` and, if the
pointer changed, calls `adapters.overlay_head(model, head_state_dict)` — this
walks the Detect module and copies the head tensors in-place. No restart, no
memory spike, no interruption to sampling. Fallback: if `current.json` is
missing or unreadable the base model runs untouched (byte-identical).

---

## 4. VM commands cheatsheet

### 4.1 SSH in

Three ways in, all reach the same `turkey-collector`:

```bash
# From your machine (once: gcloud auth login && gcloud config set project turkey-footfall)
gcloud compute ssh turkey-collector --zone=us-east1-c

# Browser SSH: Console → Compute Engine → SSH button next to the instance
# Mobile: Google Cloud app → Compute Engine → SSH
```

### 4.2 Service management

```bash
sudo systemctl status  collector          # is it running? (want: active (running))
sudo systemctl restart collector          # after a code change or env edit
sudo systemctl stop    collector          # deliberate pause (billing keeps ticking)
sudo systemctl start   collector          # resume
sudo systemctl status  digest.timer       # scheduled report timer
sudo systemctl list-timers                # all timers, next fire times
```

### 4.3 Logs

```bash
sudo journalctl -u collector -f                              # tail live
sudo journalctl -u collector -n 200                          # last 200 lines
sudo journalctl -u collector --since "15 min ago"            # windowed
sudo journalctl -u collector --since "6h" | grep -iE "oom|Killed"   # oom hunt

# One-liner: success/miss counts over the last 15 min
sudo journalctl -u collector --since "15 min ago" \
  | grep -oE "slot_[0-9] \([a-z0-9_]+\): (person|MISS)" | sort | uniq -c | sort -rn
```

### 4.4 Deploy new code

```bash
# The standard path — safe with force-pushed history rewrites
sudo git -C /opt/turkey-footfall fetch origin main && \
  sudo git -C /opt/turkey-footfall reset --hard origin/main && \
  sudo systemctl restart collector

# Alternative: re-run install.sh (idempotent; also refreshes venv deps)
sudo /opt/turkey-footfall/src/deploy/gcp-vm/install.sh
```

Never `sed ... | tee /etc/systemd/system/collector.service` from the repo
template — the installed unit carries machine-local `Environment=` lines
(see 3.4).

### 4.5 Health-check battery — is the VM really feeding the report?

Run this before trusting any report:

```bash
# 1. service alive
sudo systemctl status collector --no-pager | head -12

# 2. live sampling — want slot_1..4 with real counts scrolling every ~40 s
sudo journalctl -u collector -f --no-hostname | grep --line-buffered -E "slot_|MISS|country"

# 3. success/miss ratio, last 15 min
sudo journalctl -u collector --since "15 min ago" \
  | grep -oE "slot_[0-9] \([a-z0-9_]+\): (person|MISS)" | sort | uniq -c | sort -rn

# 4. memory headroom
sudo systemctl show collector -p MemoryCurrent -p MemoryMax && free -h

# 5. genuine oom kills only (ignore googlevideo HLS URL noise)
sudo journalctl -u collector --since "6h" | grep -iE "oom-kill|Killed process|out of memory"

# 6. IBB proxy env is wired (values redacted; want 2)
sudo grep -c -E "IBB_PROXY_URL|IBB_PROXY_SECRET" /etc/turkey-footfall/proxy.env

# 7. DECISIVE end-to-end — grab a real Turkey frame right now
sudo bash -c 'set -a; . /etc/turkey-footfall/proxy.env; set +a; \
  cd /opt/turkey-footfall/src && timeout 90 .venv/bin/python - <<PY
from app.cameras import CAMERAS
from app.detect_core import resolve_stream, grab_burst
url = resolve_stream(CAMERAS["taksim_yeni"])
frames = grab_burst(url, n=2, stride=10)
print("frames grabbed:", len(frames), "shape:", frames[0].shape if frames else None)
PY'
# want: frames grabbed: 2 shape: (1080, 1920, 3)

# 8. deployed code is current
sudo git -C /opt/turkey-footfall log --oneline -1
```

### 4.6 Memory / disk / swap

```bash
free -h                                # RAM + swap use
df -h /                                # root disk (30 GB total, expect <10 GB used)
sudo du -sh /opt/turkey-footfall/src/.venv     # ~2 GB, normal
sudo du -sh /opt/turkey-footfall/src/web/snapshots  # anomaly + review cache
```

### 4.7 Rotate the external IP

Sometimes a fresh IP clears a CDN rate-block. Deleting and re-adding the
access config swaps the IP with no reboot:

```bash
NAME=$(gcloud compute instances describe turkey-collector --zone=us-east1-c \
  --format="value(networkInterfaces[0].accessConfigs[0].name)")
gcloud compute instances delete-access-config turkey-collector \
  --zone=us-east1-c --access-config-name="$NAME"
gcloud compute instances add-access-config turkey-collector --zone=us-east1-c
gcloud compute instances describe turkey-collector --zone=us-east1-c \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

### 4.8 Full rebuild from zero

The VM is disposable by design. Three secrets survive: the Firebase Admin key
(re-mint from Firebase Console → Service accounts → Generate new private
key), the IBB relay secret (`wrangler secret put PROXY_SECRET`), and the
Gmail app password (`myaccount.google.com/apppasswords`).

**Path A: rebuild on GCP (identical machine)**

```bash
# 1. create the machine
gcloud compute instances create turkey-collector \
  --project=turkey-footfall --zone=us-east1-c \
  --machine-type=e2-micro \
  --image-family=debian-13 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-standard

# 2. bootstrap (packages, repo, venv, Firebase key from Secret Manager, units)
gcloud compute ssh turkey-collector --zone=us-east1-c --project=turkey-footfall \
  --command='curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh | sudo bash'

# 3. swap, env files, digest timer (see 3.3)
```

**Path B: rebuild on any other Linux provider**

Requirements: Debian 12/13 or Ubuntu, x86_64, 1 GB+ RAM (with the swap
step), ~20 GB disk, outbound internet. The collector listens on nothing.
`install.sh` assumes a GCP image; on another provider, run its steps
manually and drop the Firebase key in by hand:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
  git python3 python3-venv python3-pip \
  ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
  ca-certificates curl fonts-dejavu-core

sudo git clone --depth 1 https://github.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection.git /opt/turkey-footfall
cd /opt/turkey-footfall/src
sudo python3 -m venv .venv
sudo TMPDIR=/var/tmp .venv/bin/pip install --no-cache-dir -r requirements.txt

sudo mkdir -p /etc/turkey-footfall
sudo install -m 0400 -o root -g root ~/serviceAccount.json /etc/turkey-footfall/serviceAccount.json

for unit in collector.service digest.service digest.timer; do
  sudo sed -e 's|__STORAGE_BUCKET__|turkey-footfall.firebasestorage.app|g' \
           -e 's|__INSTALL_DIR__|/opt/turkey-footfall|g' \
           -e 's|__SA_PATH__|/etc/turkey-footfall/serviceAccount.json|g' \
    /opt/turkey-footfall/src/deploy/gcp-vm/$unit | sudo tee /etc/systemd/system/$unit > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now collector.service
```

Then apply Path A's swap + env files + timer steps unchanged. Istanbul
cameras work from any ASN Cloudflare can reach; residential ASNs may not
even need the relay — test with `python -m tools.probe_country --country turkey`.

### 4.9 Uninstall

```bash
sudo systemctl disable --now collector digest.timer
sudo rm /etc/systemd/system/collector.service /etc/systemd/system/digest.{service,timer}
sudo rm -rf /opt/turkey-footfall /etc/turkey-footfall
sudo systemctl daemon-reload
```

Then delete the VM from the Console.

---

## 5. The 7 live analysis layers

Since fix 2 (2026-08), any dashboard tile can morph in place into a live
stream of analyzed frames on the exact camera it is playing. Click 🔬, pick
a layer, the server spins up a `LiveSession` and pushes analyzed JPEGs at
~1/s; the client renders them inside the tile. Up to 4 sessions in the grid
(one per tile). Switching a layer on a running tile MUTATES the session —
the stream, the tracker, and every accumulator (heat, counters, gesture
tallies) survive the switch. The VM is not involved; everything runs in
`app/live_analysis.py` on the operator machine.

**Shared pipeline (`LiveSession.run`, every tick ≈ TICK_TARGET_S = 0.8 s):**

```python
frame = self._grab()                              # (a) fetch frame
if frame is None: continue
boxes = self._infer(frame)                        # (b) YOLO + gates/ROI
self.tracker.update(boxes, now)                   # (c) BurstTracker
if layer in ("pose", "gestures", "body"):
    self._pose_pass(frame, boxes)                 # (d) top-down pose, ONLY when needed
faces_list = self._faces_pass(frame) if layer == "faces" else []
self._accumulate(frame.shape, boxes, now)         # (e) heat + line counters
img = self._render(frame, faces_list, layer)      # (f) draw the layer
self._publish(img)                                # (g) JPEG for the client
```

`INFER_LOCK` serializes every model call in this process (Ultralytics
`predict` is not thread-safe on a shared model). On CPU: one active session
runs at 1-2 fps, four concurrent at 0.3-0.5 fps each — degrading gracefully
instead of thrashing.

**Camera resolution for the picked tile:** `resolve_cam` first looks up
`cam_id` in the registry (`app/cameras.py`); if that misses, it reads
`web/local_grid.json` (written by notebook cell 32) and maps the slot to a
`kind ∈ {youtube, hls, webcamera24, skyline}` dict. The exact camera the
operator is watching is the exact one that gets analyzed — no cross-wiring.

### 5.1 Paths & speeds — `draw_paths_layer`

Trails + per-track id boxes + km/h chips. This is the only layer that draws
detection boxes for every class. Per-track history is capped at
`TRAIL_MAX_PTS = 40` centroid points; a colored line is drawn through the
centroids in the track's colour (stable per id).

km/h estimate for vehicles comes from `track_stats`:

```python
real_len = VEHICLE_LENGTH_M.get(cls or "")       # 4.5 for car, 12 for bus, ...
if real_len and speeds:
    exts    = [max(b["x2"]-b["x1"], b["y2"]-b["y1"]) for b in boxes if ...]
    m_per_px = real_len / (sum(exts) / len(exts))     # pixel-scale ruler
    kmh     = round(sum(speeds)/len(speeds) * m_per_px * 3.6, 1)
```

Honest error band ±30-50 % (vehicle not always parallel to the image plane).
The report shows this only when the camera has enough statistical mass
(≥ 5 samples AND ≥ 10 % of rounds carrying one).

### 5.2 Pose & skeleton — `draw_pose_layer`

Skeletons ONLY, on people close enough for the per-crop pose pass — no
detection boxes, no vehicles. Because a street-cam pedestrian is 30-120 px
tall and a full-frame pose pass at 640 hands the model ~15 px of person and
finds nothing, this layer runs **top-down pose**: for each detector `person`
box, crop the neighborhood with 25 % padding and run YOLOv8n-pose on the
crop alone. The best pose-person inside each crop claims that box:

```python
def attach_keypoints_crops(model, frame, boxes,
                           imgsz=256, pad_frac=0.25,
                           min_box_h=40, conf=0.25) -> int:
    persons = [b for b in boxes if b.get("cls") == "person"
               and (b["y2"] - b["y1"]) >= min_box_h]
    crops, offsets = [], []
    for b in persons:
        bw, bh = b["x2"]-b["x1"], b["y2"]-b["y1"]
        px, py = bw*pad_frac, bh*pad_frac
        x1 = max(0, int(b["x1"]-px)); y1 = max(0, int(b["y1"]-py))
        x2 = min(W, int(b["x2"]+px)); y2 = min(H, int(b["y2"]+py))
        crops.append(frame[y1:y2, x1:x2])
        offsets.append((b, x1, y1))
    results = model.predict(crops, imgsz=imgsz, conf=conf, verbose=False)
    for (b, ox, oy), res in zip(offsets, results):
        if not len(res.boxes): continue
        qi = max(range(len(res.boxes.conf)), key=lambda i: res.boxes.conf[i])
        kps = res.keypoints.data.tolist()[qi]
        b["kps"] = [[x+ox, y+oy, c] for x, y, c in kps]     # back to frame coords
```

Output: 17 COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists,
hips, knees, ankles) drawn on each close-enough person. What is too far is
labeled honestly: "skeletons on 3 of 12 people, rest too far".

### 5.3 Hand gestures — `draw_gestures_layer` + `app/gestures.py`

Three arm-level gestures on the same skeletons: `hand_raised` (one wrist
above its shoulder for ≥ 3 pose frames), `both_hands_up` (both wrists above
both shoulders), `wave` (raised wrist crosses the elbow ≥ 2 times). The
session keeps a running total (`self.gesture_counts`) so the caption reads
"session: hand_raised x3, wave x1" once anyone has done anything. An empty
scene reads "no gestures detected right now" — that is honest, not a bug.

### 5.4 Body anomalies — `draw_body_layer`

Fall-detection-style live view. `label_track` (`app/behavior_labels.py`)
runs three tiers per track and returns exactly one label:

1. **Pose flags from the skeleton** (`pose_flags_of`): the shoulder-mid ↔
   hip-mid line — its angle from vertical; > `FALL_TORSO_DEG = 60°` for ≥
   `POSE_FLAG_MIN_FRAMES = 2` frames → `fall_suspect`.
2. **Course reversals** (`heading_turns`): how many > 90 ° reversals across
   the trajectory; ≥ 3 → `erratic`.
3. **Pure kinematics**: mean speed, moving fraction, net displacement over
   path — yields `running` / `walking` / `standing` / `dwelling` / `driving`
   / `parked` / `normal`.

The layer draws ONLY the alert-grade labels
(`BODY_ANOMALY_LABELS = {"fall_suspect", "erratic", "running"}`): a red or
orange box + skeleton overlay + a verdict chip on flagged people, a HUD
tally in the top-left (`persons in view: N, flagged: M`), and a red ALERT
banner while a `fall`/`erratic` flag is live.

### 5.5 Face detection — `draw_faces_layer_img` + `app/faces.py`

Face rectangles only (no embeddings, no database). The detector is
**YuNet** (OpenCV Zoo, ~230 KB ONNX), CPU-only, ~15 ms on a 960 × 540 frame.
At street-cam distance faces are often below the detector's resolution — the
caption "no faces at this distance/resolution" is honest.

### 5.6 Heat vision — `draw_heat_layer`

The whole picture changes when you pick this layer (fix 3 requirement).
Not a thermal sensor — a stylized colormap driven by brightness plus the
session's dwell accumulation:

```python
gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
signal = gray * 0.72                                      # baseline: brightness
peak   = float(np.asarray(grid).max())
if peak > 0:
    dwell  = np.sqrt(grid / peak)                         # sqrt gamma flattens peaks
    dwell  = cv2.resize(dwell, (W, H), INTER_LINEAR)
    dwell  = cv2.GaussianBlur(dwell, (0,0), sigmaX=max(2, W/96))
    signal = np.clip(signal + dwell*0.55, 0, 1)           # dwelt-in zones run hotter
out = cv2.applyColorMap((signal*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
```

`grid` is a `GRID_H × GRID_W` (27 × 48) matrix of dwelt seconds per cell;
`bump_heat` banks each detected foot point (bottom-centre of the box)
weighted by the interval since the previous tick. Switching layers away and
back keeps the accumulation — the map keeps growing.

### 5.7 Line crossing — `draw_line_layer` + `update_crossings`

A crossing line (defined per-camera in `app/cameras.py`, or the default
horizontal `DEFAULT_LINE = [[0.10, 0.62], [0.90, 0.62]]` — the sidewalk
band). Every strict sign flip of a track's foot point across the line is a
crossing event; direction (in/out) follows the A→B point order (negative →
positive side = "in"):

```python
def update_crossings(side_state, tracks, frame_shape, line, cross):
    H, W = frame_shape[:2]
    for tr in tracks:
        if tr.misses: continue
        fx = (tr.boxes[-1]["x1"] + tr.boxes[-1]["x2"]) / 2
        fy = tr.boxes[-1]["y2"]
        side = _line_side(fx/W, fy/H, line)          # signed cross product
        if side == 0: continue                       # exactly on the line: ambiguous, skip
        prev = side_state.get(tr.tid)
        side_state[tr.tid] = side
        if prev is None or prev == 0: continue
        if prev < 0 and side > 0:  cross["in"]  += 1
        elif prev > 0 and side < 0: cross["out"] += 1
```

`side == 0` (landing exactly on the line) is deliberately skipped — it is
ambiguous and would double-count jitter around the boundary. The caption
shows "IN x / OUT y (session total)".

---

## 6. The deep-window analysis (`behavior.analyze_window`)

Separate engine, on-demand: grabs a longer window (default 12 frames at
stride 12 ≈ 0.5 s apart) from one camera, runs the same gated detection per
frame, threads them into per-individual tracks with `BurstTracker`, and
returns a per-individual profile:

- `path` — foot-point trajectory (normalized, JSON-safe)
- `distance / speed` — path length, net displacement, mean/max px/s, and a
  km/h estimate for vehicles (same ±30-50 % band as the live layer)
- `moving_frac` — fraction of steps that actually moved (stood still 80 %
  of the window)
- `direction` — dominant screen direction of the net displacement
- `zones` — heatmap cells visited (ties the trajectory to the long-horizon
  dwell map)
- `nn_min / mean_px` — closest same-class neighbour over the window
  (crowding / pairing signal)
- `label` — one readable behavior verdict per individual (from
  `label_track`) with its evidence in `label_reasons`
- `gestures` — arm-level gestures over the window, pose mode only

Optional layers per request:

- `pose=1` — runs the top-down pose pass, enriches `label` and populates
  `gestures`.
- `want_faces=1` — face detection on the last frame.
- `lock=auto` or `lock=<track_id>` — draws a crosshair target lock on that
  individual and returns the normalized offset from frame center (`dx, dy` ∈
  `[-0.5, 0.5]` — the exact signal a pan/tilt controller would consume if
  local hardware existed).

CLI form:

```bash
cd /path/to/repo/src
python -m tools.analyze_window --cam taksim_yeni --pose --faces --lock auto
```

Output: annotated JPEG + JSON profile under `web/snapshots/behavior/`, LRU
40 files. The dashboard's "Analyze window" button calls `POST /api/deep-analyze?cam=<id>`.

---

## 7. The notebook — offline analytics

The main notebook is `turkey_business_activity.ipynb` (at the repo root; the
imports find `src/app/` automatically). It uses the same `detect_core` and
`reid` modules as the collector so the numbers reconcile. Twelve sections:

| # | Cell topic | What it does |
|---|---|---|
| 0 | Setup | Dependency check + `MODEL_WEIGHTS = 'yolo26m.pt'` + one-time `load_model` |
| 1 | Camera picker | Numbered catalog across all countries; operator picks 4 by number (all must share a country); live probe rejects dead picks |
| 2 | Single-frame check | Grabs one frame from the first pick and annotates it |
| 3 | Footfall time series | Sparse sample every `interval_s`; DataFrame + peak-hour chart |
| 4 | Anomalies + peak-hour profile | Robust rolling z (median + MAD × 1.4826) marks anomalies |
| 5 | Dwell / prolonged stops | Dense burst + ByteTrack for a short window; per-track dwell + movement |
| 5b | Re-identification | ReidStore over N frames; per-class unique / seen-again / regulars (≥ 3) |
| 6 | Business score | Composite `volume_median × w0 + linger_rate × w1 + consistency × w2` (empty data → honest `None` + note) |
| 7 | Live cloud dashboard | Writes `web/local_grid.json` = the picked cams, spins up `http.server` on `localhost:8000`, opens the browser |
| 8 | Compare multiple sites | Ranks the picked cameras by activity |
| 9 | Live summary | Rollup of the session + always-on visual footfall/anomaly chart |
| 10 | Accuracy calibration | 10a captures frames + predictions at 640/960; 10b interactive labelling; 10c MAE + bias per cam per size |
| 11 | Forecasting | 11a Firestore delta fetch → CSV cache; 11b 15-min grid + eligibility; 11c persistence / seasonal-naive / hour-of-week profile / closed-form ridge; 11d small GRU |

### 7.1 The local twin notebook

`turkey_business_activity_yolov8n.ipynb` (git-ignored — never in the repo)
is a hand copy of the main notebook with `MODEL_WEIGHTS = 'yolov8s.pt'`
(or `yolov8n.pt`) so the operator can see the VM's view on the same frames.
Documented in `.gitignore` line 54.

### 7.2 Section 11 forecasting — how it decides

Every fancy model must beat "the same time yesterday" (seasonal-naive) on
MAE over the last 25 % of the cache (never touched during fitting). The
ladder:

- **persistence** — `y_{t+h} = y_t`
- **seasnaive24** — `y_{t+h} = y_{t+h - 24h}` (the honesty baseline)
- **profile** — median count per local (hour-of-day) slot, or (hour-of-week)
  once the cache carries ≥ 7 days
- **ridge** — closed-form numpy ridge on lags (1, 2, 3, 4, 96), rolling
  means (4, 12), sin/cos of the target hour-of-day, per-camera one-hots
- **gru** — small GRU (hidden 32, ~15 k params) reading the last 24 h to
  emit the next 12 h; trains on CPU in well under a minute; joins the
  ladder once the cache holds enough windows

`skill = 1 - mae / mae['seasnaive24']` (positive = better). A perfectly-
stable stream (`seasnaive24 MAE = 0`) yields `n/a` rather than misleading
infinities.

---

## 8. Model choice + parameters

### 8.1 Two detectors, two runtimes

`yolov8s.pt` on the VM (since 2026-08-05, `@ 640`); `yolo26m.pt` in the
notebook (`@ 960`). The notebook is the accuracy reference; on the same
live frames: `yolov8n@512` (the pre-2026-08 VM config) found 0 people in
Taksim and 0 vehicles in Sarachane; `yolov8s@960` found 5 and 7;
`yolo26m@960` found 6 and 16 + a bus. Undercounts were the config, not the
cameras.

Model size ladder (COCO):

| Model | Params | mAP50 | CPU 1080p pass | Verdict |
|---|---|---|---|---|
| `yolov8n` | 3.2 M | 37.3 | ~120 ms | Old VM config; low recall on wide street cams |
| **`yolov8s`** | **11.2 M** | **44.9** | **~280 ms** | **Current VM (@640, since 2026-08-05)** |
| `yolov8m` | 25.9 M | 50.2 | ~700 ms | Peak RSS > 900 MB → oom-kill on e2-micro |
| `yolov8l` | 43.7 M | 52.9 | ~1400 ms | Not realistic on e2-micro |
| `yolo26m` | ~30 M | ~50 (NMS-free) | ~800 ms CPU | Notebook only |

### 8.2 The key knobs

| Env / flag | Value | Why |
|---|---|---|
| `--imgsz 640` | Was 512 until 2026-08-05 | 640 recovers small objects the 512 pass lost; ~0.39 s / pass on the VM (× 2 frames × 4 cams = ~3 s / 40 s round) |
| `--burst 2 --burst-stride 13` | Two frames ~0.5 s apart | Median kills single-frame flicker; two points feed the speed estimator |
| `--interval 40` | s | Firestore-quota bound (~17 k writes/day of 20 k) |
| `MemoryHigh=760M / MemoryMax=900M` | cgroup guardrails | Fits the 1 GB e2-micro with margin |
| `OMP_NUM_THREADS=2` | | Matches shared vCPU count; torch default oversubscribes |
| `MALLOC_ARENA_MAX=2` | | glibc per-thread arenas cost 50-150 MB of RSS on a threaded python |
| `DEFAULT_PER_CLASS_CONF` (in `detect_core.py`) | Per-class gate map | Nightly `night_adjusted_conf(+0.08)` + per-camera boosts learned by review |
| `EXTRA_CLASSES` env | `bird, dog, cat, backpack, handbag, suitcase, umbrella` | Feeds unattended-object watch + "other objects" report line |
| `FALL_CHECK=1` env | | Person-loiter → one pose pass on that crop; horizontal torso → "Possible FALL" |

### 8.3 Per-camera confidence calibration

`tools/calibrate_conf.py` reads the operator's verdict history, computes a
per-`(camera, class)` confusion matrix, and picks the lowest threshold that
achieves **precision ≥ 0.90** with **≥ 30 verdicts** — writes it to
`data/per_camera_conf.json`. `cameras._merge_per_camera_conf()` runs AFTER
`_merge_confidence_boost` and overrides it per pair. A calibrated pair is
tagged `source=calibration` in the Learning-proof panel.

---

## 9. Anomalies + reporting

The collector runs a set of deterministic anomaly gates per round and per
camera. Each gate has an explicit trigger + a debounce window so the digest
does not spam.

| Gate | Trigger | Debounce |
|---|---|---|
| Extreme load | Person / vehicle count above a rolling robust-z threshold | 3 rounds |
| Camera obscured | Mean brightness drops below the camera's night floor while the clock says day | 5 rounds |
| Camera dark | Sample MISSes exceed the rest-and-probe schedule | 3 rounds |
| Loiter | Same person track stays inside a box for ≥ camera's `loiter_s` | Cap 10/day/cam |
| Returning visitor | Same OSNet identity seen at ≥ 1.2 × box-scale distance from previous sighting | Person only, ≥ 64 px floor |
| Unattended object | Bag / suitcase without an owner-nearby person for ≥ 90 s | Owner-nearby gate |
| Fall suspect | Person loiter + horizontal torso from one pose pass | Under `FALL_CHECK=1` |
| Crowd rush | Sudden speed × density spike | 2 rounds |

Anomaly evidence is captured as an annotated JPEG under
`snapshots/anomalies/<cam>/<ts>.jpg` (24 h lifecycle in Storage). Each event
also lands in `events/` in Firestore (also 24 h TTL) so the dashboard's
"Events" strip can show it live.

**Reporting**:

- Twice-daily archive digest (12:00 + 20:00 Israel) → project archive
  mailbox only, via `digest.timer`. Uses `tools/daily_digest.py`.
- On-demand PDF from the dashboard header (private tile: "Send Report From
  VM" → `POST /api/send-report`; public tile: GitHub Actions workflow
  dispatch). Same PDF composer (`tools/report_pdf.py`), different sender.

The report is honest about statistical mass: km/h fields only publish when
there are ≥ 5 speed samples AND ≥ 10 % of rounds carrying one — otherwise
`-`.

---

## 10. Active-learning loop

Every reviewed frame becomes training data; nightly (or on-demand) a
head-only fine-tune runs on GitHub Actions; the promoted head lands in
Storage; the collector hot-swaps it with no restart. The whole loop uses
zero paid resources.

### 10.1 Uncertainty-first frame queue

Every stored box carries `uncertainty ∈ [0,1]` from `app/uncertainty.py`:

```
uncertainty = 0.6 * margin + 0.4 * flip_delta
```

- `margin(conf, gate, span=0.25)` — 1.0 at the class gate, falling linearly
  to 0 at `gate ± span`. Cheap: every box already has `conf` and the
  effective gate the burst ran with.
- `flip_delta` (optional, sampled bursts only) — one extra pass on the
  horizontally-flipped frame; per-box IoU-matched conf delta. Costs one
  inference on ~1-in-5 bursts on ONE camera when `UNCERTAINTY_FLIP=1`.

Persisted: frames → sidecar JSON `metadata.boxes[i].uncertainty`; crops →
filename suffix `_uNN` (e.g. `..._u87.jpg` = 0.87). The review UI's
`labels.frame_uncertainty` prefers the persisted value; missing → margin
fallback. The naive random frame sampler no longer exists.

### 10.2 BADGE crop sampler

`app/badge.py`: hand-rolled k-means++ init picks a diverse batch weighted by
uncertainty — OSNet embeddings as direction, uncertainty as magnitude. Env
switch `REVIEW_SAMPLER=badge|naive` (default `naive`); per-request override
`?strategy=` on `/api/review-sample`. Review rows record `sampler` +
`uncertainty_at_selection` so the naive-vs-BADGE efficiency replay can run
offline.

### 10.3 Head-only fine-tune + promotion gate

`tools/train_head.py` wraps `yolo detect train` with the backbone frozen
(`freeze=<all-but-head>`), mosaic/mixup off, HSV + flip on, ≤ 10 epochs
early-stop. Emits `data/adapters/<cam>/head_<ts>.pt` — Detect-head tensors
only (~4-6 MB).

`tools/promote_adapter.py` runs `val` on the exporter's chronological
90/10 split for both the baseline and the candidate; gate:

- `ΔmAP50 ≥ +0.5` percentage points, AND
- No class drops > 2 pp (person / car: 0 pp — the counts that drive every
  report).

Pass → atomic `current` pointer update + `history.jsonl` append.
Fail → `gate.log` line. `--rollback` restores the previous pointer.

Transport: the operator's local labels + frames flow to Storage via
`app/training_sync.py` (batched, ledger-diffed); GitHub Actions
(`.github/workflows/train.yml`) trains on free public-repo runners; the
promoted head lands in Storage; the collector polls `current.json` every
30 rounds and hot-swaps in place.

**Byte-identical fallback**: missing / unreadable `current.json` → base
model runs untouched. No adapter is present at rest; the head is loaded
into memory only if a promoted one exists and validates.

### 10.4 "Labels vs quality" curve

`GET /api/al-curve` reads `history.jsonl` (+ Firestore mirror
`training_events`, TTL 30 d, one write per promotion) and the dashboard
draws a Chart.js line: labels_total on X, mAP50 on Y, rejected candidates
greyed, baseline dashed. The chart fills in after a week of nightly runs.

---

## 11. Firebase project setup

The dashboard is `onSnapshot`-live — every collector write reaches the
browser instantly, no polling. The setup, once per project:

**1. Create the project.** `console.firebase.google.com` → Add project →
enable Firestore in test mode (locked before the public deploy — see step 5).

**2. Backend credentials for the collector.**

```bash
# Project settings (gear) → Service accounts → Generate new private key.
# Save the JSON outside git (gitignored if named firebase-service-account.json)
export FIREBASE_CREDENTIALS=/path/to/serviceAccount.json
pip install firebase-admin
```

**3. Run the collector against Firebase.**

```bash
python -m app.collector --backend firebase --interval 20 \
  --only konya_hukumet,kapali_carsi,misir_carsisi,eminonu,istiklal_1
```

Each round writes one history doc per camera to `footfall` and overwrites
`latest/{cam_id}`. Run on an open network (IBB/YouTube blocks restricted
sandboxes). Keep it alive with systemd / Docker / `nohup`.

**4. Web frontend.** Firebase Console → Project settings → Web app → copy
the SDK config. Create `web/firebase-config.js` with
`export const firebaseConfig = {...}`. Then:

```bash
cd src/web && python -m http.server 8000     # http://localhost:8000
```

The page subscribes with `onSnapshot` and every collector write appears
immediately.

**5. Security rules — this is what protects the DB.** Test mode lets
anyone on the internet read AND WRITE. The public web-SDK config
(`apiKey`, `projectId`) ships in every visitor's browser and is not a
secret; the security rules are.

Locked-down rules live in `src/firestore.rules`: public read on the
dashboard collections (`footfall`, `latest`, `reid_stats`, `events`,
`config`), all client writes denied (the Admin SDK bypasses rules, so the
collector is unaffected). Deploy them:

```bash
npm install -g firebase-tools    # once
firebase login
# .firebaserc: {"projects":{"default":"<your-project-id>"}}
firebase deploy --only firestore:rules
```

Then in Firebase Console → Firestore → Rules, verify writes show `if false`.

**6. TTL policies.** Firebase Console → Firestore → Time-to-live → add TTL
on `footfall.expire_at` AND `events.expire_at` (both self-prune after 24 h).

**7. App Check (anti-abuse / read-quota protection).** Rules make the data
read-only but a scraper can still burn the read quota. App Check requires
every request to carry a reCAPTCHA-v3 attestation.

Firebase Console → App Check → Apps → register the web app with the
reCAPTCHA v3 provider. Copy the site key into `web/firebase-config.js` as
`recaptchaSiteKey`; `web/app.js` initialises App Check automatically when
it is set. When confident, App Check → Firestore → Enforce.

Enable enforcement ONLY after the site key is live on the page —
otherwise enforced reads are rejected and the dashboard goes blank.

**8. Rate limit + cost cap.** Firestore Spark tier ≈ 20 k writes/day. The
collector prints the projected daily write count on startup and clamps
`--interval` to a 5 s floor. Set a budget alert in Google Cloud → Billing;
on Blaze, also set an App Engine daily spending limit — that is the real
hard cap. Firestore has no per-user request-rate limit of its own.

---

## 12. Cloudflare proxy for IBB

`kamerayayin.ibb.istanbul` refuses every Google Cloud IP range (HTTP 403)
but answers normally from any other address. A Cloudflare Worker on the
free plan (100 k requests/day, our load ~26 k/day) proxies the IBB
requests through Cloudflare's edge — a different ASN — restoring
`taksim_yeni`, `sultanahmet_1_yeni`, `eyup_sultan_yeni`,
`beyazit_meydan_yeni`.

Worker source and deploy artefacts stay in `src/deploy/cloudflare-proxy/`:
`worker.js` (the fetch handler) and `wrangler.toml` (the deploy config).

**One-time setup (~5 min):**

```bash
# 1. Free Cloudflare account: https://dash.cloudflare.com/sign-up (no card).

# 2. Install wrangler (once)
npm install -g wrangler
wrangler login                # opens a browser tab, grant access

# 3. Deploy the worker
cd src/deploy/cloudflare-proxy
wrangler deploy               # prints https://ibb-proxy.<your-subdomain>.workers.dev

# 4. Set the shared secret (any random string)
wrangler secret put PROXY_SECRET
# Suggest: openssl rand -hex 24

# 5. Wire the VM
sudo tee /etc/turkey-footfall/proxy.env > /dev/null <<EOF
IBB_PROXY_URL=https://ibb-proxy.<your-subdomain>.workers.dev
IBB_PROXY_SECRET=<the same secret you just set>
EOF
sudo chmod 600 /etc/turkey-footfall/proxy.env
sudo systemctl restart collector
```

**Verify (from the VM):**

```bash
cd /opt/turkey-footfall/src && sudo -E .venv/bin/python -m tools.probe_country --country turkey
# want: the four IBB cameras flip to LIVE; total 7/24 live if the YouTube
# tier is also installed

# spot-check the worker itself
curl -s -H "X-Proxy-Secret: <your secret>" \
  "https://ibb-proxy.<you>.workers.dev/https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8" \
  | head -3
# expect: #EXTM3U ...
# 403 from the worker → secret wrong
# 403 with an IBB body → Cloudflare itself blocked (rare)
```

**What the worker does NOT do:**

- No caching that would break liveness (`cf.cacheTtl: 4` matches the ~4 s
  HLS segment rotation).
- No proxying of other hosts (only `kamerayayin.ibb.istanbul`; everything
  else returns 403).
- No proxying of `tvkur.com` (Konya, Otogar and other Turkish webcamera24
  streams — tvkur restricts even residential ASNs, and a Cloudflare edge
  faces the same 403; those need a Turkish-IP proxy specifically, out of
  the free-tier budget).

---

## 13. GCP billing kill-switch

Auto-disables billing on `turkey-footfall` the moment a Cloud Billing
budget threshold is crossed. This is the difference between "an email at
3 AM saying you are over budget" (plain budget alert) and "services
stopped billing you three minutes after crossing $5" (this).

Source stays in `src/deploy/gcp-billing-killswitch/`: `main.py` (the Cloud
Function), `requirements.txt`.

**Prerequisites (once):**

1. **Enable APIs**: Cloud Pub/Sub, Cloud Functions, Cloud Build, Cloud
   Billing.
2. **Create a Pub/Sub topic** the budget will publish to:
   ```bash
   gcloud pubsub topics create budget-alerts --project=turkey-footfall
   ```
3. **Wire the topic into the budget**: GCP Console → Billing → Budgets &
   alerts → open the budget → Manage notifications → Connect a Pub/Sub
   topic → pick `projects/turkey-footfall/topics/budget-alerts`.
4. **Create the runtime SA**:
   ```bash
   gcloud iam service-accounts create billing-killswitch \
     --display-name "Billing kill-switch runtime" \
     --project=turkey-footfall
   ```
5. **Grant it two project-level roles**:
   ```bash
   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/billing.projectManager

   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/browser
   ```
   `billing.projectManager` has `deleteBillingAssignment` (the actual
   unlink). `browser` has `resourcemanager.projects.get` for the
   idempotency check that runs before the unlink.
6. **Force-create the Pub/Sub service agent** (skip only if this project
   has used Pub/Sub push-subscriptions before):
   ```bash
   gcloud beta services identity create --service=pubsub.googleapis.com \
     --project=turkey-footfall
   ```

**Deploy:**

```bash
cd src/deploy/gcp-billing-killswitch
gcloud functions deploy billing-killswitch \
    --gen2 \
    --project=turkey-footfall \
    --region=us-east1 \
    --runtime=python312 \
    --source=. \
    --entry-point=stop_billing \
    --trigger-topic=budget-alerts \
    --set-env-vars=PROJECT_ID=turkey-footfall \
    --service-account=billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
    --memory=256Mi \
    --timeout=60s \
    --max-instances=1

# 2-4 minutes. Then:
gcloud functions describe billing-killswitch --gen2 --region=us-east1
# want: state: ACTIVE
```

**Grant the trigger SA `run.invoker`** on the Cloud Run service backing the
gen2 function (otherwise every Pub/Sub delivery is rejected):

```bash
gcloud functions add-invoker-policy-binding billing-killswitch \
  --gen2 --region=us-east1 \
  --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com
```

**Prove it works**:

```bash
gcloud pubsub topics publish budget-alerts \
    --message='{"budgetDisplayName":"test","costAmount":999,"budgetAmount":1}'

gcloud functions logs read billing-killswitch --gen2 --region=us-east1 --limit=20
# want: "billing DISABLED on turkey-footfall"
# Billing console should show "Billing account: None"
```

**Re-enable billing after the test**: GCP Console → Billing → Link this
project to a billing account.

**What it does NOT do:** does not delete resources (the VM, Firestore data,
Storage bucket, and the function itself all remain — they simply stop
generating billable events until the billing account is re-linked); does
not touch free-tier services (the e2-micro keeps running); does not care
which threshold crossed (Google publishes at every configured threshold —
50/90/100/120 %; the function only unlinks when `costAmount ≥
budgetAmount`).

Cost of the kill-switch itself: zero — one Pub/Sub message per threshold
cross, one Cloud Function invocation (2 M/month free), Cloud Storage for
the function source (Always Free).

---

## 14. Troubleshooting / FAQ

**"The dashboard's per-tile counts are ‘from Ns ago’ and the label is red."**
The collector is not keeping up with `--interval`. Run the health-check
battery (§4.5); if memory is fine but CPU is saturated, drop to
`--weights yolov8n.pt --imgsz 768` in the ExecStart line (§3.4).

**"Live analysis on a picked skyline camera 404s."** Fixed in the audit
pass — `_cam_from_slot` now handles `kind="skyline"` slots from
`web/local_grid.json`. Old pre-fix versions failed with
`ValueError("no analyzable stream")`.

**"The daily digest e-mail never arrives."** Check `/etc/turkey-footfall/digest.env`
exists with a real Gmail app password (`sudo cat` as root); check
`sudo systemctl status digest.timer` for the next fire time; run
`sudo systemctl start digest.service` for an immediate manual run.

**"Turkey cameras always MISS from the VM."** IBB geo-blocks Google Cloud
ASNs. Either wire the Cloudflare Worker (§12) or accept that the grid
falls through to Thailand / Japan / USA until IBB unblocks. `tools/probe_country
--country turkey` shows live status of each Turkey camera.

**"How do I add a new camera?"** Edit `src/app/cameras.py`: pick a stable
`cam_id`, fill in the `kind` (`hls | youtube | webcamera24 | skyline`), the
URL and page, the display name, and any `roi` / `roi_exclude` / `line`
overrides. `python -m tools.probe_country --country <c>` verifies it. No
VM change needed — a `git pull + systemctl restart` picks it up.

**"How do I take the VM offline for a week?"**
`gcloud compute instances stop turkey-collector --zone=us-east1-c` —
Firestore keeps the last 24 h (TTL), dashboard shows the last known state.
`gcloud compute instances start ...` when back.

**"How much does this really cost per month?"** $0 in normal operation. The
e2-micro is Always Free; Firestore Spark tier writes stay under 20 k/day;
Firebase Storage stays under the 5 GB free limit (~50 MB active with 24 h
TTL); egress from GCP to Firebase (same region) is free. The kill-switch
guards against surprise overages (§13).

**"Twin notebook is missing after clone."** By design — see `.gitignore:54`.
The twin (`turkey_business_activity_yolov8n.ipynb`) is local-only: copy the
main notebook and change `MODEL_WEIGHTS = 'yolov8s.pt'`.

---

## 15. Appendix: design decisions taken

Retrospective. Line-by-line engineering review of the original Active-
Learning SPEC against what the codebase and production environment
actually needed. Each verdict says what survived, what was replaced, and
why. Kept here for the reader who wants the WHY behind the current shape.

**D1 — MC-Dropout uncertainty**: REJECTED. YOLOv8 detection has zero
`nn.Dropout` modules — a T=10 stochastic-pass variance would be exactly 0.
Replacement (WS1, shipped): margin against the effective per-class gate
(0.6) + one-pass flip delta on sampled bursts (0.4). Same downstream
contract, near-zero cost.

**D2 — LoRA-via-peft**: REPLACED. peft-wrapping Ultralytics' Detect
breaks attribute access (`stride`, `nc`, `reg_max`), EMA deep-copies, and
checkpoint pickling. Native `yolo detect train freeze=<all-but-head>`
delivers the same "small artifact, frozen backbone" outcome without exotic
deps. The head-only `.pt` (~4-6 MB) is the "adapter".

**D3 — COCO export**: SUPERSEDED. `tools/export_labels.py` already emits a
YOLO-format dataset (chronological 90/10 split, verdict mapping incl.
relabel + operator-added misses). Ultralytics trains from this natively; a
COCO converter can be added later if any external tool needs it.

**D4 — BADGE embeddings**: UPGRADED input. OSNet ONNX now ships in-repo
and is the default embedder everywhere (auto-detected). BADGE gets 512-d
identity-grade vectors on day one; k-means++ init is hand-rolled (~30
lines) — sklearn stays off the VM.

**D5 — Architecture option B (split VM / external trainer)**: CONFIRMED.
`app/pool_sync.py` already moves artefacts VM↔Storage↔operator with
manifests, batching, and public URLs; the training round-trip reuses it
under prefix `training/`.

**D6 — Bit-identical fallback**: TRIVIALLY SATISFIED. Head-overlay
loading means "no adapter file" = base model untouched — byte-identical.
No identity-LoRA gymnastics.

**D7 — VM resource envelope**: TIGHTENED after live oom-kill incidents.
Standing envelope for every VM-side addition: `MALLOC_ARENA_MAX=2`,
`OMP_NUM_THREADS=2`, 2 GB `/swapfile` present, any new per-round compute
keeps the measured round under ~30 s, any new upload path batches (≤ 40
objects/pass), Firestore stays under 20 k writes/day.

**D8 — mAP wording**: ADJUSTED. mAP targets measured with Ultralytics
`val` on the exporter's chronological val split. The "40 %-fewer-labels"
headline is measured naive-vs-BADGE on label-count-matched checkpoints as
a chronological comparison; a two-camera A/B is optional stretch, not a
gate.

**D9 — Trainer host + adapter retention**: operator-decided at kickoff.
Current defaults: GitHub Actions for training (public-repo runners, free);
adapter retention = full history in `history.jsonl`.

---
