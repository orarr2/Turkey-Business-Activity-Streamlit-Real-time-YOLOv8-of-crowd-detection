# Project Guide: Business Activity / Live Footfall

Operational reference for the VM-based footfall analysis system. The Hebrew
twin of this document is [`PROJECT_GUIDE_HE.md`](PROJECT_GUIDE_HE.md); same
chapter numbering.

---

## Quick navigation

1. [What the project does](#1-what-the-project-does)
2. [Architecture: where each piece runs](#2-architecture-where-each-piece-runs)
3. [The VM: deep dive](#3-the-vm-deep-dive)
4. [VM commands cheatsheet](#4-vm-commands-cheatsheet)
5. [The 10 live analysis layers](#5-the-10-live-analysis-layers)
6. [The deep-window analysis (`behavior.analyze_window`)](#6-the-deep-window-analysis-behavioranalyze_window)
7. [The notebook: offline analytics](#7-the-notebook-offline-analytics)
8. [Model choice + parameters](#8-model-choice--parameters)
9. [Anomalies + reporting](#9-anomalies--reporting)
10. [Active-learning loop](#10-active-learning-loop)
11. [Firebase project setup](#11-firebase-project-setup)
12. [Cloudflare proxy for IBB](#12-cloudflare-proxy-for-ibb)
13. [GCP billing kill-switch](#13-gcp-billing-kill-switch)
14. [Troubleshooting / FAQ](#14-troubleshooting--faq)

---

## 1. What the project does

The system turns 4 public street cameras into a quantitative time series:

> **live HLS stream -> YOLOv8 frame inference -> counts + appearance re-ID ->
> Firestore -> real-time web dashboard + Jupyter analytics.**

Every sampling round (40 s in the shipped cloud deployment; `--interval` sets
it), the collector grabs a burst from each active camera, runs YOLO on each
frame, computes gated counts + appearance signatures + anomaly gates, then
writes the result to Firestore. The web dashboard subscribes with
`onSnapshot`; every collector write appears immediately.

The grid is **country-generic**: it always runs 4 cameras from ONE country
and rotates through a priority ladder (**Turkey -> Thailand -> Japan -> USA**),
falling to the next country only when the active one goes fully dark. Turkey
is the project's subject; from GCP the Istanbul IBB feed is geo-blocked, so
the grid usually runs the foreign benches (YouTube-Live-backed street cams)
until a Cloudflare Worker on a non-Google ASN restores IBB (see chapter 12).

Outputs:

- **Footfall**: people / vehicles per camera per round.
- **Re-identification**: same-person / same-vehicle recognition over time
  (OSNet embeddings when the ONNX file is present, HSV histogram otherwise).
- **Anomalies**: extreme load, camera obstruction, blackouts, loitering,
  returning visitor, unattended object, fall suspect.
- **Typical vehicle speed** in km/h (published only when the camera has
  statistical mass: >= 5 samples and >= 10 % of rounds carrying one; plazas
  without real traffic show `-` rather than a fabricated number).
- **PDF report** by e-mail on demand (dashboard "Send Report" button); a
  scheduled twice-a-day digest goes to the project archive mailbox only.

Everything runs on free tiers: open-source model, GCP `e2-micro` on Always
Free ($0/month), GitHub Actions on a public repo, Firebase Spark plan.

### 1.1 The country fallback ladder

`CountryDirector` manages two nested ladders:

- **Country ladder (priority)**: Turkey -> Thailand -> Japan -> USA.
  Grid runs 4 cameras from ONE country; moves to the next country only
  when the active one cannot deliver a single live camera. A dead
  single camera does not shift the grid; a bench camera from the SAME
  country backfills.
- **Camera ladder (per country)**: `CameraPool` walks the list and
  keeps the first 4 live cameras assigned each round; a camera that
  misses 3 consecutive samples rests 15 min; `tvkur` (Konya) cameras
  rest after one miss.
- **Host-level breaker** (`HostBreaker`): 4 consecutive 403/429 ->
  20-min rest for every camera on that host, then a single probe
  re-opens. A blocking CDN gets ~3 requests/hour instead of ~120.
- **Pre-report recovery**: a few minutes before each scheduled report
  (12:00 and 20:00 Israel time) the collector re-probes higher
  priority countries.

Day/night gates in the report use each **camera's** timezone (the US
bench alone spans Eastern / Central / Pacific).

---

## 2. Architecture: where each piece runs

```
GCP e2-micro VM (1 GB RAM, 24/7)
  collector.py: loop { grab_frame -> YOLO -> count -> re-ID -> events }
  4 cameras in parallel, ~40 s per round
        |
        v (Firestore + Firebase Storage)
  Dashboard in browser (onSnapshot)          On-demand report
  (any static host or localhost:8000)        (dashboard button ->
                                              workflow_dispatch -> PDF)
        ^                                          v
  operator labels                              Inbox notification
        v
  training_sync -> GitHub Actions train-head (free) -> promotion gate
        -> promoted head to Storage
        -> Collector hot-loads next 30 rounds (no restart)
```

Terms:

- **VM**: `e2-micro` on GCP (2 shared vCPUs, 1 GB RAM), Always Free.
- **Firestore**: managed NoSQL. Dashboard subscribes with `onSnapshot`.
- **Firebase Storage**: object bucket for JPEG snapshots + JSON heatmap
  exports. 24 h lifecycle rule on `snapshots/`.
- **GitHub Actions**: free CI runner where the head fine-tune runs; the
  promoted Detect head lands in Storage and the collector hot-swaps
  in-place.

The dashboard is a pure consumer; all state lives in Firestore, and
TTL prunes each history collection to 24 h.

---

## 3. The VM: deep dive

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
Cost        : $0/month (Always Free; one e2-micro per account, us-central1 /
              us-east1 / us-west1 only)
```

Notes:

- **0.25 vCPU guaranteed**: base allocation is a quarter of a core; bursts
  to ~1 vCPU when the host has headroom. Rounds occasionally take ~25 %
  longer under shared-tenant load.
- **1 GB RAM**: an 11 M-parameter model + 4 HLS decoders + OSNet requires
  a 2 GB `/swapfile` (added by hand, not by `install.sh`). Observed peak
  RSS approx 273 MB.
- **Region us-east1-c**: Always Free is restricted to us-central1 /
  us-east1 / us-west1. RTT to Istanbul is ~150 ms; sampling is every 40 s,
  latency has no effect.

### 3.2 Folder layout on disk

```
/opt/turkey-footfall/              <- repo clone (git pull refreshes it)
|-- src/
|   |-- app/                       <- Python modules (collector, detect, tracker, reid, heatmap, faces, pose, gestures, behavior, live_analysis, dashboard_server, alerts, adapters, ...)
|   |-- tools/                     <- CLI helpers (analyze_window, calibrate_conf, probe_country, daily_digest, train_head, promote_adapter, ...)
|   |-- data/
|   |   |-- reid.db                <- SQLite of OSNet embeddings
|   |   |-- osnet_x0_25_msmt17.onnx <- re-ID model (~5 MB)
|   |   |-- confidence_boost.json  <- learned per-(cam,cls) gate nudges
|   |   |-- blacklist_auto.json    <- auto-blacklist polygons
|   |   |-- per_camera_conf.json   <- precision-calibrated gates
|   |   `-- adapters/              <- head-only fine-tuned artifacts
|   |       |-- current.json       <- pointer to active head
|   |       |-- history.jsonl      <- promotion / rejection log
|   |       `-- head_run<N>.pt
|   |-- web/
|   |   |-- snapshots/             <- live snapshot cache (review_frames LRU 500, live_samples LRU 1000, entities LRU 400, anomalies 24 h TTL, heatmaps)
|   |   `-- firebase-config.js     <- public web SDK keys (safe in git)
|   |-- .venv/                     <- Python virtualenv (~2 GB, dominated by torch)
|   `-- deploy/                    <- install.sh, systemd unit templates, worker.js
`-- yolov8s.pt                     <- detection weights (ultralytics auto-downloads)

/etc/turkey-footfall/              <- protected config (root:root)
|-- serviceAccount.json            <- Firebase Admin SDK key (0400)
|-- proxy.env                      <- IBB_PROXY_URL, IBB_PROXY_SECRET (0600)
`-- digest.env                     <- GMAIL_USER, GMAIL_APP_PASSWORD (0600)

/etc/systemd/system/
|-- collector.service
|-- digest.service
`-- digest.timer

/swapfile                          <- 2 GB swap (manual step; see 3.7)
```

The collector runs as root because it must read `serviceAccount.json`
(mode 0400 root:root). Everything under `web/snapshots/` is regenerated
within a round; safe to delete for a clean-slate restart.

### 3.3 Installation

**Prerequisites (once, in the GCP Console):**

1. Enable billing on the project (needed even for free-tier VMs).
2. Enable APIs: Compute Engine, Secret Manager, Cloud Storage.
3. Secret Manager -> create secret `firebase-sa`, paste the Firebase Admin
   SDK JSON as the value.
4. Grant `<PROJECT_NUMBER>-compute@developer.gserviceaccount.com` the role
   **Secret Manager Secret Accessor** on `firebase-sa`.
5. Firestore console -> Time-to-live -> add TTL on `footfall.expire_at`
   and `events.expire_at`.
6. Firebase Console -> Storage -> Get started; then GCP -> Cloud Storage
   -> the bucket -> Lifecycle -> delete files under `snapshots/` after
   1 day.

**Create the VM (Console -> Compute Engine -> Create instance):**
name `turkey-collector`; region `us-east1` (or `us-central1`,
`us-west1`); type `e2-micro`; boot disk Debian 12, Standard PD, 30 GB;
HTTP/HTTPS firewall unchecked; default service account.

**Install the collector**: SSH in, then:

```bash
curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh \
  | sudo bash
```

Idempotent; re-running refreshes code. Steps: `apt-get` packages
(`git`, `python3-venv`, `ffmpeg`, OpenCV libs, `fonts-dejavu-core`);
clone/reset `/opt/turkey-footfall`; create `.venv` and
`pip install -r requirements.txt` with `TMPDIR=/var/tmp`; pull
`serviceAccount.json` from Secret Manager (mode 0400 root:root);
detect the Firebase Storage bucket
(`<project>.firebasestorage.app` first, then legacy
`<project>.appspot.com`); render systemd unit templates with `sed`
and install to `/etc/systemd/system/`; enable and start
`collector.service`; install `digest.service` and `digest.timer`
(timer enables only if `/etc/turkey-footfall/digest.env` exists).

**Post-install one-time steps `install.sh` does NOT cover:**

```bash
# 2 GB swap (the 1 GB VM needs it; observed peak RSS approx 273 MB)
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

### 3.4 `collector.service`: the systemd unit

The template lives in `src/deploy/gcp-vm/collector.service`; `install.sh`
renders it with `sed -e 's|__INSTALL_DIR__|/opt/turkey-footfall|g' ...`
before writing to `/etc/systemd/system/`. Key lines:

```ini
Environment=OMP_NUM_THREADS=2
Environment=MALLOC_ARENA_MAX=2
Environment=FIREBASE_CREDENTIALS=/etc/turkey-footfall/serviceAccount.json
Environment=FIREBASE_STORAGE_BUCKET=<detected at install time>
Environment=REID_MODEL=/opt/turkey-footfall/src/data/osnet_x0_25_msmt17.onnx
EnvironmentFile=-/etc/turkey-footfall/proxy.env
Environment=EXTRA_CLASSES=bird:0.35,dog:0.35,cat:0.40,backpack:0.35,handbag:0.40,suitcase:0.35,umbrella:0.35
Environment=FALL_CHECK=1

ExecStart=/opt/turkey-footfall/src/.venv/bin/python \
  -m app.collector --weights yolov8s.pt --interval 40 --imgsz 640 \
  --burst 2 --burst-stride 13

MemoryHigh=760M
MemoryMax=900M
```

Env notes:

- `OMP_NUM_THREADS=2` matches the shared vCPU count; torch default
  oversubscribes.
- `MALLOC_ARENA_MAX=2`: glibc per-thread arenas cost 50-150 MB of RSS
  on a threaded Python process.
- `EnvironmentFile=-...proxy.env`: optional IBB relay + secret; when
  absent the collector runs unchanged (IBB stays 403 from GCP).
- `EXTRA_CLASSES`: bags feed the unattended-object watch; animals +
  parasols get "other objects" report line. Detection-only.
- `FALL_CHECK=1`: person-loiter triggers ONE pose pass on that person's
  crop; horizontal torso upgrades the alert to "Possible FALL".
- `MemoryHigh/MemoryMax` cgroup guardrails. If the journal shows
  reclaim throttling or oom-kills, fall back to
  `--weights yolov8n.pt --imgsz 768` in ExecStart.

**Code updates**: `git pull` + `systemctl restart` suffices. The installed
unit carries **machine-local `Environment=` lines** (`FIREBASE_CREDENTIALS`,
`FIREBASE_STORAGE_BUCKET`, `REID_MODEL`) that are deliberately NOT in the
repo template. Overwriting the installed unit from the template DROPS those
lines and the collector crash-loops with
`FileNotFoundError: Firebase service-account JSON not found`. To change a
flag, edit the installed unit in place:

```bash
sudo sed -i 's#--weights yolov8s.pt --interval 40 --imgsz 640#--weights yolov8s.pt --interval 40 --imgsz 512#' \
  /etc/systemd/system/collector.service
sudo systemctl daemon-reload && sudo systemctl restart collector
```

### 3.5 `digest.service` + `digest.timer`

Daily digest is on-demand only (via the dashboard button); the scheduled
timer writes to the project archive mailbox only. Enabling it needs
`/etc/turkey-footfall/digest.env` with `GMAIL_USER` and
`GMAIL_APP_PASSWORD`.

### 3.6 The main loop

`app/collector.py`: for each slot in `current_grid()`, `sample_slot`
grabs a `burst` (default 2 frames at `--burst-stride 13` = ~0.5 s at
25 fps), runs `detect_and_count`, applies ROI + gate filters +
auto-blacklist polygons, aggregates by median; writes Firestore;
accumulates heatmap; runs anomaly gates; captures a review frame if
due; reloads overrides + adapter if due; sleeps to fill `INTERVAL`.

### 3.7 The memory model

Observed steady-state (4 cams, `burst 2`, `yolov8s @ 640`): ~410 MB
RSS, 96-100 % CPU idle, swap 0-30 MB. Failure modes that require
falling back to `yolov8n @ 768`:

- Reclaim throttling (`memory pressure` in `journalctl -u collector`)
- Kernel oom-kill loops (`Killed process ... (python)` in
  `journalctl -k`)
- Rounds stretching past interval (`! round took Ns > interval` in
  the app log; dashboard's "counts from Ns ago" label turns red)

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
~17 k (4 slots x 2 writes/round x 2160 samples/day). Raising `--interval`
to 60 s brings it strictly under the quota.

### 3.9 Firebase Storage layout

```
snapshots/
|-- review_frames/<cam>/<ts>_uNN.jpg   <- review queue (LRU on VM disk, batched to Storage)
|-- live_samples/<cam>/<ts>_<cls>.jpg  <- visual-search crops
|-- entities/<eid>/<ts>.jpg            <- per-entity crops (<= 6 per entity)
|-- anomalies/<cam>/<ts>.jpg           <- anomaly evidence (24 h lifecycle)
|-- heatmaps/<cam>.jpg                 <- per-cam overlay
`-- heatmaps/<cam>.json                <- full per-daypart x per-layer grid
training/
|-- labels/                             <- operator verdicts (training_sync)
|-- frames/                             <- frames referenced by the labels
`-- adapters/<run>/head.pt              <- promoted head + metadata
```

### 3.10 Hot-swap of the Detect head

Every 30 rounds the collector polls `data/adapters/current.json` and,
if the pointer changed, calls
`adapters.overlay_head(model, head_state_dict)`; copies head tensors
in-place. No restart, no memory spike. Fallback: missing/unreadable
`current.json` -> base model runs untouched (byte-identical).

---

## 4. VM commands cheatsheet

### 4.1 SSH in

```bash
# Once: gcloud auth login && gcloud config set project turkey-footfall
gcloud compute ssh turkey-collector --zone=us-east1-c

# Or: Console -> Compute Engine -> SSH button; mobile Google Cloud app
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
# Standard path; safe with force-pushed history rewrites
sudo git -C /opt/turkey-footfall fetch origin main && \
  sudo git -C /opt/turkey-footfall reset --hard origin/main && \
  sudo systemctl restart collector

# Alternative: re-run install.sh (idempotent; also refreshes venv deps)
sudo /opt/turkey-footfall/src/deploy/gcp-vm/install.sh
```

Never `sed ... | tee /etc/systemd/system/collector.service` from the repo
template; the installed unit carries machine-local `Environment=` lines
(see 3.4).

### 4.5 Health-check battery

Run before trusting any report:

```bash
# 1. service alive
sudo systemctl status collector --no-pager | head -12

# 2. live sampling; want slot_1..4 with real counts scrolling every ~40 s
sudo journalctl -u collector -f --no-hostname | grep --line-buffered -E "slot_|MISS|country"

# 3. success/miss ratio, last 15 min
sudo journalctl -u collector --since "15 min ago" \
  | grep -oE "slot_[0-9] \([a-z0-9_]+\): (person|MISS)" | sort | uniq -c | sort -rn

# 4. memory headroom
sudo systemctl show collector -p MemoryCurrent -p MemoryMax && free -h

# 5. genuine oom kills only
sudo journalctl -u collector --since "6h" | grep -iE "oom-kill|Killed process|out of memory"

# 6. IBB proxy env is wired (want 2)
sudo grep -c -E "IBB_PROXY_URL|IBB_PROXY_SECRET" /etc/turkey-footfall/proxy.env

# 7. end-to-end: grab a real Turkey frame right now
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

The VM is disposable. Three secrets survive: the Firebase Admin key
(re-mint from Firebase Console -> Service accounts -> Generate new private
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

Requirements: Debian 12/13 or Ubuntu, x86_64, 1 GB+ RAM (with swap),
~20 GB disk, outbound internet. `install.sh` assumes a GCP image; on
another provider, run its steps manually and drop the Firebase key
in by hand:

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

Then apply Path A's swap + env files + timer steps unchanged. Verify
Istanbul cameras with
`python -m tools.probe_country --country turkey`.

### 4.9 Uninstall

```bash
sudo systemctl disable --now collector digest.timer
sudo rm /etc/systemd/system/collector.service /etc/systemd/system/digest.{service,timer}
sudo rm -rf /opt/turkey-footfall /etc/turkey-footfall
sudo systemctl daemon-reload
```

Then delete the VM from the Console.

---

## 5. The 10 live analysis layers

Any dashboard tile can run a live analysis on the exact camera it is
playing. Click the analysis button, pick a layer, the server spins up
a `LiveSession`. The tile keeps playing its full-rate video; the
analysis is drawn on a transparent canvas layered above it, fed by
`GET /api/analysis/data` (overlay JSON, polled every 500 ms). The
server-annotated JPEG (`/api/analysis/frame`) is the fallback view
when the video element is not provably advancing. Up to 4 sessions in
the grid (one per tile). Switching a layer on a running tile mutates
the session; the stream, tracker, and every accumulator (heat,
counters, gesture tallies) survive the switch. Runs in
`app/live_analysis.py` on the operator machine on the shared `yolov8s`
engine.

**Shared pipeline (`LiveSession.run`, tick floor TICK_TARGET_S = 0.8 s):**
grab -> infer (YOLO + gates/ROI, batched) -> tracker update
(BurstTracker BYTE-style) -> conditional pose/plates/faces pass ->
accumulate (heat + line + zone clocks) -> render JPEG fallback ->
publish overlay JSON + events.

`INFER_LOCK` serializes every model call in this process (Ultralytics
`predict` is not thread-safe on a shared model); an inference batcher
coalesces concurrent sessions' frames (collect window 0.10 s). Cadence
with four concurrent sessions and four playing videos: a new tick per
camera every ~12-15 s; a single session alone: ~1-2 s per tick. The
video never drops; only the overlay cadence stretches.

**Interpolation of boxes between ticks:** the 15 fps canvas loop picks
the two ticks that bracket the video's `playingDate` and does a per-track
linear interpolation between their box positions (keypoints lerp
joint-by-joint). Past the newest tick, the loop falls back to
extrapolation along the tracker velocity (vx, vy px/s, EMA-smoothed);
extrapolation window scales adaptively to ~1.3x the session's measured
tick interval; past that window, boxes fade (alpha 1 -> 0.35 over
~0.7x the window). Display gates: a track appears after 2 consecutive
hits at conf >= 0.40, disappears after 1 missed tick; train/boat/airplane
are suppressed.

**Camera resolution:** `resolve_cam` looks up `cam_id` in
`app/cameras.py`; if that misses, it reads `web/local_grid.json` and
maps the slot to a `kind in {youtube, hls, webcamera24, skyline}` dict.

### 5.1 Paths & speeds: `draw_paths_layer`

Trails + per-track id boxes + km/h chips. The only layer that draws
detection boxes for every class. Per-track history is capped at
`TRAIL_MAX_PTS = 40` centroid points; a colored line is drawn through
the centroids in the track's colour (stable per id).

km/h estimate for vehicles: `track_stats` computes the pixel-scale ruler
from a real-world length lookup (`VEHICLE_LENGTH_M`: 4.5 for car, 12
for bus, etc.) divided by the mean bbox extent, then multiplies mean
speed_px_s by that scale and by 3.6. Error band +/- 30-50 %. The report
shows this only when the camera has >= 5 samples AND >= 10 % of rounds
carrying one. On the live JPEG the km/h chip renders only at >= 8 km/h
with >= 5 sightings.

On the live canvas the same layer labels every track with a speed TIER
in body-lengths per second: `blps = speed_px_s / bbox_diagonal_px`.
Tiers: `static` < 0.05 (only after >= 3 hits and >= 4 s of track age),
`slow` < 0.25, `moving` < 0.8, `fast` >= 0.8 (fires a hot-trail event).

### 5.2 Pose & skeleton: `draw_pose_layer`

Skeletons only, no detection boxes, no vehicles. Runs **top-down pose**:
for each detector `person` box (height >= 40 px), crop with 25 %
padding and run YOLOv8n-pose on the crop alone
(`attach_keypoints_crops`, imgsz 256, conf 0.25). Output: 17 COCO
keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
ankles) drawn on each close-enough person. Far persons are reported
as "skeletons on N of M people, rest too far".

### 5.3 Hand gestures: `draw_gestures_layer` + `app/gestures.py`

Three arm-level gestures: `hand_raised` (one wrist above its shoulder
for >= 3 pose frames), `both_hands_up` (both wrists above both
shoulders), `wave` (raised wrist crosses the elbow >= 2 times). The
session keeps a running total (`self.gesture_counts`) so the caption
reads "session: hand_raised x3, wave x1". An empty scene reads
"no gestures detected right now".

### 5.4 Body anomalies: `draw_body_layer`

`label_track` (`app/behavior_labels.py`) runs three tiers per track and
returns exactly one label:

1. **Pose flags** (`pose_flags_of`): shoulder-mid to hip-mid line angle
   from vertical > `FALL_TORSO_DEG = 60` for >= `POSE_FLAG_MIN_FRAMES = 2`
   frames -> `fall_suspect`.
2. **Course reversals** (`heading_turns`): >= 3 reversals > 100 deg
   across the trajectory -> `erratic`. Jitter guards: significant step
   must cover >= 35 % of the object's own diagonal
   (`TURN_MIN_BODY_FRAC`), and `erratic` requires `moving_frac` >= 0.30.
   Mounted riders (person box >= 45 % inside a vehicle box) excluded.
3. **Pure kinematics**: mean speed, moving fraction, net displacement;
   yields `running` / `walking` / `standing` / `dwelling` / `driving` /
   `parked` / `normal`.

Draws ONLY `BODY_ANOMALY_LABELS = {"fall_suspect", "erratic", "running"}`:
red/orange box + skeleton overlay + verdict chip on flagged people,
HUD tally in the top-left, red ALERT banner while a `fall`/`erratic`
flag is live.

### 5.5 Face detection: `draw_faces_layer_img` + `app/faces.py`

Face rectangles only (no embeddings, no database). The detector is
**YuNet** (OpenCV Zoo, ~230 KB ONNX), CPU-only, ~15 ms on a 960 x 540
frame. At street-cam distance faces are often below the detector's
resolution; the caption "no faces at this distance/resolution" is
displayed then.

### 5.6 Heat vision: `draw_heat_layer`

Stylized INFERNO colormap: baseline signal is brightness (`gray * 0.72`)
plus session dwell accumulation (`sqrt(grid/peak) * 0.55`), resized to
frame size and Gaussian-blurred (sigmaX = max(2, W/96)). `grid` is a
`GRID_H x GRID_W` (27 x 48) matrix of dwelt seconds per cell; `bump_heat`
banks each detected foot point weighted by the interval since the
previous tick. Switching layers away and back keeps the accumulation.

### 5.7 Line crossing: `draw_line_layer` + `update_crossings`

A crossing line defined per-camera in `app/cameras.py`, default
horizontal `DEFAULT_LINE = [[0.10, 0.62], [0.90, 0.62]]` (sidewalk band).
Every strict sign flip of a track's foot point across the line is a
crossing event; direction (in/out) follows the A->B point order
(negative -> positive side = "in"). Landing exactly on the line
(`side == 0`) is skipped. A per-track 2 s cooldown
(`CROSSING_COOLDOWN_S`) swallows a foot point trembling across the
boundary. Every crossing appends a JSONL row
(`data/crossings/<cam>.jsonl`, last 50 kept) plus a padded crop
snapshot. The line JSON is re-read every 5 s.

### 5.8 Zone & loitering: `loiter`

User-drawn polygons on the tile (kind `loiter`, dwell threshold
5-3600 s; defaults 300 s person / 900 s vehicle, per-camera overridable).
A confirmed track whose FOOT POINT sits inside a polygon starts that
zone's clock; the presence streak tolerates a single missed tick;
crossing the threshold flips the zone into alert and fires a hot-trail
event on the edge (not continuously). Zones hot-reload from
`data/zones/<cam>.json` every 5 s, same contract as the line.

### 5.9 Parking occupancy: `parking`

Polygons of kind `parking`: a spot is `occupied` when a stationary
vehicle-class track covers >= 30 % of it (asymmetric hysteresis: 2
positive ticks to flip occupied, 4 to flip free). The event fires only
on the occupied/free FLIP. Occupancy + loiter dwell are computed once
per tick and shared between the JPEG render and the JSON publish
(cached on the frame's capture stamp).

**Trackerless probe**: every 12 s the parking layer re-detects each spot
on a 2x-upscaled crop of the spot itself (`_parking_probe`, imgsz 320,
vehicle classes only); a fresh hit feeds the same per-spot hysteresis
as a track candidate. This covers parked two-wheelers at night that
never clear the tracker's confirmation gates.

### 5.10 License plates (LPR): `plates` + `app/plates.py`

Two stages plus a per-track cache:

1. **Plate detection**: `yolov8n-plate` (OpenVINO preferred). Runs only
   inside vehicle boxes wide enough to carry a readable plate: vehicle
   >= 96 px (motorcycles >= 72), plate box >= 32 px, conf 0.30, at
   most 3 vehicles per tick (widest first), at most 6 OCR attempts per
   track.
2. **OCR**: plate crop (2x cubic-upscaled when < 128 px wide) runs
   through `plate_ocr_global.onnx`, a CTC recognizer with a 9-slot,
   country-agnostic alphabet `0-9 A-Z _`. Out-of-alphabet script
   (Thai/Japanese glyphs) is reported as unreadable.

Read accepted at OCR conf >= 0.45 with >= 4 characters; each track
keeps its best read and re-tries until conf >= 0.70 or the try budget
runs out. Sharpness gate: Laplacian variance >= 45 on the plate crop,
else the OCR is skipped and the try refunded. Closest-approach
preference: no budget spent once an unread vehicle shrinks below 85 %
of its own peak width. Static retry: an exhausted-but-unread track
gets a fresh try budget every 120 s. Layer envelope reports the funnel
"N vehicles, M in plate range (>= 96 px), K read". A first successful
read fires a hot-trail event with the text.

### 5.11 The hot trail and the Investigation tab

Every layer feeds a per-camera event ring (50 events): plate read, line
crossing, loiter alert, parking flip, new gesture, body flag, skeleton
acquired, fast mover, face-count rise (>= 30 s apart), heat hotspot
(>= 120 s apart). The strip under the video polls
`GET /api/analysis/events` every 2.5 s. The save button on a chip
POSTs `/api/analysis/event/save`: the server writes the full annotated
frame to `web/snapshots/detections/<cam>_<id>.jpg` and appends a
manifest row (`saved.json`, capped 500). The **Investigation tab**
renders that manifest permanently (20 s refresh).

---

## 6. The deep-window analysis (`behavior.analyze_window`)

Separate engine, on-demand: grabs a window (default 12 frames at
stride 12, ~ 0.5 s apart) from one camera, runs the same gated
detection per frame, threads them into per-individual tracks with
`BurstTracker`, and returns a per-individual profile:

- `path`: foot-point trajectory (normalized, JSON-safe)
- `distance / speed`: path length, net displacement, mean/max px/s,
  km/h for vehicles (+/- 30-50 %)
- `moving_frac`: fraction of steps that moved
- `direction`: dominant screen direction of the net displacement
- `zones`: heatmap cells visited
- `nn_min / mean_px`: closest same-class neighbour over the window
- `label`: behavior verdict per individual (`label_track`); evidence
  in `label_reasons`
- `gestures`: arm-level gestures (pose mode only)

Optional per request: `pose=1` (top-down pose, enriches `label` and
populates `gestures`); `want_faces=1` (face detection on last frame);
`lock=auto` or `lock=<track_id>` (crosshair target lock, returns
normalized offset from frame center `dx, dy` in `[-0.5, 0.5]`).

CLI:

```bash
cd /path/to/repo/src
python -m tools.analyze_window --cam taksim_yeni --pose --faces --lock auto
```

Output: annotated JPEG + JSON profile under `web/snapshots/behavior/`,
LRU 40 files. Dashboard's "Analyze window" button calls
`POST /api/deep-analyze?cam=<id>`.

---

## 7. The notebook: offline analytics

The notebook is `turkey_business_activity_yolov8s.ipynb` (at the repo
root; the imports find `src/app/` automatically). It uses the same
`detect_core` and `reid` modules as the collector so the numbers
reconcile.

| # | Cell topic | What it does |
|---|---|---|
| 0 | Setup | Dependency check + `MODEL_WEIGHTS = 'yolov8s.pt'` + one-time `load_model` |
| 1 | Camera picker | Numbered catalog across all countries; operator picks 4 by number (all must share a country); live probe rejects dead picks |
| 2 | Single-frame check | Grabs one frame from the first pick and annotates it |
| 3 | Footfall time series | Sparse sample every `interval_s`; DataFrame + peak-hour chart |
| 4 | Anomalies + peak-hour profile | Robust rolling z (median + MAD x 1.4826) marks anomalies |
| 5 | Dwell / prolonged stops | Dense burst + ByteTrack for a short window; per-track dwell + movement |
| 5b | Re-identification | ReidStore over N frames; per-class unique / seen-again / regulars (>= 3) |
| 6 | Business score | Composite `volume_median * w0 + linger_rate * w1 + consistency * w2` (empty data yields `None` + note) |
| 7 | Live cloud dashboard | Writes `web/local_grid.json` = the picked cams, spins up `http.server` on `localhost:8000`, opens the browser |
| 8 | Compare multiple sites | Ranks the picked cameras by activity |
| 9 | Live summary | Rollup of the session + always-on visual footfall/anomaly chart |
| 10 | Accuracy calibration | 10a captures frames + predictions; 10b interactive labelling; 10c MAE + bias per cam per size |
| 11 | Forecasting | 11a Firestore delta fetch -> CSV cache; 11b 15-min grid + eligibility; 11c persistence / seasonal-naive / hour-of-week profile / closed-form ridge; 11d small GRU |

### 7.1 Section 11 forecasting: how it decides

Every model must beat "the same time yesterday" (seasonal-naive) on MAE
over the last 25 % of the cache (never touched during fitting). The
ladder:

- **persistence**: `y_{t+h} = y_t`
- **seasnaive24**: `y_{t+h} = y_{t+h - 24h}` (the reference baseline)
- **profile**: median count per local (hour-of-day) slot, or
  (hour-of-week) once the cache carries >= 7 days
- **ridge**: closed-form numpy ridge on lags (1, 2, 3, 4, 96), rolling
  means (4, 12), sin/cos of the target hour-of-day, per-camera one-hots
- **gru**: small GRU (hidden 32, ~15 k params) reading the last 24 h to
  emit the next 12 h; trains on CPU in well under a minute; joins the
  ladder once the cache holds enough windows

`skill = 1 - mae / mae['seasnaive24']` (positive = better). A perfectly
stable stream (`seasnaive24 MAE = 0`) yields `n/a` rather than
misleading infinities.

---

## 8. Model choice + parameters

### 8.1 Detector

`yolov8s.pt` on the VM at `imgsz 640` (memory-fallback `yolov8n.pt @ 768`).
The notebook loads `yolov8s.pt` at higher `imgsz` for offline analytics.

Model size ladder (COCO):

| Model | Params | mAP50 | CPU 1080p | Verdict |
|---|---|---|---|---|
| `yolov8n` | 3.2 M | 37.3 | ~120 ms | Memory-fallback only |
| **`yolov8s`** | **11.2 M** | **44.9** | **~280 ms** | **Current VM (@ 640)** |
| `yolov8m` | 25.9 M | 50.2 | ~700 ms | RSS > 900 MB -> oom on e2-micro |
| `yolov8l` | 43.7 M | 52.9 | ~1400 ms | Not realistic on e2-micro |

### 8.2 The key knobs

| Env / flag | Value | Why |
|---|---|---|
| `--imgsz 640` | | Recovers small objects the 512 pass lost; ~0.39 s / pass on the VM (x 2 frames x 4 cams = ~3 s / 40 s round) |
| `--burst 2 --burst-stride 13` | Two frames ~0.5 s apart | Median kills single-frame flicker; two points feed the speed estimator |
| `--interval 40` | s | Firestore-quota bound (~17 k writes/day of 20 k) |
| `MemoryHigh=760M / MemoryMax=900M` | cgroup guardrails | Fits the 1 GB e2-micro with margin |
| `OMP_NUM_THREADS=2` | | Matches shared vCPU count; torch default oversubscribes |
| `MALLOC_ARENA_MAX=2` | | glibc per-thread arenas cost 50-150 MB of RSS on a threaded python |
| `DEFAULT_PER_CLASS_CONF` (in `detect_core.py`) | Per-class gate map | Nightly `night_adjusted_conf(+0.08)` + per-camera boosts learned by review |
| `EXTRA_CLASSES` env | `bird, dog, cat, backpack, handbag, suitcase, umbrella` | Feeds unattended-object watch + "other objects" report line |
| `FALL_CHECK=1` env | | Person-loiter -> one pose pass on that crop; horizontal torso -> "Possible FALL" |

### 8.3 Per-camera confidence calibration

`tools/calibrate_conf.py` reads the operator's verdict history, computes a
per-`(camera, class)` confusion matrix, and picks the lowest threshold
that achieves **precision >= 0.90** with **>= 30 verdicts**; writes it to
`data/per_camera_conf.json`. `cameras._merge_per_camera_conf()` runs
AFTER `_merge_confidence_boost` and overrides it per pair. A calibrated
pair is tagged `source=calibration` in the Learning-proof panel.

---

## 9. Anomalies + reporting

The collector runs a set of deterministic anomaly gates per round and
per camera. Each gate has an explicit trigger + a debounce window.

| Gate | Trigger | Debounce |
|---|---|---|
| Extreme load | Person / vehicle count above a rolling robust-z threshold | 3 rounds |
| Camera obscured | Mean brightness drops below the camera's night floor while the clock says day | 5 rounds |
| Camera dark | Sample MISSes exceed the rest-and-probe schedule | 3 rounds |
| Loiter | Same person track stays inside a box for >= camera's `loiter_s` | Cap 10/day/cam |
| Returning visitor | Same OSNet identity seen at >= 1.2 x box-scale distance from previous sighting | Person only, >= 64 px floor |
| Unattended object | Bag / suitcase without an owner-nearby person for >= 90 s | Owner-nearby gate |
| Fall suspect | Person loiter + horizontal torso from one pose pass | Under `FALL_CHECK=1` |
| Crowd rush | Sudden speed x density spike | 2 rounds |

Anomaly evidence is captured as an annotated JPEG under
`snapshots/anomalies/<cam>/<ts>.jpg` (24 h lifecycle in Storage). Each
event also lands in `events/` in Firestore (also 24 h TTL) so the
dashboard's "Events" strip can show it live.

**Reporting**:

- Twice-daily archive digest (12:00 + 20:00 Israel) -> project archive
  mailbox only, via `digest.timer`. Uses `tools/daily_digest.py`.
- On-demand PDF from the dashboard header (private tile: "Send Report
  From VM" -> `POST /api/send-report`; public tile: GitHub Actions
  workflow dispatch). Same PDF composer (`tools/report_pdf.py`),
  different sender.

The report publishes km/h fields only when there are >= 5 speed samples
AND >= 10 % of rounds carrying one; otherwise `-`.

---

## 10. Active-learning loop

Every reviewed frame becomes training data; nightly (or on-demand) a
head-only fine-tune runs on GitHub Actions; the promoted head lands in
Storage; the collector hot-swaps it with no restart. The loop uses zero
paid resources.

### 10.1 Uncertainty-first frame queue

Every stored box carries `uncertainty in [0,1]` from
`app/uncertainty.py`: `uncertainty = 0.6 * margin + 0.4 * flip_delta`.

- `margin(conf, gate, span=0.25)`: 1.0 at the class gate, falling
  linearly to 0 at `gate +/- span`.
- `flip_delta` (optional, sampled bursts, `UNCERTAINTY_FLIP=1`): one
  extra pass on the horizontally-flipped frame; per-box IoU-matched
  conf delta. ~1-in-5 bursts on ONE camera.

Persisted: frames -> sidecar JSON `metadata.boxes[i].uncertainty`;
crops -> filename suffix `_uNN` (`..._u87.jpg` = 0.87). The review UI's
`labels.frame_uncertainty` prefers the persisted value; missing falls
back to the margin.

### 10.2 BADGE crop sampler

`app/badge.py`: hand-rolled k-means++ init picks a diverse batch
weighted by uncertainty; OSNet embeddings as direction, uncertainty as
magnitude. Env switch `REVIEW_SAMPLER=badge|naive` (default `naive`);
per-request override `?strategy=` on `/api/review-sample`. Review rows
record `sampler` + `uncertainty_at_selection` so the naive-vs-BADGE
efficiency replay can run offline.

### 10.3 Head-only fine-tune + promotion gate

`tools/train_head.py` wraps `yolo detect train` with the backbone
frozen (`freeze=<all-but-head>`), mosaic/mixup off, HSV + flip on,
<= 10 epochs early-stop. Emits `data/adapters/<cam>/head_<ts>.pt`;
Detect-head tensors only (~4-6 MB).

`tools/promote_adapter.py` runs `val` on the exporter's chronological
90/10 split for both the baseline and the candidate; gate:

- `delta mAP50 >= +0.5` percentage points, AND
- No class drops > 2 pp (person / car: 0 pp; those counts drive every
  report).

Pass -> atomic `current` pointer update + `history.jsonl` append.
Fail -> `gate.log` line. `--rollback` restores the previous pointer.

Transport: the operator's local labels + frames flow to Storage via
`app/training_sync.py` (batched, ledger-diffed); GitHub Actions
(`.github/workflows/train.yml`) trains on free public-repo runners;
the promoted head lands in Storage; the collector polls `current.json`
every 30 rounds and hot-swaps in place.

**Byte-identical fallback**: missing / unreadable `current.json` ->
base model runs untouched. No adapter is present at rest; the head is
loaded into memory only if a promoted one exists and validates.

### 10.4 "Labels vs quality" curve

`GET /api/al-curve` reads `history.jsonl` (+ Firestore mirror
`training_events`, TTL 30 d, one write per promotion) and the dashboard
draws a Chart.js line: labels_total on X, mAP50 on Y, rejected
candidates greyed, baseline dashed.

---

## 11. Firebase project setup

The dashboard is `onSnapshot`-live: every collector write reaches the
browser instantly, no polling. Setup, once per project:

**1. Create the project.** `console.firebase.google.com` -> Add project
-> enable Firestore in test mode (locked before the public deploy; see
step 5).

**2. Backend credentials.** Project settings -> Service accounts ->
Generate new private key; save the JSON outside git. Set
`FIREBASE_CREDENTIALS=/path/to/serviceAccount.json` and
`pip install firebase-admin`.

**3. Run the collector against Firebase.**

```bash
python -m app.collector --backend firebase --interval 20 \
  --only konya_hukumet,kapali_carsi,misir_carsisi,eminonu,istiklal_1
```

Each round writes one history doc per camera to `footfall` and
overwrites `latest/{cam_id}`. Keep it alive with systemd / Docker /
`nohup`.

**4. Web frontend.** Firebase Console -> Project settings -> Web app
-> copy the SDK config. Create `web/firebase-config.js` with
`export const firebaseConfig = {...}`. Then:

```bash
cd src/web && python -m http.server 8000     # http://localhost:8000
```

**5. Security rules.** Test mode lets anyone on the internet read AND
WRITE. The public web-SDK config (`apiKey`, `projectId`) ships in every
visitor's browser and is not a secret; the security rules are.

Locked-down rules live in `src/firestore.rules`: public read on the
dashboard collections (`footfall`, `latest`, `reid_stats`, `events`,
`config`), all client writes denied (the Admin SDK bypasses rules, so
the collector is unaffected). Deploy them:

```bash
npm install -g firebase-tools    # once
firebase login
# .firebaserc: {"projects":{"default":"<your-project-id>"}}
firebase deploy --only firestore:rules
```

Then in Firebase Console -> Firestore -> Rules, verify writes show
`if false`.

**6. TTL policies.** Firebase Console -> Firestore -> Time-to-live ->
add TTL on `footfall.expire_at` AND `events.expire_at` (both self-prune
after 24 h).

**7. App Check.** Rules make the data read-only but a scraper can burn
the read quota. App Check requires every request to carry a
reCAPTCHA-v3 attestation. Firebase Console -> App Check -> Apps ->
register the web app with the reCAPTCHA v3 provider. Copy the site key
into `web/firebase-config.js` as `recaptchaSiteKey`; `web/app.js`
initialises App Check automatically when it is set. Enable enforcement
(App Check -> Firestore -> Enforce) ONLY after the site key is live
on the page; otherwise enforced reads are rejected and the dashboard
goes blank.

**8. Rate limit + cost cap.** Firestore Spark tier ~ 20 k writes/day.
The collector prints the projected daily write count on startup and
clamps `--interval` to a 5 s floor. Set a budget alert in Google Cloud
-> Billing; on Blaze, also set an App Engine daily spending limit as
the hard cap.

---

## 12. Cloudflare proxy for IBB

`kamerayayin.ibb.istanbul` refuses every Google Cloud IP range (HTTP
403) but answers normally from any other address. A Cloudflare Worker
on the free plan (100 k requests/day, load ~26 k/day) proxies IBB
requests through Cloudflare's edge (a different ASN), restoring
`taksim_yeni`, `sultanahmet_1_yeni`, `eyup_sultan_yeni`,
`beyazit_meydan_yeni`.

Source in `src/deploy/cloudflare-proxy/`: `worker.js` and
`wrangler.toml`.

**One-time setup:**

```bash
# Free Cloudflare account (no card): https://dash.cloudflare.com/sign-up
npm install -g wrangler
wrangler login

cd src/deploy/cloudflare-proxy
wrangler deploy               # prints https://ibb-proxy.<subdomain>.workers.dev
wrangler secret put PROXY_SECRET     # e.g. openssl rand -hex 24

sudo tee /etc/turkey-footfall/proxy.env > /dev/null <<EOF
IBB_PROXY_URL=https://ibb-proxy.<subdomain>.workers.dev
IBB_PROXY_SECRET=<same secret>
EOF
sudo chmod 600 /etc/turkey-footfall/proxy.env
sudo systemctl restart collector
```

**Verify (from the VM):**

```bash
cd /opt/turkey-footfall/src && sudo -E .venv/bin/python -m tools.probe_country --country turkey
# want: four IBB cameras flip to LIVE

curl -s -H "X-Proxy-Secret: <your secret>" \
  "https://ibb-proxy.<you>.workers.dev/https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8" \
  | head -3
# expect: #EXTM3U ...
# 403 from the worker -> secret wrong
```

**Not proxied:**

- Other hosts (only `kamerayayin.ibb.istanbul`; else 403).
- `tvkur.com` (Konya, Otogar and other Turkish webcamera24 streams;
  tvkur restricts even residential ASNs, and a Cloudflare edge faces
  the same 403; those need a Turkish-IP proxy specifically).
- Caching that would break liveness (`cf.cacheTtl: 4` matches the
  ~4 s HLS segment rotation).

---

## 13. GCP billing kill-switch

Auto-disables billing on `turkey-footfall` the moment a Cloud Billing
budget threshold is crossed.

Source stays in `src/deploy/gcp-billing-killswitch/`: `main.py` (the
Cloud Function), `requirements.txt`.

**Prerequisites (once):**

1. **Enable APIs**: Cloud Pub/Sub, Cloud Functions, Cloud Build,
   Cloud Billing.
2. **Create the Pub/Sub topic and runtime SA, grant roles, seed Pub/Sub
   agent**:
   ```bash
   gcloud pubsub topics create budget-alerts --project=turkey-footfall

   gcloud iam service-accounts create billing-killswitch \
     --display-name "Billing kill-switch runtime" \
     --project=turkey-footfall

   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/billing.projectManager
   gcloud projects add-iam-policy-binding turkey-footfall \
     --member=serviceAccount:billing-killswitch@turkey-footfall.iam.gserviceaccount.com \
     --role=roles/browser

   gcloud beta services identity create --service=pubsub.googleapis.com \
     --project=turkey-footfall
   ```
   `billing.projectManager` has `deleteBillingAssignment` (the actual
   unlink). `browser` has `resourcemanager.projects.get` for the
   idempotency check.
3. **Wire the topic into the budget**: GCP Console -> Billing ->
   Budgets & alerts -> open the budget -> Manage notifications ->
   Connect a Pub/Sub topic -> pick
   `projects/turkey-footfall/topics/budget-alerts`.

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

**Grant the trigger SA `run.invoker`** on the Cloud Run service backing
the gen2 function (otherwise every Pub/Sub delivery is rejected):

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

**Re-enable billing after the test**: GCP Console -> Billing -> Link
this project to a billing account.

**What it does NOT do:** does not delete resources (the VM, Firestore
data, Storage bucket, and function itself all remain and stop
generating billable events until the billing account is re-linked);
does not touch free-tier services; does not care which threshold
crossed (unlinks only when `costAmount >= budgetAmount`).

Cost of the kill-switch: zero (Pub/Sub messages, Cloud Function
invocations 2 M/month free, Cloud Storage function source Always Free).

---

## 14. Troubleshooting / FAQ

**"The dashboard's per-tile counts are 'from Ns ago' and the label is
red."** The collector is not keeping up with `--interval`. Run the
health-check battery (section 4.5); if memory is fine but CPU is
saturated, drop to `--weights yolov8n.pt --imgsz 768` in the ExecStart
line (section 3.4).

**"Live analysis on a picked skyline camera 404s."** `_cam_from_slot`
handles `kind="skyline"` slots from `web/local_grid.json`. Old pre-fix
versions failed with `ValueError("no analyzable stream")`.

**"The daily digest e-mail never arrives."** Check
`/etc/turkey-footfall/digest.env` exists with a real Gmail app password
(`sudo cat` as root); check `sudo systemctl status digest.timer` for the
next fire time; run `sudo systemctl start digest.service` for an
immediate manual run.

**"Turkey cameras always MISS from the VM."** IBB geo-blocks Google
Cloud ASNs. Either wire the Cloudflare Worker (chapter 12) or accept
that the grid falls through to Thailand / Japan / USA until IBB
unblocks. `tools/probe_country --country turkey` shows live status of
each Turkey camera.

**"How do I add a new camera?"** Edit `src/app/cameras.py`: pick a
stable `cam_id`, fill in the `kind` (`hls | youtube | webcamera24 |
skyline`), the URL and page, the display name, and any `roi` /
`roi_exclude` / `line` overrides.
`python -m tools.probe_country --country <c>` verifies it. No VM change
needed; a `git pull + systemctl restart` picks it up.

**"How do I take the VM offline for a week?"**
`gcloud compute instances stop turkey-collector --zone=us-east1-c`;
Firestore keeps the last 24 h (TTL), dashboard shows the last known
state. `gcloud compute instances start ...` when back.

**"How much does this really cost per month?"** $0 in normal operation.
The e2-micro is Always Free; Firestore Spark tier writes stay under
20 k/day; Firebase Storage stays under the 5 GB free limit (~50 MB
active with 24 h TTL); egress from GCP to Firebase (same region) is
free. The kill-switch guards against surprise overages (chapter 13).
