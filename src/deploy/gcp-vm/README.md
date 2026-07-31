# Deploy the collector on a small GCP VM

The cloud collector runs the same `app/collector.py` you know locally, but as a
systemd service on a small always-on VM.

> **Machine sizing — measured on this deployment:** the VM is an `e2-micro`
> (1 GB, Always Free — $0/month). Its measured peak is **~635 MB RSS** at the
> default `--imgsz 960` (3-frame burst), inside the shipped
> `MemoryHigh=700M`/`MemoryMax=850M` caps. Memory peaks can vary by torch
> build; if `journalctl -u collector` ever shows reclaim throttling or
> oom-kill restarts (rounds of minutes, dashboard numbers frozen between
> updates), add `--imgsz 640` to `ExecStart` in
> `/etc/systemd/system/collector.service`, then run
> `sudo systemctl daemon-reload && sudo systemctl restart collector` —
> roughly half the inference memory, at the cost of small/distant-object
> recall. Check `systemctl status collector | grep -i memory` after a day.

## Prerequisites (do these once, from the GCP Console at console.cloud.google.com)

1. **Switch to your Firebase project.** Top-of-page project picker → select the
   project that hosts your Firestore (`turkey-footfall`), NOT `My First Project`.
2. **Enable billing.** Billing → Link a billing account (credit card). The
   `e2-micro` we create is Always Free — no charge — but GCP requires billing
   to be enabled on the project even for free-tier VMs.
3. **Enable APIs.** APIs & Services → Enable: `Compute Engine API`,
   `Secret Manager API`, `Cloud Storage API`.
4. **Upload the service-account JSON to Secret Manager.**
   Secret Manager → Create secret → Name `firebase-sa`, secret value = paste
   the JSON contents of your Firebase Admin SDK key.
5. **Grant the VM's default service account read access to the secret.**
   Secret Manager → click `firebase-sa` → Permissions → Add principal →
   `<PROJECT_NUMBER>-compute@developer.gserviceaccount.com` →
   role `Secret Manager Secret Accessor`.
6. **Enable Firestore TTL on `footfall.expire_at`.**
   Firebase Console → Firestore Database → Time-to-live → Add TTL policy →
   Collection ID `footfall`, timestamp field `expire_at`.
7. **Enable Firebase Storage.** Firebase Console → Storage → Get started
   (default region is fine, matching your Firestore region is best).
8. **Add a Storage lifecycle rule to delete snapshots after 24h.**
   GCP Console → Cloud Storage → click the Firebase Storage bucket → Lifecycle
   → Add rule → Action: Delete → Condition: Age = 1 day, Prefix = `snapshots/`.

## Create the VM

Console → Compute Engine → VM instances → CREATE INSTANCE:

- **Name**: `turkey-collector`
- **Region**: `us-central1` (required for Always Free — also `us-east1` or `us-west1`)
- **Zone**: any `-a` zone in that region
- **Machine configuration**: series `E2`, machine type **`e2-micro`**
  (Always Free — $0/month; the service file's memory caps are sized for it,
  see the note above)
- **Boot disk**: Debian 12, **Standard persistent disk**, size **30 GB**
- **Firewall**: leave both HTTP/HTTPS unchecked — the collector doesn't listen
- **Identity and API access**: keep the default service account, "Allow default access"
- Click **Create**

Wait ~30 seconds for the VM to reach "Running".

## Install the collector

Click the **SSH** button next to the VM (works from the mobile app too), then paste:

```bash
curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh \
  | sudo bash
```

The script:

1. Installs Python 3, ffmpeg, and the OpenCV system libraries.
2. Clones this repo into `/opt/turkey-footfall`.
3. Creates a Python venv and pip-installs `requirements.txt`.
4. Fetches your Firebase service-account JSON from Secret Manager into
   `/etc/turkey-footfall/serviceAccount.json` (root:root, 0400).
5. Detects your Firebase Storage bucket from the JSON.
6. Installs `collector.service` under systemd and starts it.

You should see the collector's first output within ~30 seconds:

```bash
sudo journalctl -u collector -f
```

Look for `Firebase backend initialized. Storage: ON` followed by
`[TS] slot_konya_hukumet (konya_hukumet): person=X vehicles=Y ...` every
sampling round (40 s with the shipped service file). If you instead see
`! round took Ns > interval` lines, the machine can't keep up with the
configured `--interval`/`--imgsz` — the dashboard tiles refresh every N
seconds in that state, and the per-tile "counts from Ns ago" label on the
dashboard turns red.

## Connecting to the VM

Three ways in, any time - all reach the same `turkey-collector` instance
(zone `us-east1-c`, project `turkey-footfall`):

1. **Browser SSH (no setup, what the maintainer uses)** - Console →
   Compute Engine → VM instances → the **SSH** button next to
   `turkey-collector`. Opens `ssh.cloud.google.com` in a new tab with a
   full terminal. Also has **UPLOAD FILE / DOWNLOAD FILE** buttons.
2. **`gcloud` CLI (from your own machine)** - once, `gcloud auth login &&
   gcloud config set project turkey-footfall`; then, any time:
   ```bash
   gcloud compute ssh turkey-collector --zone=us-east1-c
   ```
   (First run creates and uploads an SSH key automatically.)
3. **Mobile** - the Google Cloud app (iOS/Android) → Compute Engine →
   `turkey-collector` → **SSH**.

The VM's external IP is **ephemeral** and changes on stop/start. Read the
current one, or rotate it (a fresh IP sometimes clears a CDN rate-block),
from Cloud Shell or `gcloud`:

```bash
# current external IP
gcloud compute instances describe turkey-collector --zone=us-east1-c \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
# rotate it (detach + reattach the access config -> new IP, no reboot)
NAME=$(gcloud compute instances describe turkey-collector --zone=us-east1-c \
  --format="value(networkInterfaces[0].accessConfigs[0].name)")
gcloud compute instances delete-access-config turkey-collector \
  --zone=us-east1-c --access-config-name="$NAME"
gcloud compute instances add-access-config turkey-collector --zone=us-east1-c
```

### Health check - is the VM really feeding the report?

The twice-daily report is only as good as the collector. This battery
proves each link end to end; run it after any change or whenever a report
looks off:

```bash
# 1. service alive, not crash-looping (want: active (running), uptime in min/h)
sudo systemctl status collector --no-pager | head -12

# 2. live sampling - want slot_1..4 on taksim/beyazit/sarachane/sultanahmet
#    with real person/vehicle counts scrolling every ~40s (Ctrl+C to stop)
sudo journalctl -u collector -f --no-hostname | grep --line-buffered -E "slot_|MISS|country"

# 3. success vs miss, last 15 min (the digit in sultanahmet_1_yeni needs 0-9 in the class)
sudo journalctl -u collector --since "15 min ago" \
  | grep -oE "slot_[0-9] \([a-z0-9_]+\): (person|MISS)" | sort | uniq -c | sort -rn

# 4. memory headroom on the 1 GB e2-micro (want current < max, available > 0)
sudo systemctl show collector -p MemoryCurrent -p MemoryMax && free -h

# 5. real out-of-memory kills only (want: NO 'oom-kill'/'Killed process' lines;
#    ignore googlevideo HLS URL noise from any YouTube era)
sudo journalctl -u collector --since "6 hours ago" | grep -iE "oom-kill|Killed process|out of memory"

# 6. the IBB proxy env that keeps Turkey alive is wired (values redacted here)
sudo grep -c -E "IBB_PROXY_URL|IBB_PROXY_SECRET" /etc/turkey-footfall/proxy.env   # want: 2

# 7. DECISIVE end-to-end: the machine grabs a real Turkey frame right now
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

## Managing the collector from your phone

Google Cloud app (iOS/Android) → Compute Engine → `turkey-collector`:

- **Start / Stop / Reset** buttons at the top of the instance detail page.
- **SSH** button opens an in-app terminal for the checks below.
- **Logs** link opens Cloud Logging with the VM pre-selected.

Common commands once you're SSH'd in from the phone:

```bash
sudo systemctl status  collector   # is it running?
sudo systemctl restart collector   # after a code change
sudo journalctl -u     collector -n 100      # last 100 log lines
sudo journalctl -u     collector -f          # tail live
# deploy new code (fetch+reset also survives history rewrites, unlike pull):
sudo git -C /opt/turkey-footfall fetch origin main && \
  sudo git -C /opt/turkey-footfall reset --hard origin/main && \
  sudo systemctl restart collector
```

> **Refreshing the VM — do NOT re-render the systemd unit.** New code is
> picked up by `git pull` + `systemctl restart` ALONE; the unit file at
> `/etc/systemd/system/collector.service` never needs rebuilding for a code
> change. The installed unit carries **machine-local `Environment=` lines**
> (`FIREBASE_CREDENTIALS`, `FIREBASE_STORAGE_BUCKET`, `REID_MODEL`) that are
> deliberately NOT in the repo template `collector.service` (they hold your
> paths, not the project's). Overwriting the installed unit from the template
> (`sed ... | tee /etc/systemd/system/collector.service`) DROPS those lines
> and the collector then crash-loops with `FileNotFoundError: Firebase
> service-account JSON not found`. If you must change a flag (e.g. add
> `--weights yolov8n.pt`), edit the installed unit IN PLACE:
> ```bash
> sudo sed -i 's#-m app.collector#-m app.collector --weights yolov8n.pt#' \
>   /etc/systemd/system/collector.service   # only if --weights not already there
> sudo systemctl daemon-reload && sudo systemctl restart collector
> ```
> — never a wholesale copy from the repo.

## YouTube cameras from the VM: the PO-token provider

YouTube starves Google-datacenter IPs: every YouTube-backed camera (the
Turkey YT tier, all of Thailand/Japan/USA) resolves a stream from the VM
but receives no video data ("empty frame - stream: opened but produced
no frames"), while the same streams play 1080p from any residential IP.
The documented remedy is attaching a PO (proof-of-origin) token, minted
by the bgutil provider - unauthenticated "cold" tokens, no Google
account, no cookies to expire.

One-shot setup (script mode - a short-lived node process per stream
resolution, nothing resident; the 1 GB e2-micro cannot afford a
standing server):

```bash
sudo bash /opt/turkey-footfall/src/deploy/gcp-vm/setup_pot_provider.sh
```

The script installs node, builds the provider under
`/opt/bgutil-ytdlp-pot-provider`, pip-installs the yt-dlp plugin into
the venv, appends `YT_POT_SCRIPT` + `YT_PLAYER_CLIENTS=web,mweb,android,ios`
to `/etc/turkey-footfall/proxy.env` (the EnvironmentFile the service
already loads) and restarts the collector. Remove those two lines from
the env file and restart to disable. Success is not guaranteed - YouTube
moves this fence regularly - which is why the whole path is opt-in env
config on top of an unchanged default.

## Costs to watch

- **VM**: `e2-micro` is **$0** on the Always Free tier (us-central1 /
  us-east1 / us-west1). Set a **budget alert** so you catch anything weird.
- **Firestore writes**: at the shipped `--interval 40`:
  `4 slots × 3 writes/sample × 2160 samples/day ≈ 26k writes/day`. Blaze free
  tier allows 20k/day; the overflow costs pennies/month. If you want it
  strictly free, raise `--interval` to 60s (edit the `ExecStart` in
  `collector.service` and `systemctl daemon-reload`).
- **Storage**: at ~50MB active with 24h TTL — well under the 5GB free tier.
- **Egress from GCP**: the collector only *writes* to Firebase (same Google
  region if you kept the default) — no external egress.

## Uninstall

```bash
sudo systemctl disable --now collector
sudo rm /etc/systemd/system/collector.service
sudo rm -rf /opt/turkey-footfall /etc/turkey-footfall
sudo systemctl daemon-reload
```
Then delete the VM from Cloud Console.
