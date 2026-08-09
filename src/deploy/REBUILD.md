# Rebuild from zero - disaster recovery

The collector VM is disposable. If it is deleted, corrupted, or the
account hosting it is lost, everything below recreates an identical
deployment in under 30 minutes - on GCP or on any other Linux provider.
What must survive is only:

1. **This repository** (code, installer, systemd unit templates, this guide).
2. **Three secrets** - none stored in git, every one re-mintable from its
   console (see [Secrets](#secrets)).

The VM's *disk content* is deliberately not backed up: collector state
regenerates within one sampling round, Firestore holds only short-term
data by design, and the long-term archive is the notebook's local CSV
cache.

## The deployment at a glance

| Item | Value |
|---|---|
| GCP project | `turkey-footfall` |
| Instance / zone | `turkey-collector` / `us-east1-c` |
| Machine | `e2-micro` (Always Free: one per account, in us-east1 / us-west1 / us-central1) |
| Image | family `debian-13`, project `debian-cloud` |
| Boot disk | 30 GB `pd-standard` (Always Free ceiling) |
| Swap | 2 GB `/swapfile` (manual step - not in install.sh) |
| Storage bucket | `turkey-footfall.firebasestorage.app` |
| Firebase key | Secret Manager secret `firebase-sa`; on disk `/etc/turkey-footfall/serviceAccount.json` (0400) |
| Config dir | `/etc/turkey-footfall/` - `serviceAccount.json`, `proxy.env` (0600), `digest.env` (0600) |
| systemd units | `collector.service`, `digest.service` + `digest.timer` (12:00 + 20:00 Asia/Jerusalem -> project archive mailbox) |
| IBB relay | Cloudflare Worker, `https://ibb-proxy.<subdomain>.workers.dev` |

## Secrets

All three are re-mintable in minutes; keep private copies outside git if
you want a rebuild with zero console visits.

1. **Firebase Admin SDK key** (`serviceAccount.json`) - Firebase Console
   -> Project settings -> Service accounts -> Generate new private key.
   GCP path stores it in Secret Manager as `firebase-sa`; any-provider
   path places the file directly.
2. **IBB relay shared secret** - set on the Worker with
   `wrangler secret put PROXY_SECRET`
   (from `src/deploy/cloudflare-proxy/`), and mirrored in `proxy.env`:

   ```
   IBB_PROXY_URL=https://ibb-proxy.<subdomain>.workers.dev
   IBB_PROXY_SECRET=<same value>
   ```

3. **Gmail app password** (digest emails) - create at
   https://myaccount.google.com/apppasswords, then `digest.env`:

   ```
   GMAIL_USER=<gmail address>
   GMAIL_APP_PASSWORD=<16-char app password>
   ```

To snapshot the live values off a running VM into a private kit:

```bash
gcloud compute ssh turkey-collector --zone=us-east1-c --project=turkey-footfall \
  --quiet --command='sudo install -m 600 -o $USER /etc/turkey-footfall/serviceAccount.json /tmp/sa.json;
                     sudo install -m 600 -o $USER /etc/turkey-footfall/proxy.env /tmp/proxy.env'
gcloud compute scp turkey-collector:/tmp/sa.json ./serviceAccount.json \
  turkey-collector:/tmp/proxy.env ./proxy.env --zone=us-east1-c --project=turkey-footfall
gcloud compute ssh turkey-collector --zone=us-east1-c --project=turkey-footfall \
  --quiet --command='shred -u /tmp/sa.json /tmp/proxy.env'
```

## Path A - rebuild on GCP (identical machine)

One-time project prerequisites (billing, APIs, Secret Manager grant,
Firestore TTL, Storage + lifecycle rule) are documented in
[gcp-vm/README.md](gcp-vm/README.md#prerequisites-do-these-once-from-the-gcp-console-at-consolecloudgooglecom);
they survive VM loss and normally need nothing.

```bash
# 1. create the machine (exact spec of the live one)
gcloud compute instances create turkey-collector \
  --project=turkey-footfall --zone=us-east1-c \
  --machine-type=e2-micro \
  --image-family=debian-13 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-standard

# 2. bootstrap (packages, repo, venv, Firebase key from Secret Manager, units)
gcloud compute ssh turkey-collector --zone=us-east1-c --project=turkey-footfall \
  --command='curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh | sudo bash'
```

Then SSH in and finish the three steps install.sh does not cover:

```bash
# 3. swap - the 1 GB box needs it (live machine runs 2 GB and uses it)
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. env files (fill from your private copies / re-minted secrets)
sudo tee /etc/turkey-footfall/proxy.env  > /dev/null <<'EOF'
IBB_PROXY_URL=https://ibb-proxy.<subdomain>.workers.dev
IBB_PROXY_SECRET=<secret>
EOF
sudo tee /etc/turkey-footfall/digest.env > /dev/null <<'EOF'
GMAIL_USER=<gmail address>
GMAIL_APP_PASSWORD=<app password>
EOF
sudo chmod 600 /etc/turkey-footfall/proxy.env /etc/turkey-footfall/digest.env

# 5. enable the archive report timer, reload the collector with the proxy env
sudo systemctl enable --now digest.timer
sudo systemctl restart collector
```

Verify with the full
[health-check battery](gcp-vm/README.md#health-check---is-the-vm-really-feeding-the-report) -
the decisive check is the end-to-end "grab a real Turkey frame now".

## Path B - rebuild on any other provider

Requirements: Debian 12/13 or Ubuntu, 1 GB+ RAM (with the swap step),
x86_64, ~20 GB disk, outbound internet. The collector listens on
nothing - no inbound ports, no firewall work.

`install.sh` assumes a GCP image (gcloud + Secret Manager), so run its
steps manually; only the key delivery differs:

```bash
# packages
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
  git python3 python3-venv python3-pip \
  ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
  ca-certificates curl fonts-dejavu-core

# code + venv (pip tmp on the real disk - /tmp is a tiny tmpfs on 1 GB hosts)
sudo git clone --depth 1 https://github.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection.git /opt/turkey-footfall
cd /opt/turkey-footfall/src
sudo python3 -m venv .venv
sudo TMPDIR=/var/tmp .venv/bin/pip install --no-cache-dir -r requirements.txt

# Firebase key - place the JSON you minted/copied (no Secret Manager here)
sudo mkdir -p /etc/turkey-footfall
sudo install -m 0400 -o root -g root ~/serviceAccount.json /etc/turkey-footfall/serviceAccount.json

# render + install the systemd units from the repo templates
for unit in collector.service digest.service digest.timer; do
  sudo sed -e 's|__STORAGE_BUCKET__|turkey-footfall.firebasestorage.app|g' \
           -e 's|__INSTALL_DIR__|/opt/turkey-footfall|g' \
           -e 's|__SA_PATH__|/etc/turkey-footfall/serviceAccount.json|g' \
    /opt/turkey-footfall/src/deploy/gcp-vm/$unit | sudo tee /etc/systemd/system/$unit > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now collector.service
```

Then apply Path A steps 3-5 unchanged (swap, env files, timer). The
Istanbul cameras work from any ASN Cloudflare can reach - the relay was
built against Google-Cloud blocking, and other datacenter ranges may not
even need it; test first with `tools.probe_country --country turkey`.

## Satellite pieces (rebuilt independently of the VM)

- **IBB Cloudflare Worker** - full 5-minute setup in
  [cloudflare-proxy/README.md](cloudflare-proxy/README.md)
  (`wrangler deploy` + `wrangler secret put PROXY_SECRET`).
- **Billing killswitch** (GCP only, optional guard) -
  [gcp-billing-killswitch/README.md](gcp-billing-killswitch/README.md).
- **Firebase project from scratch** (only if the whole Google account is
  lost): [../docs/firebase_setup.md](../docs/firebase_setup.md), then the
  prerequisites list in [gcp-vm/README.md](gcp-vm/README.md), then update
  `src/web/firebase-config.js` with the new web config and let install.sh
  re-detect the bucket.

## Notes from the live machine (captured 2026-07-31)

- Swap: 2 GB `/swapfile`, observed peak 273 MB used - treat it as required.
- The live box also carries
  `/etc/systemd/system/collector.service.d/proxy.conf`, a legacy drop-in
  that duplicates the `EnvironmentFile=-/etc/turkey-footfall/proxy.env`
  line already in the shipped unit; rebuilds do not need it.
- The installed `collector.service` carries machine-local `Environment=`
  lines - after a rebuild, never overwrite it from the repo template on a
  running box (details in
  [gcp-vm/README.md](gcp-vm/README.md#managing-the-collector-from-your-phone)).
- Routine code updates never need this guide:
  `sudo git -C /opt/turkey-footfall fetch origin main && sudo git -C /opt/turkey-footfall reset --hard origin/main && sudo systemctl restart collector`.
