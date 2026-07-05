# Deploy the collector on a small GCP VM

The cloud collector runs the same `app/collector.py` you know locally, but as a
systemd service on a small always-on VM.

> **Machine sizing — measured, not theoretical:** the default is the
> **`e2-micro` (1 GB, Always Free — $0/month)**. This project's live e2-micro
> measured **~635 MB RSS** at `--imgsz 960` under the 700M/850M caps — it
> fits. But peaks vary by torch build: some environments measure ~820 MB, and
> a process living above `MemoryHigh` gets permanently reclaim-throttled —
> rounds stretch to minutes and the dashboard numbers freeze between updates
> (the #1 cause of "the counts don't match the live video"). The installer
> therefore auto-fits any <1.5 GB box with `--imgsz 640` +
> `MemoryHigh=700M`/`MemoryMax=850M`. If your own journal shows no
> throttling, you can remove `--imgsz 640` from `ExecStart` to get 960's
> small-object recall; if it does throttle, keep 640 — or pay for an
> **`e2-small` (2 GB, ~$13/month)**, which the unit template's higher caps
> (1300M/1600M) are sized for. Check
> `systemctl status collector | grep -i memory` after a day of runtime.

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
  (Always Free — $0/month; the installer auto-fits its memory caps, see the
  sizing note above. Choose `e2-small` (~$13/month) only if you explicitly
  want guaranteed 960-input headroom)
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

## Costs to watch

- **VM**: `e2-micro` (the default) is **$0** on the Always Free tier
  (us-central1 / us-east1 / us-west1). `e2-small` is an optional ~$13/month
  upgrade — nothing in this repo requires it. Set a **budget alert** so you
  catch anything weird.
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
