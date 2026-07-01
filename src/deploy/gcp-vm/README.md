# Deploy the collector on GCP e2-micro (Always Free)

The cloud collector runs the same `app/collector.py` you know locally, but as a
systemd service on a tiny always-on VM. Total cost: **$0/month** on Google's
Always Free tier for the `e2-micro` machine.

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
- **Machine configuration**: series `E2`, machine type **`e2-micro`** (exactly this — anything larger is billed)
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
`[TS] slot_konya_hukumet (konya_hukumet): person=X vehicles=Y ...` every 20s.

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
cd /opt/turkey-footfall && sudo git pull && sudo systemctl restart collector   # deploy new code
```

## Costs to watch

- **e2-micro in us-central1**: $0 as long as you have only one and stay in the
  free region. Set a **budget alert at $1/month** so you catch anything weird.
- **Firestore writes**: `4 slots × 3 writes/sample × 4320 samples/day ≈ 52k writes/day`.
  Blaze free tier allows 20k/day; the overflow costs ~$0.06/day = ~$1.8/month
  in the worst case. If you want it strictly free, raise `--interval` to 60s
  (edit the `ExecStart` in `collector.service` and `systemctl daemon-reload`).
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
