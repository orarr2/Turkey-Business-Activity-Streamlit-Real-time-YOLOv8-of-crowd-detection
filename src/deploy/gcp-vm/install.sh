#!/usr/bin/env bash
# One-shot installer for the Turkey Footfall collector on a fresh Debian 12
# GCP e2-micro. Run once via:
#
#   curl -sSL https://raw.githubusercontent.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection/main/src/deploy/gcp-vm/install.sh | sudo bash
#
# Assumes:
#   - The VM's default service account has "Secret Manager Secret Accessor" on
#     the `firebase-sa` secret (see README.md step 5).
#   - Firebase Storage is enabled (README.md step 7).
#
# Idempotent: safe to re-run to refresh the code and restart the service.

set -euo pipefail

REPO_URL="https://github.com/orarr2/Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection.git"
REPO_BRANCH="main"
INSTALL_DIR="/opt/turkey-footfall"
CFG_DIR="/etc/turkey-footfall"
SA_PATH="${CFG_DIR}/serviceAccount.json"
SECRET_NAME="firebase-sa"

log() { printf "\n=== %s ===\n" "$*"; }

if [[ $EUID -ne 0 ]]; then
  echo "install.sh must run as root (use sudo)." >&2
  exit 1
fi

log "1/6 installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    git python3 python3-venv python3-pip \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    ca-certificates curl

# gcloud CLI is preinstalled on GCP VM images; verify it's present.
if ! command -v gcloud >/dev/null; then
  echo "gcloud CLI not found — this installer assumes a GCP VM image." >&2
  exit 1
fi

log "2/6 cloning the repo into ${INSTALL_DIR}"
if [[ -d "${INSTALL_DIR}/.git" ]]; then
  git -C "${INSTALL_DIR}" fetch --depth 1 origin "${REPO_BRANCH}"
  git -C "${INSTALL_DIR}" reset --hard "origin/${REPO_BRANCH}"
else
  git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
fi

log "3/6 setting up Python venv + dependencies"
cd "${INSTALL_DIR}/src"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# /tmp on e2-micro is tmpfs (RAM-backed, ~500MB on 1GB RAM). Pytorch/ultralytics
# wheels (~800MB unpacked) blow it up mid-install with `OSError: [Errno 28]
# No space left on device`. Point pip at /var/tmp which lives on the real
# 30GB disk, and use --no-cache-dir to avoid keeping the downloads around too.
export TMPDIR=/var/tmp
mkdir -p "${TMPDIR}"
.venv/bin/pip install --quiet --no-cache-dir --upgrade pip
.venv/bin/pip install --quiet --no-cache-dir -r requirements.txt

log "4/6 fetching Firebase service-account JSON from Secret Manager"
mkdir -p "${CFG_DIR}"
gcloud secrets versions access latest --secret="${SECRET_NAME}" > "${SA_PATH}"
chown root:root "${SA_PATH}"
chmod 0400 "${SA_PATH}"

# Derive the Firebase Storage bucket. Firebase used to provision buckets as
# <project_id>.appspot.com; projects created after ~Oct 2024 get
# <project_id>.firebasestorage.app instead. Probe both, pick whichever exists.
PROJECT_ID=$(python3 -c "import json; print(json.load(open('${SA_PATH}'))['project_id'])")
STORAGE_BUCKET=""
for candidate in "${PROJECT_ID}.firebasestorage.app" "${PROJECT_ID}.appspot.com"; do
  if gcloud storage buckets describe "gs://${candidate}" >/dev/null 2>&1; then
    STORAGE_BUCKET="${candidate}"
    break
  fi
done
if [[ -z "${STORAGE_BUCKET}" ]]; then
  echo "ERROR: could not find a Firebase Storage bucket for project ${PROJECT_ID}." >&2
  echo "       Enable Firebase Storage first (Firebase Console -> Storage -> Get started)," >&2
  echo "       then re-run this installer." >&2
  exit 1
fi
echo "Storage bucket: ${STORAGE_BUCKET}"

log "5/6 installing systemd unit"
UNIT_SRC="${INSTALL_DIR}/src/deploy/gcp-vm/collector.service"
UNIT_DEST="/etc/systemd/system/collector.service"
# Patch STORAGE_BUCKET into the unit file at install time so the venv sees it.
sed -e "s|__STORAGE_BUCKET__|${STORAGE_BUCKET}|g" \
    -e "s|__INSTALL_DIR__|${INSTALL_DIR}|g" \
    -e "s|__SA_PATH__|${SA_PATH}|g" \
    "${UNIT_SRC}" > "${UNIT_DEST}"
chmod 0644 "${UNIT_DEST}"
systemctl daemon-reload

log "6/6 starting collector service"
systemctl enable --now collector.service
sleep 2
systemctl --no-pager --lines=20 status collector.service || true

cat <<EOF

=== Done. ===

Follow logs:            sudo journalctl -u collector -f
Restart after edits:    sudo systemctl restart collector
Update to latest main:  sudo ${INSTALL_DIR}/src/deploy/gcp-vm/install.sh

Dashboard should show live counts within ~1 minute at:
  http://localhost:8000   (viewer notebook / local serve.py)
  wherever you host web/  (Firebase Hosting, etc.)
EOF
