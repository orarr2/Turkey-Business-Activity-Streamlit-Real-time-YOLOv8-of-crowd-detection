#!/usr/bin/env bash
# One-shot setup of the bgutil PO-token provider on the collector VM.
#
# WHY: YouTube starves Google-datacenter IPs - every YouTube-backed camera
# (the Turkey YT tier + all of Thailand/Japan/USA) resolves but receives
# no video data from the VM, while the same streams play fine from any
# residential IP. yt-dlp's documented remedy is attaching a PO (proof of
# origin) token to the session; the bgutil provider mints unauthenticated
# "cold start" tokens - no Google account, no cookies to expire.
#
# MODE: script (per-resolution node invocation). Chosen over the HTTP
# server because the e2-micro has 1 GB total and the collector already
# budgets most of it - a resident node server is asking for the
# oom-killer, while a short-lived node process only exists during the
# occasional stream resolution (resolved URLs are cached).
#
# Usage (as the operator, on the VM):
#   sudo bash /opt/turkey-footfall/src/deploy/gcp-vm/setup_pot_provider.sh
#
# Idempotent: safe to re-run; every step checks before doing.
set -euo pipefail

POT_DIR=/opt/bgutil-ytdlp-pot-provider
VENV=/opt/turkey-footfall/src/.venv
ENV_FILE=/etc/turkey-footfall/proxy.env
SCRIPT_PATH="$POT_DIR/server/build/generate_once.js"

echo "== 1/6 node + npm (apt) =="
if ! command -v node >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq nodejs npm
fi
node --version

echo "== 2/6 stop the collector during the build (npm needs the RAM) =="
systemctl stop collector || true

echo "== 3/6 fetch + build the provider =="
if [ ! -d "$POT_DIR/.git" ]; then
    git clone --depth 1 https://github.com/Brainicism/bgutil-ytdlp-pot-provider "$POT_DIR"
else
    git -C "$POT_DIR" pull --ff-only || true
fi
cd "$POT_DIR/server"
npm install --no-audit --no-fund --loglevel=error
npx tsc || npm run build
test -f "$SCRIPT_PATH" || { echo "BUILD FAILED: $SCRIPT_PATH missing"; exit 1; }

echo "== 4/6 yt-dlp plugin into the collector venv =="
"$VENV/bin/pip" install -q -U bgutil-ytdlp-pot-provider

echo "== 5/6 wire the env (EnvironmentFile the service already loads) =="
touch "$ENV_FILE"; chmod 600 "$ENV_FILE"
grep -q "^YT_POT_SCRIPT=" "$ENV_FILE" || \
    echo "YT_POT_SCRIPT=$SCRIPT_PATH" >> "$ENV_FILE"
# web/mweb are the PO-token-bearing clients; keep android/ios as fallbacks.
grep -q "^YT_PLAYER_CLIENTS=" "$ENV_FILE" || \
    echo "YT_PLAYER_CLIENTS=web,mweb,android,ios" >> "$ENV_FILE"

echo "== 6/6 restart the collector =="
systemctl start collector

echo
echo "DONE. Verify with:"
echo "  sudo bash -c 'set -a; . $ENV_FILE; set +a; \\"
echo "    cd /opt/turkey-footfall/src && \\"
echo "    .venv/bin/python -m tools.probe_country --country thailand 2>&1 | tail -6'"
echo "Expected: th_* cameras flip to LIVE. If they stay dead, run the"
echo "same probe with --country turkey to confirm the IBB path still works,"
echo "and send the output back for diagnosis."
