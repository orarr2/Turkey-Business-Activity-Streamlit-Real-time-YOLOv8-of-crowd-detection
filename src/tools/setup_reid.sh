#!/usr/bin/env bash
# One-shot OSNet re-ID model setup.
#
# The default appearance embedder is an HSV histogram (color signature).
# It's dependency-free but blind to lighting/pose changes and matches by
# color rather than identity. OSNet is a small (~5 MB) purpose-built
# re-ID CNN that keeps working across lighting shifts and different
# angles - the piece the histogram fundamentally cannot do.
#
# Usage (from the src/ directory):
#     bash tools/setup_reid.sh
# Then restart the collector + dashboard server. app.reid_embed picks
# up the ONNX automatically when REID_MODEL points at it.
#
# The script tries a couple of well-known mirrors; if none work (proxy
# blocks, no network), it falls back to printing the manual "produce
# your own with torchreid" instructions and exits 1.

set -eu

# Model choice: osnet_x0_25 is the smallest torchreid variant that still
# outperforms histograms on identity. ~5 MB on disk, ~5-10 ms inference
# per crop on a modern CPU (the e2-micro handles it comfortably in the
# collector's per-burst slack).
MODEL_NAME="osnet_x0_25"
OUT_DIR="data"
OUT_PATH="$OUT_DIR/${MODEL_NAME}_msmt17.onnx"

mkdir -p "$OUT_DIR"

if [ -f "$OUT_PATH" ]; then
    echo "already present: $OUT_PATH"
    echo "(delete it to re-download)"
    exit 0
fi

# Public mirrors, in order of preference. Add / edit as needed.
URLS=(
    "https://huggingface.co/deepcam-cn/reid-onnx/resolve/main/osnet_x0_25_msmt17.onnx"
    "https://media.githubusercontent.com/media/mikel-brostrom/yolo_tracking/master/tracker/weights/osnet_x0_25_msmt17.onnx"
)

TMP="$OUT_PATH.download.tmp"
for URL in "${URLS[@]}"; do
    echo "attempting: $URL"
    if command -v curl >/dev/null 2>&1; then
        if curl -sSL --fail --max-time 60 "$URL" -o "$TMP"; then
            # Sanity check: min 1MB to guard against 404 HTML
            SIZE=$(stat -c%s "$TMP" 2>/dev/null || stat -f%z "$TMP")
            if [ "$SIZE" -gt 1000000 ]; then
                mv "$TMP" "$OUT_PATH"
                echo ""
                echo "OK: $OUT_PATH ($SIZE bytes)"
                echo ""
                echo "next: set REID_MODEL in your systemd unit or shell:"
                echo "    export REID_MODEL=\$PWD/$OUT_PATH"
                echo "    sudo systemctl restart collector.service"
                exit 0
            fi
            rm -f "$TMP"
        fi
    fi
done

rm -f "$TMP"
cat <<'EOF'
--------------------------------------------------------------
Automatic download failed (proxy / no public network / mirror
rotation). You have two options:

1) Fetch OSNet manually from a host that can reach GitHub or
   HuggingFace, then scp it onto the VM:

       # on a machine with internet access
       curl -L -o osnet_x0_25_msmt17.onnx \
           https://huggingface.co/deepcam-cn/reid-onnx/resolve/main/osnet_x0_25_msmt17.onnx
       scp osnet_x0_25_msmt17.onnx <vm>:~/.../src/data/

2) Produce it yourself with torchreid (one-time, needs a Python
   env with torch + torchreid installed):

       import torchreid
       from torchreid.utils import load_pretrained_weights
       m = torchreid.models.build_model('osnet_x0_25', num_classes=1000)
       load_pretrained_weights(m, 'osnet_x0_25_msmt17')
       m.eval()
       import torch
       torch.onnx.export(m, torch.randn(1, 3, 256, 128),
                         'osnet_x0_25_msmt17.onnx',
                         input_names=['input'], output_names=['embedding'],
                         opset_version=13)

Then place the .onnx at data/osnet_x0_25_msmt17.onnx and re-run
this script (it will short-circuit on the already-present file).
--------------------------------------------------------------
EOF
exit 1
