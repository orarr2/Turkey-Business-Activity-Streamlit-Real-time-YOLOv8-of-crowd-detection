"""Export an OSNet re-ID model to ONNX for the collector's --reid-model flag.

Run this ONCE on a machine with internet access (it downloads pretrained
weights), then copy the .onnx to the VM:

    pip install torchreid gdown           # only needed for this script
    python -m tools.export_osnet --out osnet_x0_25.onnx
    scp osnet_x0_25.onnx vm:/opt/turkey-footfall/src/data/
    # on the VM: add  --reid-model data/osnet_x0_25.onnx  to ExecStart

Variants (accuracy vs CPU cost, all fine with onnxruntime on 2 vCPU):
    osnet_x0_25   ~0.2 GFLOPs  - default, ~5-10 ms/crop, big upgrade already
    osnet_x0_5    ~0.4 GFLOPs
    osnet_x1_0    ~1.0 GFLOPs  - strongest, use if crops/sample stay < ~20

The exported network takes NCHW float32 (ImageNet-normalized RGB, 256x128)
and returns one feature vector per crop; app/reid_embed.py's OsnetEmbedder
handles that preprocessing at runtime.
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="osnet_x0_25",
                    help="torchreid model name (osnet_x0_25 / x0_5 / x1_0)")
    ap.add_argument("--out", default=None,
                    help="output .onnx path (default: <model>.onnx)")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()
    out = args.out or f"{args.model}.onnx"

    import torch
    try:
        import torchreid
    except ImportError:
        print("torchreid is not installed - run:  pip install torchreid gdown")
        return 2

    model = torchreid.models.build_model(
        name=args.model, num_classes=1000, pretrained=True)
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width)
    torch.onnx.export(
        model, dummy, out,
        input_names=["images"], output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {args.model} -> {out}")
    print(f"Use with:  python -m app.collector --reid-model {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
