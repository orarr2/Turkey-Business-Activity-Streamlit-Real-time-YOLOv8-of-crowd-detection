"""Export a YOLOv8 detector to ONNX / OpenVINO for faster CPU inference.

Why: PyTorch eager inference is the collector's biggest CPU cost. OpenVINO
(and to a lesser degree ONNX Runtime) run the same network 2-3x faster on
x86, which buys either a faster round (fresher dashboard numbers) or a BIGGER
model (yolov8s - noticeably better small/distant-object recall on these wide
street scenes) in the same time budget.

Usage (on a machine with internet access; artifacts are plain directories/
files you can rsync to the VM):

    # best accuracy-per-CPU-second combo for the collector:
    python -m tools.export_model --weights yolov8s.pt --format openvino --half

    # INT8 quantization (another ~1.5-2x on top, slight accuracy cost;
    # needs a calibration dataset yaml, coco128 works fine):
    python -m tools.export_model --weights yolov8s.pt --format openvino --int8

    # portable single-file alternative runnable via onnxruntime:
    python -m tools.export_model --weights yolov8s.pt --format onnx

Then point the collector at the exported artifact - ultralytics loads
exported models transparently, nothing else changes:

    python -m app.collector --weights yolov8s_openvino_model/ --imgsz 960
    (or edit ExecStart in deploy/gcp-vm/collector.service accordingly)

Verify the speedup on the target machine with --bench.
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="source .pt weights (yolov8n.pt / yolov8s.pt / ...)")
    ap.add_argument("--format", choices=("onnx", "openvino"), default="openvino")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="input size to optimize for (the collector's --imgsz)")
    ap.add_argument("--half", action="store_true",
                    help="FP16 weights (openvino; ~2x smaller, same speed on CPU)")
    ap.add_argument("--int8", action="store_true",
                    help="INT8 quantization (openvino; needs --data for calibration)")
    ap.add_argument("--data", default="coco128.yaml",
                    help="calibration dataset yaml for --int8")
    ap.add_argument("--bench", metavar="MODEL", default=None,
                    help="skip export; benchmark MODEL (a .pt, .onnx or an "
                         "openvino dir) on this machine and exit")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.bench:
        import numpy as np
        model = YOLO(args.bench)
        frame = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
        model.predict(frame, imgsz=args.imgsz, verbose=False)      # warmup
        t0 = time.time()
        n = 10
        for _ in range(n):
            model.predict(frame, imgsz=args.imgsz, conf=0.3,
                          classes=[0, 1, 2, 3, 5, 7], verbose=False)
        dt = (time.time() - t0) / n * 1000
        print(f"{args.bench}: {dt:.0f} ms/frame @ imgsz={args.imgsz}")
        return 0

    model = YOLO(args.weights)
    kwargs: dict = dict(format=args.format, imgsz=args.imgsz)
    if args.format == "openvino":
        kwargs["half"] = args.half
        if args.int8:
            kwargs.update(int8=True, data=args.data)
    out = model.export(**kwargs)
    print(f"\nExported: {out}")
    print("Run the collector with:  python -m app.collector "
          f"--weights {out} --imgsz {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
