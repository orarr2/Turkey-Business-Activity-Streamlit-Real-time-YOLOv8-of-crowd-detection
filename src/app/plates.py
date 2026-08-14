"""License-plate reading (LPR) on top of vehicle detections - layer 10.

Scope, stated up front: this reads the PLATE STRING of vehicles the
detector already found. It keeps no plate database, matches nothing
against watchlists, and adds no new heavy dependency - both models ride
the OpenVINO runtime the detection engine already brought.

Two-stage, both tiny and both on crops only:

  * plate DETECTION - a yolov8n plate detector (~6 MB;
    Koushim/yolov8-license-plate-detection, MIT) run per VEHICLE CROP,
    same top-down economics as the pose pass: a street-cam vehicle is
    100-400 px wide, so a dedicated crop multiplies the plate's
    effective resolution instead of handing the model a 6 px smear;
  * plate OCR - fast-plate-ocr's cct_xs_relu_v1_global head (~2 MB ONNX,
    MIT), compiled directly by OpenVINO (core.read_model handles ONNX -
    no onnxruntime dependency). Latin alphabet + digits, 9 slots.

Operating envelope (honesty over demo-magic): the OCR model is the
COUNTRY-GENERIC one on purpose - the picker runs Thailand today, Turkey,
Japan and the USA on other days, so the reader must not assume a plate
grammar. Digits 0-9 and Latin letters cover the registration NUMBER on
all four (Turkish and US plates are fully in-alphabet; Thai and
Japanese plates carry a local-script line that no Latin head can read -
their digit groups still resolve). Below MIN_VEHICLE_W px of vehicle
width the plate is physically sub-legible at 480p and the pass skips
the vehicle; the layer's envelope note states how many vehicles were in
range so an empty overlay reads as "too far", not "broken".

Per-track caching happens in the live session (a plate does not change
mid-track): one accepted read per track id, bounded retries while the
vehicle is close enough.
"""
from __future__ import annotations

import os
import threading
import time

# Weights: bare names resolve against the process CWD (the project root,
# same convention as the detection/pose weights). Both are gitignored
# binaries - see the module docstring for their public sources.
PLATE_WEIGHTS_DEFAULT = "yolov8n-plate.pt"
PLATE_OCR_DEFAULT = "plate_ocr_global.onnx"

# Plate-detector confidence floor on the vehicle crop. Permissive on
# purpose: the crop IS a vehicle (the detector vouched for it), so a
# plate-shaped hit is almost certainly the plate; the OCR confidence
# gate downstream is the real filter.
PLATE_CONF = 0.30
# fast-plate-ocr cct_xs_relu_v1_global contract (from its shipped
# plate_config.yaml, inlined so the .yaml need not live in the repo):
OCR_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
OCR_PAD = "_"
OCR_SLOTS = 9
OCR_W, OCR_H = 128, 64
# Acceptance gates: mean per-slot confidence and a minimum of 4 readable
# characters (shorter "reads" on far crops are noise letters). 0.45 (was
# 0.60): night footage never clears 0.60 on sub-60px plates, so the gate
# rejected every read the operator could still eyeball-verify. The chip
# SHOWS the confidence, so a 0.5 read is presented as what it is.
OCR_MIN_CONF = 0.45
OCR_MIN_CHARS = 4
# Vehicle box narrower than this (px) puts the plate under ~8 px of
# height at 480p street distance - upscaling artifacts, not glyphs.
# 96 (not higher): the pass should TRY mid-range vehicles too - the
# plate-width and OCR-confidence gates downstream reject the unreadable
# ones, so the range limit is physics (source resolution), not policy.
MIN_VEHICLE_W = 96
# Motorcycles get a lower floor: the bike itself is narrow, but its
# plate fills a much larger FRACTION of the box than a car's does.
MIN_VEHICLE_W_MOTO = 72
# Plate crop narrower than this (px) is skipped even on a wide vehicle
# (plate at an extreme angle or partially occluded).
MIN_PLATE_W = 32
# Bounded work per tick: closest (widest) unread vehicles first.
MAX_VEHICLES_PER_TICK = 3
# Give up on a track after this many failed read attempts; a vehicle
# that stayed unreadable for 6 close-range ticks is angled/blurred.
MAX_TRIES_PER_TRACK = 6
# Minimum Laplacian variance of the plate crop before OCR is attempted -
# below this the crop is motion-smeared (night exposure) and any read
# would be a hallucination. Skipped crops refund their try.
PLATE_SHARPNESS_MIN = 45.0
# An exhausted-but-unread track gets a fresh try budget this often. A
# parked vehicle's track lives for hours; without the reset its only
# chances were the session's first few ticks.
STATIC_RETRY_S = 120.0

PLATE_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

_det_model = None
_ocr = None
_LOAD_LOCK = threading.Lock()


def load_plate_model(weights: str | None = None):
    """Load (once) the plate detector; prefers a sibling
    `<stem>_openvino_model` export exactly like the detection and pose
    loaders. Export once with:
        YOLO("yolov8n-plate.pt").export(format="openvino", imgsz=256)
    """
    global _det_model
    if _det_model is not None:
        return _det_model
    with _LOAD_LOCK:
        if _det_model is None:
            from ultralytics import YOLO
            w = weights or os.environ.get("PLATE_WEIGHTS",
                                          PLATE_WEIGHTS_DEFAULT)
            if str(w).endswith(".pt"):
                ov_dir = str(w)[:-3] + "_openvino_model"
                if os.path.isdir(ov_dir):
                    w = ov_dir
                    print(f"plates: OpenVINO engine loaded ({ov_dir})")
            # task= silences the exported engine's guess-the-task warning.
            _det_model = YOLO(w, task="detect")
    return _det_model


class _OvOcr:
    """The fast-plate-ocr ONNX head compiled by OpenVINO.

    Input introspected at load: NHWC [1,64,128,3] (keras export) or NCHW,
    uint8 or float - either way the graph carries its own /255 rescaling,
    so raw 0..255 pixel values go in. Output [1, 9, 37]: one softmax (or
    logit - normalized here if needed) row per plate slot."""

    def __init__(self, path: str):
        import numpy as np
        import openvino as ov
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(path), "CPU")
        inp = self.compiled.input(0)
        self.out = self.compiled.output(0)
        shape = [int(d) for d in inp.get_shape()]
        self.nchw = len(shape) == 4 and shape[1] == 3
        self.uint8 = "u8" in inp.get_element_type().get_type_name()
        self._np = np

    def read(self, plate_bgr) -> tuple[str, float]:
        """OCR one plate crop -> (text, mean_conf). Empty text on junk."""
        import cv2
        np = self._np
        # Small crops go through a 2x cubic upscale BEFORE the model-size
        # resize: direct linear 40px -> 128px loses the stroke edges the
        # head needs; cubic-then-area keeps them measurably sharper.
        if plate_bgr.shape[1] < OCR_W:
            plate_bgr = cv2.resize(plate_bgr, (OCR_W * 2, OCR_H * 2),
                                   interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(plate_bgr, (OCR_W, OCR_H),
                         interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img if self.uint8 else img.astype("float32")
        if self.nchw:
            x = x.transpose(2, 0, 1)
        y = self.compiled([x[None]])[self.out]
        y = np.asarray(y).reshape(OCR_SLOTS, len(OCR_ALPHABET))
        # Keras heads usually export with softmax baked in; normalize
        # defensively when rows are logits.
        if not (0.99 <= float(y[0].sum()) <= 1.01):
            e = np.exp(y - y.max(axis=1, keepdims=True))
            y = e / e.sum(axis=1, keepdims=True)
        idx = y.argmax(axis=1)
        confs = y[np.arange(OCR_SLOTS), idx]
        chars = [OCR_ALPHABET[i] for i in idx]
        text = "".join(c for c in chars if c != OCR_PAD)
        used = [c for ch, c in zip(chars, confs) if ch != OCR_PAD]
        conf = float(np.mean(used)) if used else 0.0
        if len(text) < OCR_MIN_CHARS or conf < OCR_MIN_CONF:
            return "", conf
        return text, conf


def load_ocr(path: str | None = None) -> _OvOcr:
    global _ocr
    if _ocr is not None:
        return _ocr
    with _LOAD_LOCK:
        if _ocr is None:
            _ocr = _OvOcr(path or os.environ.get("PLATE_OCR",
                                                 PLATE_OCR_DEFAULT))
            print("plates: OCR head compiled (OpenVINO, "
                  f"{OCR_SLOTS} slots, Latin+digits)")
    return _ocr


def attach_plates(det_model, ocr: _OvOcr, frame, tracker,
                  reads: dict) -> tuple[int, int]:
    """Read plates for the tracker's open vehicle tracks, in place.

    `reads` is the session's per-track cache: tid -> {"text", "conf",
    "tries"}. A cached accepted read is stamped onto the track's current
    box for free; only close-enough unread tracks cost inference, capped
    at MAX_VEHICLES_PER_TICK widest-first. Returns (in_range, new_reads).
    """
    H, W = frame.shape[:2]
    in_range = new_reads = 0
    candidates = []
    open_tids = set()
    for tr in (tracker.open if tracker else []):
        if tr.cls not in PLATE_VEHICLE_CLASSES:
            continue
        open_tids.add(tr.tid)
        b = tr.boxes[-1]
        entry = reads.get(tr.tid)
        if entry and entry.get("text"):
            b["plate"] = entry["text"]
            b["plate_conf"] = entry["conf"]
        bw = b["x2"] - b["x1"]
        floor = (MIN_VEHICLE_W_MOTO if tr.cls == "motorcycle"
                 else MIN_VEHICLE_W)
        if bw < floor:
            continue
        in_range += 1
        # Keep re-reading an already-read track until the read is GOOD
        # (>=0.70) or the try budget runs out - the best read across all
        # attempts wins, so one lucky sharp frame upgrades a marginal one.
        if entry and entry.get("conf", 0) >= 0.70:
            continue
        if entry and entry.get("tries", 0) >= MAX_TRIES_PER_TRACK:
            if entry.get("text"):
                continue
            # Long-lived UNREAD track (a parked vehicle): the first budget
            # was spent on whatever frames the session opened with (audit
            # 2026-08-14: a legible parked-scooter plate stayed unread all
            # night). Grant a fresh budget every STATIC_RETRY_S - light
            # and occlusion change.
            t0 = entry.setdefault("t_giveup", time.time())
            if time.time() - t0 < STATIC_RETRY_S:
                continue
            entry["tries"] = 0
            entry.pop("t_giveup", None)
        candidates.append((bw, tr.tid, b))
    # Cache hygiene: forget tracks the tracker itself dropped.
    for tid in [t for t in reads if t not in open_tids]:
        reads.pop(tid, None)
    if not candidates:
        return in_range, 0

    import cv2  # noqa: F401  (cv2 import kept local, like the pose pass)
    from app.detect_core import _PREDICT_LOCK
    candidates.sort(reverse=True)
    for bw, tid, b in candidates[:MAX_VEHICLES_PER_TICK]:
        x1 = max(0, int(b["x1"]) - 4); y1 = max(0, int(b["y1"]) - 4)
        x2 = min(W, int(b["x2"]) + 4); y2 = min(H, int(b["y2"]) + 4)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        entry = reads.setdefault(tid, {"text": "", "conf": 0.0, "tries": 0})
        # Closest-approach: while the vehicle still GROWS on screen every
        # tick is a better shot than the last; once it shrinks below 85%
        # of its own peak width the best frame has already passed - stop
        # spending the try budget on frames that can only be worse.
        _w_max = entry.get("w_max", 0)
        entry["w_max"] = max(_w_max, bw)
        if not entry.get("text") and _w_max and bw < 0.85 * _w_max:
            continue
        entry["tries"] += 1
        with _PREDICT_LOCK:
            res = det_model.predict(crop, imgsz=256, conf=PLATE_CONF,
                                    verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        confs = [float(c) for c in res.boxes.conf.tolist()]
        qi = max(range(len(confs)), key=confs.__getitem__)
        px1, py1, px2, py2 = [int(v) for v in res.boxes.xyxy.tolist()[qi]]
        if px2 - px1 < MIN_PLATE_W:
            continue
        plate = crop[max(0, py1):py2, max(0, px1):px2]
        if plate.size == 0:
            continue
        # Motion-blur gate: OCR on a smeared night plate can only
        # hallucinate. Laplacian variance is a cheap sharpness proxy;
        # below the floor, skip the OCR and REFUND the try - the next
        # tick may catch the same plate sharp (per-tick cost stays
        # bounded by MAX_VEHICLES_PER_TICK either way).
        _g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        if float(cv2.Laplacian(_g, cv2.CV_64F).var()) < PLATE_SHARPNESS_MIN:
            entry["tries"] -= 1
            continue
        text, conf = ocr.read(plate)
        # Best-read-wins: a later attempt only replaces the stored read
        # when its confidence is higher.
        if text and conf > entry.get("conf", 0):
            if not entry.get("text"):
                new_reads += 1
            entry["text"] = text
            entry["conf"] = round(conf, 2)
            b["plate"] = text
            b["plate_conf"] = entry["conf"]
            b["plate_box"] = [x1 + px1, y1 + py1, x1 + px2, y1 + py2]
    return in_range, new_reads
