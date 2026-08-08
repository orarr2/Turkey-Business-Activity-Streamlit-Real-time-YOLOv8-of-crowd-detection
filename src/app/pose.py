"""Skeleton keypoints for person detections (COCO-17, opt-in).

The counting pipeline only needs boxes; this module adds WHAT THE BODY IS
DOING on top of them - the input for the behavior labels ("fell over",
"crouching") and the gesture pass ("hand raised", "waving") that boxes
alone cannot answer.

Design constraints, in order:

  * the detection model stays the single source of truth for WHO exists.
    The pose model (yolov8n-pose by default) runs as a SECOND pass and its
    skeletons are matched onto the detector's `person` boxes by IoU - a
    pose-person with no matching detection is discarded, never counted.
    Counts can therefore never change because pose ran;
  * additive, in place - a matched box dict gains a `"kps"` key
    ([[x, y, conf] x 17]) exactly like the tracker's `track_id` and the
    speed pass's `kmh`. Every existing consumer (counts, ROI filters,
    draw_boxes, re-ID) reads only the keys it knows, so nothing downstream
    needs changing;
  * lazy + optional - the model loads on first use only. The 24/7 VM
    round never loads it unless the operator passes `--pose` (two YOLO
    models + the OSNet embedder do not fit the 1 GB e2-micro; the flag is
    for >=2 GB hosts). The natural home is the on-demand deep window
    (`behavior.analyze_window(pose=True)`), which already accepts the cost
    of extra inference per click.

This module imports nothing heavy at module level so the pure matching
logic stays unit-testable on any machine.
"""
from __future__ import annotations

import os

# Weights for the pose pass. Any ultralytics *-pose checkpoint works; nano
# matches the collector's detection tier and auto-downloads on first use.
POSE_WEIGHTS_DEFAULT = "yolov8n-pose.pt"
# A skeleton claims a detector person box only above this IoU - below it
# the two models are looking at different people.
POSE_MATCH_IOU = 0.50
# Pose-pass confidence floor. Deliberately permissive: a pose-person that
# fails to match any detection is dropped anyway, so a loose gate here
# only costs a few wasted matches, while a tight one loses skeletons for
# people the detector already vouched for.
POSE_CONF = 0.25
# A keypoint below this confidence is not drawn and not used by the
# gesture/label rules - street-cam limbs are often occluded and a
# hallucinated wrist would fire "hand raised" on noise.
KP_MIN_CONF = 0.30

# COCO-17 keypoint order (what every ultralytics pose checkpoint emits).
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# Indices used by gestures.py / behavior_labels.py - named here once so
# the rule modules never carry magic numbers.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Bones, grouped so the overlay reads at a glance even on a small crop:
# head/neck green, arms blue, torso magenta, legs orange (BGR).
_BONE_GROUPS = (
    (((NOSE, L_SHOULDER), (NOSE, R_SHOULDER)),               (80, 175, 76)),
    (((L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
      (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)),            (246, 130, 60)),
    (((L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP),
      (R_SHOULDER, R_HIP), (L_HIP, R_HIP)),                  (200, 60, 200)),
    (((L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
      (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)),                   (0, 160, 255)),
)
SKELETON = tuple(edge for edges, _color in _BONE_GROUPS for edge in edges)

_model = None


def load_pose_model(weights: str | None = None):
    """Load (once) and return the pose model. `POSE_WEIGHTS` env overrides
    the default; ultralytics downloads the checkpoint on first use."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(weights or os.environ.get("POSE_WEIGHTS",
                                                POSE_WEIGHTS_DEFAULT))
    return _model


def _iou(a: dict, b: dict) -> float:
    """IoU of two {"x1","y1","x2","y2"} dicts. Local copy so this module
    (and its tests) never import cv2/numpy through detect_core."""
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_poses(person_boxes: list[dict], pose_dets: list[dict],
                min_iou: float = POSE_MATCH_IOU) -> list[tuple[int, int]]:
    """Greedy best-IoU pairing of detector person boxes with pose-model
    person boxes. Returns (person_idx, pose_idx) pairs; each side is used
    at most once. Pure python - unit-tested without any model."""
    cands: list[tuple[float, int, int]] = []
    for pi, pb in enumerate(person_boxes):
        for qi, qb in enumerate(pose_dets):
            iou = _iou(pb, qb)
            if iou >= min_iou:
                cands.append((iou, pi, qi))
    cands.sort(reverse=True)
    used_p: set[int] = set()
    used_q: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for iou, pi, qi in cands:
        if pi in used_p or qi in used_q:
            continue
        used_p.add(pi)
        used_q.add(qi)
        pairs.append((pi, qi))
    return pairs


def attach_keypoints(model, frame, boxes: list[dict],
                     imgsz: int = 640, conf: float = POSE_CONF) -> int:
    """Run the pose pass on `frame` and attach `"kps"` to every matched
    `person` box IN PLACE. Returns how many boxes gained a skeleton.

    One inference per call; skipped entirely when the frame holds no
    person detections (the common night-street case)."""
    persons = [b for b in boxes if b.get("cls") == "person"]
    if not persons:
        return 0
    res = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
        return 0
    pose_dets: list[dict] = []
    for bb, kps in zip(res.boxes, res.keypoints.data.tolist()):
        x1, y1, x2, y2 = (float(v) for v in bb.xyxy[0].tolist())
        pose_dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "kps": [[round(x, 1), round(y, 1), round(c, 3)]
                    for x, y, c in kps],
        })
    pairs = match_poses(persons, pose_dets)
    for pi, qi in pairs:
        persons[pi]["kps"] = pose_dets[qi]["kps"]
    return len(pairs)


def attach_keypoints_crops(model, frame, boxes: list[dict],
                           imgsz: int = 256, pad_frac: float = 0.25,
                           min_box_h: int = 40,
                           conf: float = POSE_CONF) -> int:
    """Top-down pose: one pass PER PERSON CROP instead of one full-frame
    pass. Attaches `"kps"` (frame coordinates) in place; returns matches.

    Why: on a street cam a pedestrian is 30-120 px tall; a full-frame pose
    pass at 640 hands the model ~15 px of person and finds nothing, so
    gestures/fall/crouch got no input at all on exactly the scenes they
    exist for. Cropping each detector box (padded 25%) and letting the
    pose model see it at `imgsz` multiplies the effective resolution per
    person by 5-15x. The detector stays the source of truth for WHO
    exists: only its `person` boxes are cropped, and the best pose-person
    inside each crop claims that box - there is nothing else it could be.
    Boxes shorter than `min_box_h` px are skipped - below that even a
    dedicated crop holds no limbs, only upscaling artifacts.

    Cost: one small inference per person, batched into a single
    model.predict call. Deep-window-only economics, same as the full-frame
    variant it replaces there.
    """
    persons = [b for b in boxes
               if b.get("cls") == "person"
               and (b["y2"] - b["y1"]) >= min_box_h]
    if not persons:
        return 0
    H, W = frame.shape[:2]
    crops, offsets = [], []
    for b in persons:
        bw, bh = b["x2"] - b["x1"], b["y2"] - b["y1"]
        px, py = bw * pad_frac, bh * pad_frac
        x1 = max(0, int(b["x1"] - px)); y1 = max(0, int(b["y1"] - py))
        x2 = min(W, int(b["x2"] + px)); y2 = min(H, int(b["y2"] + py))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crops.append(frame[y1:y2, x1:x2])
        offsets.append((b, x1, y1))
    if not crops:
        return 0
    results = model.predict(crops, imgsz=imgsz, conf=conf, verbose=False)
    matched = 0
    for (b, ox, oy), res in zip(offsets, results):
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            continue
        # The crop is one person's neighborhood; take the pose-person with
        # the highest box confidence (bystanders clipped at the crop edge
        # score lower and lose).
        confs = [float(c) for c in res.boxes.conf.tolist()]
        qi = max(range(len(confs)), key=confs.__getitem__)
        kps = res.keypoints.data.tolist()[qi]
        b["kps"] = [[round(x + ox, 1), round(y + oy, 1), round(c, 3)]
                    for x, y, c in kps]
        matched += 1
    return matched


def draw_skeleton(img, boxes: list[dict], min_conf: float = KP_MIN_CONF):
    """Draw the skeleton of every box carrying `"kps"` onto `img` (in
    place, returns it). Call AFTER detect_core.draw_boxes - the bones sit
    on top of the box annotation, same layering as the trail overlay."""
    import cv2

    for b in boxes:
        kps = b.get("kps")
        if not kps:
            continue
        for edges, color in _BONE_GROUPS:
            for i, j in edges:
                xi, yi, ci = kps[i]
                xj, yj, cj = kps[j]
                if ci < min_conf or cj < min_conf:
                    continue
                cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)),
                         color, 2, cv2.LINE_AA)
        for x, y, c in kps:
            if c >= min_conf:
                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1,
                           cv2.LINE_AA)
    return img
