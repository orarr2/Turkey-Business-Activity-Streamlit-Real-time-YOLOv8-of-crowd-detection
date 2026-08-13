"""Pose-to-detection matching: pure geometry, no model.

Run from src/:  python -m pytest tests -q
"""
from app.pose import (KEYPOINT_NAMES, POSE_MATCH_IOU, SKELETON, _iou,
                      match_poses)


def _box(x, y, w=40, h=90):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h}


def test_skeleton_indices_are_valid():
    assert len(KEYPOINT_NAMES) == 17
    assert all(0 <= i < 17 and 0 <= j < 17 for i, j in SKELETON)


def test_iou_identical_and_disjoint():
    a = _box(10, 10)
    assert _iou(a, dict(a)) == 1.0
    assert _iou(a, _box(500, 500)) == 0.0


def test_perfect_overlap_pairs_up():
    persons = [_box(0, 0), _box(200, 0)]
    poses = [dict(_box(200, 0), kps=[]), dict(_box(0, 0), kps=[])]
    pairs = match_poses(persons, poses)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_greedy_prefers_best_iou():
    # One pose box overlapping two persons - the tighter fit wins.
    persons = [_box(0, 0, w=40, h=90), _box(10, 5, w=40, h=90)]
    poses = [_box(10, 5, w=40, h=90)]
    pairs = match_poses(persons, poses)
    assert pairs == [(1, 0)]


def test_below_threshold_stays_unmatched():
    persons = [_box(0, 0)]
    # Slight overlap, IoU well under the gate.
    poses = [_box(35, 80)]
    assert _iou(persons[0], poses[0]) < POSE_MATCH_IOU
    assert match_poses(persons, poses) == []


def test_each_side_used_once():
    persons = [_box(0, 0)]
    poses = [_box(0, 0), _box(1, 1)]     # both would match
    pairs = match_poses(persons, poses)
    assert len(pairs) == 1
