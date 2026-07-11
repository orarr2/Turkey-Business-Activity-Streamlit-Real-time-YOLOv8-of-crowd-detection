"""Burst-based vehicle speed estimation: geometry-only, no model needed."""
from app.detect_core import estimate_speeds, summarize_speeds

SHAPE = (720, 1280, 3)


def _box(x, y, w=100, h=50, cls="car", conf=0.8):
    return {"x1": float(x), "y1": float(y),
            "x2": float(x + w), "y2": float(y + h),
            "cls": cls, "conf": conf}


def test_moving_car_speed_math():
    # 100px-long car = 4.5 m ruler -> 0.045 m/px. 150 px/frame at dt=1s
    # (stride 25 @ 25fps) over 3 frames -> 6.75 m/s = 24.3 km/h.
    frames = [[_box(100, 300)], [_box(250, 300)], [_box(400, 300)]]
    out = estimate_speeds(frames, SHAPE, stride=25, fps=25.0)
    assert len(out) == 1
    assert out[0]["cls"] == "car" and out[0]["points"] == 3
    assert abs(out[0]["kmh"] - 24.3) < 0.5


def test_parked_car_clamps_to_zero():
    frames = [[_box(100, 300)], [_box(101, 300)], [_box(100, 301)]]
    out = estimate_speeds(frames, SHAPE, stride=25, fps=25.0)
    assert len(out) == 1 and out[0]["kmh"] == 0.0


def test_impossible_speed_dropped_as_mismatch():
    # tiny 20px box (0.225 m/px) jumping 430 px/frame -> ~350 km/h -> fused
    # pair of different vehicles, not a measurement.
    frames = [[_box(100, 300, w=20, h=10)], [_box(530, 300, w=20, h=10)]]
    out = estimate_speeds(frames, SHAPE, stride=25, fps=25.0)
    assert out == []


def test_persons_are_ignored():
    frames = [[_box(100, 300, cls="person")], [_box(150, 300, cls="person")]]
    assert estimate_speeds(frames, SHAPE, stride=25, fps=25.0) == []


def test_single_frame_burst_no_estimates():
    assert estimate_speeds([[_box(100, 300)]], SHAPE) == []


def test_bus_uses_its_own_length():
    # Same pixel geometry as the car test but a 12 m bus -> speed scales
    # by 12/4.5.
    frames = [[_box(100, 300, cls="bus")], [_box(250, 300, cls="bus")],
              [_box(400, 300, cls="bus")]]
    out = estimate_speeds(frames, SHAPE, stride=25, fps=25.0)
    assert len(out) == 1
    assert abs(out[0]["kmh"] - 24.3 * (12.0 / 4.5)) < 1.5


def test_iou_fallback_catches_budget_broken_match():
    """A vehicle whose centroid jump exceeds the track budget (small frame)
    but whose boxes still overlap gets its speed via the IoU fallback -
    coverage toward 'every matched vehicle carries a speed'."""
    shape = (200, 200, 3)   # diag ~283 -> centroid budget ~85px
    a = {"x1": 0.0, "y1": 0.0, "x2": 300.0, "y2": 80.0, "cls": "car", "conf": 0.8}
    b = {"x1": 100.0, "y1": 0.0, "x2": 400.0, "y2": 80.0, "cls": "car", "conf": 0.8}
    out = estimate_speeds([[a], [b]], shape, stride=25, fps=25.0)
    assert len(out) == 1
    # 100px over a 300px~=4.5m ruler in 1s -> 1.5 m/s = 5.4 km/h
    assert abs(out[0]["kmh"] - 5.4) < 0.3


def test_summary_medians_over_moving_only():
    speeds = [{"cls": "car", "kmh": 30.0, "points": 3, "box": {}},
              {"cls": "car", "kmh": 0.0, "points": 3, "box": {}},   # parked
              {"cls": "bus", "kmh": 20.0, "points": 2, "box": {}},
              {"cls": "car", "kmh": 50.0, "points": 2, "box": {}}]
    s = summarize_speeds(speeds)
    assert s["tracked"] == 4 and s["moving"] == 3
    assert s["median_kmh"] == 30.0 and s["max_kmh"] == 50.0
    assert s["per_class"] == {"bus": 20.0, "car": 50.0} or \
           s["per_class"]["car"] in (30.0, 50.0)   # even-count median = upper
    assert summarize_speeds([]) is None
