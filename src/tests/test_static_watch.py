"""Static-object watch: settle >= 5 min, then detect the departure.

Run from src/:  python -m pytest tests -q
"""
from app.static_watch import StaticWatch

SHAPE = (360, 640)
BOX = {"x1": 100, "y1": 100, "x2": 180, "y2": 160}
BOX_FAR = {"x1": 400, "y1": 200, "x2": 480, "y2": 260}


def _det(box=BOX, cls="car", conf=0.8):
    return dict(box, cls=cls, conf=conf)


def _feed(w, cam, dets, t, luma=120.0):
    return w.observe(cam, dets, SHAPE, luma=luma, now=t)


def _settle(w, cam="cam", box=BOX, cls="car", conf=0.8,
            t0=1000.0, step=60.0, n=7):
    """Feed enough same-spot sightings to settle an anchor; returns the
    time after the last one."""
    t = t0
    for _ in range(n):
        evs = _feed(w, cam, [_det(box, cls, conf)], t)
        assert evs == []
        t += step
    return t


def test_settles_then_departure_fires_after_two_misses():
    w = StaticWatch()
    t = _settle(w)                                  # 6 min of sightings
    assert w.counts("cam")["settled"] == 1
    assert _feed(w, "cam", [], t) == []             # miss 1: not yet
    evs = _feed(w, "cam", [], t + 60)               # miss 2: departed
    assert len(evs) == 1
    ev = evs[0]
    assert ev["kind"] == "static_departed"
    assert ev["cls"] == "car"
    assert ev["dwell_sec"] >= 300
    assert ev["conf_median"] == 0.8
    assert w.counts("cam") == {"anchors": 0, "settled": 0}
    # And it never re-fires - the anchor is gone.
    assert _feed(w, "cam", [], t + 120) == []


def test_single_miss_then_reappearance_keeps_the_anchor():
    w = StaticWatch()
    t = _settle(w)
    assert _feed(w, "cam", [], t) == []             # one occluded sample
    assert _feed(w, "cam", [_det()], t + 60) == []  # it's back
    assert w.counts("cam")["settled"] == 1


def test_unsettled_candidate_fizzles_silently():
    w = StaticWatch()
    _feed(w, "cam", [_det()], 1000.0)               # one sighting only
    assert _feed(w, "cam", [], 1060.0) == []
    assert _feed(w, "cam", [], 1120.0) == []        # no event, ever
    assert w.counts("cam")["anchors"] == 0


def test_short_stay_never_settles():
    """Four sightings inside two minutes: continuity yes, five minutes no."""
    w = StaticWatch()
    t = 1000.0
    for _ in range(4):
        _feed(w, "cam", [_det()], t)
        t += 30.0
    assert w.counts("cam")["settled"] == 0
    assert _feed(w, "cam", [], t) == []
    assert _feed(w, "cam", [], t + 30) == []        # fizzles, no event


def test_dark_frame_suppresses_miss_counting():
    w = StaticWatch()
    t = _settle(w)
    # Camera goes dark: losing sight of the anchor proves nothing.
    assert _feed(w, "cam", [], t, luma=10.0) == []
    assert _feed(w, "cam", [], t + 60, luma=10.0) == []
    assert w.counts("cam")["settled"] == 1
    # Light returns and the object is still there.
    assert _feed(w, "cam", [_det()], t + 120) == []
    assert w.counts("cam")["settled"] == 1


def test_scene_wipe_suppresses_mass_departure():
    """Both settled anchors vanish at once = camera cut, not two exits."""
    w = StaticWatch()
    t = 1000.0
    for _ in range(7):
        _feed(w, "cam", [_det(BOX), _det(BOX_FAR)], t)
        t += 60.0
    assert w.counts("cam")["settled"] == 2
    for i in range(4):                              # nothing, repeatedly
        assert _feed(w, "cam", [], t + 60 * i) == []
    assert w.counts("cam")["settled"] == 2          # both still anchored


def test_one_of_many_departing_still_fires():
    """One anchor leaving while the other stays IS a departure."""
    w = StaticWatch()
    t = 1000.0
    for _ in range(7):
        _feed(w, "cam", [_det(BOX), _det(BOX_FAR)], t)
        t += 60.0
    assert _feed(w, "cam", [_det(BOX_FAR)], t) == []
    evs = _feed(w, "cam", [_det(BOX_FAR)], t + 60)
    assert [e["cls"] for e in evs] == ["car"]
    assert w.counts("cam")["settled"] == 1


def test_evidence_floor_blocks_weak_stays():
    """Median conf below the class gate: the 'car' that only exists at
    0.22 on a loosened gate is a shadow - it must never settle."""
    w = StaticWatch(evidence_gates={"car": 0.35})
    t = _settle(w, conf=0.22)
    assert w.counts("cam")["settled"] == 0
    assert _feed(w, "cam", [], t) == []
    assert _feed(w, "cam", [], t + 60) == []        # no event


def test_moving_object_never_anchors():
    w = StaticWatch()
    t = 1000.0
    for i in range(8):
        box = {"x1": 100 + 90 * i, "y1": 100,
               "x2": 180 + 90 * i, "y2": 160}       # IoU 0 vs previous
        _feed(w, "cam", [_det(box)], t)
        t += 60.0
    assert w.counts("cam")["settled"] == 0


def test_prune_drops_stale_anchors():
    w = StaticWatch()
    t = _settle(w)
    assert w.prune(max_age_sec=3600, now=t + 4000) == 1
    assert w.counts("cam")["anchors"] == 0


def test_per_camera_isolation():
    w = StaticWatch()
    t = _settle(w, cam="camA")
    _feed(w, "camB", [_det(BOX_FAR)], t)
    # camB's misses cannot depart camA's anchor.
    assert _feed(w, "camB", [], t + 60) == []
    assert _feed(w, "camB", [], t + 120) == []
    assert w.counts("camA")["settled"] == 1
