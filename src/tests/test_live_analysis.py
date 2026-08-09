"""fix 2 live-analysis engine: accumulators, layer semantics, manager.

Run from src/:  python -m pytest tests -q
"""
import json
import threading

import numpy as np
import pytest

from app import live_analysis as la
from app.tracker import Track

SHAPE = (360, 640)


def _box(x, y, w=30, h=60, cls="person", conf=0.9):
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "cls": cls, "conf": conf}


def _kps_for(b, conf=0.9):
    """17 plausible keypoints inside a box (COCO order not important for
    drawing - draw_skeleton only needs x,y,conf per index)."""
    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    cx = (x1 + x2) / 2
    return [[cx + (i % 3 - 1) * 5, y1 + (y2 - y1) * (i / 16.0), conf]
            for i in range(17)]


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

def test_update_crossings_counts_in_then_out():
    # DEFAULT_LINE is horizontal at y=0.62: crossing downward = "in".
    line = la.DEFAULT_LINE
    tr = Track(1, _box(300, 180 - 60), 0.0)          # foot y=180 (0.50) above
    sides, cross = {}, {"in": 0, "out": 0}
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross == {"in": 0, "out": 0}              # first sighting: no flip
    tr.add(_box(300, 252 - 60), 1.0)                 # foot y=252 (0.70) below
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross == {"in": 1, "out": 0}
    tr.add(_box(300, 180 - 60), 2.0)                 # back up
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert cross == {"in": 1, "out": 1}


def test_update_crossings_skips_coasting_tracks():
    line = la.DEFAULT_LINE
    tr = Track(1, _box(300, 120), 0.0)
    tr.misses = 1                                    # coasting - no fresh box
    sides, cross = {}, {}
    la.update_crossings(sides, [tr], SHAPE, line, cross)
    assert sides == {} and cross == {}


def test_bump_heat_and_grid_from_tracks():
    grid = [[0.0] * la.GRID_W for _ in range(la.GRID_H)]
    b = _box(320 - 15, 180 - 60)                     # foot at frame center
    la.bump_heat(grid, [b], SHAPE, 2.5)
    assert sum(v for row in grid for v in row) == pytest.approx(2.5)
    tr = Track(1, b, 0.0)
    tr.add(_box(400, 200), 1.0)
    g2 = la.grid_from_tracks([tr], SHAPE)
    assert sum(v for row in g2 for v in row) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Camera resolution
# ---------------------------------------------------------------------------

def test_resolve_cam_registry():
    cam = la.resolve_cam("taksim_yeni")
    assert cam["id"] == "taksim_yeni"
    assert cam["url"].startswith("http")


def test_resolve_cam_local_slots(tmp_path):
    p = tmp_path / "local_grid.json"
    p.write_text(json.dumps({"slots": [
        {"slot_id": "local_0", "placeholder_name": "Sukhumvit Rd",
         "placeholder_embed": "https://www.youtube.com/embed/Q71sLS8h9a4?x=1"},
        {"slot_id": "local_1", "placeholder_name": "Konya",
         "placeholder_hls": "/tvkur/abc123/master.m3u8"},
    ]}), encoding="utf-8")
    yt = la.resolve_cam("local_0", grid_path=p)
    assert yt["kind"] == "youtube"
    assert yt["url"] == "https://www.youtube.com/watch?v=Q71sLS8h9a4"
    hls = la.resolve_cam("local_1", grid_path=p)
    assert hls["kind"] == "hls"
    assert hls["url"] == "https://content.tvkur.com/l/abc123/master.m3u8"
    with pytest.raises(ValueError):
        la.resolve_cam("local_9", grid_path=p)


# ---------------------------------------------------------------------------
# Layer semantics (fix 2: each layer draws ONLY its own information)
# ---------------------------------------------------------------------------

CAP_H = 60   # everything below this row must be untouched by caption-only layers


def test_pose_layer_draws_no_detection_boxes():
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    person = _box(100, 100)                          # no kps -> too far
    car = _box(400, 200, w=100, h=40, cls="car")
    out = la.draw_pose_layer(img.copy(), [person, car])
    assert out[CAP_H:].sum() == 0                    # no boxes, no skeleton
    person["kps"] = _kps_for(person)
    out2 = la.draw_pose_layer(img.copy(), [person, car])
    assert out2[CAP_H:].sum() > 0                    # skeleton appeared
    # ...and the car region still has no annotation.
    assert out2[200:240, 400:500].sum() == 0


def test_gestures_layer_honest_when_empty():
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    person = _box(100, 100)
    person["kps"] = _kps_for(person)
    person["track_id"] = 1
    stats = {1: {"id": 1, "gestures": []}}
    out = la.draw_gestures_layer(img.copy(), [person], stats, {})
    assert out[CAP_H:].sum() == 0                    # nothing to show, says so
    stats[1]["gestures"] = ["hand_up"]
    out2 = la.draw_gestures_layer(img.copy(), [person], stats, {"hand_up": 1})
    assert out2[CAP_H:].sum() > 0                    # skeleton + chip


def test_body_layer_flags_only_anomalies():
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    walker = _box(100, 100); walker["track_id"] = 1
    faller = _box(300, 100); faller["track_id"] = 2
    stats = {1: {"id": 1, "label": "walking", "alert": False},
             2: {"id": 2, "label": "fall_suspect", "alert": True}}
    out = la.draw_body_layer(img.copy(), [walker, faller], stats)
    assert out[100:170, 90:140].sum() == 0           # walker undrawn
    assert out[100:170, 295:340].sum() > 0           # faller boxed


def test_line_layer_draws_line_and_counts():
    img = np.zeros((*SHAPE, 3), dtype=np.uint8)
    out = la.draw_line_layer(img.copy(), la.DEFAULT_LINE, {"in": 3, "out": 1})
    y = int(0.62 * SHAPE[0])
    assert out[y - 3:y + 4].sum() > 0                # the line itself
    assert out[:30].sum() > 0                        # the caption


def test_heat_layer_empty_and_hot():
    img = np.full((*SHAPE, 3), 40, dtype=np.uint8)
    grid = [[0.0] * la.GRID_W for _ in range(la.GRID_H)]
    out = la.draw_heat_layer(img.copy(), grid)
    assert (out[CAP_H:] == img[CAP_H:]).all()        # zero grid: photo intact
    grid[la.GRID_H // 2][la.GRID_W // 2] = 50.0
    out2 = la.draw_heat_layer(img.copy(), grid)
    assert not (out2[CAP_H:] == img[CAP_H:]).all()   # overlay visible


def test_one_shot_render_layer_matches_live_semantics():
    from app.behavior import render_layer
    frames = [np.zeros((*SHAPE, 3), dtype=np.uint8) for _ in range(3)]
    tr = Track(1, _box(100, 100), 0.0)
    tr.add(_box(160, 100), 0.5)
    tr.boxes[-1]["track_id"] = 1
    pose = render_layer(frames, [tr], [], "pose")
    assert pose[CAP_H:].sum() == 0                   # no kps -> no boxes drawn
    line = render_layer(frames, [tr], [], "line", cam_id=None)
    assert line[:30].sum() > 0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class _StubSession:
    def __init__(self, cam, model, layer):
        self.cam = cam
        self.cam_id = cam["id"]
        self.cam_name = cam.get("name", cam["id"])
        self.model = model
        self.layer = layer
        self.last_poll = 0.0
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.note = "starting"
        self._alive = True

    def start(self):
        pass

    def is_alive(self):
        return self._alive


@pytest.fixture()
def stub_manager(monkeypatch):
    monkeypatch.setattr(la, "LiveSession", _StubSession)
    return la.LiveAnalysisManager()


def test_manager_rejects_unknown_layer(stub_manager):
    with pytest.raises(ValueError):
        stub_manager.start("taksim_yeni", "xray", model=None)


def test_manager_switch_keeps_session(stub_manager):
    a = stub_manager.start("taksim_yeni", "heat", model=None)
    assert a["switched"] is False and a["active"] == 1
    b = stub_manager.start("taksim_yeni", "gestures", model=None)
    assert b["switched"] is True and b["active"] == 1
    fr = stub_manager.frame("taksim_yeni")
    assert fr["layer"] == "gestures" and fr["jpeg"] is None


def test_manager_caps_sessions_and_reaps(stub_manager):
    cams = ["taksim_yeni", "beyazit_meydan_yeni", "sarachane_yeni",
            "sultanahmet_1_yeni"]
    for c in cams:
        stub_manager.start(c, "paths", model=None)
    with pytest.raises(la.BusyError):
        stub_manager.start("konya_hukumet", "paths", model=None)
    # One session dies -> its slot frees on the next start.
    stub_manager._sessions["taksim_yeni"]._alive = False
    ok = stub_manager.start("konya_hukumet", "paths", model=None)
    assert ok["active"] == 4
    assert stub_manager.frame("taksim_yeni") is None


def test_manager_stop(stub_manager):
    stub_manager.start("taksim_yeni", "line", model=None)
    assert stub_manager.stop("taksim_yeni") is True
    assert stub_manager.stop("taksim_yeni") is False
    assert stub_manager.frame("taksim_yeni") is None
