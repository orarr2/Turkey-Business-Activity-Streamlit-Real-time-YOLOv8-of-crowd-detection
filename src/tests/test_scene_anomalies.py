"""Operator-defined anomalies, night gates, entity gallery, plain header."""
import numpy as np
import pytest

import app.collector as collector
from app.collector import check_scene_anomalies, weighted_vehicle_load
from app.detect_core import night_adjusted_conf, DEFAULT_PER_CLASS_CONF
from app.model_metrics import header_line


@pytest.fixture(autouse=True)
def _fresh_cooldowns():
    collector._SCENE_ANOMALY_LAST.clear()
    collector._LAST_LUMA.clear()
    yield


def test_night_bump_raises_every_gate_clamped():
    g = night_adjusted_conf(DEFAULT_PER_CLASS_CONF)
    for cls, v in DEFAULT_PER_CLASS_CONF.items():
        assert g[cls] == pytest.approx(min(0.8, v + 0.08))
    assert night_adjusted_conf({"car": 0.79})["car"] == 0.8


def test_weighted_vehicle_load():
    assert weighted_vehicle_load({"car": 2, "bus": 1}) == pytest.approx(4.5)
    assert weighted_vehicle_load({}) == 0.0


def test_extreme_load_fires_with_cooldown():
    counts = {"person": 60}
    v1 = check_scene_anomalies("camA", counts, [], (720, 1280), 120.0, now=1000.0)
    assert v1 and v1[0]["kind"] == "extreme_load" and v1[0]["metric"] == "person"
    # inside cooldown -> silent
    assert check_scene_anomalies("camA", counts, [], (720, 1280), 120.0,
                                 now=1100.0) == []
    # past cooldown -> fires again
    assert check_scene_anomalies("camA", counts, [], (720, 1280), 120.0,
                                 now=1000.0 + 1801)[0]["kind"] == "extreme_load"


def test_vehicle_load_extreme():
    counts = {"person": 2, "bus": 10, "truck": 6}   # load 40 >= 38
    v = check_scene_anomalies("camB", counts, [], (720, 1280), 120.0, now=5.0)
    assert v and v[0]["metric"] == "vehicles"


def test_obstruction_giant_box():
    big = [{"x1": 0, "y1": 0, "x2": 1000, "y2": 500, "cls": "bus"}]  # 54% of frame
    v = check_scene_anomalies("camC", {"person": 0}, big, (720, 1280), 120.0,
                              now=7.0)
    assert v and v[0]["kind"] == "camera_obstructed"
    small = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "cls": "car"}]
    assert check_scene_anomalies("camD", {"person": 0}, small, (720, 1280),
                                 120.0, now=9.0) == []


def test_camera_dark_transition_only():
    # first sample just records the baseline
    assert check_scene_anomalies("camE", {}, [], (720, 1280), 120.0, now=1.0) == []
    # bright -> near-black = alarm
    v = check_scene_anomalies("camE", {}, [], (720, 1280), 10.0, now=2.0)
    assert v and v[0]["kind"] == "camera_dark"
    # night staying dark is NOT an alarm
    assert check_scene_anomalies("camE", {}, [], (720, 1280), 8.0, now=3.0) == []


def test_entity_gallery_caps(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from app import entity_gallery as eg
    monkeypatch.setattr(eg, "PER_ENTITY_MIN_GAP_S", 0.0)
    monkeypatch.setattr(eg, "PER_ENTITY_CAP", 3)
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    box = {"x1": 10, "y1": 10, "x2": 100, "y2": 100}
    for _ in range(5):
        assert eg.save_sighting("camA", 7, frame, box, tmp_path)
    crops = list((tmp_path / "entities" / "camA" / "7").glob("*.jpg"))
    assert len(crops) == 3                    # per-entity cap enforced
    items = eg.list_sightings("camA", 7, tmp_path)
    assert len(items) == 3 and items[0]["url"].startswith("/snapshots/entities/")


def test_header_line_plain_language():
    m = {"tp": 3, "fp": 0, "fn": 21, "n_precision": 3, "n_recall": 24,
         "accuracy": 1.0, "recall": 0.125}
    line = header_line(m, {"adjusted_cls": 6, "updated_at": ""})
    assert "right on 3 of 3 boxes you checked" in line
    assert "21 objects it missed" in line
    assert "learning is ON" in line and "6 detection thresholds" in line
    assert "precision pending" not in line and "recall" not in line
    empty = header_line({"tp": 0, "fp": 0, "fn": 0,
                         "n_precision": 0, "n_recall": 0})
    assert "no feedback yet" in empty
