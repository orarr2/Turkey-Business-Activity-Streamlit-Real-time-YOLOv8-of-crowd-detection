"""User-drawn counting-line override + crossing event log.

Round-trip: save_line writes a JSON the collector's resolve_line picks up
(beating the CAMERAS entry). Malformed input is rejected. The event log
grows bounded and returns newest-first."""
import json
from pathlib import Path

import pytest

from app import cameras, live_analysis


@pytest.fixture(autouse=True)
def _isolate_lines_dir(tmp_path, monkeypatch):
    """Send every write for this test to a fresh tmp dir - no state leaks
    between tests and no touching the real data/lines/."""
    d = tmp_path / "lines"
    monkeypatch.setattr(cameras, "_lines_dir", lambda: d)
    # crossings log lives elsewhere but pin it too so tests are hermetic.
    c = tmp_path / "crossings"
    monkeypatch.setattr(live_analysis, "_crossings_dir", lambda: c)
    return d


def test_resolve_line_falls_back_to_cameras_catalog(monkeypatch):
    """No override file -> whatever CAMERAS[cam]['line'] is."""
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    assert cameras.resolve_line("camA") == [[0.1, 0.5], [0.9, 0.5]]
    assert cameras.resolve_line("camB_unknown") is None


def test_save_line_persists_and_overrides(monkeypatch):
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    cameras.save_line("camA", [[0.2, 0.4], [0.8, 0.6]])
    assert cameras.resolve_line("camA") == [[0.2, 0.4], [0.8, 0.6]]


def test_save_line_rejects_bad_shape():
    """Every failure mode we can express in the spec must 400 out cleanly
    - one point, three points, out-of-range coords, non-numeric."""
    for bad in [
        None,
        [],
        [[0.5, 0.5]],                            # one point
        [[0.1, 0.5], [0.9, 0.5], [0.5, 0.5]],   # three points
        [[1.5, 0.5], [0.9, 0.5]],                # x > 1
        [[-0.1, 0.5], [0.9, 0.5]],               # x < 0
        [["a", "b"], [0.9, 0.5]],                # non-numeric
    ]:
        with pytest.raises(ValueError):
            cameras.save_line("camX", bad)


def test_clear_line_returns_to_catalog(monkeypatch):
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    cameras.save_line("camA", [[0.2, 0.4], [0.8, 0.6]])
    assert cameras.resolve_line("camA") == [[0.2, 0.4], [0.8, 0.6]]
    assert cameras.clear_line("camA") is True
    assert cameras.resolve_line("camA") == [[0.1, 0.5], [0.9, 0.5]]
    # Idempotent: clearing again just says False.
    assert cameras.clear_line("camA") is False


def test_malformed_override_falls_back_silently(monkeypatch):
    """A garbage override file must degrade to the CAMERAS default rather
    than crashing the collector on the next round."""
    monkeypatch.setitem(cameras.CAMERAS, "camA",
                        {"line": [[0.1, 0.5], [0.9, 0.5]], "country": "turkey"})
    d = cameras._lines_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "camA.json").write_text('{"line": "not-a-list"}')
    assert cameras.resolve_line("camA") == [[0.1, 0.5], [0.9, 0.5]]


def test_crossing_event_log_bounded_and_newest_first():
    class _T:
        def __init__(self, tid, cls="person"):
            self.tid = tid; self.cls = cls
            self.boxes = [{"x1": 10, "y1": 20, "x2": 40, "y2": 80}]
    # Write more than CROSSING_LOG_KEEP events; verify bound + order.
    for i in range(live_analysis.CROSSING_LOG_KEEP + 30):
        live_analysis.log_crossing_event(
            "camA", "in" if i % 2 == 0 else "out", _T(tid=i), frame=None)
    events = live_analysis.read_crossing_events("camA", limit=200)
    assert len(events) == live_analysis.CROSSING_LOG_KEEP
    # Newest-first: the very last logged is at index 0.
    assert events[0]["tid"] == live_analysis.CROSSING_LOG_KEEP + 30 - 1


def test_read_events_missing_returns_empty():
    assert live_analysis.read_crossing_events("nobody", limit=10) == []
