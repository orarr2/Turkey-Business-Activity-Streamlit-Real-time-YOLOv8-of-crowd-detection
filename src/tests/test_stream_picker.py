"""SlotStreamPicker fallback behaviour.

Pins the 2026-07-16 Konya outage lessons:
  * a dead prefix is walked one step per `max_failures` misses;
  * the periodic primary probe, when it misses, snaps straight back to the
    fallback that was delivering frames (one lost sample) instead of
    re-walking the whole dead prefix (which cost ~10 minutes of MISS out of
    every 15 during the outage).
"""
import time

from app.collector import SlotStreamPicker

SLOT = {
    "slot_id": "slot_test",
    "display_area": "Test Area",
    "primary": "cam_a",
    "fallbacks": ["cam_b", "cam_c", "cam_d"],
}


def make_picker(**kw):
    kw.setdefault("max_failures", 3)
    kw.setdefault("retry_minutes", 15)
    return SlotStreamPicker(SLOT, **kw)


def miss_times(picker, n):
    changed = None
    for _ in range(n):
        picker.current_cam()
        changed = picker.record_result(False)
    return changed


def test_starts_on_primary():
    p = make_picker()
    assert p.current_cam() == "cam_a"


def test_advances_one_step_after_max_failures():
    p = make_picker()
    assert miss_times(p, 2) is None          # 2 misses: still on primary
    assert p.current_cam() == "cam_a"
    assert miss_times(p, 1) == "cam_b"       # 3rd miss: advance
    assert p.current_cam() == "cam_b"


def test_walks_whole_dead_prefix_to_live_tail():
    p = make_picker()
    miss_times(p, 3)                          # a -> b
    miss_times(p, 3)                          # b -> c
    miss_times(p, 3)                          # c -> d
    assert p.current_cam() == "cam_d"
    p.record_result(True)                     # d delivers frames
    assert p.last_good_idx == 3


def test_success_pins_current_camera():
    p = make_picker()
    miss_times(p, 3)
    assert p.current_cam() == "cam_b"
    p.record_result(True)
    for _ in range(10):
        assert p.current_cam() == "cam_b"
        p.record_result(True)


def test_stays_on_last_link_when_whole_chain_dead():
    p = make_picker()
    miss_times(p, 50)
    assert p.current_cam() == "cam_d"         # clamped at the tail


def test_periodic_probe_retries_primary():
    p = make_picker()
    miss_times(p, 3)                          # on cam_b
    p.record_result(True)
    p.last_primary_check = time.time() - p.retry_seconds - 1
    assert p.current_cam() == "cam_a"         # probe fires


def test_probe_miss_snaps_back_to_last_good_fallback():
    """THE outage fix: primary probe misses once -> back to the working
    fallback immediately, not a 3-miss walk through every dead link."""
    p = make_picker()
    miss_times(p, 3)                          # a -> b (a is dead)
    miss_times(p, 3)                          # b -> c (b is dead)
    p.current_cam()
    p.record_result(True)                     # c delivers frames
    p.last_primary_check = time.time() - p.retry_seconds - 1
    assert p.current_cam() == "cam_a"         # probe primary
    changed = p.record_result(False)          # primary still dead: ONE miss
    assert changed == "cam_c"                 # snapped straight back
    assert p.current_cam() == "cam_c"
    p.record_result(True)                     # and it still works


def test_probe_success_stays_on_primary():
    p = make_picker()
    miss_times(p, 3)
    p.current_cam()
    p.record_result(True)                     # cam_b good
    p.last_primary_check = time.time() - p.retry_seconds - 1
    assert p.current_cam() == "cam_a"
    p.record_result(True)                     # primary recovered
    assert p.current_cam() == "cam_a"         # stay on primary
    assert p.last_good_idx == 0


def test_no_snap_back_without_known_good_fallback():
    """Fresh start, nothing ever worked: probe miss must keep walking the
    chain (there is nowhere better to snap back to)."""
    p = make_picker()
    miss_times(p, 3)                          # a -> b, nothing good yet
    p.last_primary_check = time.time() - p.retry_seconds - 1
    assert p.current_cam() == "cam_a"         # probe resets to primary
    p.record_result(False)                    # 1 miss - no snap-back target
    assert p.current_cam() == "cam_a"         # still walking normally
    miss_times(p, 2)                          # complete the 3 misses
    assert p.current_cam() == "cam_b"
