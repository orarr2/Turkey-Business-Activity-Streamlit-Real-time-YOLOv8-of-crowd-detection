"""Rolling-window anomaly engine (median + MAD robust z).

Run from src/:  python -m pytest tests -q
"""
import pytest

from app.collector import AnomalyTracker, robust_stats


def make_tracker(**kw):
    base = dict(metric="person", window=30, warmup=10,
                z_spike=3.5, z_drop=3.0, min_value=5, min_delta=5.0,
                drop_min_baseline=8.0, cooldown_sec=300)
    base.update(kw)
    return AnomalyTracker(**base)


def feed(tracker, key, values):
    last = (False, {})
    for v in values:
        last = tracker.push_and_check(key, v)
    return last


def test_robust_stats_ignores_outlier():
    med, spread = robust_stats([1, 2, 3, 4, 100])
    assert med == 3
    assert spread == pytest.approx(1.4826, rel=1e-6)


def test_warmup_never_flags():
    t = make_tracker()
    for v in [50, 0, 50, 0, 50, 0, 50, 0, 50]:
        flagged, dbg = t.push_and_check("s", v)
        assert not flagged
        assert dbg["reason"] == "warmup"


def test_spike_flagged_on_steady_baseline():
    t = make_tracker()
    feed(t, "s", [5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 5, 6, 4, 5])
    flagged, dbg = t.push_and_check("s", 25)
    assert flagged, dbg
    assert dbg["kind"] == "spike"


def test_outliers_in_window_do_not_mask_next_spike():
    # Two 50s contaminate the window. With mean/std the spread inflates to
    # ~11 and z(25) ~ 1.5 - the event is silently missed. Median/MAD keeps the
    # baseline at 5 and still flags it.
    t = make_tracker(cooldown_sec=0)
    feed(t, "s", [5, 6, 5, 4, 5, 50, 6, 5, 4, 5, 50, 5, 6, 5, 4, 5, 6, 5])
    flagged, dbg = t.push_and_check("s", 25)
    assert flagged, dbg
    assert dbg["kind"] == "spike"
    assert dbg["median"] == 5


def test_small_crowd_below_min_value_not_flagged():
    t = make_tracker()
    feed(t, "s", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    flagged, _ = t.push_and_check("s", 4)   # below min_value=5
    assert not flagged


def test_drop_flagged_only_below_busy_baseline():
    t = make_tracker()
    feed(t, "busy", [12, 13, 12, 11, 12, 13, 12, 11, 12, 13, 12, 12])
    flagged, dbg = t.push_and_check("busy", 0)
    assert flagged, dbg
    assert dbg["kind"] == "drop"

    t2 = make_tracker()
    feed(t2, "quiet", [2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 2])
    flagged, _ = t2.push_and_check("quiet", 0)   # baseline < 8 -> silent
    assert not flagged


def test_cooldown_suppresses_repeat():
    t = make_tracker(cooldown_sec=300)
    feed(t, "s", [5] * 12)
    flagged, _ = t.push_and_check("s", 25)
    assert flagged
    flagged2, dbg2 = t.push_and_check("s", 30)
    assert not flagged2
    assert dbg2["reason"] == "cooldown"
    assert dbg2["suppressed_kind"] == "spike"


def test_seed_skips_warmup():
    t = make_tracker()
    kept = t.seed("s", [5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 5])
    assert kept == 12
    flagged, dbg = t.push_and_check("s", 25)
    assert flagged, dbg
    assert dbg["kind"] == "spike"


def test_seed_drops_nones_and_caps_to_window():
    t = make_tracker(window=5)
    kept = t.seed("s", [1, None, 2, 3, None, 4, 5, 6, 7])
    assert kept == 5   # window cap; Nones (missed samples) ignored


def test_none_sample_is_a_noop():
    t = make_tracker()
    flagged, dbg = t.push_and_check("s", None)
    assert not flagged
    assert dbg["reason"] == "no_sample"


def test_legacy_kwargs_still_accepted():
    t = AnomalyTracker(window=30, z_threshold=4.0, min_people=6,
                       min_delta=5.0, min_std=0.0, spike_only=True,
                       cooldown_sec=300)
    assert t.z_spike == 4.0
    assert t.min_value == 6
