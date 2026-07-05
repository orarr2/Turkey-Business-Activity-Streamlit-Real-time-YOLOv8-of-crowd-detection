"""Rolling-window anomaly engine (median + MAD robust z).

Run from src/:  python -m pytest tests -q
"""
import pytest

from app.collector import AnomalyTracker, robust_stats


def make_tracker(**kw):
    # confirm_samples=1 keeps the single-push legacy tests readable; the
    # confirmation rule has its own dedicated tests below.
    base = dict(metric="person", window=30, warmup=10,
                z_spike=3.5, z_drop=3.0, min_value=5, min_delta=5.0,
                drop_min_baseline=8.0, cooldown_sec=300, confirm_samples=1)
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


# ---- operational gates: confirmation + scene-relative delta -----------------

def test_one_sample_blip_needs_confirmation():
    """A single abnormal sample (bus unloading, decode glitch) must NOT flag;
    the same level persisting on the next sample must."""
    t = make_tracker(confirm_samples=2)
    feed(t, "s", [5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 5])
    flagged, dbg = t.push_and_check("s", 25)
    assert not flagged
    assert dbg["reason"] == "pending_confirmation"
    assert dbg["candidate_kind"] == "spike"
    flagged, dbg = t.push_and_check("s", 24)
    assert flagged, dbg
    assert dbg["kind"] == "spike"


def test_transient_blip_never_flags():
    """Spike for one sample, back to normal - the pending candidate dies."""
    t = make_tracker(confirm_samples=2)
    feed(t, "s", [5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 5])
    flagged, _ = t.push_and_check("s", 25)   # blip
    assert not flagged
    flagged, dbg = t.push_and_check("s", 5)  # gone
    assert not flagged
    # a fresh blip later still needs its own confirmation
    flagged, dbg = t.push_and_check("s", 25)
    assert not flagged
    assert dbg["reason"] == "pending_confirmation"
    assert dbg["streak"] == 1


def test_confirmation_requires_same_kind():
    """A drop candidate does not confirm a spike candidate."""
    t = make_tracker(confirm_samples=2, drop_min_baseline=5.0, min_value=5)
    feed(t, "s", [10, 11, 10, 9, 10, 11, 10, 9, 10, 11, 10, 10])
    flagged, dbg = t.push_and_check("s", 30)   # spike candidate (streak 1)
    assert not flagged and dbg["candidate_kind"] == "spike"
    flagged, dbg = t.push_and_check("s", 0)    # drop candidate resets streak
    assert not flagged
    assert dbg["reason"] == "pending_confirmation"
    assert dbg["candidate_kind"] == "drop"
    assert dbg["streak"] == 1


def test_relative_delta_scales_with_scene_level():
    """On a median-10 street the effective floor is max(5, 0.8*10) = 8:
    a +6 move (which cleared the old absolute-only gates) stays silent,
    a +9 move flags."""
    t = make_tracker(rel_delta=0.8, min_delta=5.0, min_value=5,
                     mad_floor=1.0, confirm_samples=1)
    feed(t, "s", [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    flagged, dbg = t.push_and_check("s", 16)   # delta 6 < 0.8*10
    assert not flagged
    assert dbg["min_delta_effective"] == 8.0
    flagged, dbg = t.push_and_check("s", 19)   # delta 9 >= 8
    assert flagged, dbg


def test_mad_floor_damps_static_scene():
    """Constant window -> MAD 0. With the 2.0 floor a +7 move on a static
    scene needs z = 7/2 = 3.5 - exactly at threshold - instead of z=7."""
    t = make_tracker(mad_floor=2.0, min_value=5, min_delta=5.0,
                     rel_delta=0.0, confirm_samples=1)
    feed(t, "s", [3] * 12)
    flagged, dbg = t.push_and_check("s", 9)    # z = 6/2 = 3.0 < 3.5
    assert not flagged
    flagged, dbg = t.push_and_check("s", 10)   # z = 7/2 = 3.5 >= 3.5
    assert flagged, dbg


def test_windows_are_independent_per_key():
    """slot|cam keys: a fallback swap gets a fresh warmup instead of scoring
    the quiet cam against the busy cam's window."""
    t = make_tracker(confirm_samples=1)
    feed(t, "slot_a|busy_cam", [18, 20, 19, 18, 20, 19, 18, 20, 19, 18, 20, 19])
    # swap to the quiet cam: same slot, different key -> warmup, no drop storm
    flagged, dbg = t.push_and_check("slot_a|quiet_cam", 2)
    assert not flagged
    assert dbg["reason"] == "warmup"


def test_stale_window_is_cleared_not_scored(monkeypatch):
    """A window unfed for hours (slot was on a fallback, stream was down)
    describes another regime: on the next push it must re-warm, not fire a
    fake spike/drop against the stale baseline."""
    import app.collector as collector
    now = [1_000.0]
    monkeypatch.setattr(collector.time, "time", lambda: now[0])
    t = make_tracker(confirm_samples=1, stale_after_sec=600)
    # quiet 05:00 baseline
    for v in [2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 2]:
        now[0] += 40
        t.push_and_check("slot|primary", v)
    # slot spends 6h on a fallback, primary recovers at a busy hour
    now[0] += 6 * 3600
    flagged, dbg = t.push_and_check("slot|primary", 15)
    assert not flagged
    assert dbg["reason"] == "warmup"
    assert dbg.get("window_was_stale")


def test_fresh_window_not_treated_as_stale(monkeypatch):
    import app.collector as collector
    now = [1_000.0]
    monkeypatch.setattr(collector.time, "time", lambda: now[0])
    t = make_tracker(confirm_samples=1, stale_after_sec=600)
    for v in [5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 5]:
        now[0] += 40
        t.push_and_check("k", v)
    now[0] += 40
    flagged, dbg = t.push_and_check("k", 25)
    assert flagged, dbg
