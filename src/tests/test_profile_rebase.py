"""HourlyProfile self-healing rebase: a detector regime change (counts jump
because thresholds got tuned looser) must re-converge the baseline in ~a day
instead of crying "hourly spike" for weeks.
"""
import datetime as dt
import math

import app.collector as collector
from app.collector import HourlyProfile

TS = dt.datetime(2026, 7, 10, 9, 30, tzinfo=dt.timezone.utc)
BUCKET = HourlyProfile.bucket_of(TS)[0]


def _seeded_profile(**kw):
    prof = HourlyProfile(min_samples=30, cooldown_sec=0, **kw)
    prof.load_payload("cam", {"metrics": {"vehicles": {
        BUCKET: {"n": 120, "mean": 2.5, "m2": 120 * 0.25},   # std = 0.5
    }}})
    return prof


def _fake_clock(monkeypatch, start=1_000.0):
    t = [start]
    monkeypatch.setattr(collector.time, "time", lambda: t[0])
    return t


def test_sustained_spikes_trigger_one_rebase(monkeypatch):
    t = _fake_clock(monkeypatch)
    prof = _seeded_profile()
    rebases = 0
    for _ in range(HourlyProfile.REBASE_AFTER + 3):
        t[0] += 60
        _, dbg = prof.check("cam", "vehicles", TS, 12.0,
                            min_delta=3, drop_min_baseline=5)
        rebases += bool(dbg.get("rebased"))
    assert rebases == 1
    cell = prof._slots["cam"]["vehicles"][BUCKET]
    assert cell[0] == HourlyProfile.REBASE_N
    # std preserved through the n cut (m2 scaled by the same factor)
    assert math.sqrt(cell[2] / cell[0]) == 0.5


def test_rebase_rearms_only_after_window(monkeypatch):
    t = _fake_clock(monkeypatch)
    prof = _seeded_profile()
    rebases = 0
    # two waves of spikes inside the same 24h window -> still one rebase
    for _ in range(HourlyProfile.REBASE_AFTER * 3):
        t[0] += 60
        _, dbg = prof.check("cam", "vehicles", TS, 12.0,
                            min_delta=3, drop_min_baseline=5)
        rebases += bool(dbg.get("rebased"))
    assert rebases == 1
    # past the window, sustained disagreement may rebase again
    t[0] += HourlyProfile.REBASE_WINDOW_SEC + 1
    for _ in range(HourlyProfile.REBASE_AFTER):
        t[0] += 60
        _, dbg = prof.check("cam", "vehicles", TS, 12.0,
                            min_delta=3, drop_min_baseline=5)
        rebases += bool(dbg.get("rebased"))
    assert rebases == 2


def test_occasional_spikes_do_not_rebase(monkeypatch):
    t = _fake_clock(monkeypatch)
    prof = _seeded_profile()
    for _ in range(HourlyProfile.REBASE_AFTER - 1):   # below the bar
        t[0] += 3600
        _, dbg = prof.check("cam", "vehicles", TS, 12.0,
                            min_delta=3, drop_min_baseline=5)
        assert not dbg.get("rebased")
    assert prof._slots["cam"]["vehicles"][BUCKET][0] == 120


def test_rebase_accelerates_convergence(monkeypatch):
    t = _fake_clock(monkeypatch)
    rebased = _seeded_profile()
    control = _seeded_profile()
    # trigger the rebase on one profile only
    for _ in range(HourlyProfile.REBASE_AFTER):
        t[0] += 60
        rebased.check("cam", "vehicles", TS, 12.0,
                      min_delta=3, drop_min_baseline=5)
    # both now learn the same new regime (12 vehicles every sample)
    for _ in range(60):
        rebased.update("cam", "vehicles", TS, 12.0)
        control.update("cam", "vehicles", TS, 12.0)
    _, _, _, mean_r, _ = rebased.stats("cam", "vehicles", TS)
    _, _, _, mean_c, _ = control.stats("cam", "vehicles", TS)
    assert mean_r > mean_c + 1.0     # markedly faster toward 12
    assert mean_r > 5.0
