"""Hour-of-week contextual baseline (Welford profile).

Run from src/:  python -m pytest tests -q
"""
import datetime as dt
import math
import random

import pytest

from app.collector import ANOMALY_METRICS, HourlyProfile

UTC = dt.timezone.utc
# Wednesday 10:30 UTC -> 13:30 Turkey local (UTC+3): dow=2, hour=13.
WED = dt.datetime(2026, 7, 1, 10, 30, tzinfo=UTC)


def gates():
    g = ANOMALY_METRICS["person"]
    return dict(min_delta=g["min_delta"], drop_min_baseline=g["drop_min_baseline"])


def test_bucket_uses_turkey_local_time():
    bucket, label = HourlyProfile.bucket_of(WED)
    assert bucket == "2_13"
    assert label == "Wed 13:00"


def test_welford_matches_direct_mean_std():
    p = HourlyProfile()
    rng = random.Random(7)
    vals = [rng.uniform(0, 20) for _ in range(200)]
    for v in vals:
        p.update("s", "person", WED, v)
    _, _, n, mean, std = p.stats("s", "person", WED)
    assert n == 200
    assert mean == pytest.approx(sum(vals) / 200)
    direct = math.sqrt(sum((v - mean) ** 2 for v in vals) / 200)
    assert std == pytest.approx(direct, rel=1e-9)


def test_contextual_spike_and_drop():
    p = HourlyProfile(min_samples=10, cooldown_sec=0)
    for v in [11, 12, 13, 12, 11, 12, 13, 12, 11, 12, 13, 12]:
        p.update("s", "person", WED, v)
    flagged, dbg = p.check("s", "person", WED, 30, **gates())
    assert flagged, dbg
    assert dbg["kind"] == "contextual_spike"
    flagged, dbg = p.check("s", "person", WED, 1, **gates())
    assert flagged, dbg
    assert dbg["kind"] == "contextual_drop"
    flagged, _ = p.check("s", "person", WED, 13, **gates())
    assert not flagged


def test_drop_needs_busy_bucket():
    p = HourlyProfile(min_samples=10, cooldown_sec=0)
    for _ in range(15):
        p.update("s", "person", WED, 3)   # mean 3 < drop_min_baseline 8
    flagged, _ = p.check("s", "person", WED, 0, **gates())
    assert not flagged


def test_bucket_warmup_gate():
    p = HourlyProfile(min_samples=10)
    for _ in range(9):
        p.update("s", "person", WED, 12)
    flagged, dbg = p.check("s", "person", WED, 40, **gates())
    assert not flagged
    assert dbg["reason"] == "bucket_warmup"


def test_other_hour_bucket_is_independent():
    p = HourlyProfile(min_samples=10)
    for _ in range(20):
        p.update("s", "person", WED, 12)
    thu = WED + dt.timedelta(days=1)
    flagged, dbg = p.check("s", "person", thu, 30, **gates())
    assert not flagged
    assert dbg["reason"] == "bucket_warmup"


def test_contextual_cooldown():
    p = HourlyProfile(min_samples=10, cooldown_sec=1800)
    for _ in range(12):
        p.update("s", "person", WED, 12)
    flagged, _ = p.check("s", "person", WED, 40, **gates())
    assert flagged
    flagged2, dbg2 = p.check("s", "person", WED, 45, **gates())
    assert not flagged2
    assert dbg2["reason"] == "cooldown"
    assert dbg2["suppressed_kind"] == "contextual_spike"


def test_payload_round_trip():
    p = HourlyProfile()
    for v in [3, 4, 5, 6, 3, 4]:
        p.update("s", "person", WED, v)
        p.update("s", "vehicles", WED, v * 2)
    payload = p.to_payload("s")
    q = HourlyProfile()
    loaded = q.load_payload("s", payload)
    assert loaded == 2   # 2 metrics x 1 bucket
    assert q.stats("s", "person", WED) == p.stats("s", "person", WED)
    assert q.stats("s", "vehicles", WED) == p.stats("s", "vehicles", WED)


def test_load_payload_tolerates_junk():
    q = HourlyProfile()
    loaded = q.load_payload("s", {"metrics": {"person": {
        "2_13": {"n": 5, "mean": 4.0, "m2": 2.0},
        "bad":  {"n": "x"},
        "2_14": None,
    }}})
    assert loaded == 1
