"""CameraPool - the shared priority ladder behind the 4 grid slots.

Pins the operator's spec as revised over 2026-07-16..17:
  Turkey ladder (now IBB-first): taksim -> sultanahmet -> eyup -> beyazit,
  THEN the four Konya cams, THEN the Turkish tail.
Four DISTINCT cameras every round, dead cameras rest on a cooldown,
probation probes cost one sample, an all-dead pool holds STEADY on the
top of the ladder (no tile churn), and forced samples of resting cameras
never push their recovery further out.

tvkur (Konya) cams are fast-fail probes - ONE miss rests them even on
first contact, and an all-tvkur miss round is exempt from the politeness
backoff.

The pool itself is country-agnostic (it walks whatever ordered list it is
given); these tests drive it with Turkey's ladder via FALLBACK_POOL.
"""
import time

from app.cameras import FALLBACK_POOL, GRID_SLOTS, country_pool
from app.collector import CameraPool

# Turkey ladder, new order: IBB four on top, Konya four below.
IBB4 = ["taksim_yeni", "sultanahmet_1_yeni", "eyup_sultan_yeni",
        "beyazit_meydan_yeni"]
KONYA = ["konya_hukumet", "otogar_kavsagi", "konya_kulturpark",
         "konya_millet_caddesi"]


def make_pool(**kw):
    kw.setdefault("max_failures", 3)
    kw.setdefault("retry_minutes", 15)
    return CameraPool(FALLBACK_POOL, n_slots=4, **kw)


def kill(pool, cam, now):
    """Feed enough misses to rest the camera."""
    for _ in range(pool.max_failures):
        pool.record(cam, False, now=now)


def test_pool_layout_matches_operator_spec():
    assert FALLBACK_POOL[:4] == IBB4                  # IBB first now
    assert FALLBACK_POOL[4:8] == KONYA
    assert FALLBACK_POOL[8] == "buyuk_camlica_yeni"   # Turkish tail head
    assert len(FALLBACK_POOL) == len(set(FALLBACK_POOL))
    assert country_pool("turkey") == FALLBACK_POOL
    for s in GRID_SLOTS:
        chain = [s["primary"], *s["fallbacks"]]
        assert set(chain) == set(FALLBACK_POOL)


def test_all_healthy_assigns_the_four_ibb():
    pool = make_pool()
    assert pool.assign(now=1000) == IBB4


def test_assignment_always_distinct():
    pool = make_pool()
    now = 1000
    for _ in range(60):
        cams = pool.assign(now=now)
        assert len(cams) == 4 and len(set(cams)) == 4
        for c in cams:
            pool.record(c, False, now=now)
        now += 40


def test_dead_ibb_promotes_the_konya_four_in_order():
    pool = make_pool()
    now = 1000
    for cam in IBB4:
        kill(pool, cam, now)
    assert pool.assign(now=now) == KONYA


def test_partial_ibb_outage_mixes_tiers_in_priority_order():
    pool = make_pool()
    now = 1000
    kill(pool, "sultanahmet_1_yeni", now)
    kill(pool, "beyazit_meydan_yeni", now)
    assert pool.assign(now=now) == [
        "taksim_yeni", "eyup_sultan_yeni",
        "konya_hukumet", "otogar_kavsagi",
    ]


def test_dead_top_eight_keeps_walking_the_catalog():
    pool = make_pool()
    now = 1000
    for cam in IBB4 + KONYA:
        kill(pool, cam, now)
    got = pool.assign(now=now)
    assert got == FALLBACK_POOL[8:12]   # buyuk_camlica, ince_minareli, ...
    assert len(set(got)) == 4


def test_everything_dead_holds_steady_on_top_of_ladder():
    """All-dead pool: the padding is deterministic (pool priority order),
    so the grid stops churning tiles and sits on the four IBB cams."""
    pool = make_pool()
    now = 1000
    for cam in FALLBACK_POOL:
        kill(pool, cam, now)
    first = pool.assign(now=now + 1)
    assert first == IBB4
    t = now + 1
    for _ in range(10):
        cams = pool.assign(now=t)
        assert cams == IBB4
        for c in cams:
            pool.record(c, False, now=t)
        t += 40


def test_forced_sample_of_resting_camera_does_not_extend_cooldown():
    pool = make_pool(retry_minutes=15)
    now = 1000
    kill(pool, "taksim_yeni", now)
    rest_until = pool.cooldown_until["taksim_yeni"]
    pool.record("taksim_yeni", False, now=now + 40)       # forced padding miss
    assert pool.cooldown_until["taksim_yeni"] == rest_until


def test_cooldown_expiry_reprobes_higher_priority():
    pool = make_pool(retry_minutes=15)
    now = 1000
    for cam in IBB4:
        kill(pool, cam, now)
    assert pool.assign(now=now) == KONYA
    later = now + 15 * 60 + 1
    assert pool.assign(now=later)[0] == "taksim_yeni"


def test_probation_cameras_rest_after_a_single_miss():
    pool = make_pool(retry_minutes=15)
    now = 1000
    for cam in IBB4:
        kill(pool, cam, now)
    for cam in KONYA:
        pool.record(cam, True, now=now)
    later = now + 15 * 60 + 1
    probe = pool.assign(now=later)
    assert probe == IBB4
    for cam in probe:
        pool.record(cam, False, now=later)    # still dead: ONE miss each
    assert pool.assign(now=later + 1) == KONYA


def test_recovered_camera_is_fully_rehabilitated():
    pool = make_pool()
    now = 1000
    kill(pool, "taksim_yeni", now)
    later = now + pool.retry_seconds + 1
    pool.record("taksim_yeni", True, now=later)
    assert pool.assign(now=later + 1)[0] == "taksim_yeni"
    pool.record("taksim_yeni", False, now=later + 2)
    assert pool.assign(now=later + 3)[0] == "taksim_yeni"


def test_record_ignores_unknown_camera():
    pool = make_pool()
    pool.record("not_in_pool", False, now=1000)
    assert pool.assign(now=1000) == IBB4


def test_fast_fail_cameras_rest_after_one_miss():
    """tvkur/Konya fast-fail: one miss and the camera is out this round."""
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    for cam in IBB4:                     # clear the IBB tier out of the way
        kill(pool, cam, now)
    assert pool.assign(now=now) == KONYA
    for cam in KONYA:
        pool.record(cam, False, now=now)      # ONE miss each
    # Konya rests immediately; ladder walks to the Turkish tail.
    assert pool.assign(now=now + 1) == FALLBACK_POOL[8:12]


def test_fast_fail_does_not_touch_other_tiers():
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    # One miss on an IBB (non-fast-fail) cam must NOT rest it.
    pool.record("taksim_yeni", False, now=now)
    assert pool.assign(now=now)[0] == "taksim_yeni"


def test_recovered_fast_fail_cam_is_still_one_strike():
    """Asserted on cooldown state directly so it doesn't depend on where
    Konya sits in the ladder: one miss rests a fast-fail cam, a success
    fully revives it, and the next single miss rests it again."""
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    pool.record("konya_hukumet", False, now=now)          # rests immediately
    assert pool.cooldown_until["konya_hukumet"] > now
    later = now + pool.retry_seconds + 1
    pool.record("konya_hukumet", True, now=later)         # revived
    assert pool.cooldown_until["konya_hukumet"] == 0.0
    pool.record("konya_hukumet", False, now=later + 2)    # ONE miss again
    assert pool.cooldown_until["konya_hukumet"] > later + 2


def test_all_fast_fail_round_detection():
    pool = make_pool(fast_fail=KONYA)
    assert pool.all_fast_fail(KONYA)
    assert not pool.all_fast_fail(KONYA[:3] + ["taksim_yeni"])
    assert not pool.all_fast_fail([])
