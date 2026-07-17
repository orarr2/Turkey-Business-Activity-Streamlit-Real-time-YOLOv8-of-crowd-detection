"""CameraPool - the shared priority ladder behind the 4 grid slots.

Pins the operator's spec (2026-07-16, revised same day):
  tier 1: four Konya cams;
  tier 2: taksim -> sultanahmet -> eyup sultan -> beyazit meydani;
  tier 3: rest of the catalog, walked until a camera delivers.
Four DISTINCT cameras every round, dead cameras rest on a cooldown,
probation probes cost one sample, an all-dead pool holds STEADY on the
top of the ladder (no tile churn), and forced samples of resting cameras
never push their recovery further out.

2026-07-17 revision: tvkur (Konya) cams are fast-fail probes - ONE miss
rests them even on first contact, and an all-tvkur miss round is exempt
from the politeness backoff, so the whole Konya sweep costs a single
round (well under the operator's 2-minute ceiling) before the ladder
reaches the Istanbul tier.
"""
import time

from app.cameras import FALLBACK_POOL, GRID_SLOTS
from app.collector import CameraPool

KONYA = ["konya_hukumet", "otogar_kavsagi", "konya_kulturpark",
         "konya_millet_caddesi"]
IBB4 = ["taksim_yeni", "sultanahmet_1_yeni", "eyup_sultan_yeni",
        "beyazit_meydan_yeni"]


def make_pool(**kw):
    kw.setdefault("max_failures", 3)
    kw.setdefault("retry_minutes", 15)
    return CameraPool(FALLBACK_POOL, n_slots=4, **kw)


def kill(pool, cam, now):
    """Feed enough misses to rest the camera."""
    for _ in range(pool.max_failures):
        pool.record(cam, False, now=now)


def test_pool_layout_matches_operator_spec():
    assert FALLBACK_POOL[:4] == KONYA
    assert FALLBACK_POOL[4:8] == IBB4
    assert FALLBACK_POOL[8] == "buyuk_camlica_yeni"   # tier-3 head
    assert len(FALLBACK_POOL) == len(set(FALLBACK_POOL))
    for s in GRID_SLOTS:
        chain = [s["primary"], *s["fallbacks"]]
        assert set(chain) == set(FALLBACK_POOL)


def test_all_healthy_assigns_the_four_konya():
    pool = make_pool()
    assert pool.assign(now=1000) == KONYA


def test_assignment_always_distinct():
    pool = make_pool()
    now = 1000
    for _ in range(60):
        cams = pool.assign(now=now)
        assert len(cams) == 4 and len(set(cams)) == 4
        for c in cams:
            pool.record(c, False, now=now)
        now += 40


def test_dead_konya_promotes_the_ibb_four_in_order():
    pool = make_pool()
    now = 1000
    for cam in KONYA:
        kill(pool, cam, now)
    assert pool.assign(now=now) == IBB4


def test_partial_konya_outage_mixes_tiers_in_priority_order():
    pool = make_pool()
    now = 1000
    kill(pool, "otogar_kavsagi", now)
    kill(pool, "konya_millet_caddesi", now)
    assert pool.assign(now=now) == [
        "konya_hukumet", "konya_kulturpark",
        "taksim_yeni", "sultanahmet_1_yeni",
    ]


def test_dead_ibb_four_keeps_walking_the_catalog():
    pool = make_pool()
    now = 1000
    for cam in KONYA + IBB4:
        kill(pool, cam, now)
    got = pool.assign(now=now)
    assert got == FALLBACK_POOL[8:12]   # buyuk_camlica, ince_minareli, ...
    assert len(set(got)) == 4


def test_everything_dead_holds_steady_on_top_of_ladder():
    """All-dead pool: the padding is deterministic (pool priority order),
    so the grid stops churning tiles and sits on the four Konya cams."""
    pool = make_pool()
    now = 1000
    for cam in FALLBACK_POOL:
        kill(pool, cam, now)
    first = pool.assign(now=now + 1)
    assert first == KONYA
    # ...and it STAYS there round after round.
    t = now + 1
    for _ in range(10):
        cams = pool.assign(now=t)
        assert cams == KONYA
        for c in cams:
            pool.record(c, False, now=t)
        t += 40


def test_forced_sample_of_resting_camera_does_not_extend_cooldown():
    """A padded (still-resting) camera that misses must keep its original
    recovery time - otherwise an all-dead pool slides every cooldown
    forward forever and nothing ever gets re-probed."""
    pool = make_pool(retry_minutes=15)
    now = 1000
    kill(pool, "konya_hukumet", now)
    rest_until = pool.cooldown_until["konya_hukumet"]
    pool.record("konya_hukumet", False, now=now + 40)     # forced padding miss
    assert pool.cooldown_until["konya_hukumet"] == rest_until


def test_cooldown_expiry_reprobes_higher_priority():
    pool = make_pool(retry_minutes=15)
    now = 1000
    for cam in KONYA:
        kill(pool, cam, now)
    assert pool.assign(now=now) == IBB4
    later = now + 15 * 60 + 1
    assert pool.assign(now=later)[0] == "konya_hukumet"


def test_probation_cameras_rest_after_a_single_miss():
    pool = make_pool(retry_minutes=15)
    now = 1000
    for cam in KONYA:
        kill(pool, cam, now)
    for cam in IBB4:
        pool.record(cam, True, now=now)
    later = now + 15 * 60 + 1
    probe = pool.assign(now=later)
    assert probe == KONYA
    for cam in probe:
        pool.record(cam, False, now=later)    # still dead: ONE miss each
    assert pool.assign(now=later + 1) == IBB4


def test_recovered_camera_is_fully_rehabilitated():
    pool = make_pool()
    now = 1000
    kill(pool, "konya_hukumet", now)
    later = now + pool.retry_seconds + 1
    pool.record("konya_hukumet", True, now=later)
    assert pool.assign(now=later + 1)[0] == "konya_hukumet"
    pool.record("konya_hukumet", False, now=later + 2)
    assert pool.assign(now=later + 3)[0] == "konya_hukumet"


def test_record_ignores_unknown_camera():
    pool = make_pool()
    pool.record("not_in_pool", False, now=1000)
    assert pool.assign(now=1000) == KONYA


def test_fast_fail_cameras_rest_after_one_miss():
    """Operator spec 2026-07-17: a dead Konya backend costs ONE round, so
    the very next assignment is already the Istanbul tier."""
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    for cam in KONYA:
        pool.record(cam, False, now=now)
    assert pool.assign(now=now + 1) == IBB4


def test_fast_fail_does_not_touch_other_tiers():
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    for cam in KONYA:
        pool.record(cam, False, now=now)
    # One miss on an IBB cam must NOT rest it - full 3-strike grace stays.
    pool.record("taksim_yeni", False, now=now + 1)
    assert pool.assign(now=now + 2)[0] == "taksim_yeni"


def test_recovered_fast_fail_cam_is_still_one_strike():
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    pool.record("konya_hukumet", False, now=now)          # rests immediately
    later = now + pool.retry_seconds + 1
    pool.record("konya_hukumet", True, now=later)         # revived
    assert pool.assign(now=later + 1)[0] == "konya_hukumet"
    pool.record("konya_hukumet", False, now=later + 2)    # one miss again
    assert pool.assign(now=later + 3)[0] != "konya_hukumet"


def test_all_fast_fail_round_detection():
    """The main loop skips the politeness backoff exactly for rounds made
    of nothing but low-risk tvkur probes."""
    pool = make_pool(fast_fail=KONYA)
    assert pool.all_fast_fail(KONYA)
    assert not pool.all_fast_fail(KONYA[:3] + ["taksim_yeni"])
    assert not pool.all_fast_fail([])
