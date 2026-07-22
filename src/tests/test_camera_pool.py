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

# Turkey ladder, operator-approved 2026-07-21: 3 YouTube-Live cameras on
# top (the only ones that clear the GCP geo-block on the IBB CDN), then
# IBB four, then Konya four, then the rest of the Turkish tail.
YT3 = ["tr_bulancak_meydan", "tr_golden_horn", "tr_giresun_kalesi"]
IBB4 = ["taksim_yeni", "beyazit_meydan_yeni", "sarachane_yeni",
        "sultanahmet_1_yeni"]
KONYA = ["konya_hukumet", "otogar_kavsagi", "konya_kulturpark",
         "konya_millet_caddesi"]
TOP4 = YT3 + IBB4[:1]      # what an all-healthy pool serves on n_slots=4


def make_pool(**kw):
    kw.setdefault("max_failures", 3)
    kw.setdefault("retry_minutes", 15)
    return CameraPool(FALLBACK_POOL, n_slots=4, **kw)


def kill(pool, cam, now):
    """Feed enough misses to rest the camera."""
    for _ in range(pool.max_failures):
        pool.record(cam, False, now=now)


def test_pool_layout_matches_operator_spec():
    assert FALLBACK_POOL[:3] == YT3                   # YouTube tier first
    assert FALLBACK_POOL[3:7] == IBB4
    assert FALLBACK_POOL[7:11] == KONYA
    assert FALLBACK_POOL[11] == "buyuk_camlica_yeni"  # Turkish tail head
    assert len(FALLBACK_POOL) == len(set(FALLBACK_POOL))
    assert country_pool("turkey") == FALLBACK_POOL
    for s in GRID_SLOTS:
        chain = [s["primary"], *s["fallbacks"]]
        assert set(chain) == set(FALLBACK_POOL)


def test_all_healthy_assigns_the_top_of_the_ladder():
    pool = make_pool()
    assert pool.assign(now=1000) == TOP4


def test_assignment_always_distinct():
    pool = make_pool()
    now = 1000
    for _ in range(60):
        cams = pool.assign(now=now)
        assert len(cams) == 4 and len(set(cams)) == 4
        for c in cams:
            pool.record(c, False, now=now)
        now += 40


def test_dead_top_promotes_next_tier_in_order():
    pool = make_pool()
    now = 1000
    for cam in YT3 + IBB4:
        kill(pool, cam, now)
    assert pool.assign(now=now) == KONYA


def test_partial_ibb_outage_still_serves_yt_first():
    pool = make_pool()
    now = 1000
    # Kill the 3rd + 4th IBB cams of the new order (sarachane, sultanahmet_1).
    kill(pool, "sarachane_yeni", now)
    kill(pool, "sultanahmet_1_yeni", now)
    # With YT3 healthy, they always fill slots 1-3; slot 4 goes to the
    # first surviving IBB camera (taksim, since sarachane+sultanahmet rest).
    assert pool.assign(now=now) == YT3 + ["taksim_yeni"]


def test_yt_dead_partial_ibb_mixes_tiers_in_priority_order():
    pool = make_pool()
    now = 1000
    for cam in YT3:
        kill(pool, cam, now)
    kill(pool, "beyazit_meydan_yeni", now)
    kill(pool, "sultanahmet_1_yeni", now)
    assert pool.assign(now=now) == [
        "taksim_yeni", "sarachane_yeni",
        "konya_hukumet", "otogar_kavsagi",
    ]


def test_dead_top_tiers_keep_walking_the_catalog():
    pool = make_pool()
    now = 1000
    for cam in YT3 + IBB4 + KONYA:
        kill(pool, cam, now)
    got = pool.assign(now=now)
    assert got == FALLBACK_POOL[11:15]   # buyuk_camlica, ince_minareli, ...
    assert len(set(got)) == 4


def test_everything_dead_holds_steady_on_top_of_ladder():
    """All-dead pool: the padding is deterministic (pool priority order),
    so the grid stops churning tiles and sits on the top of the ladder."""
    pool = make_pool()
    now = 1000
    for cam in FALLBACK_POOL:
        kill(pool, cam, now)
    first = pool.assign(now=now + 1)
    assert first == TOP4
    t = now + 1
    for _ in range(10):
        cams = pool.assign(now=t)
        assert cams == TOP4
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
    for cam in YT3 + IBB4:
        kill(pool, cam, now)
    assert pool.assign(now=now) == KONYA
    later = now + 15 * 60 + 1
    # The reprobe pulls the highest-priority tier - now YT3 - back first.
    assert pool.assign(now=later)[0] == YT3[0]


def test_probation_cameras_rest_after_a_single_miss():
    """Proven-dead cams (killed and then rehabilitated by a
    cooldown expiry) rest after ONE further miss - bypass the
    max_failures grace period."""
    pool = make_pool(retry_minutes=15)
    now = 1000
    kill(pool, "taksim_yeni", now)
    assert pool.proven_dead["taksim_yeni"]
    later = now + pool.retry_seconds + 1        # cooldown expired
    assert pool.cooldown_until["taksim_yeni"] <= later
    pool.record("taksim_yeni", False, now=later)   # ONE miss
    assert pool.cooldown_until["taksim_yeni"] > later


def test_recovered_camera_is_fully_rehabilitated():
    """A successful frame after cooldown wipes proven_dead so the
    camera regains the FULL grace (max_failures misses allowed again)."""
    pool = make_pool()
    now = 1000
    kill(pool, "taksim_yeni", now)
    later = now + pool.retry_seconds + 1
    pool.record("taksim_yeni", True, now=later)      # revived
    assert not pool.proven_dead["taksim_yeni"]
    assert pool.cooldown_until["taksim_yeni"] == 0.0
    pool.record("taksim_yeni", False, now=later + 2) # 1 of max_failures
    # Rehabilitated -> ONE miss should NOT rest it.
    assert pool.cooldown_until["taksim_yeni"] == 0.0


def test_record_ignores_unknown_camera():
    pool = make_pool()
    pool.record("not_in_pool", False, now=1000)
    assert pool.assign(now=1000) == TOP4


def test_fast_fail_cameras_rest_after_one_miss():
    """tvkur/Konya fast-fail: one miss and the camera is out this round."""
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    for cam in YT3 + IBB4:                    # clear the YT+IBB tiers first
        kill(pool, cam, now)
    assert pool.assign(now=now) == KONYA
    for cam in KONYA:
        pool.record(cam, False, now=now)      # ONE miss each
    # Konya rests immediately; ladder walks to the Turkish tail.
    assert pool.assign(now=now + 1) == FALLBACK_POOL[11:15]


def test_fast_fail_does_not_touch_other_tiers():
    pool = make_pool(fast_fail=KONYA)
    now = 1000
    # One miss on an IBB (non-fast-fail) cam must NOT rest it. Asserted
    # on the cooldown directly so it doesn't depend on where taksim sits
    # in the pool head (YT3 sits above IBB in the operator-approved
    # ladder as of 2026-07-21).
    pool.record("taksim_yeni", False, now=now)
    assert pool.cooldown_until["taksim_yeni"] == 0.0


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
