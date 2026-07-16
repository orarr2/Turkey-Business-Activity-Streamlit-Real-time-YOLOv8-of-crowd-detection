"""CameraPool - the shared priority ladder behind the 4 grid slots.

Pins the operator's 2026-07-16 spec:
  tier 1: four Konya cams; tier 2: sultanahmet -> beyazit -> eyup ->
  buyuk_camlica; tier 3: rest of the catalog. Four DISTINCT cameras every
  round, dead cameras rest on a cooldown, probation cameras cost one
  sample per probe, and the grid never idles even with everything dead.
"""
import time

from app.cameras import FALLBACK_POOL, GRID_SLOTS
from app.collector import CameraPool

KONYA = ["konya_hukumet", "otogar_kavsagi", "konya_kulturpark",
         "konya_millet_caddesi"]
IBB4 = ["sultanahmet_1_yeni", "beyazit_meydan_yeni", "eyup_sultan_yeni",
        "buyuk_camlica_yeni"]


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
    assert len(FALLBACK_POOL) == len(set(FALLBACK_POOL))
    # every pool entry must be reachable from every slot's chain
    for s in GRID_SLOTS:
        chain = [s["primary"], *s["fallbacks"]]
        assert set(chain) == set(FALLBACK_POOL)


def test_all_healthy_assigns_the_four_konya():
    pool = make_pool()
    assert pool.assign(now=1000) == KONYA


def test_assignment_always_distinct():
    pool = make_pool()
    now = 1000
    for round_ in range(60):
        cams = pool.assign(now=now)
        assert len(cams) == 4 and len(set(cams)) == 4
        for c in cams:                       # everything keeps missing
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
        "sultanahmet_1_yeni", "beyazit_meydan_yeni",
    ]


def test_dead_ibb_four_keeps_walking_the_catalog():
    pool = make_pool()
    now = 1000
    for cam in KONYA + IBB4:
        kill(pool, cam, now)
    got = pool.assign(now=now)
    assert got == FALLBACK_POOL[8:12]        # taksim, ince_minareli, ...
    assert len(set(got)) == 4


def test_everything_dead_still_samples_four_cameras():
    pool = make_pool()
    now = 1000
    for cam in FALLBACK_POOL:
        kill(pool, cam, now)
    got = pool.assign(now=now)
    assert len(got) == 4 and len(set(got)) == 4


def test_cooldown_expiry_reprobes_higher_priority():
    pool = make_pool(retry_minutes=15)
    now = 1000
    for cam in KONYA:
        kill(pool, cam, now)
    assert pool.assign(now=now) == IBB4
    later = now + 15 * 60 + 1
    assert pool.assign(now=later)[:4][0] == "konya_hukumet"


def test_probation_cameras_rest_after_a_single_miss():
    """The snap-back economics: when the cooldown expires, the proven-dead
    Konya four are re-probed for ONE round (one miss each) and then the
    healthy tier-2 set returns for the whole next cooldown window."""
    pool = make_pool(retry_minutes=15)
    now = 1000
    for cam in KONYA:
        kill(pool, cam, now)
    for cam in IBB4:                          # tier 2 is delivering
        pool.record(cam, True, now=now)
    later = now + 15 * 60 + 1
    probe = pool.assign(now=later)
    assert probe == KONYA                     # all four probed together
    for cam in probe:
        pool.record(cam, False, now=later)    # still dead: ONE miss each
    assert pool.assign(now=later + 1) == IBB4  # straight back to tier 2


def test_recovered_camera_is_fully_rehabilitated():
    pool = make_pool()
    now = 1000
    kill(pool, "konya_hukumet", now)
    later = now + pool.retry_seconds + 1
    pool.record("konya_hukumet", True, now=later)     # probe succeeds
    assert pool.assign(now=later + 1)[0] == "konya_hukumet"
    # and one later miss does NOT insta-rest it (grace restored)
    pool.record("konya_hukumet", False, now=later + 2)
    assert pool.assign(now=later + 3)[0] == "konya_hukumet"


def test_record_ignores_unknown_camera():
    pool = make_pool()
    pool.record("not_in_pool", False, now=1000)       # must not raise
    assert pool.assign(now=1000) == KONYA
