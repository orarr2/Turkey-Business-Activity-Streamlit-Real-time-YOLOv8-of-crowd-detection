"""HostBreaker - the host-level circuit breaker for access blocks.

Pins the 2026-07-17 incident response: kamerayayin returned HTTP 403 on
EVERY playlist (address block), yet per-camera strikes kept the collector
knocking dozens of times an hour. The breaker rests the WHOLE host after
`threshold` consecutive 403/429s, forgives the per-camera strikes (the
cameras were never dead), probes with ONE request when the rest expires,
and reopens the instant the host answers - so an unblock window like the
one observed at 21:15 the night before is caught within minutes.
"""
from app.cameras import FALLBACK_POOL
from app.collector import CameraPool, HostBreaker

KONYA = ["konya_hukumet", "otogar_kavsagi", "konya_kulturpark",
         "konya_millet_caddesi"]
IBB4 = ["taksim_yeni", "beyazit_meydan_yeni", "sarachane_yeni",
        "sultanahmet_1_yeni"]

def _host(cam: str) -> str:
    if cam.startswith(("konya", "otogar")):
        return "content.tvkur.com"
    if cam.startswith("tr_"):
        # YouTube-Live tier added 2026-07-21: those cams resolve through
        # yt-dlp, not the IBB CDN, so they must not share the IBB host bucket.
        return "youtube.com"
    return "kamerayayin.ibb.istanbul"


HOST_OF = {c: _host(c) for c in FALLBACK_POOL}
IBB_ALL = [c for c, h in HOST_OF.items() if h == "kamerayayin.ibb.istanbul"]


def make_breaker(**kw):
    kw.setdefault("threshold", 4)
    kw.setdefault("rest_minutes", 20)
    return HostBreaker(HOST_OF, **kw)


def test_trips_after_threshold_consecutive_403s():
    br = make_breaker()
    now = 1000
    events = [br.note(c, False, 403, now=now) for c in IBB4]
    assert events == [None, None, None, "tripped"]
    assert set(br.blocked_cams(now=now + 1)) == set(IBB_ALL)


def test_konya_404s_do_not_feed_the_ibb_counter():
    br = make_breaker()
    now = 1000
    for c in IBB4[:3]:
        br.note(c, False, 403, now=now)
    # A tvkur 404 in between must not reset OR advance the IBB count...
    assert br.note("konya_hukumet", False, 404, now=now) is None
    # ...so the 4th IBB refusal still trips.
    assert br.note("beyazit_meydan_yeni", False, 403, now=now) == "tripped"


def test_non_block_failure_resets_the_count():
    br = make_breaker()
    now = 1000
    for c in IBB4[:3]:
        br.note(c, False, 403, now=now)
    br.note("taksim_yeni", False, None, now=now)   # timeout: host answered? not a 403
    for c in IBB4[:3]:
        assert br.note(c, False, 403, now=now) is None
    assert br.blocked_cams(now=now) == set()


def test_rest_expiry_frees_exactly_one_probe_camera():
    br = make_breaker()
    now = 1000
    for c in IBB4:
        br.note(c, False, 403, now=now)
    probing = now + br.rest_seconds + 1
    blocked = br.blocked_cams(now=probing)
    free = set(IBB_ALL) - blocked
    assert free == {"taksim_yeni"}          # highest-priority cam probes


def test_probe_refused_rearms_the_rest():
    br = make_breaker()
    now = 1000
    for c in IBB4:
        br.note(c, False, 403, now=now)
    probing = now + br.rest_seconds + 1
    assert br.note("taksim_yeni", False, 403, now=probing) == "rearmed"
    assert set(br.blocked_cams(now=probing + 1)) == set(IBB_ALL)


def test_probe_success_reopens_the_whole_host():
    br = make_breaker()
    now = 1000
    for c in IBB4:
        br.note(c, False, 403, now=now)
    probing = now + br.rest_seconds + 1
    assert br.note("taksim_yeni", True, None, now=probing) == "reopened"
    assert br.blocked_cams(now=probing + 1) == set()


def test_forced_sample_during_rest_does_not_extend_it():
    br = make_breaker()
    now = 1000
    for c in IBB4:
        br.note(c, False, 403, now=now)
    rest_until = br.rest_until["kamerayayin.ibb.istanbul"]
    assert br.note("eyup_sultan_yeni", False, 403, now=now + 60) is None
    assert br.rest_until["kamerayayin.ibb.istanbul"] == rest_until


def test_pool_assign_skips_blocked_cams_and_pads_unblocked_first():
    pool = CameraPool(FALLBACK_POOL, n_slots=4)
    now = 1000
    # The whole IBB host is blocked by the breaker; with IBB out, the pool
    # must skip every blocked cam. Post-2026-07-21 the YT3 tier sits above
    # IBB so it fills slots 1-3 before falling to Konya for the last slot.
    YT3 = ["tr_bulancak_meydan", "tr_golden_horn", "tr_giresun_kalesi"]
    picked = pool.assign(now=now, blocked=set(IBB_ALL))
    assert picked == YT3 + [KONYA[0]]
    assert not set(picked) & set(IBB_ALL)


def test_pool_forgive_wipes_strikes_and_cooldowns():
    pool = CameraPool(FALLBACK_POOL, n_slots=4)
    now = 1000
    # Kill YT3 first so IBB actually reaches the assignment; the breaker
    # test predates the YT tier and pinned the head to taksim.
    YT3 = ["tr_bulancak_meydan", "tr_golden_horn", "tr_giresun_kalesi"]
    for cam in YT3 + IBB4:
        for _ in range(pool.max_failures):
            pool.record(cam, False, now=now)
    assert pool.assign(now=now)[0] != "taksim_yeni"
    pool.forgive(IBB4)
    # YT3 still rests; assignment fills from IBB (forgiven) + KONYA head.
    got = pool.assign(now=now)
    assert got[:4] == IBB4
    # Forgiven cams regain the FULL grace (proven_dead cleared).
    pool.record("taksim_yeni", False, now=now + 1)
    assert pool.assign(now=now + 2)[0] == "taksim_yeni"


def test_forgive_ignores_unknown_camera():
    pool = CameraPool(FALLBACK_POOL, n_slots=4)
    pool.forgive(["not_in_pool"])
    # Post-2026-07-21: the YT3 tier sits above IBB in the pool head.
    YT3 = ["tr_bulancak_meydan", "tr_golden_horn", "tr_giresun_kalesi"]
    assert pool.assign(now=1000) == YT3 + IBB4[:1]
