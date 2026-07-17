"""CountryDirector - the country-generic grid controller (2026-07-17).

The grid runs 4 cameras from ONE country; the director stays on a country
while it can field live cameras (backfilling a dead camera from deeper in
the SAME country's bench) and only advances to the next country when the
active one is fully dark. Higher-priority countries are re-probed before
each report so Turkey reclaims the grid the moment its block lifts.

These tests drive the real catalog's country pools but are pure control
logic - no network.
"""
from app.cameras import COUNTRY_ORDER, country_pool
from app.collector import CountryDirector, _minutes_to_next_report


def make_director(**kw):
    pools = {c: country_pool(c) for c in COUNTRY_ORDER}
    kw.setdefault("n_slots", 4)
    return CountryDirector(pools, COUNTRY_ORDER, **kw)


def kill_country(director, country, now):
    """Rest every camera of a country so it reads as fully dark."""
    pool = director.pools[country]
    for cam in list(pool.pool):
        for _ in range(pool.max_failures):
            pool.record(cam, False, now=now)


def test_starts_on_top_priority_country():
    d = make_director()
    assert d.active == "turkey"
    country, cams = d.assign(now=1000)
    assert country == "turkey"
    assert len(cams) == 4 and len(set(cams)) == 4
    assert cams[0] == "taksim_yeni"       # Turkey ladder is IBB-first


def test_full_turkey_stays_on_turkey():
    d = make_director()
    assert d.maybe_advance(now=1000) is None
    assert d.active == "turkey"


def test_dark_turkey_advances_to_thailand():
    d = make_director()
    now = 1000
    kill_country(d, "turkey", now)
    assert d.live_count("turkey", now) == 0
    switch = d.maybe_advance(now)
    assert switch == ("turkey", "thailand")
    assert d.active == "thailand"
    _, cams = d.assign(now)
    assert cams[0] == "th_sukhumvit"


def test_single_dead_camera_does_not_advance_country():
    """Operator spec: a dead camera backfills from the same country's bench;
    the country is only abandoned when it can field NO live cameras."""
    d = make_director()
    now = 1000
    pool = d.pools["turkey"]
    # Kill the four IBB primaries; Konya + tail are still live.
    for cam in ["taksim_yeni", "sultanahmet_1_yeni", "eyup_sultan_yeni",
                "beyazit_meydan_yeni"]:
        for _ in range(pool.max_failures):
            pool.record(cam, False, now=now)
    assert d.live_count("turkey", now) >= 4
    assert d.maybe_advance(now) is None      # stayed on Turkey
    _, cams = d.assign(now)
    assert cams[0] == "konya_hukumet"        # backfilled from the same country


def test_advances_past_multiple_dark_countries():
    d = make_director()
    now = 1000
    for c in ("turkey", "thailand", "japan"):
        kill_country(d, c, now)
    switch = d.maybe_advance(now)
    assert switch == ("turkey", "usa")
    assert d.active == "usa"


def test_everything_dark_holds_on_active():
    d = make_director()
    now = 1000
    for c in COUNTRY_ORDER:
        kill_country(d, c, now)
    assert d.maybe_advance(now) is None       # nobody live -> hold steady
    assert d.active == "turkey"


def test_host_block_counts_as_not_live():
    """A 403/429 block trips the host breaker; those cameras are not live
    even though their per-camera cooldown is clean."""
    d = make_director(breaker_threshold=4)
    now = 1000
    br = d.breakers["turkey"]
    # IBB and Konya sit on different hosts; block them all with 403s.
    for cam in d.pools["turkey"].pool:
        for _ in range(4):
            d.record(cam, False, 403, now)
    assert d.live_count("turkey", now) == 0
    assert d.maybe_advance(now)[1] == "thailand"


def test_countries_above_orders_recovery_candidates():
    d = make_director()
    d.switch_to("japan")
    assert d.countries_above() == ["turkey", "thailand"]
    d.switch_to("turkey")
    assert d.countries_above() == []          # nothing higher than the top


def test_switch_to_forgives_strikes():
    d = make_director()
    now = 1000
    kill_country(d, "turkey", now)
    d.switch_to("thailand")
    # Turkey recovered upstream: switching back must start it clean.
    d.switch_to("turkey")
    assert d.live_count("turkey", now) == len(d.pools["turkey"].pool)
    _, cams = d.assign(now)
    assert cams[0] == "taksim_yeni"


def test_record_routes_to_named_country():
    d = make_director()
    now = 1000
    # Record misses against Thailand while active country is Turkey.
    for cam in d.pools["thailand"].pool:
        for _ in range(d.pools["thailand"].max_failures):
            d.record(cam, False, None, now, country="thailand")
    assert d.live_count("thailand", now) == 0
    assert d.live_count("turkey", now) == len(d.pools["turkey"].pool)


def test_minutes_to_next_report_is_bounded():
    # Reports at 12:00 and 20:00 -> the longest wait to the NEXT report is
    # the 20:00->12:00 gap (16h); never more than a full day.
    for ts in (1784313381, 1784350000, 1784300000):
        m = _minutes_to_next_report(ts, "12:00,20:00")
        assert 0 <= m <= 24 * 60
