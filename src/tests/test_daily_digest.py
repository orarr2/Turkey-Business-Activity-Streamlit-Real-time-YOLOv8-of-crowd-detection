"""Situation-report compose layer: aggregation, peaks, Hebrew rendering."""
import datetime as dt

from tools.daily_digest import (aggregate_events, compose_digest,
                                footfall_stats, stale_from_latest)

_NOON = dt.datetime(2026, 7, 11, 12, 0)
_EVE = dt.datetime(2026, 7, 11, 20, 0)


def test_aggregate_events_groups_per_kind_and_camera():
    evs = [
        {"kind": "camera_obstructed", "cam_name": "Otogar",
         "ts": "2026-07-11T05:00:00Z"},
        {"kind": "camera_obstructed", "cam_name": "Otogar",
         "ts": "2026-07-11T08:00:00Z"},
        {"kind": "returning", "cam_id": "konya_hukumet",
         "ts": "2026-07-11T07:00:00Z"},
    ]
    groups = aggregate_events(evs)
    assert len(groups) == 2
    assert groups[0]["kind"] == "camera_obstructed"       # newest first
    assert groups[0]["count"] == 2
    assert groups[0]["last_ts"] == "2026-07-11T08:00:00Z"
    assert groups[0]["kind_he"] == "חסימת מצלמה"
    assert groups[1]["kind_he"] == "מבקר חוזר"
    assert aggregate_events([]) == []


def test_footfall_stats_peaks_and_speed():
    rows = [
        {"cam_name": "Hukumet", "person": 3, "vehicles": 2,
         "ts": "2026-07-11T04:00:00Z"},
        {"cam_name": "Hukumet", "person": 9, "vehicles": 1,
         "ts": "2026-07-11T06:30:00Z",
         "speeds": {"median_kmh": 42.5, "max_kmh": 127.0}},   # max = outlier
        {"cam_name": "Otogar", "person": 1, "vehicles": 7,
         "ts": "2026-07-11T05:00:00Z"},
        {"cam_name": "Hukumet", "person": None, "vehicles": None,
         "ts": "2026-07-11T07:00:00Z"},                     # failed round
    ]
    stats = footfall_stats(rows)
    assert stats[0]["cam"] == "Hukumet"                    # highest peak first
    assert stats[0]["peak_person"] == 9
    assert stats[0]["peak_person_ts"] == "2026-07-11T06:30:00Z"
    assert stats[0]["peak_vehicles"] == 2
    assert stats[0]["typ_kmh"] == 42.5      # median-of-medians, not the 127 outlier
    assert stats[0]["samples"] == 3
    assert stats[1]["cam"] == "Otogar" and stats[1]["peak_vehicles"] == 7


def test_stale_slots_flagged():
    now = dt.datetime(2026, 7, 11, 12, 0, tzinfo=dt.timezone.utc)
    latest = [
        {"cam_name": "Fresh", "slot": "s1", "ts": "2026-07-11T11:58:00Z"},
        {"cam_name": "Stuck", "slot": "s2", "ts": "2026-07-11T10:30:00Z"},
        {"cam_name": "NoTs", "slot": "s3"},
    ]
    stale = stale_from_latest(latest, now_utc=now)
    assert [s["cam"] for s in stale] == ["Stuck"]
    assert stale[0]["age_min"] == 90
    # a camera that once ran but is no longer in the grid is history, not
    # an alarm (the catalog-only tram cam false-alarmed at 1462 min live)
    latest.append({"cam_name": "OldTram", "slot": "gone",
                   "ts": "2026-07-10T12:00:00Z"})
    stale = stale_from_latest(latest, now_utc=now,
                              active_slots={"s1", "s2", "s3"})
    assert [s["cam"] for s in stale] == ["Stuck"]


def test_compose_full_report_hebrew():
    groups = aggregate_events([
        {"kind": "extreme_load", "cam_name": "Millet",
         "ts": "2026-07-11T06:00:00Z"},
        {"kind": "extreme_load", "cam_name": "Millet",
         "ts": "2026-07-11T07:00:00Z"},
    ])
    stats = footfall_stats([{"cam_name": "Millet", "person": 55,
                             "vehicles": 12, "ts": "2026-07-11T06:00:00Z"}])
    training = {"event": "gate", "promoted": False, "candidate": "head_run2.pt",
                "at": "2026-07-11T10:09:52Z",
                "reasons": ["mAP50 gain +0.00pp < required +0.50pp"]}
    subject, text, html = compose_digest(_NOON, 12, groups, stats,
                                         training, [])
    assert subject == "קוניה - דוח צהריים 11.07"
    assert "עומס חריג" in text and "(x2)" in text
    assert "עד 55 אנשים" in text
    assert "נדחה בשער" in text and "head_run2.pt" in text
    assert "כל המצלמות מדווחות כסדרן" in text
    assert 'dir="rtl"' in html and "x2" in html

    # evening + stale camera + quiet window
    subject2, text2, _ = compose_digest(
        _EVE, 12, [], [], None, [{"cam": "Otogar", "age_min": 45}])
    assert "דוח ערב" in subject2
    assert "שקט - לא נרשם אף חריג" in text2
    assert "Otogar" in text2 and "45 דקות" in text2
    assert "עוד לא רצה ריצת אימון" in text2


def test_promoted_line():
    training = {"event": "gate", "promoted": True, "candidate": "head_run9.pt",
                "at": "2026-07-12T01:30:00Z", "reasons": ["+1.2pp"]}
    _, text, _ = compose_digest(_NOON, 12, [], [], training, [])
    assert "קודם ראש חדש" in text and "head_run9.pt" in text
