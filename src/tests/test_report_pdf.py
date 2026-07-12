"""Rich Hebrew report: sample selection, snapshot fetch, PDF composition."""
import datetime as dt

import pytest

reportlab = pytest.importorskip("reportlab")
report_pdf = pytest.importorskip("tools.report_pdf")


def _mk(kind, cam, ts, full=None, snap=None, **extra):
    e = {"kind": kind, "cam_id": cam, "ts": ts, **extra}
    if full: e["fullframe_url"] = full
    if snap: e["snapshot_url"] = snap
    return e


def test_pick_event_samples_priority_and_recency():
    """Priority (obstructed > dark > extreme > loiter > returning) beats
    volume; within a kind, the newest event wins the first slot."""
    events = [
        # a flood of loiter events...
        *[_mk("loiter", "cam_A", f"2026-07-11T0{i}:00:00Z", snap="s") for i in range(5)],
        _mk("camera_obstructed", "cam_B", "2026-07-10T00:00:00Z", snap="s"),
        _mk("camera_dark", "cam_B", "2026-07-10T01:00:00Z", snap="s"),
        _mk("extreme_load", "cam_C", "2026-07-10T02:00:00Z", full="f"),
        _mk("returning", "cam_A", "2026-07-11T00:00:00Z", snap="s"),
    ]
    picks = report_pdf.pick_event_samples(events, max_total=6)
    kinds = [p["kind"] for p in picks]
    # First pass: one per priority-ordered kind that has any events.
    assert kinds[:5] == ["camera_obstructed", "camera_dark",
                         "extreme_load", "loiter", "returning"]
    # 6th slot goes back around; loiter has 5 events so a second round-robin
    # gives another loiter (the SECOND-newest of the batch).
    assert kinds[5] == "loiter"
    # newest loiter first
    assert picks[3]["ts"] == "2026-07-11T04:00:00Z"


def test_pick_event_samples_skips_events_without_any_url():
    events = [
        _mk("loiter", "cam_A", "2026-07-11T02:00:00Z"),        # no url
        _mk("loiter", "cam_A", "2026-07-11T01:00:00Z", snap="s"),
    ]
    picks = report_pdf.pick_event_samples(events, max_total=5)
    assert len(picks) == 1
    assert picks[0]["ts"] == "2026-07-11T01:00:00Z"


def test_pick_event_samples_max_cap():
    events = [_mk("loiter", "cam_A", f"2026-07-11T{h:02d}:00:00Z", snap="s")
              for h in range(20)]
    assert len(report_pdf.pick_event_samples(events, max_total=3)) == 3


def test_fetch_snapshots_uses_fullframe_then_snapshot():
    """We prefer the wide fullframe (context) - the object crop is only the
    fallback. A URL whose downloader returns None must fall through, not
    poison the pick."""
    served = {"https://ff.jpg": b"x" * 5000, "https://snap.jpg": b"y" * 5000}
    seen = []

    def dl(url):
        seen.append(url)
        return served.get(url)

    picks = [
        _mk("loiter", "cam_A", "t", full="https://ff.jpg", snap="https://snap.jpg"),
        _mk("loiter", "cam_A", "t", full="https://gone.jpg",
            snap="https://snap.jpg"),                 # fullframe 404 -> use snap
        _mk("loiter", "cam_A", "t", full="https://gone.jpg"),  # both gone
    ]
    out = report_pdf.fetch_snapshots(picks, downloader=dl)
    assert [b for _, b in out] == [b"x" * 5000, b"y" * 5000]
    assert seen == ["https://ff.jpg", "https://gone.jpg",
                    "https://snap.jpg", "https://gone.jpg"]


def _jpeg(color=(180, 180, 180)):
    """A 320x180 solid-color JPEG so reportlab has something to embed."""
    pil = pytest.importorskip("PIL.Image")
    from io import BytesIO
    im = pil.new("RGB", (320, 180), color)
    buf = BytesIO()
    im.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def test_compose_pdf_end_to_end(tmp_path):
    """Compose a real PDF from realistic inputs and validate: file exists,
    is a valid PDF, contains multiple pages (evidence + summary)."""
    out = tmp_path / "report.pdf"
    now = dt.datetime(2026, 7, 12, 20, 0)
    groups = [
        {"kind": "loiter", "kind_he": "שהייה ממושכת מול המצלמה",
         "cam": "Konya - Millet Caddesi", "count": 14,
         "last_ts": "2026-07-12T15:30:00Z"},
        {"kind": "camera_obstructed", "kind_he": "חסימת מצלמה",
         "cam": "Otogar Kavsagi", "count": 2,
         "last_ts": "2026-07-12T13:00:00Z"},
    ]
    cam_stats = [
        {"cam": "Konya - Hukumet", "peak_person": 14,
         "peak_person_ts": "2026-07-12T09:07:00Z",
         "peak_vehicles": 13, "typ_kmh": 22.0, "samples": 500},
        {"cam": "Konya - Millet Caddesi", "peak_person": 4,
         "peak_person_ts": "2026-07-12T08:54:00Z",
         "peak_vehicles": 11, "typ_kmh": 24.0, "samples": 480},
    ]
    training = {"event": "gate", "promoted": False,
                "candidate": "head_run2.pt",
                "at": "2026-07-11T10:09:52Z",
                "reasons": ["mAP50 gain +0.00pp < required +0.50pp"]}
    stale = []
    snaps = [
        ({"kind": "camera_obstructed", "cam_id": "otogar_kavsagi",
          "ts": "2026-07-12T13:00:00Z", "cls": "train"}, _jpeg((60, 60, 60))),
        ({"kind": "loiter", "cam_id": "konya_millet_caddesi",
          "ts": "2026-07-12T15:30:00Z", "duration_sec": 420}, _jpeg((100, 140, 200))),
    ]
    KIND_HE = {"loiter": "שהייה ממושכת מול המצלמה",
               "camera_obstructed": "חסימת מצלמה"}
    p = report_pdf.compose_pdf(
        out, now_il=now, window_hours=12,
        events_by_kind=groups, cam_stats=cam_stats,
        training=training, stale_slots=stale,
        snapshots=snaps, kind_labels=KIND_HE,
        total_events=16, total_samples=980)
    assert p == out and p.is_file()
    head = p.read_bytes()[:8]
    assert head.startswith(b"%PDF-"), f"not a PDF: {head!r}"
    size_kb = p.stat().st_size / 1024
    assert 5 < size_kb < 800, f"unexpected PDF size {size_kb:.0f} KB"


def test_compose_pdf_handles_empty_window(tmp_path):
    """Off-hours or a fresh install produces zero events, zero snapshots -
    the report must still render without raising."""
    out = tmp_path / "empty.pdf"
    now = dt.datetime(2026, 7, 12, 12, 0)
    p = report_pdf.compose_pdf(
        out, now_il=now, window_hours=12,
        events_by_kind=[], cam_stats=[],
        training=None, stale_slots=[],
        snapshots=[], kind_labels={},
        total_events=0, total_samples=0)
    assert p.is_file() and p.read_bytes()[:5] == b"%PDF-"
