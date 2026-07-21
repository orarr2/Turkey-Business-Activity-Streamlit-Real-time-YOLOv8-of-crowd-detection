"""Twice-daily situation report, emailed to the operator's phone.

    python -m tools.daily_digest --dry-run      # compose + print, no send
    python -m tools.daily_digest                # compose + send via Gmail

Runs on the VM from a systemd timer (deploy/gcp-vm/digest.timer, 12:00 and
20:00 Asia/Jerusalem) - the operator's PC plays no part. Gmail's phone app
turns the mail into the push notification the operator asked for.

Sections (all sourced from what the collector already writes):
  * scene events from Firestore `events`, aggregated per (kind, camera)
    with a count - same aggregation the dashboard table uses;
  * activity peaks per camera from `footfall` (max people / vehicles, top
    measured speed) over the window;
  * the latest trainer verdict from Storage `training/history.jsonl`;
  * collector health from `latest` (a slot whose newest sample is old
    means a stuck stream - say so plainly).

Credentials: FIREBASE_CREDENTIALS (service-account json - present on the
VM at /etc/turkey-footfall/serviceAccount.json) for Firestore;
GMAIL_USER + GMAIL_APP_PASSWORD (+ optional DIGEST_TO) from
/etc/turkey-footfall/digest.env for SMTP. The app password is a Google
"App password" (requires 2-step verification) - never the account password.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import smtplib
import tempfile
import urllib.request
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent

WINDOW_HOURS_DEFAULT = 12
STALE_SLOT_MIN = 10          # newest sample older than this = stuck stream

# Delta reporting: the Learning line used to repeat the same running totals
# every 12 hours ("36 frames, 3 confirmed, 167 objects added") even when
# nothing new had been reviewed - misleading progress signal. We now stash
# the last report's snapshot on disk and report only the CHANGE.
LAST_REPORT_STATE = _SRC_ROOT / "data" / ".digest_last_report.json"

# Operator-facing labels for anomaly kinds. English throughout: the report
# ships to the operator's Gmail and inline-previews on a plain LTR client;
# every earlier Hebrew rendering (dashboard-side or bidi-shaped PDF) is
# retired in favor of a single, unambiguous English pass.
KIND_LABELS = {
    "extreme_load":      "Extreme load",
    "camera_obstructed": "Camera blocked",
    "camera_dark":       "View went dark",
    "loiter":            "Loitering",
    "returning":         "Returning visitor",
}


def _israel_now() -> dt.datetime:
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("Asia/Jerusalem"))


# ---- pure compose helpers (unit-tested, no I/O) -----------------------------

def aggregate_events(events: list[dict]) -> list[dict]:
    """(kind, camera) -> {ref, label, cam, count, last_ts, latest_event},
    newest group first. `ref` is a 1-based row number the report uses to
    pair the aggregated row with the specific snapshot pulled for it -
    without the ref, the operator has to eyeball the caption to figure
    out which table row an image belongs to."""
    groups: dict[tuple, dict] = {}
    for e in events:
        kind = str(e.get("kind") or "?")
        cam = str(e.get("cam_name") or e.get("cam_id") or e.get("slot") or "?")
        g = groups.setdefault((kind, cam), {
            "kind": kind, "label": KIND_LABELS.get(kind, kind), "cam": cam,
            "count": 0, "last_ts": "", "latest_event": None})
        g["count"] += 1
        ts = str(e.get("ts") or "")
        if ts > g["last_ts"]:
            g["last_ts"] = ts
            g["latest_event"] = e
    out = sorted(groups.values(), key=lambda g: g["last_ts"], reverse=True)
    for i, g in enumerate(out, start=1):
        g["ref"] = i
    return out


def footfall_stats(records: list[dict]) -> list[dict]:
    """Per camera: samples, peak people/vehicles (+when), typical speed.

    Speed is the MEDIAN over the window of each round's median moving
    speed. Any max-flavored stat over ~1000 rounds is guaranteed to
    surface fused-pair noise (observed live: every city camera "peaked"
    at 114-128 km/h, all sitting just under the 130 sanity cap). The
    median-of-medians reads as what the operator actually wants to know:
    how fast traffic typically flows there.
    """
    cams: dict[str, dict] = {}
    # Keep the display-name key working for tests that predate cam_id, but
    # the PDF reads cam_id off c["cam_id"] to partition by country.
    for r in records:
        cam_id = str(r.get("cam_id") or r.get("cam_name") or "?")
        cam = str(r.get("cam_name") or cam_id)
        c = cams.setdefault(cam_id, {"cam": cam, "cam_id": cam_id,
                                  "samples": 0, "misses": 0,
                                  "peak_person": 0, "peak_person_ts": "",
                                  "peak_vehicles": 0, "typ_kmh": 0.0,
                                  "_spd": []})
        p = r.get("person")
        v = r.get("vehicles")
        # A "sample" is a round that actually produced usable frames. A MISS
        # (empty frame / decode error) still gets logged with ok=0 but must
        # not inflate the sample count - otherwise 12h of dead streams looks
        # identical to 12h of real data in the report.
        is_good = (r.get("ok") == 1
                   or isinstance(p, (int, float))
                   or isinstance(v, (int, float)))
        if is_good:
            c["samples"] += 1
        else:
            c["misses"] += 1
        if isinstance(p, (int, float)) and p > c["peak_person"]:
            c["peak_person"] = int(p)
            c["peak_person_ts"] = str(r.get("ts") or "")
        if isinstance(v, (int, float)) and v > c["peak_vehicles"]:
            c["peak_vehicles"] = int(v)
        spd = (r.get("speeds") or {}).get("median_kmh")
        if isinstance(spd, (int, float)) and spd > 0:
            c["_spd"].append(float(spd))
    for c in cams.values():
        spds = sorted(c.pop("_spd"))
        c["typ_kmh"] = spds[len(spds) // 2] if spds else 0.0
    return sorted(cams.values(), key=lambda c: c["peak_person"], reverse=True)


def _cam_country(cam_id) -> str | None:
    """Country of a physical camera, from the catalog; None when unknown
    (keeps the digest importable in minimal test envs)."""
    try:
        from app.cameras import CAMERAS
        return (CAMERAS.get(str(cam_id)) or {}).get("country")
    except Exception:
        return None


def _fmt_ts(ts_iso: str) -> str:
    """UTC ISO -> Israel-time HH:MM (the operator lives on this clock)."""
    try:
        from zoneinfo import ZoneInfo
        t = dt.datetime.strptime(str(ts_iso)[:19], "%Y-%m-%dT%H:%M:%S")
        t = t.replace(tzinfo=dt.timezone.utc).astimezone(ZoneInfo("Asia/Jerusalem"))
        return t.strftime("%H:%M")
    except (ValueError, TypeError):
        return str(ts_iso)[:16]


def _load_last_report() -> dict:
    try:
        return json.loads(LAST_REPORT_STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save_last_report(state: dict) -> None:
    try:
        LAST_REPORT_STATE.parent.mkdir(parents=True, exist_ok=True)
        LAST_REPORT_STATE.write_text(json.dumps(state, indent=1))
    except OSError as e:
        print(f"digest: could not persist last-report state: {e}")


def _training_lines(training: dict | None,
                    reviews: dict | None,
                    prev: dict | None = None) -> list[str]:
    """Delta-based Learning line: only the CHANGE since the last digest.

    The old phrasing repeated the running totals every 12 hours, so a report
    with no new labels still read 'you have labeled 36 frames' - misleading
    progress signal. We now stash the last snapshot on disk and describe
    what moved. First-ever run prints the current totals as the baseline.
    """
    lines: list[str] = []
    r = reviews or {}
    prev = prev or {}
    cur_frames = r.get("frames_labeled", 0)
    cur_conf = r.get("boxes_confirmed", 0)
    cur_rej = r.get("boxes_rejected", 0)
    cur_miss = r.get("missed_marked", 0)

    if not prev:
        if cur_frames:
            lines.append(f"So far you have labeled {cur_frames} frames "
                         f"({cur_conf} confirmed, {cur_rej} rejected, "
                         f"{cur_miss} objects you added). Future reports "
                         f"will only mention new work since the last one.")
        else:
            lines.append("No frames labeled yet - open the Reinforcement "
                         "Learning tab in the dashboard and start tagging.")
    else:
        d_frames = cur_frames - prev.get("frames_labeled", 0)
        d_conf = cur_conf - prev.get("boxes_confirmed", 0)
        d_rej = cur_rej - prev.get("boxes_rejected", 0)
        d_miss = cur_miss - prev.get("missed_marked", 0)
        if d_frames > 0 or d_conf > 0 or d_rej > 0 or d_miss > 0:
            lines.append(f"Since the last report you labeled {d_frames} new "
                         f"frame(s) (+{d_conf} confirmed, +{d_rej} rejected, "
                         f"+{d_miss} objects added). Running total: "
                         f"{cur_frames} frames.")
        else:
            lines.append(f"No new labels since the last report - "
                         f"{cur_frames} frames on file. The trainer needs "
                         f"more diverse examples before the next promotion.")

    prev_train = prev.get("last_training") if prev else None
    if training and training != prev_train:
        when = str(training.get("at") or "")[:10]
        cand = training.get("candidate") or training.get("file") or "?"
        if training.get("promoted"):
            lines.append(f"Cloud training on {when} PROMOTED a new "
                         f"detection head ({cand}); the VM has already "
                         f"picked it up.")
        else:
            lines.append(f"Cloud training on {when} did not clear the gate "
                         f"({cand}). More diverse labels will give the "
                         f"next run something new to learn from.")
    elif not training:
        lines.append("No cloud training run has executed yet. Trigger "
                     "'train-head' in the GitHub Actions tab once you have "
                     "20+ labeled frames.")
    return lines


# The collector is country-generic: it runs 4 cameras from ONE country and
# rotates through a ladder when a country goes dark. The report therefore
# names whichever country the grid is CURRENTLY watching (read from
# config/grid.country), not a hardcoded "Konya".
_COUNTRY_LABELS = {"turkey": "Turkey", "thailand": "Thailand",
                   "japan": "Japan", "usa": "USA"}


def _country_label(grid: dict | None) -> str:
    c = str((grid or {}).get("country") or "")
    return _COUNTRY_LABELS.get(c, c.title() if c else "Live grid")


def _country_from_sample(sample: dict) -> str | None:
    """Best-effort country of one footfall sample: catalog first, then the
    cam-id prefix as a fallback for samples on cameras since removed."""
    try:
        from app.cameras import CAMERAS
        c = (CAMERAS.get(str(sample.get("cam_id"))) or {}).get("country")
    except Exception:
        c = None
    if c:
        return c
    cid = str(sample.get("cam_id") or "")
    if cid.startswith("th_"):
        return "thailand"
    if cid.startswith("jp_"):
        return "japan"
    if cid.startswith("us_"):
        return "usa"
    return "turkey" if cid else None      # legacy Turkey cams have no prefix


def dominant_country(footfall: list[dict]) -> str | None:
    """The country that owned the WINDOW - not the moment the digest fired.

    When the collector rotates through the ladder overnight (Thailand ->
    Japan -> USA), a 12h digest that titles itself by the current grid can
    read as "USA report" even though 99% of the samples were Thai. Weight
    by ok-samples so a two-round USA probe with no frames cannot outvote
    hours of live Thai data."""
    counts: dict[str, int] = {}
    for r in footfall:
        if r.get("ok") != 1:
            continue
        c = _country_from_sample(r)
        if c:
            counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _grid_cameras(grid: dict | None) -> list[dict]:
    """The cameras the VM is running RIGHT NOW, from config/grid - each with
    the display name, city and country the collector last published."""
    out = []
    for s in (grid or {}).get("slots") or []:
        out.append({
            "name": s.get("active_cam_name") or s.get("active_cam") or "?",
            "city": s.get("city") or "",
            "country": s.get("country") or "",
        })
    return out


def compose_digest(now_il: dt.datetime, window_hours: int,
                   event_groups: list[dict], cam_stats: list[dict],
                   training: dict | None, stale_slots: list[dict],
                   reviews: dict | None = None,
                   grid: dict | None = None,
                   dominant: str | None = None,
                   prev: dict | None = None,
                   ) -> tuple[str, str, str]:
    """Returns (subject, plain_text, html). English throughout, phone-first.
    `grid` is config/grid - the collector's CURRENT country; `dominant` is
    the country that owned the reporting WINDOW. When they disagree (an
    overnight ladder walk fired the digest on a probe country) we title by
    the data majority and mention the current grid as a footnote."""
    part = "Midday report" if now_il.hour < 16 else "Evening report"
    grid_label = _country_label(grid)
    label = (_COUNTRY_LABELS.get(dominant, dominant.title())
             if dominant else grid_label)
    subject = f"{label} - {part} {now_il.strftime('%d.%m')}"

    lines: list[str] = [f"{part} - last {window_hours} hours", ""]
    html: list[str] = [
        '<div style="font-family:Arial,sans-serif;font-size:15px">',
        f"<h2 style='margin:0 0 12px'>{subject}</h2>",
    ]

    # Which grid is the collector on right now (country-generic - it may have
    # fallen through to another country while the primary was geo-blocked).
    grid_cams = _grid_cameras(grid)
    if dominant and dominant != (grid or {}).get("country"):
        note = (f"This window's data is mostly from {label}. The grid moved "
                f"to {grid_label} at digest time - the live section below "
                f"reflects the current grid, not the window's majority.")
        lines.append(note)
        lines.append("")
        html.append(f"<p style='color:#475569;font-size:13px'>{note}</p>")
    if grid_cams:
        lines.append(f"Live grid ({grid_label}) - {len(grid_cams)} cameras:")
        html.append(f"<h3 style='margin:14px 0 6px'>Live grid - {grid_label}</h3>")
        html.append("<table cellpadding='4' style='border-collapse:collapse'>")
        for gc in grid_cams:
            where = f" ({gc['city']})" if gc["city"] else ""
            lines.append(f"  - {gc['name']}{where}")
            html.append(
                f"<tr><td style='border-bottom:1px solid #ddd'>{gc['name']}</td>"
                f"<td style='border-bottom:1px solid #ddd'>{gc['city']}</td>"
                f"<td style='border-bottom:1px solid #ddd'>{gc['country']}</td></tr>")
        html.append("</table>")
        lines.append("")

    # health first - a stuck camera changes how everything below reads
    if stale_slots:
        for s in stale_slots:
            reason = s.get("reason")
            if reason:
                w = f"WARNING: camera {s['cam']} is {reason}"
            else:
                w = (f"WARNING: camera {s['cam']} has not reported for "
                     f"{s['age_min']} minutes")
            lines.append("! " + w)
            html.append(f"<p style='color:#b00'><b>{w}</b></p>")
    elif cam_stats and all(c["peak_person"] == 0 and c["peak_vehicles"] == 0
                           for c in cam_stats):
        # Every camera's latest doc looks fresh yet nothing was ever
        # detected across the window - the streams are almost certainly
        # dead or geo-blocked upstream. Without this guard the report
        # says "reporting normally" while every peak is 0.
        w = (f"WARNING: {len(cam_stats)} camera(s) reporting but detected "
             "0 people and 0 vehicles across the entire window - streams "
             "may be dead, geo-blocked or obscured. Check the VM journal "
             "for repeated MISS lines.")
        lines.append("! " + w)
        html.append(f"<p style='color:#b00'><b>{w}</b></p>")
    else:
        lines.append("All cameras reporting normally.")
        html.append("<p>All cameras reporting normally.</p>")
    lines.append("")

    lines.append("Anomalies:")
    html.append("<h3 style='margin:14px 0 6px'>Anomalies</h3>")
    if event_groups:
        html.append("<table cellpadding='4' style='border-collapse:collapse'>")
        for g in event_groups:
            t = _fmt_ts(g["last_ts"])
            ctry = _cam_country(g.get("cam"))
            tag = (f" [{ctry}]"
                   if ctry and ctry != (grid or {}).get("country") else "")
            row = (f"{g['label']} - {g['cam']}{tag}"
                   + (f" (x{g['count']})" if g["count"] > 1 else "")
                   + f", latest {t}")
            lines.append("  - " + row)
            html.append(
                f"<tr><td style='border-bottom:1px solid #ddd'>{g['label']}</td>"
                f"<td style='border-bottom:1px solid #ddd'>{g['cam']}{tag}</td>"
                f"<td style='border-bottom:1px solid #ddd'>x{g['count']}</td>"
                f"<td style='border-bottom:1px solid #ddd'>{t}</td></tr>")
        html.append("</table>")
    else:
        lines.append("  Quiet - no anomalies in this window.")
        html.append("<p>Quiet - no anomalies in this window.</p>")
    lines.append("")

    # Peaks: only cameras that DELIVERED frames. The collector's ladder
    # walk probes dead cameras across every country and each probe writes
    # an ok=0 doc - listing those as zero-rows buried the real story under
    # 20+ lines of noise (operator complaint, 2026-07-18). Cameras from
    # countries other than the active one (earlier legs of this window)
    # get their own compact table instead of silently mixing in.
    visible = [c for c in cam_stats if c.get("samples", 0) > 0]
    dark = len(cam_stats) - len(visible)
    active_c = (grid or {}).get("country")
    act_rows = [c for c in visible
                if _cam_country(c.get("cam_id")) in (active_c, None)]
    other_rows = [c for c in visible if c not in act_rows]

    def _peaks_table(rows, title):
        lines.append(f"{title}:")
        html.append(f"<h3 style='margin:14px 0 6px'>{title}</h3>")
        html.append("<table cellpadding='4' style='border-collapse:collapse'>"
                    "<tr><th align='left'>Camera</th><th>Peak people</th>"
                    "<th>Peak vehicles</th><th>Typical traffic</th></tr>")
        for c in rows:
            when = f" at {_fmt_ts(c['peak_person_ts'])}" if c["peak_person_ts"] else ""
            spd = (f"~{c['typ_kmh']:.0f} km/h typical"
                   if c["typ_kmh"] > 0 else "-")
            lines.append(f"  - {c['cam']}: up to {c['peak_person']} people{when}, "
                         f"up to {c['peak_vehicles']} vehicles, {spd}")
            html.append(f"<tr><td>{c['cam']}</td>"
                        f"<td align='center'>{c['peak_person']}{when}</td>"
                        f"<td align='center'>{c['peak_vehicles']}</td>"
                        f"<td align='center'>{spd}</td></tr>")
        html.append("</table>")

    if act_rows:
        _peaks_table(act_rows, f"Activity peaks - {label}")
    if other_rows:
        _peaks_table(other_rows,
                     "Earlier in this window (before the grid settled here)")
    if not visible:
        lines.append("Activity peaks:")
        lines.append("  No footfall samples in this window - check the VM journal.")
        html.append("<h3 style='margin:14px 0 6px'>Activity peaks</h3>"
                    "<p>No footfall samples in this window - check the VM journal.</p>")
    if dark:
        note = (f"{dark} more camera(s) were probed but delivered no frames "
                f"(dead / geo-blocked) - omitted above.")
        lines.append("  " + note)
        html.append(f"<p style='color:#777;font-size:13px'>{note}</p>")
    lines.append("")

    lines.append("Learning:")
    html.append("<h3 style='margin:14px 0 6px'>Learning</h3>")
    for line in _training_lines(training, reviews, prev=prev):
        lines.append("  " + line)
        html.append(f"<p>{line}</p>")

    html.append("</div>")
    return subject, "\n".join(lines), "".join(html)


# ---- data fetch (VM side) ----------------------------------------------------

def _firestore():
    import firebase_admin
    from firebase_admin import credentials, firestore
    cred = os.environ.get("FIREBASE_CREDENTIALS")
    if not cred or not Path(cred).is_file():
        raise SystemExit("FIREBASE_CREDENTIALS must point at the "
                         "service-account json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred))
    return firestore.client()


def _scene_events_from_footfall(footfall: list[dict],
                                already: list[dict]) -> list[dict]:
    """Materialize scene anomalies (is_anomaly footfall rows) as event dicts,
    with a per-(cam, kind) cooldown so a persistent obstruction doesn't
    show up once per sample. Skips (cam, kind) pairs already present in the
    events feed - the collector's own write path landed 2026-07-21; anything
    from before that only lives in footfall."""
    SCENE_KINDS = ("extreme_load", "camera_obstructed", "camera_dark")
    have = {(str(e.get("cam_id")), str(e.get("kind")))
            for e in already if e.get("kind") in SCENE_KINDS}
    out: list[dict] = []
    last_by = {}
    for r in sorted(footfall, key=lambda x: str(x.get("ts") or "")):
        if not r.get("is_anomaly"):
            continue
        a = r.get("anomaly") or {}
        kind = str(a.get("kind") or "")
        cam = str(r.get("cam_id") or "")
        key = (cam, kind)
        if kind not in SCENE_KINDS or key in have:
            continue
        ts = str(r.get("ts") or "")
        if ts <= last_by.get(key, ""):
            continue
        last_by[key] = ts
        out.append({
            "kind":          kind,
            "slot":          r.get("slot"),
            "cam_id":        cam,
            "cam_name":      r.get("cam_name"),
            "ts":            ts,
            "metric":        a.get("metric"),
            "observed":      a.get("observed"),
            "expected":      a.get("expected"),
            "snapshot_url":  r.get("snapshot_url"),
            "fullframe_url": r.get("snapshot_annotated_url")
                             or r.get("snapshot_url"),
        })
    return out


def fetch_window(db, window_hours: int
                 ) -> tuple[list[dict], list[dict], list[dict], set[str], dict]:
    cutoff = (dt.datetime.now(dt.timezone.utc)
              - dt.timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [d.to_dict() for d in
              db.collection("events").where("ts", ">=", cutoff).stream()]
    footfall = [d.to_dict() for d in
                db.collection("footfall").where("ts", ">=", cutoff).stream()]
    latest = [d.to_dict() for d in db.collection("latest").stream()]
    grid = (db.collection("config").document("grid").get().to_dict() or {})
    active = {str(s.get("slot_id") or "") for s in grid.get("slots") or []}
    # Backfill scene anomalies from footfall so the report surfaces
    # camera_obstructed / camera_dark / extreme_load regardless of whether
    # the collector was writing them into `events` (that path only landed
    # 2026-07-21).
    events = events + _scene_events_from_footfall(footfall, events)
    return events, footfall, latest, active, grid


def stale_from_latest(latest: list[dict],
                      now_utc: dt.datetime | None = None,
                      active_slots: set[str] | None = None,
                      window_ok_by_slot: dict | None = None) -> list[dict]:
    """Slots whose newest sample is old, OR fresh but last round was a MISS.

    Two failure modes surface here:
      * traditional stale: the collector stopped writing (`ts` gone old);
      * silent-miss: collector keeps writing but the recent samples all
        decoded to empty frames. The previous rule flagged any latest
        doc with `ok=0` - which caught momentary single-round misses on
        healthy cameras and read as "camera dead" (Sukhumvit at 20:00
        with thousands of ok samples got that label because the last
        round happened to miss). `window_ok_by_slot` lets us require
        BOTH a fresh latest miss AND < HEALTHY_OK_MIN successful rounds
        across the window; healthy cameras with one bad round are quiet.

    `active_slots` filters to the CURRENT grid - `latest` keeps documents
    of cameras that once ran and were since removed (observed live: the
    catalog-only tram camera showed up as '1462 minutes stale'), and
    those are history, not alarms.
    """
    HEALTHY_OK_MIN = 20        # samples in the window that count as healthy
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    out = []
    for d in latest:
        if active_slots is not None and str(d.get("slot") or "") not in active_slots:
            continue
        ts = str(d.get("ts") or "")
        try:
            t = dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S") \
                  .replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        age_min = int((now_utc - t).total_seconds() // 60)
        traditional_stale = age_min >= STALE_SLOT_MIN
        # ok is only set to 0 by collector.py on a MISS; a legacy record
        # without the field is treated as "unknown" (not flagged).
        silent_miss = d.get("ok") == 0
        # Healthy latest with a single miss? Do not alarm. Only flag when
        # the WINDOW itself carries <HEALTHY_OK_MIN ok rounds for the slot.
        ok_in_window = ((window_ok_by_slot or {})
                        .get(str(d.get("slot") or ""), None))
        if silent_miss and ok_in_window is not None \
                and ok_in_window >= HEALTHY_OK_MIN:
            silent_miss = False
        if traditional_stale or silent_miss:
            entry = {"cam": str(d.get("cam_name") or d.get("cam_id")
                                or d.get("slot") or "?"),
                     "age_min": age_min}
            if silent_miss and not traditional_stale:
                entry["reason"] = "producing empty frames"
            out.append(entry)
    return sorted(out, key=lambda s: -s["age_min"])


def fetch_last_training() -> dict | None:
    """Last gate record from the public cumulative history in Storage."""
    try:
        from app.pool_sync import _bucket_name, _http_get
        import time as _t
        bucket = os.environ.get("FIREBASE_STORAGE_BUCKET") or _bucket_name()
        if not bucket:
            return None
        raw = _http_get(f"https://storage.googleapis.com/{bucket}/"
                        f"training/history.jsonl?t={int(_t.time())}")
        last = None
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("event") in ("gate", "promoted"):
                last = rec
        return last
    except Exception:
        return None            # trainer never ran / offline - not an error


def fetch_review_stats() -> dict | None:
    """Operator-side labeling progress, from the reviews store the dashboard
    uploads to training/reviews.json at every Submit. Feeds the "you have
    labeled N frames" line - the training-status section was reading as
    'the model rejected itself'; the plain count reframes it as progress."""
    try:
        from app.pool_sync import _bucket_name, _http_get
        import time as _t
        bucket = os.environ.get("FIREBASE_STORAGE_BUCKET") or _bucket_name()
        if not bucket:
            return None
        raw = _http_get(f"https://storage.googleapis.com/{bucket}/"
                        f"training/reviews.json?t={int(_t.time())}")
        data = json.loads(raw.decode("utf-8"))
        frame_reviews = data.get("frame_reviews") or []
        crops = data.get("reviews") or []
        confirmed = sum(1 for fr in frame_reviews
                        for v in (fr.get("box_verdicts") or {}).values()
                        if v == "correct")
        rejected = sum(1 for fr in frame_reviews
                       for v in (fr.get("box_verdicts") or {}).values()
                       if v == "wrong")
        relabeled = sum(1 for fr in frame_reviews
                        for v in (fr.get("box_verdicts") or {}).values()
                        if isinstance(v, str) and v.startswith("relabel:"))
        missed = sum(len(fr.get("missed_detections") or [])
                     for fr in frame_reviews)
        return {"frames_labeled": len(frame_reviews),
                "crop_reviews":   len(crops),
                "boxes_confirmed": confirmed,
                "boxes_rejected":  rejected,
                "boxes_relabeled": relabeled,
                "missed_marked":  missed}
    except Exception:
        return None


# ---- send --------------------------------------------------------------------

def send_gmail(subject: str, text: str, html: str,
               attachments: list[Path] | None = None) -> None:
    """Send via Gmail SMTP. The 'mixed' outer wraps the visible body
    (alternative: text + html) and any attachments - the phone Gmail app
    then treats each attachment as a downloadable file rather than inline
    content, which is exactly what we want for the PDF report."""
    user = os.environ.get("GMAIL_USER")
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("DIGEST_TO") or user
    if not user or not pwd:
        raise SystemExit("GMAIL_USER and GMAIL_APP_PASSWORD must be set "
                         "(see /etc/turkey-footfall/digest.env)")

    outer = MIMEMultipart("mixed")
    outer["Subject"] = subject
    outer["From"] = user
    outer["To"] = to

    body = MIMEMultipart("alternative")
    body.attach(MIMEText(text, "plain", "utf-8"))
    body.attach(MIMEText(html, "html", "utf-8"))
    outer.attach(body)

    for path in attachments or []:
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            print(f"digest: skipping attachment {path}: {e}")
            continue
        part = MIMEBase("application", "pdf")
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="{Path(path).name}"')
        outer.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(user, pwd)
        s.sendmail(user, [to], outer.as_string())


def _build_pdf(now_il: dt.datetime, window_hours: int,
               event_groups: list[dict],
               cam_stats: list[dict], training: dict | None,
               reviews: dict | None,
               stale_slots: list[dict], footfall_count: int,
               total_events: int,
               out_path: Path, grid: dict | None = None,
               dominant: str | None = None,
               prev: dict | None = None,
               latest: list[dict] | None = None) -> Path | None:
    """Compose the PDF; returns the path or None if reportlab is missing
    (the plain-text mail still ships in that case, which is the pre-PDF
    behavior and better than silently dropping the whole run)."""
    try:
        from tools import report_pdf
    except ImportError as e:
        print(f"digest: reportlab not available ({e}) - "
              f"sending text-only mail")
        return None
    from app.pool_sync import _bucket_name
    bucket = os.environ.get("FIREBASE_STORAGE_BUCKET") or _bucket_name()
    picked_groups = report_pdf.pick_group_samples(event_groups)
    snapshots = report_pdf.fetch_snapshots_for_groups(picked_groups, bucket)
    # Grid thumbnails: the "live view" annotated frame the collector
    # publishes for each active slot (24h Storage lifecycle - always fresh).
    grid_thumbs = report_pdf.fetch_grid_thumbnails(grid, latest or [],
                                                   bucket)
    grid_label = (_COUNTRY_LABELS.get(dominant, dominant.title())
                  if dominant else _country_label(grid))
    return report_pdf.compose_pdf(
        out_path,
        now_il=now_il, window_hours=window_hours,
        events_by_kind=event_groups, cam_stats=cam_stats,
        training=training, stale_slots=stale_slots,
        snapshots=snapshots, kind_labels=KIND_LABELS,
        total_events=total_events, total_samples=footfall_count,
        training_lines=_training_lines(training, reviews, prev=prev),
        country_label=grid_label,
        grid_country_label=_country_label(grid),
        grid_cameras=_grid_cameras(grid),
        grid_thumbnails=grid_thumbs,
        dominant=dominant,
        current_country=(grid or {}).get("country"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window-hours", type=int, default=WINDOW_HOURS_DEFAULT)
    ap.add_argument("--dry-run", action="store_true",
                    help="compose and save the PDF locally; do not send")
    ap.add_argument("--pdf-out", default=None,
                    help="destination path for the PDF "
                         "(default: temp file when sending, "
                         "./daily_digest.pdf on --dry-run)")
    args = ap.parse_args()

    db = _firestore()
    events, footfall, latest, active, grid = fetch_window(db, args.window_hours)
    groups = aggregate_events(events)
    cam_stats = footfall_stats(footfall)
    training = fetch_last_training()
    reviews = fetch_review_stats()
    prev_state = _load_last_report()
    # Per-slot healthy-round tally for the aggregate stale check - a slot
    # with 100+ ok rounds this window is not "producing empty frames" just
    # because the last one happened to miss.
    window_ok_by_slot: dict = {}
    for r in footfall:
        if r.get("ok") == 1:
            k = str(r.get("slot") or "")
            window_ok_by_slot[k] = window_ok_by_slot.get(k, 0) + 1
    stale = stale_from_latest(latest, active_slots=active or None,
                              window_ok_by_slot=window_ok_by_slot)
    dominant = dominant_country(footfall)
    now_il = _israel_now()

    subject, text, html = compose_digest(
        now_il, args.window_hours, groups, cam_stats, training, stale,
        reviews=reviews, grid=grid, dominant=dominant,
        prev=prev_state)

    # Country slug for the PDF filename: the DATA majority when it exists,
    # not the momentary grid - matches the report's subject line.
    country_slug = dominant or str((grid or {}).get("country") or "grid")
    default_pdf = (Path("./daily_digest.pdf") if args.dry_run
                   else Path(tempfile.mkdtemp(prefix="digest-"))
                        / f"{country_slug}_report_{now_il.strftime('%Y%m%d_%H%M')}.pdf")
    pdf_path = Path(args.pdf_out) if args.pdf_out else default_pdf
    built = _build_pdf(now_il, args.window_hours, groups, cam_stats,
                       training, reviews, stale, len(footfall),
                       total_events=len(events), out_path=pdf_path,
                       grid=grid, dominant=dominant, prev=prev_state,
                       latest=latest)

    if args.dry_run:
        print(f"SUBJECT: {subject}\n\n{text}")
        if built:
            print(f"\nPDF: {built.resolve()} "
                  f"({built.stat().st_size / 1024:.0f} KB)")
        return
    send_gmail(subject, text, html,
               attachments=[built] if built else None)
    # Snapshot for the NEXT report's delta line. Only persist after a
    # successful send so a crashed run doesn't retroactively silence the
    # unsent report's numbers.
    _save_last_report({
        "frames_labeled":  (reviews or {}).get("frames_labeled", 0),
        "boxes_confirmed": (reviews or {}).get("boxes_confirmed", 0),
        "boxes_rejected":  (reviews or {}).get("boxes_rejected", 0),
        "missed_marked":   (reviews or {}).get("missed_marked", 0),
        "last_training":   training,
        "sent_at":         now_il.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    print(f"digest sent: {subject} ({len(events)} events, "
          f"{len(footfall)} samples"
          + (f", PDF {built.stat().st_size // 1024} KB attached"
             if built else ", text only") + ")")


if __name__ == "__main__":
    main()
