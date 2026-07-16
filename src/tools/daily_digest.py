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
    for r in records:
        cam = str(r.get("cam_name") or r.get("cam_id") or "?")
        c = cams.setdefault(cam, {"cam": cam, "samples": 0, "misses": 0,
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


def _fmt_ts(ts_iso: str) -> str:
    """UTC ISO -> Israel-time HH:MM (the operator lives on this clock)."""
    try:
        from zoneinfo import ZoneInfo
        t = dt.datetime.strptime(str(ts_iso)[:19], "%Y-%m-%dT%H:%M:%S")
        t = t.replace(tzinfo=dt.timezone.utc).astimezone(ZoneInfo("Asia/Jerusalem"))
        return t.strftime("%H:%M")
    except (ValueError, TypeError):
        return str(ts_iso)[:16]


def _training_lines(training: dict | None,
                    reviews: dict | None) -> list[str]:
    """One or two friendly sentences summarizing where the learning loop
    stands. The old phrasing ('rejected at gate - head_run2.pt') read as
    'the model is broken' - it isn't; the gate did its job on a tiny
    dataset. Frame it as 'here's how many labels you have, here's what
    the trainer did with them'."""
    lines: list[str] = []
    r = reviews or {}
    if r.get("frames_labeled"):
        stats = (f"You have labeled {r['frames_labeled']} frames "
                 f"({r.get('boxes_confirmed', 0)} confirmed, "
                 f"{r.get('boxes_rejected', 0)} rejected, "
                 f"{r.get('missed_marked', 0)} objects you added).")
        lines.append(stats)
    else:
        lines.append("No frames labeled yet - open the Reinforcement "
                     "Learning tab in the dashboard and start tagging.")
    if training:
        when = str(training.get("at") or "")[:10]
        cand = training.get("candidate") or training.get("file") or "?"
        if training.get("promoted"):
            lines.append(f"Cloud training on {when} promoted a new "
                         f"detection head ({cand}). The VM has already "
                         f"picked it up.")
        else:
            lines.append(f"Cloud training on {when} did not improve on the "
                         f"baseline yet ({cand}). More diverse labels will "
                         f"give the next run something new to learn from.")
    else:
        lines.append("No cloud training run has executed yet. Trigger "
                     "'train-head' in the GitHub Actions tab once you have "
                     "20+ labeled frames.")
    return lines


def compose_digest(now_il: dt.datetime, window_hours: int,
                   event_groups: list[dict], cam_stats: list[dict],
                   training: dict | None, stale_slots: list[dict],
                   reviews: dict | None = None,
                   ) -> tuple[str, str, str]:
    """Returns (subject, plain_text, html). English throughout, phone-first."""
    part = "Midday report" if now_il.hour < 16 else "Evening report"
    subject = f"Konya - {part} {now_il.strftime('%d.%m')}"

    lines: list[str] = [f"{part} - last {window_hours} hours", ""]
    html: list[str] = [
        '<div style="font-family:Arial,sans-serif;font-size:15px">',
        f"<h2 style='margin:0 0 12px'>{subject}</h2>",
    ]

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
            row = (f"{g['label']} - {g['cam']}"
                   + (f" (x{g['count']})" if g["count"] > 1 else "")
                   + f", latest {t}")
            lines.append("  - " + row)
            html.append(
                f"<tr><td style='border-bottom:1px solid #ddd'>{g['label']}</td>"
                f"<td style='border-bottom:1px solid #ddd'>{g['cam']}</td>"
                f"<td style='border-bottom:1px solid #ddd'>x{g['count']}</td>"
                f"<td style='border-bottom:1px solid #ddd'>{t}</td></tr>")
        html.append("</table>")
    else:
        lines.append("  Quiet - no anomalies in this window.")
        html.append("<p>Quiet - no anomalies in this window.</p>")
    lines.append("")

    lines.append("Activity peaks:")
    html.append("<h3 style='margin:14px 0 6px'>Activity peaks</h3>")
    if cam_stats:
        html.append("<table cellpadding='4' style='border-collapse:collapse'>"
                    "<tr><th align='left'>Camera</th><th>Peak people</th>"
                    "<th>Peak vehicles</th><th>Typical traffic</th></tr>")
        for c in cam_stats:
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
    else:
        lines.append("  No footfall samples in this window - check the VM journal.")
        html.append("<p>No footfall samples in this window - check the VM journal.</p>")
    lines.append("")

    lines.append("Learning:")
    html.append("<h3 style='margin:14px 0 6px'>Learning</h3>")
    for line in _training_lines(training, reviews):
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


def fetch_window(db, window_hours: int
                 ) -> tuple[list[dict], list[dict], list[dict], set[str]]:
    cutoff = (dt.datetime.now(dt.timezone.utc)
              - dt.timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [d.to_dict() for d in
              db.collection("events").where("ts", ">=", cutoff).stream()]
    footfall = [d.to_dict() for d in
                db.collection("footfall").where("ts", ">=", cutoff).stream()]
    latest = [d.to_dict() for d in db.collection("latest").stream()]
    grid = (db.collection("config").document("grid").get().to_dict() or {})
    active = {str(s.get("slot_id") or "") for s in grid.get("slots") or []}
    return events, footfall, latest, active


def stale_from_latest(latest: list[dict],
                      now_utc: dt.datetime | None = None,
                      active_slots: set[str] | None = None) -> list[dict]:
    """Slots whose newest sample is old, OR fresh but last round was a MISS.

    Two failure modes surface here:
      * traditional stale: the collector stopped writing (`ts` gone old);
      * silent-miss: collector keeps writing but every round decodes to
        empty frames - `ts` stays fresh but `ok=0`. Without the second
        check the report reads "reporting normally" while every peak is 0.

    `active_slots` filters to the CURRENT grid - `latest` keeps documents
    of cameras that once ran and were since removed (observed live: the
    catalog-only tram camera showed up as '1462 minutes stale'), and
    those are history, not alarms.
    """
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
               out_path: Path) -> Path | None:
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
    return report_pdf.compose_pdf(
        out_path,
        now_il=now_il, window_hours=window_hours,
        events_by_kind=event_groups, cam_stats=cam_stats,
        training=training, stale_slots=stale_slots,
        snapshots=snapshots, kind_labels=KIND_LABELS,
        total_events=total_events, total_samples=footfall_count,
        training_lines=_training_lines(training, reviews))


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
    events, footfall, latest, active = fetch_window(db, args.window_hours)
    groups = aggregate_events(events)
    cam_stats = footfall_stats(footfall)
    training = fetch_last_training()
    reviews = fetch_review_stats()
    stale = stale_from_latest(latest, active_slots=active or None)
    now_il = _israel_now()

    subject, text, html = compose_digest(
        now_il, args.window_hours, groups, cam_stats, training, stale,
        reviews=reviews)

    default_pdf = (Path("./daily_digest.pdf") if args.dry_run
                   else Path(tempfile.mkdtemp(prefix="digest-"))
                        / f"konya_report_{now_il.strftime('%Y%m%d_%H%M')}.pdf")
    pdf_path = Path(args.pdf_out) if args.pdf_out else default_pdf
    built = _build_pdf(now_il, args.window_hours, groups, cam_stats,
                       training, reviews, stale, len(footfall),
                       total_events=len(events), out_path=pdf_path)

    if args.dry_run:
        print(f"SUBJECT: {subject}\n\n{text}")
        if built:
            print(f"\nPDF: {built.resolve()} "
                  f"({built.stat().st_size / 1024:.0f} KB)")
        return
    send_gmail(subject, text, html,
               attachments=[built] if built else None)
    print(f"digest sent: {subject} ({len(events)} events, "
          f"{len(footfall)} samples"
          + (f", PDF {built.stat().st_size // 1024} KB attached"
             if built else ", text only") + ")")


if __name__ == "__main__":
    main()
