"""Rich English PDF report - the visual evidence layer over daily_digest.

The plain-text/HTML digest is fine for a phone glance ("Kulturpark loitering
x14"), but every ambiguous event begs the same question: WHICH crop? Which
frame? WHICH pole did the model call a person? The events collection already
carries a public ``fullframe_url`` per event; this module downloads those
into a paginated PDF - tables for the counts, actual images for the
top-priority scenes - and returns the file so daily_digest can attach it to
the email.

Design decisions:

* Layout - A4 portrait, English throughout, LTR. Section headers left-
  aligned; page title centered.
* Image selection - one representative event per anomaly KIND, priority-
  ordered (obstructed > dark > extreme_load > loiter > returning). Cap at
  8 pictures so the PDF stays under ~1 MB for Gmail's inline preview.
* Failure mode - never raise. A blank snapshot URL, a slow bucket, or a
  PIL error degrades the section (skip the image) but the caller still
  gets a PDF path back.
"""
from __future__ import annotations

import datetime as dt
from io import BytesIO
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# Kinds in order of "operator wants to see this first" - a blocked camera is
# strictly more urgent than yet another loiter alert. Kinds not listed here
# get appended in first-seen order.
KIND_PRIORITY = (
    "camera_obstructed", "camera_dark", "extreme_load",
    "loiter", "returning",
)

# Cap so the emailed PDF stays inline-previewable on the phone.
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 4 * 1024 * 1024
IMAGE_DOWNLOAD_TIMEOUT_S = 15
# Downscale + re-encode: 8 fullframe HD snapshots at q=75 -> ~600 KB PDF vs
# 3+ MB raw, so Gmail's phone preview still opens instantly.
PDF_IMAGE_MAX_WIDTH_PX = 900
PDF_IMAGE_QUALITY = 75


def pick_event_samples(events: list[dict], max_total: int = MAX_IMAGES
                       ) -> list[dict]:
    """Best representative events across kinds.

    For each kind that appears in the window we take the LAST occurrence
    (so time-of-day makes sense in the caption); we cycle across kinds in
    the priority order above, then round-robin any extras. A kind with no
    events contributes nothing. Missing snapshot_url disqualifies an event
    from the visual layer (the row still counts in the aggregation table)."""
    by_kind: dict[str, list[dict]] = {}
    for e in events:
        if not e.get("snapshot_url") and not e.get("fullframe_url"):
            continue
        by_kind.setdefault(str(e.get("kind") or "?"), []).append(e)
    for group in by_kind.values():
        group.sort(key=lambda e: str(e.get("ts") or ""), reverse=True)

    kinds_ordered = [k for k in KIND_PRIORITY if k in by_kind] + \
                    [k for k in by_kind if k not in KIND_PRIORITY]
    picks: list[dict] = []
    round_idx = 0
    while len(picks) < max_total and kinds_ordered:
        progress = False
        for k in list(kinds_ordered):
            if round_idx < len(by_kind[k]):
                picks.append(by_kind[k][round_idx])
                progress = True
                if len(picks) >= max_total:
                    break
        if not progress:
            break
        round_idx += 1
    return picks


def _http_bytes(url: str, timeout: float = IMAGE_DOWNLOAD_TIMEOUT_S
                ) -> bytes | None:
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "digest/2"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return data if 1024 < len(data) < MAX_IMAGE_BYTES else None
    except Exception:
        return None


def _shrink_jpeg(data: bytes,
                 max_width: int = PDF_IMAGE_MAX_WIDTH_PX,
                 quality: int = PDF_IMAGE_QUALITY) -> bytes:
    """Downscale + re-encode. Silently returns the input on any PIL error,
    since a slightly-heavy PDF still ships where a raised exception would
    drop the whole report."""
    try:
        from PIL import Image as PILImage
        im = PILImage.open(BytesIO(data))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if im.width > max_width:
            new_h = int(im.height * max_width / im.width)
            im = im.resize((max_width, new_h), PILImage.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return data


def fetch_snapshots(picks: list[dict],
                    downloader=_http_bytes,
                    shrink: bool = True) -> list[tuple[dict, bytes]]:
    """Download each pick's image. Prefer fullframe_url (context), fall back
    to snapshot_url (crop-only). Silently drops picks whose downloads fail."""
    out: list[tuple[dict, bytes]] = []
    for e in picks:
        for key in ("fullframe_url", "snapshot_url"):
            url = e.get(key)
            if not url:
                continue
            data = downloader(url)
            if data:
                out.append((e, _shrink_jpeg(data) if shrink else data))
                break
    return out


def _fmt_ts(ts_iso: str) -> str:
    """UTC ISO -> Israel-time HH:MM (the operator's clock)."""
    try:
        from zoneinfo import ZoneInfo
        t = dt.datetime.strptime(str(ts_iso)[:19], "%Y-%m-%dT%H:%M:%S")
        t = t.replace(tzinfo=dt.timezone.utc).astimezone(ZoneInfo("Asia/Jerusalem"))
        return t.strftime("%H:%M")
    except (ValueError, TypeError):
        return str(ts_iso)[:16]


def _fmt_date(now: dt.datetime) -> str:
    return now.strftime("%d.%m.%Y")


def _table_style() -> TableStyle:
    return TableStyle([
        ("FONTNAME",       (0, 0),  (-1, -1), "Helvetica"),
        ("FONTNAME",       (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0),  (-1, -1), 10),
        ("BACKGROUND",     (0, 0),  (-1, 0),  colors.HexColor("#0f172a")),
        ("TEXTCOLOR",      (0, 0),  (-1, 0),  colors.white),
        ("ALIGN",          (0, 0),  (-1, 0),  "CENTER"),
        ("ALIGN",          (0, 1),  (0, -1),  "LEFT"),
        ("ALIGN",          (1, 1),  (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0),  (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0),  (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0),  (-1, -1), 6),
        ("GRID",           (0, 0),  (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1),  (-1, -1), [colors.white,
                                               colors.HexColor("#f8fafc")]),
    ])


def _event_caption(e: dict, kind_labels: dict[str, str]) -> str:
    """One-line English caption for the image evidence page."""
    kind = str(e.get("kind") or "?")
    label = kind_labels.get(kind, kind)
    cam = str(e.get("cam_name") or e.get("cam_id") or e.get("slot") or "?")
    parts: list[str] = [label, cam, _fmt_ts(e.get("ts"))]
    dur = e.get("duration_sec")
    if isinstance(dur, (int, float)) and dur > 0:
        parts.append(f"{int(dur)} sec")
    cls = e.get("cls")
    if cls:
        parts.append(str(cls))
    return " · ".join(parts)


def compose_pdf(out_path: str | Path, *,
                now_il: dt.datetime,
                window_hours: int,
                events_by_kind: list[dict],
                cam_stats: list[dict],
                training: dict | None,
                stale_slots: list[dict],
                snapshots: Iterable[tuple[dict, bytes]],
                kind_labels: dict[str, str],
                total_events: int,
                total_samples: int) -> Path:
    """Compose the phone-oriented English PDF and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20,
                                alignment=TA_CENTER, leading=26, spaceAfter=10),
        "sub":   ParagraphStyle("sub", fontName="Helvetica", fontSize=11,
                                alignment=TA_CENTER, leading=14, spaceAfter=16,
                                textColor=colors.HexColor("#475569")),
        "h":     ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14,
                                alignment=TA_LEFT, spaceBefore=14,
                                spaceAfter=6, textColor=colors.HexColor("#0f172a")),
        "body":  ParagraphStyle("body", fontName="Helvetica", fontSize=11,
                                alignment=TA_LEFT, leading=15),
        "cap":   ParagraphStyle("cap", fontName="Helvetica", fontSize=10,
                                alignment=TA_LEFT,
                                textColor=colors.HexColor("#475569"),
                                spaceAfter=10),
        "warn":  ParagraphStyle("warn", fontName="Helvetica-Bold", fontSize=11,
                                alignment=TA_LEFT, spaceAfter=4,
                                textColor=colors.HexColor("#b91c1c")),
        "ok":    ParagraphStyle("ok", fontName="Helvetica", fontSize=11,
                                alignment=TA_LEFT,
                                textColor=colors.HexColor("#166534")),
    }

    story = []
    part = "Midday Report" if now_il.hour < 16 else "Evening Report"
    story.append(Paragraph(f"Konya - Activity Summary - {part}",
                           styles["title"]))
    story.append(Paragraph(f"{_fmt_date(now_il)}  ·  last {window_hours} hours",
                           styles["sub"]))

    # KPI cards - two columns
    events_word = "anomaly" if total_events == 1 else "anomalies"
    samples_word = "sample" if total_samples == 1 else "samples"
    kpi_rows = [[f"{total_events} {events_word}",
                 f"{total_samples} {samples_word}"]]
    kpi = Table(kpi_rows, colWidths=[9*cm, 9*cm])
    kpi.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 13),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (0, 0),   colors.HexColor("#fef3c7")),
        ("BACKGROUND", (1, 0), (1, 0),   colors.HexColor("#dbeafe")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX",        (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("LINEAFTER",  (0, 0), (0, 0),   0.25, colors.HexColor("#cbd5e1")),
    ]))
    story.append(kpi)

    story.append(Paragraph("Camera Status", styles["h"]))
    if stale_slots:
        for s in stale_slots:
            story.append(Paragraph(
                f"⚠  {s['cam']} - not reporting for {s['age_min']} minutes",
                styles["warn"]))
    else:
        story.append(Paragraph("✓  All cameras active and reporting normally",
                               styles["ok"]))

    # Camera names in these tables are long ("Konya - Hukumet Meydani /
    # Sarraflar Yeralti Carsisi"); wrapping them as Paragraph flowables
    # keeps the cell inside its column width instead of running over its
    # neighbor. Plain strings do not wrap in reportlab tables.
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=10,
                                alignment=TA_LEFT, leading=12)

    # Aggregated anomalies table
    story.append(Paragraph("Anomalies by Type and Camera", styles["h"]))
    if events_by_kind:
        header = ["Type", "Camera", "Last", "Count"]
        body = [header]
        for g in events_by_kind:
            body.append([Paragraph(g["label"], cell_style),
                         Paragraph(g["cam"], cell_style),
                         _fmt_ts(g["last_ts"]), str(g["count"])])
        tbl = Table(body, colWidths=[5.5*cm, 6.5*cm, 2.2*cm, 2.4*cm],
                    repeatRows=1)
        tbl.setStyle(_table_style())
        story.append(tbl)
    else:
        story.append(Paragraph("Quiet - no anomalies in this window.",
                               styles["body"]))

    # Camera activity peaks
    story.append(Paragraph("Activity Peaks by Camera", styles["h"]))
    if cam_stats:
        header = ["Camera", "Peak People", "Peak Vehicles", "Typical Traffic"]
        body = [header]
        for c in cam_stats:
            people = str(c["peak_person"])
            if c["peak_person_ts"]:
                people += f" at {_fmt_ts(c['peak_person_ts'])}"
            spd = f"~{c['typ_kmh']:.0f} km/h" if c["typ_kmh"] > 0 else "-"
            body.append([Paragraph(c["cam"], cell_style),
                         people, str(c["peak_vehicles"]), spd])
        tbl = Table(body, colWidths=[7*cm, 3.6*cm, 2.5*cm, 3.5*cm],
                    repeatRows=1)
        tbl.setStyle(_table_style())
        story.append(tbl)
    else:
        story.append(Paragraph("No footfall samples in this window.",
                               styles["body"]))

    # Training status
    story.append(Paragraph("Model Training Status", styles["h"]))
    if training:
        verdict = ("PROMOTED new head" if training.get("promoted")
                   else "rejected at gate")
        cand = training.get("candidate") or training.get("file") or "?"
        when = str(training.get("at") or "")[:10]
        story.append(Paragraph(
            f"Last training run ({when}): {verdict} - {cand}",
            styles["body"]))
        for r in (training.get("reasons") or [])[:3]:
            story.append(Paragraph(f"• {r}", styles["cap"]))
    else:
        story.append(Paragraph("No cloud training run has executed yet.",
                               styles["body"]))

    # Visual evidence pages
    snap_list = list(snapshots)
    if snap_list:
        story.append(PageBreak())
        story.append(Paragraph(
            "Anomaly Samples - Original Camera Snapshots",
            styles["h"]))
        story.append(Paragraph(
            "One representative snapshot per anomaly type, saved to the "
            "cloud with the camera, time, and additional details.",
            styles["cap"]))
        cap_style = ParagraphStyle(
            "img_cap", fontName="Helvetica", fontSize=10, alignment=TA_LEFT,
            leading=13, textColor=colors.HexColor("#334155"))
        for e, jpeg in snap_list:
            img = Image(BytesIO(jpeg), width=15*cm, height=8.5*cm,
                        kind="proportional")
            caption = Paragraph(_event_caption(e, kind_labels), cap_style)
            frame = Table([[img], [caption]], colWidths=[16.2*cm])
            frame.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX",           (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
                ("LINEABOVE",     (0, 1), (-1, 1),  0.5,  colors.HexColor("#e2e8f0")),
                ("TOPPADDING",    (0, 0), (-1, 0),  8),
                ("BOTTOMPADDING", (0, 0), (-1, 0),  6),
                ("TOPPADDING",    (0, 1), (-1, 1),  8),
                ("BOTTOMPADDING", (0, 1), (-1, 1),  10),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
                ("ALIGN",         (0, 1), (-1, 1),  "LEFT"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(frame)
            story.append(Spacer(1, 0.4*cm))
    else:
        story.append(Paragraph(
            "No images attached (image bucket is empty or events lack "
            "snapshot_url).", styles["cap"]))

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm,
                            title=f"Konya activity report {_fmt_date(now_il)}",
                            author="turkey-collector")
    doc.build(story)
    return out_path
