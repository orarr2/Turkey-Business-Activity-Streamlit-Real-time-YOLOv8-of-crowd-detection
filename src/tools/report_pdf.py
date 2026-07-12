"""Rich Hebrew PDF report - the visual evidence layer over daily_digest.

The plain-text/HTML digest is fine for a phone glance ("Kulturpark loitering
x14"), but every ambiguous event begs the same question: WHICH crop? Which
frame? WHICH pole did the model call a person? The events collection already
carries a public ``fullframe_url`` per event; this module downloads those
into a paginated Hebrew PDF - tables for the counts, actual images for the
top-priority scenes - and returns the file so daily_digest can attach it to
the email.

Design decisions:

* Layout - A4 portrait, RTL throughout. Text goes through python-bidi
  (``get_display``) once so a downstream RTL-agnostic renderer (reportlab
  is LTR-only) shows the letters in their expected visual order.
* Fonts - DejaVuSans, shipped by ``fonts-dejavu-core`` on Debian and thus
  the GCP VM's install.sh already installs it. On developer machines that
  do not have DejaVu (typical Windows), we fall through to Arial/Segoe as
  fallbacks; the built-in Helvetica has zero Hebrew glyphs so we never let
  it be the final answer if we can help it.
* Image selection - one representative event per anomaly KIND, priority-
  ordered (obstructed > dark > extreme_load > loiter > returning). Cap at
  8 pictures so the PDF stays under ~2 MB for Gmail's inline preview to
  keep working on the phone.
* Failure mode - never raise. A blank snapshot URL, a slow bucket, or a
  missing font degrades the section (skip the image, use fallback font)
  but the caller still gets a PDF path back.
"""
from __future__ import annotations

import datetime as dt
from io import BytesIO
from pathlib import Path
from typing import Iterable

try:
    from bidi.algorithm import get_display
except ImportError:                    # pragma: no cover - bidi ships with weasyprint / other
    def get_display(s):                # type: ignore[no-redef]
        return s


# ---- Hebrew shaping helpers -------------------------------------------------
#
# python-bidi 0.6 has a known bug in mixed-content strings: when a compound
# caption like "kind · cam · time · 319 שניות · cls" runs through get_display
# in one call, the LTR "319" and the RTL "שניות" get PLACED IN SEPARATE
# CELLS of the visual layout (a debug walk over its output confirms this on
# every version tested). The report shipped from the VM produced captions
# reading as ``person · שניות konya_kulturpark · 13:36 · 319 · שהייה...`` -
# each atom right on its own but the joined line unreadable.
#
# Workaround (below): compose the visual string ourselves. Reportlab is
# LTR-only, so a Hebrew-first line renders with the FIRST logical word at
# the RIGHT-MOST visual position. We hand the engine parts in REVERSE
# logical order, and call get_display only on Hebrew-containing parts.
# Pure-LTR parts (English, digits, timestamps) never leave get_display's
# domain of mishandling in the first place.

def _has_hebrew(s: str) -> bool:
    return any(0x0590 <= ord(c) < 0x0600 for c in s)


def _he(s) -> str:
    """Single-piece Hebrew reordering. Safe for headings, table cells - a
    self-contained phrase where python-bidi behaves correctly."""
    s = str(s)
    return get_display(s) if _has_hebrew(s) else s


def _he_join(parts, sep: str = " · ") -> str:
    """Compose a mixed-direction line in visual order for an LTR engine.
    See the module-level note on why we do not hand the whole string to
    get_display in one shot."""
    return sep.join(_he(p) for p in reversed(list(parts)) if p is not None and str(p) != "")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# Kinds in order of "operator wants to see this first" - a blocked camera is
# strictly more urgent than yet another loiter alert. Kinds not listed here
# get appended in first-seen order.
KIND_PRIORITY = (
    "camera_obstructed", "camera_dark", "extreme_load",
    "loiter", "returning",
)

# Cap so the emailed PDF stays inline-previewable on the phone. 8 x ~120 KB
# images + text pages sits under 1.5 MB after the re-encode below.
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 4 * 1024 * 1024
IMAGE_DOWNLOAD_TIMEOUT_S = 15
# The collector uploads full-frame HD (~1920x1080, 300-500 KB). Downscaling
# to page-width resolution and re-encoding at JPEG q=75 cuts each image to
# ~40-80 KB with no visible loss at A4 print size and turns a 3.3 MB PDF
# into a ~600 KB one - fast enough for Gmail's phone preview.
PDF_IMAGE_MAX_WIDTH_PX = 900
PDF_IMAGE_QUALITY = 75

_FONT_CANDIDATES = (
    # Debian / GCP VM (fonts-dejavu-core - always installed by install.sh):
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    # Windows dev machines - Arial has partial Hebrew coverage, David has full:
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/david.ttf", "C:/Windows/Fonts/davidbd.ttf"),
    # Common macOS location:
    ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc"),
)

_FONT_CACHE: tuple[str, str] | None = None


def _register_fonts() -> tuple[str, str]:
    """Register the first candidate whose regular font exists. Returns
    (regular, bold) reportlab font names. Falls back to Helvetica when
    nothing usable is found - the operator sees Hebrew as square boxes,
    which is a better failure mode than a crashed report at 12:00."""
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    for reg_path, bold_path in _FONT_CANDIDATES:
        if not Path(reg_path).is_file():
            continue
        reg_name = "ReportRegular"
        bold_name = "ReportBold"
        try:
            pdfmetrics.registerFont(TTFont(reg_name, reg_path))
        except Exception:
            continue
        if Path(bold_path).is_file() and bold_path != reg_path:
            try:
                pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            except Exception:
                bold_name = reg_name
        else:
            bold_name = reg_name
        _FONT_CACHE = (reg_name, bold_name)
        return _FONT_CACHE
    _FONT_CACHE = ("Helvetica", "Helvetica-Bold")
    return _FONT_CACHE


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
    """Downscale + re-encode to page-friendly resolution. Silently returns
    the input on any PIL error, since a slightly-heavy PDF still ships
    where a raised exception would drop the whole report."""
    try:
        from PIL import Image as PILImage
        buf_in = BytesIO(data)
        im = PILImage.open(buf_in)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if im.width > max_width:
            new_h = int(im.height * max_width / im.width)
            im = im.resize((max_width, new_h), PILImage.LANCZOS)
        buf_out = BytesIO()
        im.save(buf_out, "JPEG", quality=quality, optimize=True)
        return buf_out.getvalue()
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


def _fmt_ts_he(ts_iso: str) -> str:
    """UTC ISO -> Israel-time HH:MM (duplicated from daily_digest to keep this
    module self-contained; it stays small enough that copying beats coupling)."""
    try:
        from zoneinfo import ZoneInfo
        t = dt.datetime.strptime(str(ts_iso)[:19], "%Y-%m-%dT%H:%M:%S")
        t = t.replace(tzinfo=dt.timezone.utc).astimezone(ZoneInfo("Asia/Jerusalem"))
        return t.strftime("%H:%M")
    except (ValueError, TypeError):
        return str(ts_iso)[:16]


def _fmt_date_he(now: dt.datetime) -> str:
    return now.strftime("%d.%m.%Y")


def _table_style(reg: str, bold: str) -> TableStyle:
    return TableStyle([
        ("FONTNAME",       (0, 0),  (-1, -1), reg),
        ("FONTNAME",       (0, 0),  (-1, 0),  bold),
        ("FONTSIZE",       (0, 0),  (-1, -1), 10),
        ("BACKGROUND",     (0, 0),  (-1, 0),  colors.HexColor("#0f172a")),
        ("TEXTCOLOR",      (0, 0),  (-1, 0),  colors.white),
        ("ALIGN",          (0, 0),  (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0),  (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0),  (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0),  (-1, -1), 6),
        ("GRID",           (0, 0),  (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1),  (-1, -1), [colors.white,
                                               colors.HexColor("#f8fafc")]),
    ])


def _event_kind_he(e: dict, kind_labels: dict[str, str]) -> str:
    kind = str(e.get("kind") or "?")
    return kind_labels.get(kind, kind)


def _event_caption(e: dict, kind_labels: dict[str, str]) -> str:
    """Visual-order caption ready for reportlab. Composed from LOGICAL
    parts via ``_he_join``, so a Hebrew reader looking at the PDF from the
    right sees them in kind → camera → time → duration → class order."""
    kind_he = _event_kind_he(e, kind_labels)
    cam = str(e.get("cam_name") or e.get("cam_id") or e.get("slot") or "?")
    parts: list[str] = [kind_he, cam, _fmt_ts_he(e.get("ts"))]
    dur = e.get("duration_sec")
    if isinstance(dur, (int, float)) and dur > 0:
        parts.append(f"{int(dur)} שניות")
    cls = e.get("cls")
    if cls:
        parts.append(str(cls))
    return _he_join(parts)


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
    """Compose the phone-oriented Hebrew PDF and return its path."""
    reg, bold = _register_fonts()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "title": ParagraphStyle("title", fontName=bold, fontSize=22,
                                alignment=TA_CENTER, spaceAfter=4),
        "sub":   ParagraphStyle("sub", fontName=reg, fontSize=11,
                                alignment=TA_CENTER, spaceAfter=14,
                                textColor=colors.HexColor("#475569")),
        "h":     ParagraphStyle("h", fontName=bold, fontSize=14,
                                alignment=TA_RIGHT, spaceBefore=14,
                                spaceAfter=6, textColor=colors.HexColor("#0f172a")),
        "body":  ParagraphStyle("body", fontName=reg, fontSize=11,
                                alignment=TA_RIGHT, leading=15),
        "cap":   ParagraphStyle("cap", fontName=reg, fontSize=10,
                                alignment=TA_RIGHT,
                                textColor=colors.HexColor("#475569"),
                                spaceAfter=10),
        "warn":  ParagraphStyle("warn", fontName=bold, fontSize=11,
                                alignment=TA_RIGHT, spaceAfter=4,
                                textColor=colors.HexColor("#b91c1c")),
        "ok":    ParagraphStyle("ok", fontName=reg, fontSize=11,
                                alignment=TA_RIGHT,
                                textColor=colors.HexColor("#166534")),
    }

    story = []
    part = "צהריים" if now_il.hour < 16 else "ערב"
    # Parentheses render mirrored via python-bidi (unresolved bracket-pair
    # mirroring in 0.6). A dash keeps the same intent without the fight.
    story.append(Paragraph(
        _he_join(["קוניה", "סיכום פעילות", part], sep=" – "),
        styles["title"]))
    story.append(Paragraph(
        _he_join([f"חלון {window_hours} שעות אחורה", _fmt_date_he(now_il)],
                 sep="  ·  "),
        styles["sub"]))

    # Two-column snapshot: totals + health. Number + Hebrew noun uses the
    # compound-safe join so the visual layout keeps them together.
    kpi_rows = [
        [_he_join([f"{total_events}", "חריגים"], sep=" "),
         _he_join([f"{total_samples}", "דגימות"], sep=" ")],
    ]
    kpi = Table(kpi_rows, colWidths=[9*cm, 9*cm])
    kpi.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), bold),
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

    story.append(Paragraph(_he("סטטוס מצלמות"), styles["h"]))
    if stale_slots:
        for s in stale_slots:
            story.append(Paragraph(
                _he_join(["⚠", s["cam"], "לא מדווחת כבר",
                          f"{s['age_min']} דקות"], sep=" "),
                styles["warn"]))
    else:
        story.append(Paragraph(
            _he("✓  כל המצלמות פעילות ומדווחות כסדרן"), styles["ok"]))

    # Aggregated anomalies table. The header CELLS get standard get_display;
    # each body cell's text is either single-script (safe for get_display)
    # or numeric-only (kept as-is).
    story.append(Paragraph(_he("סיכום חריגים לפי סוג ומצלמה"), styles["h"]))
    if events_by_kind:
        header = [_he(h) for h in ("מספר מופעים", "אחרון", "מצלמה", "סוג חריגה")]
        body = [header]
        for g in events_by_kind:
            body.append([str(g["count"]),
                         _fmt_ts_he(g["last_ts"]),
                         _he(g["cam"]),
                         _he(g["kind_he"])])
        tbl = Table(body, colWidths=[2.4*cm, 2.2*cm, 6*cm, 6*cm],
                    repeatRows=1)
        tbl.setStyle(_table_style(reg, bold))
        story.append(tbl)
    else:
        story.append(Paragraph(_he("שקט - לא נרשם אף חריג בחלון הזה."),
                               styles["body"]))

    # Camera activity peaks. Speed cell is number+Hebrew (needs _he_join);
    # the "N ב-HH:MM" people cell is likewise mixed.
    story.append(Paragraph(_he("שיאי פעילות לפי מצלמה"), styles["h"]))
    if cam_stats:
        header = [_he(h) for h in ("תנועה אופיינית", "שיא רכבים",
                                    "שיא אנשים", "מצלמה")]
        body = [header]
        for c in cam_stats:
            if c["typ_kmh"] > 0:
                spd = _he_join([f"~{c['typ_kmh']:.0f}", "קמ״ש"], sep=" ")
            else:
                spd = "-"
            if c["peak_person_ts"]:
                people = _he_join([str(c['peak_person']), "ב-" +
                                   _fmt_ts_he(c['peak_person_ts'])], sep=" ")
            else:
                people = str(c["peak_person"])
            body.append([spd, str(c["peak_vehicles"]),
                         people, _he(c["cam"])])
        tbl = Table(body, colWidths=[3.5*cm, 2.5*cm, 4*cm, 6.6*cm],
                    repeatRows=1)
        tbl.setStyle(_table_style(reg, bold))
        story.append(tbl)
    else:
        story.append(Paragraph(_he("אין דגימות פעילות בחלון."),
                               styles["body"]))

    # Training status
    story.append(Paragraph(_he("סטטוס למידת המודל"), styles["h"]))
    if training:
        verdict_he = "קידום ראש חדש" if training.get("promoted") else "נדחה בשער"
        cand = training.get("candidate") or training.get("file") or "?"
        when = str(training.get("at") or "")[:10]
        story.append(Paragraph(
            _he_join([f"ריצת אימון אחרונה", when + ":",
                      verdict_he, cand], sep=" · "), styles["body"]))
        for r in (training.get("reasons") or [])[:3]:
            # reasons are English (mAP50 gain +0.00pp < required +0.50pp),
            # rendered as-is.
            story.append(Paragraph(f"• {r}", styles["cap"]))
    else:
        story.append(Paragraph(_he("עוד לא רצה ריצת אימון בענן."),
                               styles["body"]))

    # Visual evidence pages - one image + caption per event, wrapped in a
    # single-cell Table with soft background + border so the caption reads
    # as "belonging" to the image above it. Two per page fits comfortably.
    snap_list = list(snapshots)
    if snap_list:
        story.append(PageBreak())
        story.append(Paragraph(
            _he("דוגמאות מהחריגים - תמונות מקוריות מהמצלמות"),
            styles["h"]))
        story.append(Paragraph(_he(
            "לכל סוג חריגה מצורפת התמונה המלאה שנשמרה בענן, "
            "עם המצלמה, השעה, ופרטים נוספים."), styles["cap"]))
        cap_style = ParagraphStyle(
            "img_cap", fontName=reg, fontSize=10, alignment=TA_RIGHT,
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
                ("ALIGN",         (0, 1), (-1, 1),  "RIGHT"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(frame)
            story.append(Spacer(1, 0.4*cm))
    else:
        story.append(Paragraph(_he("אין תמונות מצורפות (הבאקט של התמונות ריק "
                                   "או שהאירועים חסרים snapshot_url)."),
                               styles["cap"]))

    # PDF metadata title is a UI string on Gmail/reader tabs; keep it
    # ASCII-safe so no reader misrenders it in the tab bar.
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm,
                            title=f"Konya activity report {_fmt_date_he(now_il)}",
                            author="turkey-collector")
    doc.build(story)
    return out_path
