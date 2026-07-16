"""Verified Turkey commercial / market / square camera catalog.

IBB streams (Istanbulu Seyret) migrated 2024-2025: the legacy `cam_trsk_*` prefix is gone,
and the current player config (bradmaxPlayerConfig in each page on istanbuluseyret.ibb.gov.tr)
points at `livestream.ibb.gov.tr/cam_turistik/b_*.stream/playlist.m3u8`. The Eminonu and
Istiklal cameras are no longer listed and have been removed.

Note: livestream.ibb.gov.tr returns HTTP 404 (not 403) for these stream paths when accessed
from non-Turkey IPs - they appear to be geo-restricted. Run from a Turkey-routed IP for live
data; on any other network you'll just see MISS rows in the collector.

Each entry: kind = one of
  "hls"          direct .m3u8 (used as-is)
  "youtube"      a YouTube / YouTube-backed page, resolved via yt-dlp
  "skyline"      a skylinewebcams.com page, resolved via detect_core.resolve_skyline
  "webcamera24"  a webcamera24.com page, resolved via detect_core.resolve_webcamera24

Optional per-entry keys:
  "page"   public webcam page (human-facing, also the resolver input for skyline/webcamera24)
  "embed"  iframe URL for the live player shown in the dashboard grid (None -> the grid
           falls back to the latest annotated YOLO frame as the "live" tile)
  "conf"   per-camera YOLO confidence override. The collector's global default
           (--conf) fits most scenes; set this on a camera whose calibration run
           (notebook section 10) shows a systematic bias - lower it when the
           camera undercounts (distant/small objects), raise it on false
           positives. Example: "conf": 0.25.

Operational-analysis keys (all coordinates NORMALIZED 0..1 relative to the
frame; run `python -m tools.roi_grid <cam_id>` to capture a frame with a
coordinate grid overlay and read the points off it):
  "roi"          include-polygon [[x,y], ...]: detections whose FOOT POINT
                 (bottom-center) falls outside are ignored entirely - use it
                 to cut parking lots / sky / a neighboring street out of the
                 business-activity counts.
  "roi_exclude"  list of exclude-polygons carved out of the ROI.
  "line"         virtual counting line [[x1,y1],[x2,y2]]: each sampling burst
                 counts tracked objects crossing it, split into in/out
                 (crossing from the right side of A->B is "in" - order the two
                 points so "in" points into your area of interest). The
                 dashboard shows the sampled flow; it is a rate indicator,
                 not a turnstile.
  "loiter_roi"   polygon for prolonged-presence alerts (defaults to the whole
                 frame when omitted while loitering is enabled).
  "loiter_person_sec" / "loiter_vehicle_sec"
                 per-camera override of how long a person/vehicle must stay
                 (matched in place across samples) before a loiter event
                 fires. Defaults: 300s person / 900s vehicle.

Example (uncomment and tune per scene):
  # "roi":  [[0.05, 0.35], [0.95, 0.35], [0.95, 0.98], [0.05, 0.98]],
  # "line": [[0.10, 0.60], [0.90, 0.55]],
  # "loiter_person_sec": 240,
"""

# Header set IBB's nginx accepts. ffmpeg/OpenCV honor this via OPENCV_FFMPEG_CAPTURE_OPTIONS.
IBB_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer":    "https://istanbuluseyret.ibb.gov.tr/",
    "Origin":     "https://istanbuluseyret.ibb.gov.tr",
}

CAMERAS = {
    # --- Konya: crowded square/market. webcamera24 entry 8043-sarraflar-yeralti-carsisi
    # embeds a tvkur.com live player; the underlying HLS master is on content.tvkur.com.
    # detect_core.grab_frame() handles the Referer/Origin requirements for this host. ---
    "konya_hukumet": {
        "name": "Konya - Hukumet Meydani / Sarraflar Yeralti Carsisi",
        "city": "Konya",
        "kind": "hls",
        "url": "https://content.tvkur.com/l/c77i84vbb2nj4i0fr80g/master.m3u8",
        "page": "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/",
        "embed": "https://player.tvkur.com/l/c77i84vbb2nj4i0fr80g",
        "type": "square/market",
    },
    # --- Istanbul: high-footfall commerce & markets (IBB public HLS, new b_* prefix) ---
    "taksim": {
        "name": "Taksim Meydani (legacy host - 404 outside Turkey)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_taksim_meydan.stream/playlist.m3u8",
        "type": "square/retail",
    },
    # --- IBB migrated to kamerayayin.ibb.istanbul (2025-2026). The new -yeni pages
    # use this domain, return CORS *, and are reachable from any country - much
    # better than the legacy livestream.ibb.gov.tr endpoints above. ---
    "taksim_yeni": {
        "name": "Taksim Meydani (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/taksim-yeni/",
        # the public page sets X-Frame-Options: SAMEORIGIN, so no iframe embed.
        # The web/ dashboard plays this HLS directly with hls.js in its own <video>.
        "embed": None,
        "type": "square/retail",
    },
    "beyazit_meydan": {
        "name": "Beyazit Meydani",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_beyazitmeydani.stream/playlist.m3u8",
        "type": "square/market-gateway",
    },
    "kapali_carsi": {
        "name": "Kapali Carsi (Grand Bazaar)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_kapalicarsi.stream/playlist.m3u8",
        "type": "market",
    },
    "misir_carsisi": {
        "name": "Misir Carsisi (Spice Bazaar)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_misircarsisi.stream/playlist.m3u8",
        "type": "market",
    },
    "sultanahmet_1": {
        "name": "Sultanahmet (legacy host - 404 outside Turkey)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_sultanahmet.stream/playlist.m3u8",
        "type": "tourist square",
    },
    "sultanahmet_1_yeni": {
        "name": "Sultanahmet (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/sultanahmet1.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/sultanahmet-1-yeni/",
        "embed": None,
        "type": "tourist square",
    },
    # --- Three more IBB kamerayayin cameras added 2026-07-14 after the
    # Konya tvkur backend went 404. All three verified HTTP 200 at add time
    # and returned as CORS: * so the web dashboard can play them directly
    # via hls.js. Their -yeni public pages are on istanbuluseyret. ---
    "beyazit_meydan_yeni": {
        "name": "Beyazit Meydani (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/beyazitmeydan.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/beyazit-meydani-yeni/",
        "embed": None,
        "type": "square/market-gateway",
    },
    "eyup_sultan_yeni": {
        "name": "Eyup Sultan (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/eyupsultan.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/eyup-sultan-yeni/",
        "embed": None,
        "type": "religious square",
    },
    "buyuk_camlica_yeni": {
        "name": "Buyuk Camlica (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/buyukcamlica.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/buyuk-camlica-yeni/",
        "embed": None,
        "type": "park/vista",
    },
    # --- Remaining kamerayayin.ibb.istanbul cameras (all HTTP 200 on
    # 2026-07-16). Tier-3 fallbacks for the collector: when Konya (tvkur)
    # AND the four preferred Istanbul cams are down, the pool keeps walking
    # this list so no slot ever settles on an empty frame. ---
    "sarachane_yeni": {
        "name": "Sarachane (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/sarachane.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/sarachane-yeni/",
        "embed": None,
        "type": "civic square",
    },
    "sultanahmet_2_yeni": {
        "name": "Sultanahmet 2 (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/sultanahmet2.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/sultanahmet-2-yeni/",
        "embed": None,
        "type": "tourist square",
    },
    "uskudar_yeni": {
        "name": "Uskudar (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/uskudar.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/uskudar-yeni/",
        "embed": None,
        "type": "square/transport",
    },
    "salacak_yeni": {
        "name": "Salacak (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/salacak.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/salacak-yeni/",
        "embed": None,
        "type": "waterfront promenade",
    },
    "kucukcekmece_yeni": {
        "name": "Kucukcekmece (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/kucukcekmece.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/kucukcekmece-yeni/",
        "embed": None,
        "type": "lakeside park",
    },
    "ulus_parki_yeni": {
        "name": "Ulus Parki (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/ulusparki.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/ulus-parki-yeni/",
        "embed": None,
        "type": "park/vista",
    },
    "pierre_lotti_yeni": {
        "name": "Pierre Lotti (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/pierreloti.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/pierre-lotti-yeni/",
        "embed": None,
        "type": "hilltop cafe/vista",
    },
    "emirgan_yeni": {
        "name": "Emirgan (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/emirgan.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/emirgan-yeni/",
        "embed": None,
        "type": "park",
    },
    "kiz_kulesi_yeni": {
        "name": "Kiz Kulesi (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/kizkulesi.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/kiz-kulesi-yeni/",
        "embed": None,
        "type": "waterfront landmark",
    },
    "hidiv_kasri_yeni": {
        "name": "Hidiv Kasri (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/hidivkasri.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/hidiv-kasri-yeni/",
        "embed": None,
        "type": "palace grounds",
    },
    "dragos_yeni": {
        "name": "Dragos (live)",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://kamerayayin.ibb.istanbul/turistikcam/dragos.stream/playlist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/dragos-yeni/",
        "embed": None,
        "type": "coastal vista",
    },
    "kadikoy": {
        "name": "Kadikoy",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_kadikoy.stream/chunklist.m3u8",
        "page": "https://istanbuluseyret.ibb.gov.tr/kadikoy/",
        # IBB pages set X-Frame-Options, so no reliable iframe embed; the dashboard grid
        # shows the latest annotated YOLO frame for this tile instead.
        "embed": None,
        "type": "commerce/transit",
    },
    "eyup_sultan": {
        "name": "Eyup Sultan",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_eyupsultan.stream/playlist.m3u8",
        "type": "tourist square",
    },
    "uskudar": {
        "name": "Uskudar",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_uskudar.stream/playlist.m3u8",
        "type": "square/transport",
    },
    # --- Otogar Kavsagi (webcamera24, entry 8044). Like the Konya cam this page embeds a
    # tvkur live player. tvkur player id resolved 2026-06: c77i91vbb2nj4i0fr81g. ---
    "otogar_kavsagi": {
        "name": "Otogar Kavsagi",
        "city": "Konya",
        "kind": "webcamera24",
        "url": "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
        "page": "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
        "embed": "https://player.tvkur.com/l/c77i91vbb2nj4i0fr81g",
        "type": "junction/transit",
    },
    # --- Konya Kulturpark (webcamera24 8058) - replaces Giresun in the grid because
    # skylinewebcams geo-blocks Israel. tvkur id: c77i6hb84cnrb6mlji3g. ---
    "konya_kulturpark": {
        "name": "Konya - Kulturpark",
        "city": "Konya",
        "kind": "webcamera24",
        "url": "https://webcamera24.com/camera/turkey/8058-kulturpark/",
        "page": "https://webcamera24.com/camera/turkey/8058-kulturpark/",
        "embed": "https://player.tvkur.com/l/c77i6hb84cnrb6mlji3g",
        "type": "park/commercial",
    },
    # --- Konya Millet Caddesi / Hastane Kavsagi (webcamera24 8046) - replaces Kadikoy
    # in the grid because IBB streams geo-block Israel. tvkur id: c77i9cfbb2nj4i0fr82g. ---
    "konya_millet_caddesi": {
        "name": "Konya - Millet Caddesi / Hastane Kavsagi",
        "city": "Konya",
        "kind": "webcamera24",
        "url": "https://webcamera24.com/camera/turkey/8046-millet-caddesi/",
        "page": "https://webcamera24.com/camera/turkey/8046-millet-caddesi/",
        "embed": "https://player.tvkur.com/l/c77i9cfbb2nj4i0fr82g",
        "type": "hospital junction / vehicular",
        # The blue keep-right sign on the traffic island reads as `person`
        # (~0.38) again and again - operator flagged it twice from live
        # screenshots. Static street furniture, so a static exclude: any
        # `person` whose foot-point lands on the island sign is dropped.
        "roi_exclude_class": {
            "person": [[[0.385, 0.17], [0.445, 0.17],
                        [0.445, 0.30], [0.385, 0.30]]],
        },
    },
    # --- Konya Ince Minareli Medrese (webcamera24 8033). The city's TRAM line
    # runs on dedicated tracks along the left of the frame (Alaaddin loop),
    # with a pedestrian plaza and a road in view - the first camera in the
    # grid that can actually produce `train`-class detections. Added after
    # the operator reported metro/tram traffic was invisible to the system
    # (no rail-view camera existed). tvkur id resolved 2026-07:
    # c77ib8vbb2nj4i0fr8bg, frame 1920x1080, reachable from any country. ---
    "konya_ince_minareli": {
        "name": "Konya - Ince Minareli Medrese (tram line)",
        "city": "Konya",
        "kind": "hls",
        "url": "https://content.tvkur.com/l/c77ib8vbb2nj4i0fr8bg/master.m3u8",
        "page": "https://webcamera24.com/camera/turkey/8033-ince-minareli-medrese/",
        "embed": "https://player.tvkur.com/l/c77ib8vbb2nj4i0fr8bg",
        "type": "tram line / plaza",
    },
    # --- Kept for runs from a Turkey-routed IP, where these unblock. Not in the grid. ---
    "giresun_gazi": {
        "name": "Giresun - Gazi Caddesi",
        "city": "Giresun",
        "kind": "skyline",
        "url": "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
        "page": "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
        "embed": "https://www.skylinewebcams.com/en/embed/turkey/giresun/giresun/gazi-street.html",
        "type": "commercial street (geo-restricted)",
    },
}

# Live dashboard grid, in display order. Each slot has a primary camera and an
# ordered fallback chain: if the collector can't grab a frame from the current
# cam N times in a row (see SlotStreamPicker in collector.py), it advances to
# the next entry. Every FALLBACK_RETRY_MINUTES the primary is retried; if it
# recovers, the picker snaps back.
#
# 2026-07 update: IBB (both livestream.ibb.gov.tr and kamerayayin.ibb.istanbul)
# tightened its geo-block to Turkey-only in early July 2026. Every IBB HLS now
# returns 403/502 from GCP us-east1 AND from any non-Turkey network - not just
# an outbound-IP issue with the VM. The two "Istanbul" grid slots have therefore
# been swapped for two more webcamera24/tvkur Konya cams that this project
# already had in the catalog and that work from any country. The IBB entries
# stay in CAMERAS above for anyone running from a Turkey-routed IP; if they
# unblock again they can be put back in this list.
#
# Rule for the fallback chains (updated 2026-07 after the 48h empty-run):
# every chain now spans MULTIPLE backends. The first three entries stay on
# webcamera24 / tvkur so a single-camera outage swaps into a sibling that
# reaches the same host; the last two are IBB `kamerayayin.ibb.istanbul`
# cameras plus `konya_ince_minareli` (also tvkur but on a different stream
# id) so a full tvkur or webcamera24 backend outage still has somewhere to
# go. Before this change every fallback lived on the same backend, so when
# tvkur went 404 all four grid slots died together and the report still
# said "reporting normally".
#
# IBB `_yeni` entries were geo-blocked from GCP us-east1 in early July 2026
# per the note above. They still belong at the tail of the chain: they cost
# nothing when they fail (the picker skips them within a few rounds) and
# unblock as soon as IBB relaxes the block or the VM is behind a TR proxy.
# One GLOBAL priority ladder, shared by all four slots (operator spec,
# 2026-07-16). The collector's CameraPool walks it top-down and assigns the
# first four cameras that are currently delivering frames - one per slot,
# never the same camera twice:
#   tier 1: the four Konya cams (webcamera24/tvkur) - the original grid;
#   tier 2: the four preferred Istanbul kamerayayin cams, in this order;
#   tier 3: every remaining kamerayayin camera from the catalog, so the
#           grid NEVER settles on an empty frame while anything is live.
FALLBACK_POOL = [
    # tier 1 - Konya (primary grid)
    "konya_hukumet", "otogar_kavsagi", "konya_kulturpark",
    "konya_millet_caddesi",
    # tier 2 - preferred Istanbul replacements (operator order, 2026-07-16)
    "taksim_yeni", "sultanahmet_1_yeni", "eyup_sultan_yeni",
    "beyazit_meydan_yeni",
    # tier 3 - the rest of the live catalog, walked one by one until a
    # camera actually delivers frames
    "buyuk_camlica_yeni", "konya_ince_minareli", "sarachane_yeni",
    "sultanahmet_2_yeni", "uskudar_yeni", "salacak_yeni",
    "kucukcekmece_yeni", "ulus_parki_yeni", "pierre_lotti_yeni",
    "emirgan_yeni", "kiz_kulesi_yeni", "hidiv_kasri_yeni", "dragos_yeni",
]

def _pool_fallbacks(primary: str) -> list[str]:
    return [c for c in FALLBACK_POOL if c != primary]

GRID_SLOTS = [
    {
        "slot_id":      "slot_konya_hukumet",
        "display_area": "Konya - Hukumet",
        "primary":      "konya_hukumet",
        "fallbacks":    _pool_fallbacks("konya_hukumet"),
    },
    {
        "slot_id":      "slot_otogar",
        "display_area": "Konya - Otogar",
        "primary":      "otogar_kavsagi",
        "fallbacks":    _pool_fallbacks("otogar_kavsagi"),
    },
    {
        "slot_id":      "slot_kulturpark",
        "display_area": "Konya - Kulturpark",
        "primary":      "konya_kulturpark",
        "fallbacks":    _pool_fallbacks("konya_kulturpark"),
    },
    {
        "slot_id":      "slot_millet_caddesi",
        "display_area": "Konya - Millet Caddesi",
        "primary":      "konya_millet_caddesi",
        "fallbacks":    _pool_fallbacks("konya_millet_caddesi"),
    },
    # konya_ince_minareli (tram-line view) stays in CAMERAS above as a
    # catalog option and a tier-3 pool entry; it is NOT the primary of a
    # dedicated slot because a fifth grid slot costs ~20% round time and
    # RAM on the e2-micro, and the operator prefers the 4-camera cadence.
]

# Backward compat for the viewer notebook / smoke tests: the four primary cams.
GRID_CAMERAS = [s["primary"] for s in GRID_SLOTS]


# --- auto-blacklist merge -----------------------------------------------------
# When the review UI accumulates enough "wrong" verdicts in a coherent screen
# area (see app/auto_blacklist.py), a polygon is appended to
# data/blacklist_auto.json. Merge those into cam["roi_exclude_class"] on
# import so the next collector burst drops the same false positives - no
# code change, no restart, no GPU.
def _merge_auto_blacklist() -> None:
    try:
        from app.auto_blacklist import load_auto_blacklist
    except ImportError:
        return    # tests or minimal envs without the module - just skip
    try:
        by_cam = load_auto_blacklist()
    except Exception:
        return
    for cam_id, cls_map in by_cam.items():
        cam = CAMERAS.get(cam_id)
        if not cam:
            continue
        existing = dict(cam.get("roi_exclude_class") or {})
        for cls, polys in cls_map.items():
            existing.setdefault(cls, []).extend(polys)
        cam["roi_exclude_class"] = existing


# --- confidence-boost merge ---------------------------------------------------
# Per-camera per-class confidence adjustment learned from the review UI's
# correct/wrong verdicts (see app/confidence_boost.py). Applied as a delta
# on top of DEFAULT_PER_CLASS_CONF; the collector clamps the effective
# value at read time. Live-reloaded by collector.main() every few rounds.
def _merge_confidence_boost() -> None:
    try:
        from app.confidence_boost import load_boosts
        from app.detect_core import DEFAULT_PER_CLASS_CONF
    except ImportError:
        return
    try:
        by_cam = load_boosts()
    except Exception:
        return
    for cam_id, cls_map in by_cam.items():
        cam = CAMERAS.get(cam_id)
        if not cam:
            continue
        pcc = dict(cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
        for cls, delta in cls_map.items():
            try:
                base = float(pcc.get(cls, DEFAULT_PER_CLASS_CONF.get(cls, 0.35)))
                pcc[cls] = max(0.10, min(0.80, base + float(delta)))
            except (TypeError, ValueError):
                continue
        cam["per_class_conf"] = pcc


def reload_review_overrides() -> None:
    """Public entry point for the collector's hot-reload timer. Re-runs
    both merges without importing/reloading the whole module."""
    _merge_auto_blacklist()
    _merge_confidence_boost()


_merge_auto_blacklist()
_merge_confidence_boost()


def active_cameras():
    """Cameras that have a usable URL (skips placeholders awaiting a YouTube id)."""
    return {k: v for k, v in CAMERAS.items() if v.get("url")}
