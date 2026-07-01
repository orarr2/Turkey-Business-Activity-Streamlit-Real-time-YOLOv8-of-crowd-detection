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
# Rule for the fallback chains: every slot stays within webcamera24 / tvkur,
# and each chain lists the other three Konya cams so a token rotation on the
# primary doesn't leave the tile empty.
GRID_SLOTS = [
    {
        "slot_id":      "slot_konya_hukumet",
        "display_area": "Konya - Hukumet",
        "primary":      "konya_hukumet",
        "fallbacks":    ["otogar_kavsagi", "konya_kulturpark", "konya_millet_caddesi"],
    },
    {
        "slot_id":      "slot_otogar",
        "display_area": "Konya - Otogar",
        "primary":      "otogar_kavsagi",
        "fallbacks":    ["konya_millet_caddesi", "konya_kulturpark", "konya_hukumet"],
    },
    {
        "slot_id":      "slot_kulturpark",
        "display_area": "Konya - Kulturpark",
        "primary":      "konya_kulturpark",
        "fallbacks":    ["konya_millet_caddesi", "konya_hukumet", "otogar_kavsagi"],
    },
    {
        "slot_id":      "slot_millet_caddesi",
        "display_area": "Konya - Millet Caddesi",
        "primary":      "konya_millet_caddesi",
        "fallbacks":    ["konya_kulturpark", "otogar_kavsagi", "konya_hukumet"],
    },
]

# Backward compat for the viewer notebook / smoke tests: the four primary cams.
GRID_CAMERAS = [s["primary"] for s in GRID_SLOTS]


def active_cameras():
    """Cameras that have a usable URL (skips placeholders awaiting a YouTube id)."""
    return {k: v for k, v in CAMERAS.items() if v.get("url")}
