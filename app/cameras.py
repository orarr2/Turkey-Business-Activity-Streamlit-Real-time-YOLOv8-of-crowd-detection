"""Verified Turkey commercial / market / square camera catalog.

IBB streams (Istanbulu Seyret) migrated 2024-2025: the legacy `cam_trsk_*` prefix is gone,
and the current player config (bradmaxPlayerConfig in each page on istanbuluseyret.ibb.gov.tr)
points at `livestream.ibb.gov.tr/cam_turistik/b_*.stream/playlist.m3u8`. The Eminonu and
Istiklal cameras are no longer listed and have been removed.

Note: livestream.ibb.gov.tr returns HTTP 404 (not 403) for these stream paths when accessed
from non-Turkey IPs — they appear to be geo-restricted. Run from a Turkey-routed IP for live
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
        "name": "Taksim Meydani",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_taksim_meydan.stream/playlist.m3u8",
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
        "name": "Sultanahmet",
        "city": "Istanbul",
        "kind": "hls",
        "url": "https://livestream.ibb.gov.tr/cam_turistik/b_sultanahmet.stream/playlist.m3u8",
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
    # --- Giresun: Gazi Caddesi, the city's main commercial street (skylinewebcams). ---
    # skylinewebcams serves a tokenized HLS (hd-auth.skylinewebcams.com/live.m3u8?a=<token>)
    # that rotates, so we resolve it from the page each cycle rather than hard-coding it.
    # Run `python -m app.detect_core --resolve giresun_gazi` on an open network to verify;
    # the resolver lives in detect_core.resolve_skyline.
    "giresun_gazi": {
        "name": "Giresun - Gazi Caddesi",
        "city": "Giresun",
        "kind": "skyline",
        "url": "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
        "page": "https://www.skylinewebcams.com/en/webcam/turkey/giresun/giresun/gazi-street.html",
        # skyline exposes an embeddable player at /embed/<slug>.html
        "embed": "https://www.skylinewebcams.com/en/embed/turkey/giresun/giresun/gazi-street.html",
        "type": "commercial street",
    },
    # --- Otogar Kavsagi (webcamera24, entry 8044). Like the Konya cam this page embeds a
    # tvkur (or sometimes YouTube) live player; the underlying HLS is resolved from the page
    # by detect_core.resolve_webcamera24. webcamera24 returns 403 to bare fetchers, so the
    # resolver sends a browser User-Agent + Referer. If the page ever stops resolving,
    # open it once, copy the player id, and pin `url`/`embed` here by hand. ---
    "otogar_kavsagi": {
        "name": "Otogar Kavsagi",
        "city": "Turkey",
        "kind": "webcamera24",
        "url": "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
        "page": "https://webcamera24.com/camera/turkey/8044-otogar-kavsagi/",
        # filled in once the tvkur/YouTube player id is known (see resolver notes above).
        "embed": None,
        "type": "junction/transit",
    },
}

# Cameras the live dashboard shows side by side (2x2 grid), in display order.
GRID_CAMERAS = ["konya_hukumet", "giresun_gazi", "otogar_kavsagi", "kadikoy"]


def active_cameras():
    """Cameras that have a usable URL (skips placeholders awaiting a YouTube id)."""
    return {k: v for k, v in CAMERAS.items() if v.get("url")}
