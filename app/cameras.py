"""Verified Turkey commercial / market / square camera catalog.

IBB streams (Istanbulu Seyret) migrated 2024-2025: the legacy `cam_trsk_*` prefix is gone,
and the current player config (bradmaxPlayerConfig in each page on istanbuluseyret.ibb.gov.tr)
points at `livestream.ibb.gov.tr/cam_turistik/b_*.stream/playlist.m3u8`. The Eminonu and
Istiklal cameras are no longer listed and have been removed.

Note: livestream.ibb.gov.tr returns HTTP 404 (not 403) for these stream paths when accessed
from non-Turkey IPs — they appear to be geo-restricted. Run from a Turkey-routed IP for live
data; on any other network you'll just see MISS rows in the collector.

Each entry: kind = "hls" (direct .m3u8) or "youtube" (resolve via yt-dlp first).
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
}


def active_cameras():
    """Cameras that have a usable URL (skips placeholders awaiting a YouTube id)."""
    return {k: v for k, v in CAMERAS.items() if v.get("url")}
