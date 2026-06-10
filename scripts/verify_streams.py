"""Verify cv2 can read a frame from each new IBB stream URL.

Output rows: cam_id, http_status, frame_shape, time_ms.
"""
import os, sys, time, ssl, urllib.request, cv2

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "tls_verify;0")
ctx = ssl._create_unverified_context()
sys.stdout.reconfigure(encoding="utf-8")

STREAMS = {
    "taksim":         "https://livestream.ibb.gov.tr/cam_turistik/b_taksim_meydan.stream/playlist.m3u8",
    "kapali_carsi":   "https://livestream.ibb.gov.tr/cam_turistik/b_kapalicarsi.stream/playlist.m3u8",
    "misir_carsisi":  "https://livestream.ibb.gov.tr/cam_turistik/b_misircarsisi.stream/playlist.m3u8",
    "beyazit_meydan": "https://livestream.ibb.gov.tr/cam_turistik/b_beyazitmeydani.stream/playlist.m3u8",
    "sultanahmet_1":  "https://livestream.ibb.gov.tr/cam_turistik/b_sultanahmet.stream/playlist.m3u8",
    "kadikoy":        "https://livestream.ibb.gov.tr/cam_turistik/b_kadikoy.stream/chunklist.m3u8",
}

for cid, url in STREAMS.items():
    # 1) check playlist HTTP
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":    "https://istanbuluseyret.ibb.gov.tr/",
        "Origin":     "https://istanbuluseyret.ibb.gov.tr",
        "Accept":     "*/*",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        r = urllib.request.urlopen(req, timeout=8, context=ctx)
        body = r.read(400).decode("utf-8", "replace")
        http = r.status
        is_hls = "#EXTM3U" in body
    except Exception as e:
        print(f"{cid:16s}  HTTP_ERR {type(e).__name__}: {str(e)[:80]}")
        continue

    # 2) try cv2 with same header (ffmpeg honors headers env / option)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "tls_verify;0|" +
        "headers;Referer: https://istanbuluseyret.ibb.gov.tr/\\r\\nOrigin: https://istanbuluseyret.ibb.gov.tr\\r\\nUser-Agent: Mozilla/5.0\\r\\n"
    )
    t0 = time.time()
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    dt = (time.time() - t0) * 1000
    shape = frame.shape if ok else None
    print(f"{cid:16s}  HTTP {http}  HLS={is_hls}  cv2_ok={ok}  shape={shape}  {dt:.0f}ms")
