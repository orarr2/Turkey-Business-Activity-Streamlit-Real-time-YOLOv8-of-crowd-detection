"""Locate the live-stream backing the Konya Sarraflar Yeralti Carsisi webcamera24 page.

Scrape the page HTML; look for YouTube IDs / iframe sources / HLS URLs / any common
livestream service patterns. Saves the raw HTML for debugging.
"""
import os, re, ssl, sys, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl._create_unverified_context()

PAGE = "https://webcamera24.com/camera/turkey/8043-sarraflar-yeralti-carsisi/"
out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(parents=True, exist_ok=True)

req = urllib.request.Request(PAGE, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})
with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
    html = r.read().decode("utf-8", "replace")

(out_dir / "konya_page.html").write_text(html, encoding="utf-8")
print(f"HTML bytes: {len(html)}")

PATTERNS = {
    "youtube_watch": r"youtube\.com/watch\?v=([A-Za-z0-9_\-]{6,})",
    "youtube_embed": r"youtube\.com/embed/([A-Za-z0-9_\-]{6,})",
    "youtu_be":      r"youtu\.be/([A-Za-z0-9_\-]{6,})",
    "youtube_id":    r'["\']videoId["\']\s*:\s*["\']([A-Za-z0-9_\-]{6,})["\']',
    "m3u8":          r"https?://[A-Za-z0-9._/?=&%+:\-]+\.m3u8[A-Za-z0-9._/?=&%+:\-]*",
    "iframe":        r"<iframe[^>]*src=[\"']([^\"']+)[\"']",
    "video_src":     r"<video[^>]*src=[\"']([^\"']+)[\"']",
    "data_attr":     r"data-(?:src|video|stream|url|youtube|cam)=[\"']([^\"']{6,200})[\"']",
}

for name, pat in PATTERNS.items():
    hits = list(set(re.findall(pat, html, re.IGNORECASE)))
    if hits:
        print(f"--- {name}")
        for h in hits[:10]:
            print("  ", h)

# Follow the player.tvkur.com iframe
iframe_match = re.search(r"<iframe[^>]*src=[\"']([^\"']*player\.tvkur\.com[^\"']*)[\"']", html, re.IGNORECASE)
if iframe_match:
    iframe_url = iframe_match.group(1)
    print(f"\n=== Following iframe: {iframe_url}")
    req2 = urllib.request.Request(iframe_url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": PAGE,
    })
    with urllib.request.urlopen(req2, timeout=15, context=ctx) as r:
        ihtml = r.read().decode("utf-8", "replace")
    (out_dir / "konya_player.html").write_text(ihtml, encoding="utf-8")
    print(f"player HTML bytes: {len(ihtml)}")
    for name, pat in PATTERNS.items():
        hits = list(set(re.findall(pat, ihtml, re.IGNORECASE)))
        if hits:
            print(f"player {name}:")
            for h in hits[:10]:
                print("  ", h)
    # also pull any JSON config / `file:` / `hls:` patterns
    for pat in [r'"file"\s*:\s*"([^"]{15,400})"',
                r'"src"\s*:\s*"([^"]{15,400})"',
                r'"hls"\s*:\s*"([^"]{15,400})"',
                r'"manifest_url"\s*:\s*"([^"]{15,400})"',
                r'manifest_url[\\\"\']*\s*[:=]\s*[\\\"\']([^\\\"\']{15,400})',
                r'streamUrl\s*[:=]\s*[\"\']([^\"\']{15,400})',
                r'https?://[A-Za-z0-9._/?=&%+:\-]+\.(?:m3u8|mpd)[A-Za-z0-9._/?=&%+:\-]*',
                r'wss?://[A-Za-z0-9._/?=&%+:\-]+',
                r'"id"\s*:\s*"([A-Za-z0-9]{16,})"']:
        for m in re.findall(pat, ihtml, re.IGNORECASE):
            print(f"player extra: {m[:200]}")
    # script srcs
    for s in re.findall(r"<script[^>]*src=[\"']([^\"']+)[\"']", ihtml, re.IGNORECASE):
        print(f"player script: {s}")
