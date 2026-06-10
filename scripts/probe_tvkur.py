"""Reverse-engineer the tvkur.com player to find the live HLS URL."""
import json, re, ssl, sys, urllib.request

sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl._create_unverified_context()
STREAM_ID = "c77i84vbb2nj4i0fr80g"

def fetch(url, **headers):
    h = {"User-Agent": "Mozilla/5.0", "Referer": "https://player.tvkur.com/"}
    h.update(headers)
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=15, context=ctx)

js = fetch("https://player.tvkur.com/assets/bundle-vjs.min.js").read().decode("utf-8", "replace")
print(f"bundle bytes: {len(js)}")

PATTERNS = [
    r"https?://[A-Za-z0-9._/?=&%+:\-{}\$]+\.m3u8[A-Za-z0-9._/?=&%+:\-{}]*",
    r"`[^`]*\$\{[^}]+\}[^`]*\.m3u8`",
    r"/api/[A-Za-z0-9_/\-]+",
    r"['\"]https?://[A-Za-z0-9._\-]+\.tvkur\.com[^'\"]*['\"]",
    r"['\"]/(?:l|live|stream)/[A-Za-z0-9_/\-]+['\"]",
    r"manifestUrl|streamUrl|hlsUrl|m3u8Path",
]
for p in PATTERNS:
    hits = list(set(re.findall(p, js)))
    if hits:
        print(f"\n--- {p}")
        for h in hits[:25]:
            print("   ", h[:300])

# Also look for the data flow: data.id -> some URL
for m in re.finditer(r"document\.data[^,;]{1,80}", js):
    print("data ref:", m.group(0))

# Guess likely API endpoints
print("\n--- Probing likely endpoints ---")
endpoints = [
    f"https://player.tvkur.com/api/streams/{STREAM_ID}",
    f"https://player.tvkur.com/api/live/{STREAM_ID}",
    f"https://player.tvkur.com/api/livestreams/{STREAM_ID}",
    f"https://player.tvkur.com/api/v1/streams/{STREAM_ID}",
    f"https://api.tvkur.com/streams/{STREAM_ID}",
    f"https://api.tvkur.com/live/{STREAM_ID}",
    f"https://player.tvkur.com/api/playback/{STREAM_ID}",
    f"https://player.tvkur.com/api/{STREAM_ID}",
    f"https://player.tvkur.com/l/{STREAM_ID}/manifest.m3u8",
    f"https://stream.tvkur.com/{STREAM_ID}/index.m3u8",
    f"https://live.tvkur.com/{STREAM_ID}/index.m3u8",
    f"https://cdn.tvkur.com/{STREAM_ID}/playlist.m3u8",
    f"https://player.tvkur.com/{STREAM_ID}.m3u8",
]
for u in endpoints:
    try:
        r = fetch(u)
        body = r.read(800).decode("utf-8", "replace")
        ct = r.headers.get("Content-Type", "")
        print(f"  OK {r.status} ({ct}) {u}")
        print(f"     {body[:200].strip()}")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} {u}")
    except Exception as e:
        print(f"  ERR {type(e).__name__} {u}")
