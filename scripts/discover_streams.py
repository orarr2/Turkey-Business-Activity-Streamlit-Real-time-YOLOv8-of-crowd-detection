"""Scrape each new IBB live-cam page for its embedded HLS (.m3u8) URL."""
import re, ssl, urllib.request

ctx = ssl._create_unverified_context()

PAGES = {
    "taksim":         "https://istanbuluseyret.ibb.gov.tr/taksim-yeni/",
    "kapali_carsi":   "https://istanbuluseyret.ibb.gov.tr/kapali-carsi-yeni/",
    "misir_carsisi":  "https://istanbuluseyret.ibb.gov.tr/misir-carsisi-canli-kamera-yeni/",
    "beyazit_meydan": "https://istanbuluseyret.ibb.gov.tr/beyazit-meydani-yeni/",
    "sultanahmet_1":  "https://istanbuluseyret.ibb.gov.tr/sultanahmet-1-yeni/",
    "kadikoy":        "https://istanbuluseyret.ibb.gov.tr/kadikoy/",
    "uskudar":        "https://istanbuluseyret.ibb.gov.tr/uskudar-yeni/",
    "saracane":       "https://istanbuluseyret.ibb.gov.tr/sarachane-yeni/",
    "eyup_sultan":    "https://istanbuluseyret.ibb.gov.tr/eyup-sultan-yeni/",
}

def fetch(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=10, context=ctx).read().decode("utf-8", "replace")

PATTERNS = [
    re.compile(r"https?://[A-Za-z0-9._/?=&%+:\-]+\.m3u8[A-Za-z0-9._/?=&%+:\-]*", re.IGNORECASE),
    re.compile(r"https?://hls2\.ibb\.gov\.tr/[A-Za-z0-9._/?=&%+:\-]+", re.IGNORECASE),
    re.compile(r"https?://livestream\.ibb\.gov\.tr/[A-Za-z0-9._/?=&%+:\-]+", re.IGNORECASE),
]
IFRAME_RE = re.compile(r"<iframe[^>]*src=[\"']([^\"']+)[\"']", re.IGNORECASE)

def scan(html):
    hits = []
    for p in PATTERNS:
        hits.extend(p.findall(html))
    return hits

SCRIPT_RE = re.compile(r"<script[^>]*src=[\"']([^\"']+)[\"']", re.IGNORECASE)
DATA_ATTR_RE = re.compile(r"data-(?:src|stream|url|file|cam|kamera)=[\"']([^\"']+)[\"']", re.IGNORECASE)
ANY_HLS_HOST = re.compile(r"https?://[A-Za-z0-9_.\-]*(?:hls2|hls|livestream|stream)[A-Za-z0-9_.\-]*\.ibb\.gov\.tr[A-Za-z0-9._/?=&%+:\-]*", re.IGNORECASE)

def deep_scan(html, base):
    hits = scan(html)
    for da in DATA_ATTR_RE.findall(html):
        hits.append(f"data-attr: {da}")
    for s in SCRIPT_RE.findall(html):
        if s.startswith("//"):
            s = "https:" + s
        elif s.startswith("/"):
            s = base.rstrip("/") + s
        if "istanbuluseyret" not in s and "ibb.gov.tr" not in s and "wp-content" not in s:
            continue
        try:
            js = fetch(s)
            for h in ANY_HLS_HOST.findall(js):
                hits.append(f"js({s.rsplit('/',1)[-1]}): {h}")
            for h in scan(js):
                hits.append(f"js({s.rsplit('/',1)[-1]}): {h}")
        except Exception:
            pass
    return hits

# Try a guess: the new HLS server is hls2.ibb.gov.tr. Probe known stream-name shapes.
GUESS_HOST = "https://hls2.ibb.gov.tr"
GUESSES = [
    "/{name}/index.m3u8", "/{name}/playlist.m3u8", "/live/{name}/index.m3u8",
    "/live/{name}/playlist.m3u8", "/cam/{name}/index.m3u8", "/cam_turistik/{name}/index.m3u8",
    "/{name}.stream/playlist.m3u8", "/{name}/{name}.m3u8",
]
NAMES = ["taksim", "taksim_yeni", "cam_trsk_taksim", "kapali_carsi", "kapali-carsi"]

for cid, url in PAGES.items():
    try:
        html = fetch(url)
    except Exception as e:
        print(f"{cid}: PAGE FETCH FAIL {type(e).__name__}: {e}")
        continue
    hits = deep_scan(html, "https://istanbuluseyret.ibb.gov.tr")
    seen = set()
    streams = [h for h in hits if not (h in seen or seen.add(h))]
    if streams:
        print(f"{cid}:")
        for s in streams[:8]:
            print(f"   -> {s}")
    else:
        print(f"{cid}: NO STREAM URL FOUND in HTML/JS ({len(html)} bytes)")

print("\n--- Probing hls2.ibb.gov.tr guesses ---")
for n in NAMES:
    for g in GUESSES:
        u = GUESS_HOST + g.format(name=n)
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            r = urllib.request.urlopen(req, timeout=4, context=ctx)
            body = r.read(200).decode("utf-8", "replace")
            print(f"OK {r.status}: {u}\n    {body[:120].strip()}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"HTTP {e.code}: {u}")
        except Exception:
            pass
