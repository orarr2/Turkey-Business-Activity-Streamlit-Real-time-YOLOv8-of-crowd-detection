"""Pull every URL-ish string out of the tvkur player bundle."""
import re, ssl, sys, urllib.request

sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl._create_unverified_context()
req = urllib.request.Request(
    "https://player.tvkur.com/assets/bundle-vjs.min.js",
    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://player.tvkur.com/"},
)
js = urllib.request.urlopen(req, timeout=20, context=ctx).read().decode("utf-8", "replace")

# every quoted string between 4 and 200 chars that includes "/" or "http" or ".m3u8"
strs = re.findall(r'"([^"\n]{4,200})"', js) + re.findall(r"'([^'\n]{4,200})'", js)
keep = []
for s in strs:
    sl = s.lower()
    if any(k in sl for k in (".m3u8", "/api/", "/l/", "/live", "stream", "tvkur", "playlist", "manifest", "hls")):
        keep.append(s)
seen = set()
for s in keep:
    if s in seen: continue
    seen.add(s)
    print(s)
