"""Find the actual XHR / fetch call the tvkur player makes to get the HLS URL."""
import re, ssl, sys, urllib.request

sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl._create_unverified_context()

req = urllib.request.Request(
    "https://player.tvkur.com/assets/bundle-vjs.min.js",
    headers={"User-Agent": "Mozilla/5.0"},
)
js = urllib.request.urlopen(req, timeout=20, context=ctx).read().decode("utf-8", "replace")

# look for fetch(... and XMLHttpRequest open paths
for pat in [
    r"fetch\([^)]{1,400}\)",
    r"\.open\([^)]{1,300}\)",
    r"document\.data[^,;}]{1,200}",
    r"data\.id[^,;}]{1,200}",
    r"\.live[^,;}]{1,80}",
    r"['\"`]https?:[^'\"`]{4,200}['\"`]",
    r"['\"`]/[A-Za-z0-9_/\-${}]{3,80}['\"`]",
]:
    print(f"\n=== {pat}")
    seen = set()
    for m in re.finditer(pat, js):
        s = m.group(0)
        if s in seen: continue
        seen.add(s)
        if any(k in s.lower() for k in ("api","live","stream","tvkur","manifest","playlist","m3u8","fetch","xhr","data","/l/","cdn","host")):
            print("  ", s[:300])
        if len(seen) > 60: break
