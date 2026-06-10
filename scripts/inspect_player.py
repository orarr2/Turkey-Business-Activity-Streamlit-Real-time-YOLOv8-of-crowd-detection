"""Dump the player-relevant HTML region of an IBB live-cam page."""
import re, ssl, urllib.request

ctx = ssl._create_unverified_context()
url = "https://istanbuluseyret.ibb.gov.tr/taksim-yeni/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=10, context=ctx).read().decode("utf-8", "replace")

# Look for video / player / jwplayer / hls / setup / cam id patterns
patterns = [r"video[^<]*", r"jwplayer[^<]*", r"hls[^<]*", r"<video[^<]*", r"player[^\"']*[\"'][^\"']{5,200}[\"']", r"playerInstance[^<]{0,200}", r"setup\([^)]{0,400}\)", r"file\s*:\s*[\"'][^\"']{15,300}[\"']", r"source\s*:\s*[\"'][^\"']{15,300}[\"']", r"data-[a-z-]+=[\"'][^\"']{15,300}[\"']"]
for p in patterns:
    m = re.findall(p, html, re.IGNORECASE)
    if m:
        print(f"--- {p}")
        for hit in m[:5]:
            print("   ", hit[:300])

# Also dump any <iframe> and <script src=...>
print("\n--- iframes ---")
for m in re.findall(r"<iframe[^>]*>", html, re.IGNORECASE):
    print(m[:300])
print("\n--- scripts ---")
for m in re.findall(r"<script[^>]*src=[\"']([^\"']+)[\"']", html, re.IGNORECASE):
    print(m)

# Save raw html for offline inspection
import os
os.makedirs("data", exist_ok=True)
with open("data/taksim_page.html", "w", encoding="utf-8") as f:
    f.write(html)
print("\nSaved raw HTML to data/taksim_page.html (", len(html), "bytes)")
