"""Pull every current IBB stream URL from istanbuluseyret.ibb.gov.tr camera pages.

The new pages embed a `bradmaxPlayerConfig_*` JS object whose `source[0].url` holds the
real HLS playlist URL.
"""
import json, re, ssl, urllib.request

ctx = ssl._create_unverified_context()

PAGES = {
    "anadolu_hisari":   "https://istanbuluseyret.ibb.gov.tr/anadolu-hisari-yeni/",
    "beyazit_kulesi_1": "https://istanbuluseyret.ibb.gov.tr/beyazit-kulesi-yeni/",
    "beyazit_kulesi_2": "https://istanbuluseyret.ibb.gov.tr/beyazit-kulesi-2-yeni/",
    "beyazit_meydan":   "https://istanbuluseyret.ibb.gov.tr/beyazit-meydani-yeni/",
    "buyuk_camlica":    "https://istanbuluseyret.ibb.gov.tr/buyuk-camlica-yeni/",
    "dragos":           "https://istanbuluseyret.ibb.gov.tr/dragos-yeni/",
    "eyup_sultan":      "https://istanbuluseyret.ibb.gov.tr/eyup-sultan-yeni/",
    "hidiv_kasri":      "https://istanbuluseyret.ibb.gov.tr/hidiv-kasri-yeni/",
    "kadikoy":          "https://istanbuluseyret.ibb.gov.tr/kadikoy/",
    "kapali_carsi":     "https://istanbuluseyret.ibb.gov.tr/kapali-carsi-yeni/",
    "kiz_kulesi":       "https://istanbuluseyret.ibb.gov.tr/kiz-kulesi-yeni/",
    "kucukcekmece":     "https://istanbuluseyret.ibb.gov.tr/kucukcekmece-yeni/",
    "metrohan":         "https://istanbuluseyret.ibb.gov.tr/metrohan-yeni/",
    "misir_carsisi":    "https://istanbuluseyret.ibb.gov.tr/misir-carsisi-canli-kamera-yeni/",
    "miniaturk":        "https://istanbuluseyret.ibb.gov.tr/miniaturk-yeni/",
    "pierre_lotti":     "https://istanbuluseyret.ibb.gov.tr/pierre-lotti-yeni/",
    "salacak":          "https://istanbuluseyret.ibb.gov.tr/salacak-yeni/",
    "saracane":         "https://istanbuluseyret.ibb.gov.tr/sarachane-yeni/",
    "sultanahmet_1":    "https://istanbuluseyret.ibb.gov.tr/sultanahmet-1-yeni/",
    "taksim":           "https://istanbuluseyret.ibb.gov.tr/taksim-yeni/",
    "ulus_parki":       "https://istanbuluseyret.ibb.gov.tr/ulus-parki-yeni/",
    "uskudar":          "https://istanbuluseyret.ibb.gov.tr/uskudar-yeni/",
}

CONFIG_RE = re.compile(r"bradmaxPlayerConfig_[a-z0-9]+\s*=\s*(\{.*?\});", re.IGNORECASE | re.DOTALL)
URL_RE = re.compile(r'"url"\s*:\s*"([^"]+\.m3u8[^"]*)"')

def fetch(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=10, context=ctx).read().decode("utf-8", "replace")

results = {}
for cid, page in PAGES.items():
    try:
        html = fetch(page)
    except Exception as e:
        print(f"{cid}: PAGE FAIL {e}")
        continue
    cfg = CONFIG_RE.search(html)
    if cfg:
        try:
            data = json.loads(cfg.group(1))
            url = data["dataProvider"]["source"][0]["url"]
        except Exception:
            url = (URL_RE.search(cfg.group(1)) or [None, None])
            url = url[1] if isinstance(url, tuple) else (url.group(1) if hasattr(url, "group") else None)
    else:
        m = URL_RE.search(html)
        url = m.group(1) if m else None

    if url:
        results[cid] = url
        print(f"{cid:18s} -> {url}")
    else:
        print(f"{cid:18s} -> NOT FOUND")

print("\n--- JSON ---")
print(json.dumps(results, indent=2))
