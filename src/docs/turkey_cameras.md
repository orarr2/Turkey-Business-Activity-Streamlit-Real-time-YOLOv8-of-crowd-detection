# Turkey Commercial Cameras + Live Architecture

Cameras and access methods used by `turkey_business_activity.ipynb` and the
`app/` collector + dashboard. Focus: **high-footfall commercial / market / square areas** for
business-activity analysis.

## Verified streams (Istanbul - IBB public HLS)

Source: Istanbul Metropolitan Municipality "Istanbulu Seyret" (`livestream.ibb.gov.tr`), public live cams.
Confirmed via the community catalog [ramazansancar/canli-kameralar](https://github.com/ramazansancar/canli-kameralar).

| Location | Type | Stream (HLS .m3u8) |
|----------|------|--------------------|
| Taksim Meydani | square / retail | `https://livestream.ibb.gov.tr/cam_turistik/cam_trsk_taksim.stream/playlist.m3u8` |
| Beyazit Meydani | square / market gateway | `.../cam_trsk_beyazit_meydan.stream/playlist.m3u8` |
| Eminonu | transport / commerce hub | `.../cam_trsk_eminonu.stream/playlist.m3u8` |
| **Kapali Carsi (Grand Bazaar)** | market | `.../cam_trsk_kapali_carsi.stream/playlist.m3u8` |
| **Misir Carsisi (Spice Bazaar)** | market | `.../cam_trsk_misir_carsisi.stream/playlist.m3u8` |
| Istiklal Caddesi | pedestrian retail street | `.../cam_trsk_istiklal_cad_1.stream/playlist.m3u8` |
| Sultanahmet | tourist square | `.../cam_trsk_sultanahmet_1.stream/playlist.m3u8` |

Full URLs live in `app/cameras.py`. The densest *commerce* (vs. pure tourism) is **Grand Bazaar, Spice
Bazaar, Eminonu, Istiklal** - start there for business-activity work.

## Other cities (non-IBB sources)

| Location | Type | Source page | `kind` | How it resolves |
|----------|------|-------------|--------|-----------------|
| Giresun - Gazi Caddesi | commercial street | `skylinewebcams.com/.../gazi-street.html` | `skyline` | `detect_core.resolve_skyline` scrapes the tokenized `hd-auth.skylinewebcams.com/live.m3u8?a=<token>` playlist from the page (token rotates, so resolved each cycle). |
| Otogar Kavsagi | junction / transit | `webcamera24.com/.../8044-otogar-kavsagi/` | `webcamera24` | `detect_core.resolve_webcamera24` finds the embedded tvkur (or YouTube) player on the page and builds its HLS master. |
| Kadikoy | commerce / transit | `istanbuluseyret.ibb.gov.tr/kadikoy/` | `hls` | Direct IBB `livestream.ibb.gov.tr/.../b_kadikoy.stream` playlist (same family as the squares above). |

These three plus **Konya - Hukumet Meydani** are `GRID_CAMERAS` in `app/cameras.py`: the four feeds the
Streamlit dashboard shows **side by side (2×2 grid) over the last 24 hours**, each with its live player and
the latest annotated YOLO frame. Start the collector for all four with:

```bash
python -m app.collector --interval 20 --only konya_hukumet,giresun_gazi,otogar_kavsagi,kadikoy
```

skylinewebcams and webcamera24 both 403 bare fetchers and rotate tokens, so the resolvers send a browser
User-Agent + Referer and run on every cycle. Verify resolution once on an open network with:

```bash
python -m app.detect_core --resolve giresun_gazi,otogar_kavsagi
```

If a page layout ever changes and a resolver returns nothing, open the page, copy the player id / m3u8 by
hand, and pin `url`/`embed` directly on the catalog entry.

## The two sources you gave me

1. **`webcamera24.com/.../sarraflar-yeralti-carsisi` (Konya, Hukumet Meydani).** webcamera24 pages are
   **YouTube-backed** - the live video is an embedded YouTube stream. To use it: open the page, copy the
   embedded YouTube watch URL, paste it into the `konya_hukumet` entry in `app/cameras.py`, set
   `kind="youtube"`. `yt-dlp` then resolves it to HLS automatically. (The page itself returns 403 to
   automated fetchers, so the YouTube id must be copied by hand once.)
2. **`istanbuluseyret.ibb.gov.tr/kameralar` (Taksim / Beyazit).** This portal is just a player UI; the
   actual feeds are the `livestream.ibb.gov.tr/.../playlist.m3u8` HLS URLs in the table above - which we
   use directly, no scraping of the portal needed.

## Network reality (why streams fail in a sandbox)

These hosts are public but reachable only from an **open network**. Restricted sandboxes (including the
environment that generated this repo) block them with an allowlist - a probe returns
`HTTP 403 x-deny-reason: host_not_allowed`. **Run the notebook and collector on your own machine / a VM /
a deployed app**, not inside a locked-down sandbox.

## Live-data architecture (the core question)

A notebook cell runs once. A live app needs **collection decoupled from display**:

```
 live streams  ->  collector.py        ->  data/footfall.db   ->  streamlit_app.py  ->  browser
                   (runs 24/7, samples       (shared store)         (reads + auto-       (always
                    every N seconds,                                 refresh 15s)         fresh)
                    writes counts)
```

- `app/collector.py` - infinite loop; samples every camera every `--interval` seconds, runs YOLO, appends
  to SQLite. Decoupled from any UI. Keep alive with `systemd` / Docker / `nohup`.
- `app/streamlit_app.py` - reads the same DB and auto-refreshes, so numbers update with no cell re-run.

**Cloud upgrade - Supabase (hosted Postgres):** replace the SQLite `INSERT` in `collector.py` with a
Supabase insert. Then the collector runs on a small VM/cron and many web clients read the same always-fresh
table. This is the path to a real multi-user app; the local SQLite version is the zero-infra starting point.

## Ethics / legality

- IBB cams are **published for the public**; we consume frames for aggregate counting, we do **not**
  rebroadcast the video.
- Store **aggregate counts only** (people/vehicles, dwell stats) - not raw frames of people. Privacy by design.
- Sample sparsely (every 15-30s for footfall); use short dense bursts only for dwell tracking.
