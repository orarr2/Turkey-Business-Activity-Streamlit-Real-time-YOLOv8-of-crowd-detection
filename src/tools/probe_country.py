"""Live camera probe: does the CURRENT host reach the streams for a country?

    python -m tools.probe_country --country turkey     # default
    python -m tools.probe_country --country thailand --timeout 15

Runs on the VM (or anywhere): resolves each catalog camera for the country
and grabs one frame - the same code path the collector uses. Prints LIVE
/ DEAD per camera plus a summary, and exits non-zero when zero cameras
delivered a frame - so a cron caller can flip flags without parsing text.

Turkey has been the driving case (IBB's HLS refuses non-Turkey IPs at the
raw-stream level even when the web page plays), but the tool is generic:
run it before assuming a country is up, and again before assuming it is
still down.
"""
from __future__ import annotations

import argparse
import time


def _classify(exc: BaseException) -> str:
    """One-word cause label for the summary line."""
    msg = str(exc).lower()
    if "403" in msg:
        return "http_403"
    if "404" in msg:
        return "http_404"
    if "429" in msg:
        return "http_429"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg and "refused" in msg:
        return "refused"
    if "geo" in msg or "region" in msg:
        return "geo_blocked"
    if "resolve" in msg or "no address" in msg or "gai" in msg:
        return "dns"
    return type(exc).__name__.lower()


def probe_country(country: str, timeout_s: float = 15.0) -> tuple[int, int]:
    """Return (n_live, n_total). Prints one line per camera.

    `timeout_s` sets the stream-open/read budget by exporting the env
    knobs detect_core reads at import time - the function itself has no
    timeout parameter; the underlying ffmpeg session uses STREAM_OPEN_TIMEOUT_MS
    / STREAM_READ_TIMEOUT_MS."""
    import os
    os.environ["STREAM_OPEN_TIMEOUT_MS"] = str(int(timeout_s * 1000))
    os.environ["STREAM_READ_TIMEOUT_MS"] = str(int(timeout_s * 1000))
    from app.cameras import COUNTRIES, CAMERAS, country_pool
    from app.detect_core import grab_frame, last_grab_error, resolve_stream
    if country not in COUNTRIES:
        raise SystemExit(f"unknown country {country!r}; "
                         f"expected one of {sorted(COUNTRIES)}")
    cams = country_pool(country)
    if not cams:
        raise SystemExit(f"no cameras registered for {country!r}")
    print(f"probing {len(cams)} camera(s) in {country} (timeout {timeout_s:.0f}s)")
    print("-" * 78)
    live = 0
    cause_counts: dict[str, int] = {}
    for cid in cams:
        cam = CAMERAS[cid]
        name = cam.get("name", cid)
        started = time.time()
        cause = ""
        frame = None
        try:
            url = resolve_stream(cam)
            frame = grab_frame(url)
        except BaseException as e:      # incl. SystemExit from resolve
            cause = _classify(e)
        dt = time.time() - started
        if frame is not None:
            live += 1
            print(f"  LIVE  {cid:<28s} {dt:5.1f}s  {name}")
        else:
            cause = cause or (last_grab_error() or "empty_frame")
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
            print(f"  DEAD  {cid:<28s} {dt:5.1f}s  {name}  [{cause}]")
    print("-" * 78)
    summary = f"{live}/{len(cams)} live"
    if cause_counts:
        top = sorted(cause_counts.items(), key=lambda kv: -kv[1])[:5]
        summary += "  causes: " + ", ".join(f"{k}={v}" for k, v in top)
    print(summary)
    return live, len(cams)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--country", default="turkey")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()
    live, total = probe_country(args.country, timeout_s=args.timeout)
    raise SystemExit(0 if live > 0 else 2)


if __name__ == "__main__":
    main()
