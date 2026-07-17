# -*- coding: utf-8 -*-
"""A readable, runnable window into EXACTLY what the cloud VM collector does
each round - without Firebase, so you can run it on any machine and watch the
same behaviour the VM shows in its journal.

The real 24/7 collector is `app/collector.py` (entry point `python -m
app.collector`); it is ~2900 lines because it also handles Firestore writes,
re-ID, anomaly gates, snapshots, the daily digest hooks, and hot-reload. THIS
file is not a copy of that logic - it *imports the real functions* and drives
them in the same order, so what you read here is faithful to the VM and cannot
drift out of sync. It prints the per-slot detection line and the country/host
fallback events, and stops after a few rounds.

Run it:
    cd src
    python -m tools.vm_run_readable            # ~4 rounds, default grid
    python -m tools.vm_run_readable --rounds 8 --country thailand

What each round does (identical to the VM's main loop):
  1. director.assign(now)          -> pick 4 live cameras from ONE country
  2. for each camera:
       resolve_stream(cam, now)    -> HLS url (YouTube via android / tvkur /
                                      IBB / cached until its token expires)
       grab_burst(url, n, stride)  -> a short burst of frames
       detect_burst(model, frames) -> YOLO counts (median over the burst)
       director.record(cam, ok...) -> feed health back (pool + host breaker)
  3. director.maybe_advance(now)   -> if the country went dark, rotate to the
                                      next country in the ladder
The only thing removed vs the VM is the Firestore/Storage write and the
re-ID/anomaly bookkeeping - the DETECTION path is the real one.
"""
import argparse
import time

from app.cameras import COUNTRY_ORDER, country_pool, CAMERAS
from app.collector import CountryDirector
from app.detect_core import (DEFAULT_IMGSZ, DEFAULT_PER_CLASS_CONF, detect_burst,
                             grab_burst, invalidate_resolved, last_grab_error,
                             last_grab_http, load_model, resolve_stream)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=4, help="how many rounds to run")
    ap.add_argument("--country", default=None,
                    help="start country (turkey/thailand/japan/usa)")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="the VM runs yolov8n; use yolov8s+ locally for accuracy")
    ap.add_argument("--imgsz", type=int, default=512,
                    help="the VM runs 512; 640/960 recover distant objects")
    ap.add_argument("--burst", type=int, default=2)
    ap.add_argument("--burst-stride", type=int, default=13)
    ap.add_argument("--interval", type=int, default=40)
    args = ap.parse_args()

    print(f"loading {args.weights} (imgsz={args.imgsz}) ...")
    model = load_model(args.weights)

    # ONE country ladder shared by the 4 slots - the exact object the VM builds.
    pools = {c: country_pool(c) for c in COUNTRY_ORDER}
    director = CountryDirector(pools, COUNTRY_ORDER, n_slots=4)
    if args.country and args.country in director.pools:
        director.switch_to(args.country)
    print("country ladder:", " -> ".join(COUNTRY_ORDER),
          f"| starting on {director.active}\n")

    for r in range(1, args.rounds + 1):
        now = time.time()
        # If the active country is dark, rotate before assigning (VM does this).
        adv = director.maybe_advance(now)
        if adv:
            print(f"  * country: {adv[0]} is dark -> switching grid to {adv[1]}")
        country, cams = director.assign(now)
        print(f"round {r}/{args.rounds}  country={country}  cams={cams}")

        for slot_i, cam_id in enumerate(cams, 1):
            cam = CAMERAS[cam_id]
            t = time.time()
            try:
                url = resolve_stream(cam, now)
                frames = grab_burst(url, n=args.burst, stride=args.burst_stride)
            except Exception:
                frames = None
            if not frames:
                invalidate_resolved(cam_id)
                _stage, http = last_grab_http()
                director.record(cam_id, False, http, now, country=country)
                print(f"    slot_{slot_i} {cam_id:22s} MISS "
                      f"({last_grab_error()})  {time.time()-t:.1f}s")
                continue
            # Same detector call the VM makes; median count over the burst.
            gates = dict(cam.get("per_class_conf") or DEFAULT_PER_CLASS_CONF)
            counts, boxes, annotated, dbg = detect_burst(
                model, frames, conf=0.30, imgsz=args.imgsz,
                roi=cam.get("roi"), roi_exclude=cam.get("roi_exclude"),
                per_class_conf=gates, burst_stride=args.burst_stride)
            director.record(cam_id, True, None, now, country=country)
            print(f"    slot_{slot_i} {cam_id:22s} "
                  f"person={counts.get('person', 0)} "
                  f"vehicles={counts.get('vehicles', 0)} "
                  f"(car={counts.get('car', 0)} bus={counts.get('bus', 0)} "
                  f"motorcycle={counts.get('motorcycle', 0)})  "
                  f"{time.time()-t:.1f}s")
        print()

    print("done. This is the same detection path the VM runs 24/7; the VM adds "
          "Firestore writes, re-ID, anomaly gates and the daily PDF report.")


if __name__ == "__main__":
    main()
