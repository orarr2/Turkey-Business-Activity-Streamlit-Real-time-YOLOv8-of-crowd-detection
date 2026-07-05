"""One-shot cleanup of orphaned slot documents in Firestore.

When GRID_SLOTS in app/cameras.py is edited to add / rename / remove a slot,
the collector starts writing to the new slot ids but the docs under the old
slot ids remain in `latest/`, `reid_stats/` and `config/grid.slots`. The
dashboard filters those out (it only renders slot_ids in the current
GRID_SLOTS), so they are harmless UX-wise, but they clutter Firestore and
never expire on their own (only `footfall/` has a TTL policy).

    export FIREBASE_CREDENTIALS=/abs/path/to/serviceAccount.json
    python cleanup_old_slots.py                 # dry-run: list what would go
    python cleanup_old_slots.py --apply         # actually delete

Safety:
  * Only deletes docs whose id is NOT in GRID_SLOTS (and config/profile_* docs
    that match neither a current camera nor a current slot).
  * Never touches the `footfall/` history collection (Firestore TTL handles that).
  * Prints every id before deleting.
"""
from __future__ import annotations

import argparse
import os
import sys

from app.cameras import GRID_SLOTS

# The collections that key by slot_id (one doc per slot, overwritten each round).
SLOT_COLLECTIONS = ("latest", "reid_stats")
CONFIG_GRID_PATH = ("config", "grid")
# Hour-of-week baselines live under config/profile_{cam_id} since the
# scene-keyed refactor (pre-refactor docs were profile_{slot_id}); any
# profile_* doc whose suffix is neither a current cam nor a current slot is
# orphaned (removed cams, renamed slots).
PROFILE_PREFIX = "profile_"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is dry-run)")
    args = ap.parse_args()

    if not os.environ.get("FIREBASE_CREDENTIALS"):
        print("FIREBASE_CREDENTIALS is not set - point it at the service-account JSON.",
              file=sys.stderr)
        return 2

    import firebase_admin
    from firebase_admin import credentials, firestore
    firebase_admin.initialize_app(credentials.Certificate(os.environ["FIREBASE_CREDENTIALS"]))
    db = firestore.client()

    current_slot_ids = {s["slot_id"] for s in GRID_SLOTS}
    current_cam_ids = {c for s in GRID_SLOTS for c in [s["primary"], *s["fallbacks"]]}
    print(f"Current GRID_SLOTS: {sorted(current_slot_ids)}\n")

    to_delete: list[tuple[str, str]] = []
    for coll in SLOT_COLLECTIONS:
        for doc in db.collection(coll).stream():
            if doc.id not in current_slot_ids:
                to_delete.append((coll, doc.id))

    # Orphaned hour-of-week profile docs: keep profile_{cam} for current cams;
    # everything else (legacy profile_{slot}, removed cams) goes.
    keep_profiles = {f"{PROFILE_PREFIX}{cid}" for cid in current_cam_ids}
    for doc in db.collection(CONFIG_GRID_PATH[0]).stream():
        if doc.id.startswith(PROFILE_PREFIX) and doc.id not in keep_profiles:
            to_delete.append((CONFIG_GRID_PATH[0], doc.id))

    # Also prune stale slots from config/grid.slots so the dashboard doesn't
    # briefly render an old tile before its latest/{id} doc arrives.
    cfg_ref = db.collection(CONFIG_GRID_PATH[0]).document(CONFIG_GRID_PATH[1])
    cfg_snap = cfg_ref.get()
    stale_in_cfg: list[str] = []
    kept_slots = None
    if cfg_snap.exists:
        data = cfg_snap.to_dict() or {}
        slots = data.get("slots") or []
        kept_slots = [s for s in slots if s.get("slot_id") in current_slot_ids]
        stale_in_cfg = [s.get("slot_id") for s in slots if s.get("slot_id") not in current_slot_ids]

    if not to_delete and not stale_in_cfg:
        print("Nothing to clean up - Firestore already matches GRID_SLOTS.")
        return 0

    print("Orphan documents (would delete):")
    for coll, docid in to_delete:
        print(f"  {coll}/{docid}")
    if stale_in_cfg:
        print(f"Stale slot entries in config/grid.slots: {stale_in_cfg}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to actually delete.")
        return 0

    print("\nApplying deletes...")
    for coll, docid in to_delete:
        db.collection(coll).document(docid).delete()
        print(f"  deleted {coll}/{docid}")

    if stale_in_cfg and kept_slots is not None:
        cfg_ref.update({"slots": kept_slots})
        print(f"  config/grid.slots rewritten with {len(kept_slots)} slot(s)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
