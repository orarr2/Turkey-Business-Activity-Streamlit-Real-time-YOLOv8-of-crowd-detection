"""Firebase (Firestore) backend for the live footfall app.

Firestore is a great fit for "live updating data": the web frontend subscribes with
onSnapshot() and receives every new write in real time — no polling. The collector
writes here; the browser updates instantly.

Collections this writes (the HTML dashboard in web/ subscribes to all three):
  - `footfall`     append-only history (one doc per sample) -> charts / analytics
  - `latest`       one doc per camera (overwritten each sample) -> cheap live KPIs
  - `reid_stats`   one doc per camera (overwritten each sample) -> the re-ID summary
                   table at the bottom of web/index.html. Holds `total_unique`,
                   `total_sightings`, `regulars`, and the per-class breakdown.

Setup (see docs/firebase_setup.md):
  1. Create a Firebase project, enable Firestore.
  2. Project settings -> Service accounts -> Generate new private key -> save JSON.
  3. export FIREBASE_CREDENTIALS=/path/to/serviceAccount.json
"""
from __future__ import annotations

import os


class FirebaseStore:
    """Thin Firestore writer. Initializes the Admin SDK once per process."""

    def __init__(self, cred_path: str | None = None,
                 history_collection: str = "footfall",
                 latest_collection: str = "latest",
                 reid_collection: str = "reid_stats"):
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_path = cred_path or os.environ.get("FIREBASE_CREDENTIALS")
        if not cred_path or not os.path.exists(cred_path):
            raise FileNotFoundError(
                "Firebase service-account JSON not found. Set FIREBASE_CREDENTIALS "
                "or pass cred_path. See docs/firebase_setup.md."
            )
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        self.db = firestore.client()
        self.history = history_collection
        self.latest = latest_collection
        self.reid = reid_collection

    def write(self, record: dict) -> None:
        """Append to history and overwrite the per-camera latest doc.

        `record` keys: ts, cam_id, cam_name, person, vehicles, counts, ok,
                       new_entities, seen_entities
        """
        self.db.collection(self.history).add(record)
        self.db.collection(self.latest).document(record["cam_id"]).set(record)

    def write_reid_stats(self, cam_id: str, stats: dict) -> None:
        """Overwrite the per-camera re-ID summary doc.

        `stats` follows ReidStore.stats() shape:
          {total_unique, total_sightings, per_class: {cls: {unique, total_sightings, regulars}}}
        The HTML dashboard sums `regulars` across classes and shows the totals.
        """
        regulars = sum(p.get("regulars", 0) or 0
                       for p in (stats.get("per_class") or {}).values())
        self.db.collection(self.reid).document(cam_id).set({
            "cam_id":          cam_id,
            "total_unique":    stats.get("total_unique", 0),
            "total_sightings": stats.get("total_sightings", 0),
            "regulars":        regulars,
            "per_class":       stats.get("per_class") or {},
        })
