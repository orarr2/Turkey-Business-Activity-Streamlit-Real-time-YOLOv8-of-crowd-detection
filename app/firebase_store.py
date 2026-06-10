"""Firebase (Firestore) backend for the live footfall app.

Firestore is a great fit for "live updating data": the web frontend subscribes with
onSnapshot() and receives every new write in real time — no polling. The collector
writes here; the browser updates instantly.

Two collections are maintained:
  - `footfall`        : append-only history (one doc per sample) -> charts / analytics
  - `latest`          : one doc per camera (overwritten each sample) -> cheap live KPIs

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
                 latest_collection: str = "latest"):
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

    def write(self, record: dict) -> None:
        """Append to history and overwrite the per-camera latest doc.

        `record` keys: ts, cam_id, cam_name, person, vehicles, counts, ok
        """
        self.db.collection(self.history).add(record)
        self.db.collection(self.latest).document(record["cam_id"]).set(record)
