"""Firebase backend (Firestore + Storage) for the live footfall app.

Firestore layout — everything is keyed by slot_id, not cam_id, so a fallback
switch doesn't fragment the dashboard's history:

  footfall/{auto}     append-only samples. Each doc carries `slot`, `cam_id`,
                      `cam_name`, `ts`, `person`, `vehicles`, `counts`, `ok`,
                      `is_anomaly`, `new_entities`, `seen_entities`, and
                      `expire_at` (24h ahead — Firestore TTL policy deletes
                      expired docs automatically).
  latest/{slot_id}    one doc per grid slot, overwritten each sample. Powers
                      the "now" KPI tiles cheaply.
  reid_stats/{slot_id} one doc per grid slot, overwritten each sample. Powers
                      the re-ID summary table at the bottom of the dashboard.
  config/grid         one doc, updated by the collector whenever a slot's
                      active cam changes. Structure:
                        { updated_at, slots: [
                            {slot_id, primary, active_cam, active_cam_name,
                             active_embed, active_hls, display_area}, ... ] }
                      The dashboard subscribes to this and re-renders when a
                      fallback happens.
  config/profile_{cam_id}
                      hour-of-week activity baseline per PHYSICAL CAMERA
                      (Welford mean/std per (dow, hour) bucket, per metric).
                      Keyed by cam - not slot - so the learned week-shape
                      belongs to the scene and survives fallback swaps.
                      Written by the collector every ~30 min, loaded on
                      startup so the contextual anomaly check survives
                      restarts. (Legacy profile_{slot_id} docs from before
                      the scene-keyed refactor are ignored; cams re-bootstrap
                      from history once and re-persist under their own key.)

Anomalous samples additionally carry an `anomaly` map:
  { kind: spike|drop|contextual_spike|contextual_drop, metric: person|vehicles,
    window: rolling|hourly, z, observed, expected, bucket? }
The dashboard renders these fields verbatim - the collector is the single
source of truth for what counts as an anomaly.

Firebase Storage layout — public bucket, 24h lifecycle rule:

  snapshots/anomalies/{slot_id}/{ts}.jpg
  snapshots/anomalies/{slot_id}/{ts}_annotated.jpg
  snapshots/returning/{slot_id}/eid{N}_seen{K}_{ts}.jpg
  snapshots/returning/{slot_id}/eid{N}_seen{K}_{ts}_full.jpg

Setup (see docs/firebase_setup.md and src/deploy/gcp-vm/README.md):
  1. Firebase console -> Firestore -> Time-to-live -> footfall.expire_at.
  2. GCP console -> Cloud Storage -> {bucket} -> Lifecycle ->
     Delete objects older than 1 day matching prefix `snapshots/`.
  3. FIREBASE_CREDENTIALS = /path/to/serviceAccount.json (Admin SDK key).
  4. FIREBASE_STORAGE_BUCKET = your-project.appspot.com (optional; if unset
     the collector falls back to disk writes for snapshots — the notebook
     path, not the cloud path).
"""
from __future__ import annotations

import datetime as dt
import os


TTL_HOURS = 24


class FirebaseStore:
    """Firestore + Storage writer, slot-keyed. Initializes Admin SDK once."""

    def __init__(self, cred_path: str | None = None,
                 storage_bucket: str | None = None,
                 history_collection: str = "footfall",
                 latest_collection: str = "latest",
                 reid_collection: str = "reid_stats",
                 config_collection: str = "config"):
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_path = cred_path or os.environ.get("FIREBASE_CREDENTIALS")
        if not cred_path or not os.path.exists(cred_path):
            raise FileNotFoundError(
                "Firebase service-account JSON not found. Set FIREBASE_CREDENTIALS "
                "or pass cred_path. See docs/firebase_setup.md."
            )
        storage_bucket = storage_bucket or os.environ.get("FIREBASE_STORAGE_BUCKET")
        app_options = {"storageBucket": storage_bucket} if storage_bucket else None
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(cred_path), app_options)
        self.db = firestore.client()
        self.history = history_collection
        self.latest  = latest_collection
        self.reid    = reid_collection
        self.config  = config_collection

        self.storage = None
        if storage_bucket:
            from firebase_admin import storage
            self.storage = storage.bucket()

    def write(self, slot_id: str, record: dict) -> None:
        """Append to history (with 24h TTL) and overwrite the per-slot latest doc.

        `record` keys: ts, cam_id, cam_name, person, vehicles, counts, ok,
                       new_entities, seen_entities, is_anomaly.
        `slot_id`: the fixed grid slot (does not change when fallback swaps cams).
        """
        expire_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=TTL_HOURS)
        history_doc = {**record, "slot": slot_id, "expire_at": expire_at}
        self.db.collection(self.history).add(history_doc)
        self.db.collection(self.latest).document(slot_id).set({**record, "slot": slot_id})

    def write_event(self, event: dict) -> None:
        """Append an operational event (loiter / returning / anomaly_push)
        to the `events` collection. Same 24h TTL model as footfall - set the
        Firestore TTL policy on events.expire_at (see docs/firebase_setup.md).
        """
        expire_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=TTL_HOURS)
        self.db.collection("events").add({**event, "expire_at": expire_at})

    def write_reid_stats(self, slot_id: str, cam_id: str, stats: dict) -> None:
        """Overwrite the per-slot re-ID summary doc. Includes cam_id so the UI
        can note whose stats it's showing when a fallback is active.
        """
        regulars = sum(p.get("regulars", 0) or 0
                       for p in (stats.get("per_class") or {}).values())
        self.db.collection(self.reid).document(slot_id).set({
            "slot":            slot_id,
            "cam_id":          cam_id,
            "total_unique":    stats.get("total_unique", 0),
            "total_sightings": stats.get("total_sightings", 0),
            "regulars":        regulars,
            "per_class":       stats.get("per_class") or {},
        })

    def write_grid_config(self, slots_meta: list[dict],
                          country: str | None = None) -> None:
        """Publish which cam is currently active in each slot.

        `slots_meta` is a list of dicts, one per slot, with at least:
          slot_id, primary, active_cam, active_cam_name, active_embed,
          active_hls, display_area, country.
        `country` names which country the grid is currently watching (the
        collector is country-generic: it runs 4 cameras from ONE country
        and rotates when that country goes dark). The dashboard/report use
        it to label the active region.
        """
        doc = {
            "updated_at": dt.datetime.now(dt.timezone.utc),
            "slots":      slots_meta,
        }
        if country is not None:
            doc["country"] = country
        self.db.collection(self.config).document("grid").set(doc)

    # ---- read/persist paths used by the collector's analysis state ---------

    def recent_history(self, since_iso: str, limit_docs: int = 2000) -> list[dict]:
        """History docs with ts >= since_iso, ascending. Single range query on
        `ts` - no composite index needed. Used on startup to reseed rolling
        anomaly windows and (once) bootstrap the hour-of-week profiles."""
        col = self.db.collection(self.history)
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            q = col.where(filter=FieldFilter("ts", ">=", since_iso))
        except ImportError:   # older google-cloud-firestore
            q = col.where("ts", ">=", since_iso)
        q = q.order_by("ts").limit(limit_docs)
        return [d.to_dict() for d in q.stream()]

    def load_slot_profile(self, key: str) -> dict | None:
        """Read the persisted hour-of-week profile for a key (None if absent).

        The collector passes cam_ids since the scene-keyed refactor; the name
        keeps the historical "slot" wording for API compatibility.
        """
        snap = self.db.collection(self.config).document(f"profile_{key}").get()
        return snap.to_dict() if snap.exists else None

    def save_slot_profile(self, key: str, payload: dict) -> None:
        """Overwrite the persisted hour-of-week profile for a key (cam_id)."""
        doc = {**payload, "updated_at": dt.datetime.now(dt.timezone.utc)}
        self.db.collection(self.config).document(f"profile_{key}").set(doc)

    def upload_snapshot(self, path: str, jpeg_bytes: bytes) -> str | None:
        """Upload JPEG bytes to Storage at `snapshots/{path}`. Return public URL.

        Returns None if Storage isn't configured (collector runs without a bucket).
        Public URL model — the Storage lifecycle rule removes the object after 24h.
        """
        if self.storage is None:
            return None
        blob = self.storage.blob(f"snapshots/{path}")
        blob.upload_from_string(jpeg_bytes, content_type="image/jpeg")
        blob.make_public()
        return blob.public_url
