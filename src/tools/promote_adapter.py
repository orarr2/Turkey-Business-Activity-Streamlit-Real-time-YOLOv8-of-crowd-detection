"""Gate a trained head against the baseline; promote it only if it earns it.

    python -m tools.promote_adapter --candidate data/adapters/head_X.pt
    python -m tools.promote_adapter --candidate ... --publish   # + upload
    python -m tools.promote_adapter --rollback                  # undo last

Both models (plain base and base+candidate head, yolov8n by default - the
VM's pinned weights) are validated on the SAME split: the chronological val
slice tools/export_labels.py wrote. Gate
(plan WS3): mAP50 must improve by >= +0.5pp AND no class may drop more than
2pp - person and car may not drop at all. Every verdict (promoted or
rejected) is appended to data/adapters/history.jsonl, so the "it rejected a
regressing run" evidence the plan's DoD asks for is a grep away.

``--publish`` uploads the promoted head + pointer + history to Firebase
Storage under ``training/`` (needs FIREBASE_CREDENTIALS +
FIREBASE_STORAGE_BUCKET, or the auto-detected key on the operator's
machine) - that is what the VM's hot-load polls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent


def _val_metrics(model, data_yaml: str, imgsz: int) -> dict:
    """Run ultralytics val, reduce to the gate's shape:
    {"map50": float, "per_class": {cls: ap50}}."""
    m = model.val(data=data_yaml, imgsz=imgsz, device="cpu", plots=False,
                  verbose=False)
    per_class: dict[str, float] = {}
    try:
        names = model.names or {}
        idxs = list(getattr(m.box, "ap_class_index", []) or [])
        ap50 = list(getattr(m.box, "ap50", []) or [])
        for ci, ap in zip(idxs, ap50):
            per_class[str(names.get(int(ci), ci))] = round(float(ap), 4)
    except Exception:
        pass
    return {"map50": round(float(m.box.map50), 4), "per_class": per_class}


def _storage_bucket():
    """Firebase Storage bucket via the Admin SDK (write access needed)."""
    import firebase_admin
    from firebase_admin import credentials, storage
    from app.pool_sync import _bucket_name
    cred = os.environ.get("FIREBASE_CREDENTIALS")
    bucket = os.environ.get("FIREBASE_STORAGE_BUCKET") or _bucket_name()
    if not cred or not Path(cred).is_file():
        raise SystemExit("--publish needs FIREBASE_CREDENTIALS pointing at "
                         "the service-account json")
    if not bucket:
        raise SystemExit("--publish: no storage bucket (env or "
                         "web/firebase-config.js)")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred),
                                      {"storageBucket": bucket})
    return storage.bucket()


def main() -> None:
    from app import adapters

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default=None,
                    help="head artifact from tools/train_head.py")
    ap.add_argument("--data", default=str(_SRC_ROOT / "data" / "labels_export"
                                          / "dataset.yaml"))
    ap.add_argument("--base", default="yolov8n.pt",
                    help="same base the head was trained on AND the VM runs "
                         "(pinned in deploy/gcp-vm/collector.service)")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--adapters-dir", default=str(adapters.ADAPTERS_DIR))
    ap.add_argument("--publish", action="store_true",
                    help="on promotion, upload head+pointer+history to "
                         "Storage training/")
    ap.add_argument("--rollback", action="store_true",
                    help="restore the previous promoted head and exit")
    args = ap.parse_args()
    adir = Path(args.adapters_dir)

    if args.rollback:
        entry = adapters.rollback(adir)
        print(f"rollback -> {entry['file'] if entry else 'base model (no pointer)'}")
        if args.publish:
            n = adapters.publish_to_storage(_storage_bucket(), adir)
            print(f"published {n} object(s) to Storage training/")
        return

    if not args.candidate:
        raise SystemExit("--candidate is required (or --rollback)")
    cand_path = Path(args.candidate)
    if not cand_path.is_file():
        raise SystemExit(f"candidate not found: {cand_path}")
    if not Path(args.data).is_file():
        raise SystemExit(f"dataset yaml not found: {args.data} - run "
                         f"tools/export_labels first")

    from ultralytics import YOLO

    print(f"promote: validating BASELINE {args.base} on {args.data}")
    base_model = YOLO(args.base)
    base_metrics = _val_metrics(base_model, args.data, args.imgsz)
    print(f"  baseline: mAP50={base_metrics['map50']} "
          f"per-class={base_metrics['per_class']}")

    print(f"promote: validating CANDIDATE {cand_path.name}")
    cand_model = YOLO(args.base)
    adapters.overlay_head(cand_model.model, adapters.load_head(cand_path))
    cand_metrics = _val_metrics(cand_model, args.data, args.imgsz)
    print(f"  candidate: mAP50={cand_metrics['map50']} "
          f"per-class={cand_metrics['per_class']}")

    ok, reasons = adapters.gate_decision(base_metrics, cand_metrics)
    record = {
        "event": "gate",
        "candidate": cand_path.name,
        "base": Path(args.base).name,
        "promoted": ok,
        "baseline": base_metrics,
        "metrics": cand_metrics,
        "reasons": reasons,
    }
    adapters.append_history(record, adir)

    if not ok:
        print("REJECTED:")
        for r in reasons:
            print(f"  - {r}")
        if args.publish:
            # The reject verdict still belongs in the cumulative cloud
            # history (D9 full retention); pointer and head stay untouched.
            adapters.publish_to_storage(_storage_bucket(), adir,
                                        history_only=True)
        _gh_summary(record)
        return

    # Keep the artifact inside adapters_dir (full retention - D9).
    dest = adir / cand_path.name
    if dest.resolve() != cand_path.resolve():
        adir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(cand_path.read_bytes())
    entry = adapters.promote(dest.name, cand_metrics, adir,
                             base=Path(args.base).name)
    print(f"PROMOTED {entry['file']}:")
    for r in reasons:
        print(f"  - {r}")
    if args.publish:
        n = adapters.publish_to_storage(_storage_bucket(), adir)
        print(f"published {n} object(s) to Storage training/ - the VM "
              f"hot-loads it within a few rounds")
    _gh_summary(record)


def _gh_summary(record: dict) -> None:
    """One table row in the GitHub Actions job summary, when running there."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        verdict = "PROMOTED" if record["promoted"] else "REJECTED"
        lines = [f"## Adapter gate: {verdict}",
                 f"- candidate: `{record['candidate']}`",
                 f"- baseline mAP50: {record['baseline']['map50']}",
                 f"- candidate mAP50: {record['metrics']['map50']}",
                 "- reasons:"]
        lines += [f"  - {r}" for r in record["reasons"]]
        lines += ["", "<details><summary>per-class AP50</summary>", "",
                  "```json",
                  json.dumps({"baseline": record["baseline"]["per_class"],
                              "candidate": record["metrics"]["per_class"]},
                             indent=1),
                  "```", "</details>"]
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()
