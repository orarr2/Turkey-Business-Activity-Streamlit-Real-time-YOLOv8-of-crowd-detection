"""Model-quality scoreboard, computed from user review verdicts.

The dashboard's "Model: X% accuracy · P(person) Y% · FP Z% · N reviews"
header line is driven by ``compute()`` in this module. All numbers come
from ``data/reviews.json`` (the ReviewStore) - nothing here reads live
inference, so a caller can trust it and cache it without invalidation
tricks.

Percent-metrics need a MINIMUM SAMPLE SIZE to be meaningful. 3/3 correct
is technically 100% but says nothing about the model - it says the user
happened to review three easy crops. Below ``MIN_REVIEWS_FOR_METRIC``
``header_line()`` reports N/A + a progress hint instead of a fabricated
percentage, so the dashboard never shows a trustworthy-looking 100%
against a two-digit sample. Same rule applies to per-class precision.

Definitions used here (crop-level, until the full-frame review UX ships
its own recall data):

  correct       = user said the label the model gave was right
  wrong         = user said the label was wrong (either wrong class OR
                  not an object at all)
  accuracy      = correct / (correct + wrong)
  precision[c]  = correct[c] / (correct[c] + wrong[c])
                  where ``c`` is the class the model originally gave
  fp_rate       = wrong / (correct + wrong)
                  = 1 - accuracy in the crop-level model

`recall` is left off intentionally: without "the model missed this object
here" verdicts we cannot count FN. The full-frame review UX (Task 26)
adds a ``missed`` verdict, at which point ``compute()`` grows an
``recall`` field per class.
"""
from __future__ import annotations

# Below this many verdicts the % metrics are treated as unavailable in the
# UI (header_line reports N/A + progress). 20 balances "quick to reach"
# against "not laughably small" - the standard error on a proportion at
# n=20 is already ~10 pp, low enough that a 60% vs 90% distinction is
# real signal rather than coin flips.
MIN_REVIEWS_FOR_METRIC = 20
# Per-class precision needs its own minimum. Kept lower than the global
# threshold since a class-specific bar of 20 would keep every per-class
# metric hidden until well past 60 total reviews.
MIN_REVIEWS_FOR_PER_CLASS = 5


def compute(review_store) -> dict:
    """Aggregate crop-level AND frame-level verdicts into a scoreboard.

    Two verdict streams feed the numbers:
    * crop reviews  - one verdict per crop (legacy UI, still counted)
    * frame reviews - many verdicts per frame + explicit missed detections,
      which is where FN (and therefore recall / F1) come from.

    When only crop reviews exist the recall / F1 fields are None - honest
    reporting rather than a fabricated denominator.
    """
    # --- crop verdicts (precision-only stream) -----------------------
    correct = 0
    wrong = 0
    per_cls: dict[str, dict[str, int]] = {}
    for r in review_store._by_path.values():  # noqa: SLF001 - deliberate
        cls = r.original_cls or "?"
        rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
        if r.verdict == "correct":
            correct += 1
            rec["tp"] += 1
        else:
            wrong += 1
            rec["fp"] += 1

    # --- frame verdicts (adds FN → recall / F1) ----------------------
    frame_reviews = getattr(review_store, "_frames_by_path", {}).values()
    for fr in frame_reviews:
        meta_boxes_by_id = {}
        try:
            from app.review_frames import load_metadata
            meta = load_metadata(fr.frame_path)
            for b in (meta or {}).get("boxes", []):
                meta_boxes_by_id[str(b["id"])] = b.get("cls", "?")
        except Exception:
            pass
        for box_id, verdict in (fr.box_verdicts or {}).items():
            cls = meta_boxes_by_id.get(str(box_id), "?")
            rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            if verdict == "correct":
                correct += 1; rec["tp"] += 1
            elif verdict == "wrong":
                wrong += 1; rec["fp"] += 1
        for miss in (fr.missed_detections or ()):
            cls = miss.get("cls") or "?"
            rec = per_cls.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            rec["fn"] += 1

    total = correct + wrong
    accuracy = correct / total if total else None
    fp_rate  = wrong / total if total else None

    # Global recall / F1 - defined only when at least one frame review
    # has landed (so FN is a real count, not zero-by-omission).
    total_fn = sum(rec["fn"] for rec in per_cls.values())
    if any(fr.missed_detections is not None for fr in frame_reviews) \
            or total_fn > 0:
        recall = correct / (correct + total_fn) if (correct + total_fn) else None
        precision = correct / (correct + wrong) if (correct + wrong) else None
        f1 = None
        if recall and precision and (recall + precision) > 0:
            f1 = 2 * precision * recall / (precision + recall)
    else:
        recall = None
        f1 = None

    classes = []
    for cls, rec in sorted(per_cls.items()):
        n = rec["tp"] + rec["fp"]
        p_c = rec["tp"] / n if n else None
        n_denom_r = rec["tp"] + rec["fn"]
        r_c = rec["tp"] / n_denom_r if n_denom_r > 0 else None
        classes.append({
            "cls":       cls,
            "n":         n,
            "precision": round(p_c, 4) if p_c is not None else None,
            "recall":    round(r_c, 4) if r_c is not None else None,
            "fn":        rec["fn"],
        })

    return {
        "total_reviews": total,
        "accuracy":      round(accuracy, 4) if accuracy is not None else None,
        "fp_rate":       round(fp_rate,  4) if fp_rate  is not None else None,
        "recall":        round(recall, 4)   if recall   is not None else None,
        "f1":            round(f1, 4)       if f1       is not None else None,
        "per_class":     classes,
    }


def header_line(metrics: dict, boost_summary: dict | None = None) -> str:
    """Human-readable one-line summary for the dashboard header.

    Kept in Python (not JS) so the same string can be logged, dumped in
    Firestore, and rendered by any client identically.
    """
    total = metrics.get("total_reviews") or 0
    if total == 0:
        return "Model: no reviews yet - use the panel below to teach the system"
    if total < MIN_REVIEWS_FOR_METRIC:
        # % on a handful of reviews is misleading (3/3 = 100% is not "the model
        # is perfect", it's "the user hit three easy ones"). Show a plain
        # progress meter until the sample is big enough to trust.
        return (f"Model: accuracy N/A - {total}/{MIN_REVIEWS_FOR_METRIC} "
                f"reviews so far (percentages appear once the sample is "
                f"large enough to be meaningful)")
    acc = metrics.get("accuracy")
    fp = metrics.get("fp_rate")
    recall = metrics.get("recall")
    f1 = metrics.get("f1")
    parts = [f"{int(round(acc * 100))}% accuracy" if acc is not None else "accuracy -"]
    per_cls = sorted((metrics.get("per_class") or []),
                     key=lambda c: c.get("n", 0), reverse=True)[:2]
    for c in per_cls:
        if c.get("precision") is not None and c.get("n", 0) >= MIN_REVIEWS_FOR_PER_CLASS:
            parts.append(f"P({c['cls']}) {int(round(c['precision'] * 100))}%")
    if recall is not None:
        parts.append(f"R {int(round(recall * 100))}%")
    if f1 is not None:
        parts.append(f"F1 {int(round(f1 * 100))}%")
    if fp is not None:
        parts.append(f"FP {int(round(fp * 100))}%")
    parts.append(f"{total} reviews")
    if boost_summary and boost_summary.get("adjusted_cls"):
        parts.append(f"tuned {boost_summary['adjusted_cls']} classes")
    return "Model: " + " · ".join(parts)
