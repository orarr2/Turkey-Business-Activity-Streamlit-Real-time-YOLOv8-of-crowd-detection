"""Model-quality scoreboard, computed from user review verdicts.

The dashboard's "Model: X% accuracy · P(person) Y% · FP Z% · N reviews"
header line is driven by ``compute()`` in this module. All numbers come
from ``data/reviews.json`` (the ReviewStore) - nothing here reads live
inference, so a caller can trust it and cache it without invalidation
tricks.

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


def compute(review_store) -> dict:
    """Aggregate all verdicts in the store into a small scoreboard dict."""
    correct = 0
    wrong = 0
    per_cls: dict[str, dict[str, int]] = {}

    for r in review_store._by_path.values():  # noqa: SLF001 - deliberate
        is_correct = r.verdict == "correct"
        cls = r.original_cls or "?"
        rec = per_cls.setdefault(cls, {"correct": 0, "wrong": 0})
        if is_correct:
            correct += 1
            rec["correct"] += 1
        else:
            wrong += 1
            rec["wrong"] += 1

    total = correct + wrong
    accuracy = correct / total if total else None
    fp_rate  = wrong / total if total else None

    classes = []
    for cls, rec in sorted(per_cls.items()):
        n = rec["correct"] + rec["wrong"]
        p = rec["correct"] / n if n else None
        classes.append({
            "cls":       cls,
            "n":         n,
            "precision": round(p, 4) if p is not None else None,
        })

    return {
        "total_reviews": total,
        "accuracy":      round(accuracy, 4) if accuracy is not None else None,
        "fp_rate":       round(fp_rate,  4) if fp_rate  is not None else None,
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
    acc = metrics.get("accuracy")
    fp = metrics.get("fp_rate")
    parts = [f"{int(round(acc * 100))}% accuracy" if acc is not None else "accuracy -"]
    # Two most-reviewed classes, in "P(cls) X%" form
    per_cls = sorted((metrics.get("per_class") or []),
                     key=lambda c: c.get("n", 0), reverse=True)[:2]
    for c in per_cls:
        if c.get("precision") is not None and c.get("n", 0) >= 3:
            parts.append(f"P({c['cls']}) {int(round(c['precision'] * 100))}%")
    if fp is not None:
        parts.append(f"FP {int(round(fp * 100))}%")
    parts.append(f"{total} reviews")
    if boost_summary and boost_summary.get("adjusted_cls"):
        parts.append(f"tuned {boost_summary['adjusted_cls']} classes")
    return "Model: " + " · ".join(parts)
