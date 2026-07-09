# Specification: Active Learning + LoRA Fine-Tuning Loop

**Status:** design specification, pre-implementation.
**Repo:** `Turkey-Business-Activity-Streamlit-Real-time-YOLOv8-of-crowd-detection`
**Target branch:** `main`

---

## 1. Purpose

Convert the existing human-in-the-loop (HITL) review panel into a real Active
Learning (AL) loop with LoRA fine-tuning of the YOLOv8 detection head and
automatic per-camera confidence calibration, so the model improves from
operator feedback instead of leaving the feedback signal on the floor.

This document is written to stand on its own: someone reading it cold should
be able to implement the change without further clarification. It states the
problem, lists functional and non-functional requirements, describes three
architecture options, recommends one, gives the full end-to-end data contracts
and APIs, sequences the implementation, calls out risks and mitigations, and
ends with open decisions that should be resolved before coding starts.

---

## 2. Current state (what already exists)

- **Collector** runs 24/7 on a GCP `e2-micro` VM (Always Free tier, 1 GB
  RAM). Every 40 s it iterates the 4 grid slots, grabs a 3-frame burst, runs
  `detect_with_boxes` (YOLOv8s + per-class conf gates + person-shape gates),
  updates re-ID / anomaly / line-crossing state, and writes to Firestore.
- **HITL surfaces already shipped:**
  - `web/snapshots/live_samples/…` — one crop every N bursts (see
    `app/live_samples.py`).
  - `web/snapshots/review_frames/…/<ts>.jpg` + sibling `.json` metadata
    (`app/review_frames.py`).
  - `data/reviews.json` (`ReviewStore`) — thread-safe on-disk store, both
    crop-level and frame-level verdicts.
  - Endpoints: `GET /api/review-sample`, `GET /api/review-frame`,
    `POST /api/review-submit`, `POST /api/review-frame-submit`.
- **Current selection strategy** (`app/labels.py::sample_crop` /
  `sample_frame`):
  - Crop: 70 % bias toward the anomaly pool, otherwise routine — then
    `random.choice` inside the chosen pool.
  - Frame: `random.choice` over every un-reviewed frame.
  - No uncertainty signal, no diversity constraint.
- **Current "learning" from reviews** is shallow:
  - `app/confidence_boost.py` nudges per-(cam, cls) confidence by
    `STEP = 0.015` on each verdict, clamped to `[0.20, 0.60]`. Merged into
    `cameras.py` on import; hot-reloaded every few collector rounds.
  - `app/auto_blacklist.py` promotes clustered wrong-verdict regions to
    `roi_exclude_class` polygons.
- **Scoreboard** (`app/model_metrics.py` +
  `dashboard_server.py::_model_metrics`) — accuracy, per-class precision,
  recall (only once a frame review has landed an FN), F1, FP %, N reviews;
  refreshed every 10 s.

The gap the improvement targets is real and matches the code: selection is
naive, "training" is threshold nudging, and there is no gate that decides
whether the model actually got better.

---

## 3. Goals

### Primary (must)
1. **Label efficiency:** hit the same mAP with ~40 % fewer labels than the
   current random sampler — this is the portfolio headline metric.
2. **Recall lift** on the operator's classes per camera, without regressing
   the COCO baseline YOLOv8s ships with (do not forget "car" while learning
   "delivery van").
3. **Adapter promoted only if it actually improves** — an automatic gate on
   a held-out validation split.
4. **Kill the known mis-fires** (lamp posts as person, signage as person,
   edge-cropped cars as bicycle) within the first few review batches per
   camera.
5. **Per-camera confidence calibration** driven by the confusion matrix of
   reviewed crops — supersedes / augments `confidence_boost.py`.

### Secondary (should)
6. **Live "labels vs mAP" chart** in the dashboard, sourced from the
   nightly training runs.
7. **One-config rollback:** if an adapter regresses in production, revert
   is a single symlink flip.
8. **Backward compatibility:** with adapters absent, `detect_with_boxes`
   output must be bit-identical to today's.

### Non-goals (v1)
- Full YOLO backbone fine-tuning (LoRA on the detection head only).
- Multi-GPU / distributed training.
- Continuous / online training (nightly batch only).
- Cross-camera transfer (each camera gets its own adapter; unification comes
  later, if ever).

---

## 4. Functional requirements

### 4.1 Per-crop uncertainty at inference

**Input:** one frame from `grab_burst`.
**Output:** for every box saved to `live_samples` / `review_frames`, add an
`uncertainty` field (float ∈ [0, 1]).

**Computation:**
- `H_cls`: entropy of the YOLO classification posterior for the box over
  `CLASSES_OF_INTEREST`: `-Σ p_i · log p_i`.
- `Var_box`: variance of the (x1, y1, x2, y2) regression across `T = 10`
  stochastic passes with MC-Dropout enabled (all `nn.Dropout` modules kept
  in `train()` mode while the rest of the model stays in `eval()`). Normalize
  by the frame diagonal.
- Aggregate: `uncertainty = sigmoid(0.5·H_cls_norm + 0.5·Var_box_norm)`.

**Persistence:** written to the crop / frame metadata already in place:
- `live_samples/…jpg` — encode in filename (`<ts>_<cls>_<confPct>_<uPct>.jpg`).
- `review_frames/….json` — new field `boxes[i].uncertainty`.

### 4.2 BADGE sampler for crops

**Location:** `app/labels.py`, new function
`sample_crop_badge(store, snapshots_root, batch_size, seed=None)`.

**Algorithm:**
1. Walk the un-reviewed crop pool. Read `uncertainty` from metadata (fall
   back to naive sampling when the field is missing — pre-4.1 crops).
2. For each candidate crop, produce an embedding via the existing embedder
   (OSNet ONNX when configured; HSV histogram otherwise, from
   `app/reid_embed.py`). Scale the embedding by the crop's uncertainty
   scalar — this is the "gradient-magnitude proxy". A real gradient against
   the LoRA head is a valid v2 upgrade once the head is being tuned in
   process, but a magnitude-scaled embedding preserves BADGE's core
   direction × magnitude property at very low cost.
3. Run only the `k-means++` **initialization** step (not full EM) on the
   scaled embeddings; the `k` centers are the picks.
4. Return the `batch_size` selected crop paths.

**Feature flag:** environment variable `REVIEW_SAMPLER=badge|naive`. Default
is `naive` during development, promoted to `badge` once the first LoRA gate
passes and the AL curve chart shows the expected efficiency.

**Endpoint:** `GET /api/review-sample?strategy=badge|naive` (query param
overrides the env var, per request). Returns `{batch: [crop_meta, …]}`. The
UI shows one crop at a time, buffering the rest client-side.

### 4.3 BADGE sampler for frames

Same shape as 4.2 but for whole frames:
- Embedding is the mean OSNet/histogram embedding over the top-K largest
  boxes in the frame (K = 5).
- Uncertainty is the max box uncertainty in the frame (peak drives the
  need to review).

Fed to the existing `add missing` flow that produces FN signal.

### 4.4 COCO export

**Path:** `src/tools/export_reviews.py`.

**Input:** `data/reviews.json` + every `web/snapshots/review_frames/*/*.json`.

**Output:** `data/coco/<cam_id>/train.json`, `val.json` (COCO format), plus
an image tree (symlinks / copies) that the COCO loader can consume.

**Split:** 90 / 10 hold-out **by timestamp** (val = last 10 % chronologically,
not random). Chronological hold-out measures drift toward the present; a
random split lets the model see near-duplicates of validation frames in
train.

**Label mapping:**
- `correct` on an existing box → keep the model's original class.
- `wrong_label` with `corrected_cls` → relabel to `corrected_cls`.
- `not_an_object` / `wrong` → drop the annotation (FP; the box goes away).
- `missed_detections` entries → add a new annotation with the operator's
  chosen class (FN → GT).

**Categories:** COCO ids 0/1/2/3/5/6/7 (person/bicycle/car/motorcycle/bus/
train/truck). Any operator-defined custom class (e.g. `delivery_van`) is
assigned an id ≥ 100.

### 4.5 LoRA training

**Path:** `src/tools/train_lora.py`.

**Input:** `data/coco/<cam_id>/{train,val}.json` and the YOLOv8s weights.
**Output:** `data/adapters/<cam_id>/head_YYYYMMDD.pt` — LoRA adapter state
dict only (~1–3 MB).

**Requirements:**
- Dependencies: `peft`, `ultralytics`.
- Wrap the **detection head only** (`model.model.model[-1]`, Ultralytics's
  Detect head). Backbone frozen (`requires_grad=False`).
- Rank `r = 8`, alpha `= 16`, learning rate `= 1e-4`, epochs `5–10` with
  early stop on val mAP, batch `= 4–8` (sized for the external trainer's
  RAM, not the VM's).
- Augmentation: `mosaic = OFF`, `mixup = OFF` (both harmful on a small
  dataset); light HSV jitter and horizontal flip ON.

**Training environment:** not the VM. See §6 for placement. LoRA-head runs
on 500–1000 images finish in 3–10 minutes on a modest GPU.

### 4.6 Adapter promotion gate

**Path:** `src/tools/promote_adapter.py`.

**Rules:**
1. **Baseline mAP:** clean YOLOv8s on the current val split (no adapter).
2. **Adapter mAP:** YOLOv8s with the candidate adapter on the same val.
3. `Δ mAP ≥ +0.5 pp` (`--promotion-threshold`, configurable).
4. For every class in `CLASSES_OF_INTEREST`: neither `precision[cls]` nor
   `recall[cls]` drops by more than `2 pp` vs baseline
   (`--per-class-regression-limit`).
5. For `person` and `car` (declared "critical"): `0 pp` regression
   allowed (`--critical-classes person,car`).

**Decision:**
- Pass → `data/adapters/<cam_id>/current` symlink is atomically pointed at
  `head_YYYYMMDD.pt`.
- Fail → the current pointer is unchanged (or `None` if there was no prior
  winner). A line is appended to `data/adapters/<cam_id>/gate.log`:
  `rejected 2026-08-15 Δ=-0.3pp reason=<>`.

**History dump:** every run (pass or fail) is appended to
`data/adapters/<cam_id>/history.jsonl`. The dashboard's AL-curve chart reads
from this file.

### 4.7 Adapter loading at inference

**Location:** `app/detect_core.py`, new helper
`load_model_with_adapters(weights, adapters_dir)`.

- Load YOLOv8s normally.
- Wrap the head in `peft` LoRA layers.
- If `<adapters_dir>/<cam_id>/current` exists → load its state dict into the
  LoRA layers. Otherwise leave LoRA as identity (effective `alpha = 0`) so
  YOLO behaves exactly as it does today.

**Contract:** with `adapters_dir=None` or with no adapter file present, the
output of `detect_with_boxes` **must be bit-identical to the current
behavior**. No regression on install.

**Per-camera model access:** the collector calls
`detect_with_boxes(model_for_cam, frame, …)`. Options:
- **Model per camera:** simpler, higher RAM. Not viable on e2-micro
  (multiple copies of YOLOv8s).
- **Shared base + swap LoRA state dict per burst:** ~10–30 ms per swap,
  acceptable inside a 40 s interval. **This is the chosen approach.**

### 4.8 Automatic per-camera confidence calibration

**Path:** `src/tools/calibrate_conf.py`.

**Input:** the confusion matrix of reviewed boxes per (cam, cls). Requires
≥ 30 reviews for the pair to fire (otherwise keep the current threshold —
avoid over-fitting a tiny sample).

**Computation:**
1. For each (cam, cls), enumerate reviewed boxes with their inference-time
   confidence and verdict.
2. Find `conf_star` such that `precision(conf_star) ≥ 0.90`
   (`--target-precision`, configurable) while maximizing surviving recall.
3. If no `conf_star ≤ MAX_CONF (0.60)` satisfies the target → do not
   update (the camera is either too noisy or reviews are still too few).
4. Persist to `data/per_camera_conf.json` as `{cam_id: {cls: conf_star}}`.

**Merge in `cameras.py`:** add `_merge_per_camera_conf()` after the existing
`_merge_confidence_boost()`. When a `conf_star` exists for a (cam, cls) pair
it **overrides** the `confidence_boost` delta: a calibrated adapter beats a
heuristic nudge. Both mechanisms coexist; the heuristic acts as a warm-up
until the calibration has enough reviews to take over.

### 4.9 "Labels vs mAP" endpoint + chart

**Endpoint:** `GET /api/al-curve?cam_id=<opt>`.
**Source:** `data/adapters/<cam_id>/history.jsonl` (plus baseline mAP).
**Response:**

```json
{
  "cam_id": "konya_hukumet",
  "baseline_map": 0.38,
  "points": [
    {"labels_total": 120, "map": 0.42, "adapter": "head_20260801.pt", "promoted": true},
    {"labels_total": 245, "map": 0.47, "adapter": "head_20260802.pt", "promoted": true},
    {"labels_total": 302, "map": 0.46, "adapter": "head_20260803.pt", "promoted": false}
  ]
}
```

UI: a Chart.js line chart, labels on X, mAP on Y, rejected points shown in
grey, promoted points in the accent color. A dashed line for the baseline.

### 4.10 Sanity + rollback

- Manual CLI: `python -m tools.promote_adapter --cam <cam_id> --rollback`
  points `current` back at the previous adapter.
- Every promote / reject writes a line to `data/adapters/<cam_id>/gate.log`
  with timestamp and Δ metrics — auditable outside the dashboard.

---

## 5. Non-functional requirements

- **Backward compatibility:** with the adapter path disabled, inference
  output is bit-identical to today.
- **VM RAM budget:** e2-micro (1 GB) does not run training. Inference stays
  as-is plus ~50–100 MB for the adapter state dict and MC-Dropout T=10 —
  which fires **only on sampled crops**, not on every burst (see 5.3).
- **VM disk budget:** ≤ 200 MB total for adapters + COCO exports (30 GB
  Always-Free disk quota, well within budget).
- **Inference latency:** ≤ 10 % increase on regular bursts. MC-Dropout is
  gated to the ~1-in-5 bursts that are actually sampled.
- **Privacy:** unchanged. Adapters store weights, never crop content.
- **Observability:** every promotion / rejection / calibration writes to
  `data/adapters/<cam>/history.jsonl` and mirrors to a Firestore
  `training_events/{auto}` doc (TTL 30 days), so the dashboard can render
  history without touching the VM disk directly.
- **Testability:** unit tests on BADGE (deterministic seed), on
  `export_reviews` (fixture reviews.json → COCO, idempotent), on
  `promote_adapter` (synthetic metrics → gate accepts / rejects correctly).

---

## 6. Three architecture options

The key decision is **where LoRA training runs**. The e2-micro is capped at
1 GB RAM and has no GPU; a PyTorch training step on YOLOv8s (even head-only)
does not fit. Three ways to split the system.

### Option A — Everything on the VM

- Collector, dashboard, BADGE sampler, `export_reviews`, `train_lora`,
  `promote_adapter`, `calibrate_conf` — all systemd units on the same host.
- Nightly cron (`03:00 Asia/Istanbul`) kicks the training chain.

**Pros:** single environment. Nothing to sync. No egress cost.

**Cons:**
- LoRA training does not fit in 1 GB RAM (head + optimizer + batch 4 ≈
  1.5–2 GB). High OOM risk.
- CPU-only training would take hours — may not finish in the nightly
  window.
- Training and the live collector compete for RAM; collector will crash.
- Escape is `e2-small` (2 GB) — leaves the Always-Free tier ($6–12/mo).

**Verdict:** not recommended for v1 while Always-Free is a hard constraint.

### Option B — Split VM (inference + AL query) + external trainer

- **VM:** everything it does today + BADGE query + `export_reviews` +
  `calibrate_conf` + adapter loading at inference.
- **External trainer** (operator laptop with GPU, Colab session, or a
  GPU-enabled CI runner): runs `train_lora.py` + `promote_adapter.py`,
  produces an adapter file, uploads it back.
- **Transport:** Firebase Storage (`data/adapters/…`), driven by the
  `firebase-admin` credential the VM already has. Alternative: adapter
  files committed to `main` (~1–3 MB `.pt` — acceptable in git).
- **Trigger:** VM's nightly cron writes a `training_jobs/<date>_<cam>`
  Firestore doc with the COCO snapshot URL. External trainer polls (or is
  subscribed) and starts the run.

**Pros:**
- Respects Always-Free. Collector never wobbles.
- Real GPU → training is minutes, not hours.
- Can start manually in Colab and automate later.

**Cons:**
- Two environments to keep working.
- Requires an available GPU (Colab free = time-limited; Colab Pro ≈
  $10/mo; CI GPU minutes cost money).
- If run manually it is HITL on top of HITL.

**Verdict: recommended.** Best fit for the project's cost model.

### Option C — Everything in cloud (VM is display only)

- Collector stays on the VM (still the cheapest way to keep the streams
  hot).
- BADGE query, export, train, promote, calibrate → Cloud Run / Cloud
  Functions (or a GPU CI job) subscribed to Firestore events.
- All state (reviews, metadata, adapters, history) flows through Firebase
  Storage + Firestore.
- Dashboard reads directly from Firestore.

**Pros:** scales to many cameras. Training runs in parallel. VM is out of
the critical path.

**Cons:**
- Cost: Cloud Run + GPU can reach tens of dollars/month.
- Orchestration complexity: Firestore triggers, retries, dead-letter queues.
- Egress fees for frames traveling to the trainer.
- Overkill for 4 cameras.

**Verdict:** parked. Revisit if the deployment grows to ~20+ cameras.

### Recommendation: **Option B**

Rationale: fits the project's Always-Free spirit, is the simplest way to
run real GPU training without hurting the live collector, and gracefully
supports a "start manual, automate later" rollout. §§7–12 assume Option B.

---

## 7. End-to-end flow (Option B)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              e2-micro VM (Always Free)                    │
│                                                                            │
│  ┌────────────┐   burst    ┌────────────────┐  boxes+uncert  ┌──────────┐│
│  │ Collector  ├──────────▶│ detect_core.py ├───────────────▶│ live_    ││
│  │ (40s loop) │           │ + MC-Dropout   │                │ samples/ ││
│  └────────────┘           │  (sampled      │                │ review_  ││
│         │                 │   bursts only) │                │ frames/  ││
│         │                 └────────────────┘                └──────────┘│
│         │                                                         │      │
│         ▼                                                         ▼      │
│  ┌────────────┐   subscribe  ┌────────────────────┐        ┌───────────┐│
│  │ Firestore  │◀─────────────│ dashboard_server   │◀───────│ Web UI    ││
│  │            │              │ + BADGE sampler    │        │ (review)  ││
│  └────────────┘              │ + /api/al-curve    │        └───────────┘│
│         ▲                    └────────────────────┘              │      │
│         │                                                        │      │
│         │                         POST /api/review-submit ◀──────┘      │
│         │                                                                │
│  ┌────────────────────┐                                                  │
│  │ nightly cron 03:00 │                                                  │
│  │  1. export_reviews │ → data/coco/<cam>/{train,val}.json               │
│  │  2. upload to      │                                                  │
│  │     Firebase       │                                                  │
│  │     Storage        │                                                  │
│  └────────────────────┘                                                  │
│         │                                                                │
└─────────┼────────────────────────────────────────────────────────────────┘
          │
          │ (Firestore trigger or manual pull)
          ▼
┌─────────────────────────────────────────────────────────────┐
│           External training host (Colab / laptop GPU)         │
│                                                                │
│  1. download COCO snapshot                                     │
│  2. train_lora.py → head_YYYYMMDD.pt                          │
│  3. promote_adapter.py:                                        │
│       baseline mAP vs adapter mAP                             │
│       per-class regression check                              │
│       if pass → upload adapter to Firebase Storage           │
│                  update "current" pointer in Firestore        │
│       if fail → log rejection to Firestore                   │
│  4. calibrate_conf.py → per_camera_conf.json → upload         │
│  5. append point to al-curve history                          │
└─────────────────────────────────────────────────────────────┘
          │
          │ (Firestore change → collector hot-reload)
          ▼
┌─────────────────────────────────────────────────────────────┐
│       VM: collector re-imports cameras.py + adapters/         │
│       Next burst uses new conf thresholds + adapter.          │
│       Dashboard "labels vs mAP" chart updates.                │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Runtime behavior (day-to-day)

**Real-time (every 40 s, per camera):**
1. `grab_burst` → 3 frames.
2. `detect_with_boxes` (with adapter loaded when available). Standard
   `.eval()` forward pass.
3. If `should_sample()` fires (`live_samples` / `review_frames`): enable
   MC-Dropout on the same frame for T=10 passes → derive per-box
   uncertainty.
4. Save crop / frame with `uncertainty` in metadata.
5. Rest of the pipeline runs unchanged (anomaly, re-ID, Firestore).

**On demand (user opens the review panel):**
6. `GET /api/review-sample?strategy=badge` → `sample_crop_badge` builds a
   batch of 30, returns the first.
7. User submits `POST /api/review-submit` → `ReviewStore.submit()` and the
   existing `confidence_boost.apply_review` fallback fires.

**Nightly (03:00 Asia/Istanbul):**
8. `export_reviews.py` writes `data/coco/<cam>/{train,val}.json` +
   image tree.
9. Cron uploads the tree to Firebase Storage under
   `training_jobs/<YYYYMMDD>/<cam_id>/`.
10. Creates a Firestore doc `training_jobs/<YYYYMMDD>_<cam_id>` with
    status `pending`.

**External trainer (Colab / laptop):**
11. Fetches the COCO snapshot from Firebase Storage, runs `train_lora.py`
    → `promote_adapter.py` → `calibrate_conf.py`.
12. If gate passes: uploads adapter + calibration to Firebase Storage,
    updates the Firestore doc to `promoted`, appends to
    `data/adapters/<cam>/history.jsonl` (mirrored to Firestore for the
    dashboard).

**Hot-reload (on the VM):**
13. Collector already calls `cameras.reload_review_overrides()` every K
    bursts. Extend it to also pull adapters + `per_camera_conf.json` from
    Firebase Storage (or a local cache updated by a lightweight polling
    daemon).

---

## 9. Data & API contracts

### 9.1 `data/reviews.json` (extended)

```json
{
  "reviews": [
    {
      "crop_path": "…",
      "verdict": "correct|wrong_label|not_an_object",
      "original_cls": "person",
      "corrected_cls": null,
      "note": null,
      "reviewed_at": "2026-07-08T14:00:00Z",
      "uncertainty_at_selection": 0.72,
      "sampler": "badge|naive"
    }
  ],
  "frame_reviews": [ /* same shape at frame level */ ]
}
```

### 9.2 `data/adapters/<cam_id>/history.jsonl`

One line per training run:

```json
{"run_at":"2026-08-15T03:00Z","adapter":"head_20260815.pt","labels_total":312,
 "baseline_map":0.42,"adapter_map":0.47,"delta_pp":0.5,
 "per_class_delta":{"person":0.4,"car":0.6,"bus":-0.1},
 "promoted":true,"reason":null}
{"run_at":"2026-08-16T03:00Z","adapter":"head_20260816.pt","labels_total":345,
 "baseline_map":0.47,"adapter_map":0.46,"delta_pp":-0.1,"promoted":false,
 "reason":"delta_pp below threshold 0.5"}
```

### 9.3 `data/per_camera_conf.json`

```json
{
  "updated_at": "2026-08-15T03:15Z",
  "cameras": {
    "konya_hukumet": {
      "person": {"conf": 0.42, "target_precision": 0.90, "n_reviews": 87},
      "car":    {"conf": 0.37, "target_precision": 0.90, "n_reviews": 54}
    }
  }
}
```

### 9.4 Firestore `training_jobs/{YYYYMMDD_camId}`

```json
{
  "cam_id": "konya_hukumet",
  "date": "2026-08-15",
  "status": "pending|running|promoted|rejected|failed",
  "labels_total": 312,
  "coco_url": "gs://…/training_jobs/20260815/konya_hukumet/train.json",
  "adapter_url": "gs://…/data/adapters/konya_hukumet/head_20260815.pt",
  "metrics": { /* same shape as history.jsonl */ },
  "created_at": "2026-08-15T03:00Z",
  "completed_at": "2026-08-15T03:12Z"
}
```

### 9.5 REST additions

- `GET /api/review-sample?strategy=badge|naive` — sampler mode override.
- `GET /api/review-frame?strategy=badge|naive` — sampler mode override.
- `GET /api/al-curve?cam_id=<opt>` — data for the chart.
- `POST /api/adapters/current?cam_id=<>&adapter=<>` — admin rollback.

### 9.6 CLI additions

- `python -m tools.export_reviews --cam <id> --out data/coco/`
- `python -m tools.train_lora --cam <id> --coco data/coco/<id>/ --out data/adapters/<id>/`
- `python -m tools.promote_adapter --cam <id> --candidate <path>`
- `python -m tools.calibrate_conf --cam <id> --target-precision 0.90`
- `python -m tools.al_curve --cam <id> --render html` — offline chart
  fallback for report screenshots.

---

## 10. Implementation sequence

| # | Step | Details | Depends on |
|---|---|---|---|
| 1 | Per-crop uncertainty | MC-Dropout wrapper in `detect_core.py`, aggregate to a scalar, write into `live_samples` + `review_frames` metadata. | — |
| 2 | BADGE crop sampler | `sample_crop_badge` in `labels.py`, feature-flagged. Keep naive fallback. | 1 |
| 3 | BADGE frame sampler | Same idea, at frame granularity. | 2 |
| 4 | `export_reviews.py` | Convert `reviews.json` + `frame_reviews` to COCO 90/10 per cam. | — |
| 5 | `train_lora.py` (manual) | Colab-ready script. First run on one camera; verify the adapter loads cleanly for inference. | 4 |
| 6 | `promote_adapter.py` | Val-split gate. Dry-run against the adapter from step 5. | 5 |
| 7 | Adapter-aware loader | `load_model_with_adapters` in `detect_core.py`. Bit-identical fallback when no adapter is present. | 6 |
| 8 | `calibrate_conf.py` | Confusion matrix → `conf_star`. Merge in `cameras.py`. | 4 |
| 9 | `/api/al-curve` + UI | Endpoint + Chart.js line. | 6 |
| 10 | Cron + nightly pipeline | Systemd timer for export. Colab notebook / laptop script for training. Firebase Storage sync. | 5, 6, 8 |
| 11 | Rollback CLI + observability | history.jsonl, gate.log, Firestore `training_events`. | 6 |
| 12 | Tests + docs | pytest fixtures, README update, notebook cell. | all |

---

## 11. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | Catastrophic forgetting of COCO baseline | LoRA on head only, backbone frozen, per-class regression gate (2 pp; 0 pp for person / car). |
| 2 | MC-Dropout adds latency to every burst | Enable T=10 **only** on the ~1-in-5 bursts that get sampled. All other bursts run identically to today. |
| 3 | LoRA training does not fit VM RAM | Architecture B: external trainer. VM only serves inference + loads adapters. |
| 4 | Adapter loaded non-atomically → collector crashes mid-burst | Load in a startup path or during idle, not during a burst. Atomic swap of the LoRA state dict once loaded. |
| 5 | Colab session drops mid-training | Snapshot per epoch → resume; if unrecoverable, reject the adapter and try tomorrow. |
| 6 | Unrepresentative reviews (operator only picks "easy" crops) | BADGE forces high-uncertainty picks. If uncertainty is missing, fall back to naive — a soft failure, not a wrong signal. |
| 7 | Firebase Storage egress cost | Adapters are ≤ 3 MB; once/day per camera → < 100 MB/mo, negligible. |
| 8 | Calibration crushes recall | `target-precision = 0.90` is deliberately modest. If a recalibration drops recall by > 5 pp on val, revert and log — auto-recovery. |
| 9 | Jitter between old `confidence_boost` and new calibration | Calibration overrides the boost delta once it fires. No addition, no compounding. |
| 10 | Adapter promoted for cam A applied to cam B during a fallback | Adapters keyed by physical `cam_id`, not slot. Loader resolves the actual current cam before selecting the adapter. |

---

## 12. Success metrics

- **Label efficiency:** BADGE reaches target mAP with 40 % fewer labels
  than naive, measured at 100 / 300 / 500 labels.
- **Absolute mAP lift:** at least +3 pp mAP@0.5 after 500 labels per
  camera, vs clean YOLOv8s.
- **False-positive kill:** the three known mis-fires drop to zero within
  three review batches per camera.
- **Adapter promotion rate:** trailing-week average between 30 % and 70 %.
  Below 30 → the gate is too strict or reviews dried up. Above 70 → the
  gate is too loose. Both are diagnostic.
- **VM stability:** no more than 10 % burst-latency overhead. No OOM. No
  need to leave the Always-Free tier.

---

## 13. Open decisions (resolve before coding)

1. **External training host:** Colab (free but manual / time-limited) or a
   laptop GPU (reliable, requires operator attention)? This shapes whether
   `train_lora.py` is a standalone script or a Colab notebook.
2. **Adapter transport:** Firebase Storage (async, uses existing
   credentials) or git-committed to `main` (~1–3 MB binaries, but rides
   the existing sync)? Trade-off is simplicity vs blob bloat.
3. **`confidence_boost` fate:** keep next to `per_camera_conf.json` (the
   calibration overrides it) or delete outright? Recommendation: keep as a
   warm-up fallback for three months, then remove.
4. **`REVIEW_SAMPLER` default:** stay `naive` during dev or flip to
   `badge` immediately after the first gate passes? Recommendation: flip
   after the first pass so the label-efficiency chart is honest.
5. **Adapter retention:** keep the full history or just the last 7 +
   current? Recommendation: last 7, plus every adapter that was ever
   `current`.
6. **VM adapter cache:** cache locally for speed (needs a sync daemon) or
   fetch from Firebase Storage on each hot-reload (lazy, always fresh)?
   Recommendation: local cache with a small polling daemon on the
   Firestore doc.
7. **A/B measurement:** run naive and BADGE on different cameras for a
   week to measure the 40 % claim honestly, or accept a chronological
   comparison? A/B is scientifically cleaner but delays the story.
8. **Custom categories in v1:** ship COCO-only, or allow operator-added
   classes (`delivery_van`, etc.) via a dropdown in the review UI from day
   one? Recommendation: defer to v2.

---

## 14. Definition of Done (v1)

- BADGE sampler is live in production; the average uncertainty of a served
  batch is > 0.6 (i.e. the operator is being served genuinely hard crops).
- The nightly cron has produced ≥ 2 successful COCO exports and triggered
  ≥ 2 training runs (manual or automatic).
- The promotion gate has both rejected at least one regressing adapter
  (proof it works) and promoted at least one improving adapter.
- The collector loads the fresh adapter on the VM without OOM and without
  losing a burst.
- The dashboard's "labels vs mAP" chart shows ≥ 5 live points per camera.
- README carries a new section: "Active Learning + LoRA loop".

---

## 15. Out of scope for v1

- Continuous / online training.
- YOLO backbone fine-tuning.
- A single unified multi-camera adapter.
- Explicit domain adaptation across cameras.
- Automatic deletion of "old" reviews (they remain useful as training
  data indefinitely).
- Third-party camera federation.
