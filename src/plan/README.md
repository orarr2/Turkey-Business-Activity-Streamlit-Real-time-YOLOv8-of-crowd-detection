# Active-Learning Upgrade - Execution Plan

Turns `src/docs/SPEC_ActiveLearning_LoRA.md` into buildable work packages.
Read `DECISIONS.md` first - it lists where and why this plan deviates from
the spec (MC-Dropout replaced, LoRA replaced by native head-freeze, COCO
export superseded by the shipped YOLO exporter, OSNet embeddings, resource
envelope). Everything here fits the standing constraints: Always-Free GCP,
main-branch-only, measured VM round time under ~30s, Firestore under the
free write quota.

## What the platform already provides (do not rebuild)

| Capability | Where | Shipped |
|---|---|---|
| Review verdicts incl. per-box relabel + operator-drawn misses | `app/labels.py`, review UI | yes |
| YOLO-format training export w/ chronological 90/10 split | `tools/export_labels.py` | yes |
| Identity-grade embeddings (OSNet ONNX, auto-detected) | `app/reid_embed.py` | yes |
| VM<->operator artifact transport w/ manifest + batching | `app/pool_sync.py` | yes |
| Honest scoreboard (per-metric sample gates) | `app/model_metrics.py` | yes |
| Per-(cam,cls) threshold nudging + auto/manual blacklist | `confidence_boost`, `auto_blacklist` | yes |
| Anomaly-profile self-rebase | `collector.HourlyProfile` | yes |

## Status update (2026-07-11, after the operator-redefinition batch)

Parts of WS1/WS2 landed EARLY, out of the planned order, driven by the
operator's queue-pacing demands:

* SHIPPED: margin-based frame uncertainty (`labels.frame_uncertainty`,
  post-hoc from stored conf vs default gates) + uncertainty-first paced
  frame queue (5 visible / rest at 30) in the review UI. The naive random
  frame sampler NO LONGER EXISTS.
* SHIPPED: per-entity sighting gallery (`app/entity_gallery.py`) and the
  batch-by-batch mistake-rate curve (`model_metrics.learning_curve`) -
  the operator-facing improvement signal WS5 planned to source from
  training runs now has an interim, verdict-based version.
* REMAINING in WS1: capture-time uncertainty in `collector.sample_slot`
  using the EFFECTIVE (boosted/night) gates rather than defaults,
  flip-delta second pass, and per-crop persistence.
* WS2 rescope: frame side is done sans flag; the crop sampler's 70%
  anomaly-pool bias premise is obsolete (statistical anomalies removed,
  scene anomalies are rare by design) - rebuild it as uncertainty+OSNet
  BADGE over live_samples instead.
* D8 consequence: the naive-vs-BADGE comparison arm must be a REPLAY
  (re-rank historical pools offline), not a live A/B - there is no naive
  mode left to run.
* Guardrail honored by the sync layer: reviewed frames are pinned on the
  operator's machine (pool_sync protects them from mirror eviction), so
  the training exporter never loses images behind verdicts.

## Workstreams

### WS1 - Per-box uncertainty (replaces SPEC 4.1)
**Goal:** every saved crop/frame box carries `uncertainty` in [0,1].
**Build:** `app/uncertainty.py`
- `margin_score(conf, gate, span=0.25) -> float` - 1.0 at the gate,
  falling linearly to 0 at `gate +- span`.
- `flip_delta(model, frame, boxes, imgsz) -> dict[box_id, float]` - one
  flipped-frame pass, IoU-match (`box_iou`, mirrored x), normalized conf
  delta; only called when the burst was selected for sampling.
- `attach_uncertainty(boxes, gates, flip=None)` - writes the blended
  field in place.
**Wire:** `collector.sample_slot` after ROI filtering, guarded by the
same `should_sample`/`should_save` cadence that already gates pool
writes; `review_frames.save_frame` and `live_samples.save_crop` persist
the field (frames: metadata json; crops: `_uNN` filename suffix).
**Tests:** margin curve endpoints; flip matching on synthetic mirrored
boxes; metadata round-trip.
**Budget:** +1 inference on ~1-in-5 bursts on ONE camera per round worst
case; measured round must stay <30s.
**DoD:** new review frames carry uncertainty for every box; collector
round-time log unchanged within noise.

### WS2 - BADGE samplers (SPEC 4.2-4.3, OSNet edition)
**Goal:** the review UI serves the crops/frames the model is most unsure
about, with diversity, instead of `random.choice`.
**Build:** `app/badge.py`
- `kmeanspp_pick(vectors, weights, k, seed) -> indices` - k-means++
  INIT only, numpy, no sklearn.
- `sample_crop_badge(store, root, batch=30)` - unreviewed pool ->
  OSNet embed (reuse SnapshotIndex cache) -> scale by uncertainty
  (fallback naive when the field is absent) -> pick k.
- `sample_frame_badge(...)` - frame embedding = mean of top-5 largest
  boxes' embeddings, uncertainty = max box uncertainty.
**Wire:** `REVIEW_SAMPLER=badge|naive` env (default naive);
`?strategy=` query override on `/api/review-sample` + `/api/review-frame`;
response gains `"sampler"` so the UI can badge it; `reviews.json` rows
gain `sampler` + `uncertainty_at_selection` (spec 9.1).
**Tests:** deterministic picks under seed; weight-0 degenerates to
spread-only; missing-uncertainty fallback.
**DoD:** with the flag on, served batch mean-uncertainty measurably
exceeds naive sampling on the same pool (assert in an integration test).

### WS3 - Train + gate + adapter (replaces SPEC 4.5-4.7)
**Goal:** nightly-able loop: export -> head-only fine-tune -> val gate ->
promoted head artifact the collector hot-loads.
**Build:** `tools/train_head.py`
- wraps `yolo detect train` with backbone frozen (`freeze=` all layers
  except Detect), mosaic/mixup off, HSV+flip on, epochs<=10 early-stop;
- emits `data/adapters/<cam>/head_YYYYMMDD.pt` = Detect-head tensors only.
`tools/promote_adapter.py`
- baseline `val` vs candidate `val` on the exporter's val split;
- gate: delta mAP50 >= +0.5pp AND no class drops >2pp (person/car 0pp);
- pass -> atomic `current` pointer update + `history.jsonl` append;
  fail -> `gate.log` line; `--rollback` restores previous pointer.
`app/detect_core.load_model(weights, adapters_dir=None, cam_id=None)`
- overlay `current` head tensors when present; absent -> untouched base
  (bit-identical, D6).
**Transport:** operator PC or Colab pulls the export via `pool_sync`
prefix `training/` (VM uploads dataset zip nightly or on demand);
promoted head uploaded back the same way; collector hot-reload extends
`reload_review_overrides` cadence to also refresh `data/adapters/`.
**Tests:** head-tensor save/load byte equivalence; gate accept/reject on
synthetic metric fixtures; loader fallback identity.
**DoD (spec 14 adapted):** >=2 training runs executed; gate has both
rejected one regressing and promoted one improving candidate; collector
serves the promoted head after a hot-reload with no restart and no OOM.

### WS4 - Per-camera confidence calibration (SPEC 4.8, unchanged)
**Build:** `tools/calibrate_conf.py` - confusion of reviewed boxes per
(cam,cls) with >=30 verdicts -> `conf_star` at target precision 0.90 ->
`data/per_camera_conf.json`; `cameras._merge_per_camera_conf()` runs
AFTER `_merge_confidence_boost` and overrides it per pair.
**Tests:** threshold search on synthetic verdict sets; merge precedence.
**DoD:** a calibrated pair shows in Learning-proof as source=calibration.

### WS5 - "Labels vs quality" curve (SPEC 4.9)
**Build:** `GET /api/al-curve` reading `history.jsonl` (+ Firestore
mirror doc `training_events` for the hosted dashboard case, TTL 30d,
write-per-run only - D7 quota-safe); Chart.js line in `index.html`:
labels_total on X, mAP50 on Y, rejected greyed, baseline dashed.
**DoD:** >=5 real points render after a week of nightly runs.

### WS6 - Automation, observability, docs (SPEC 4.10, 10-12)
systemd timer (or collector round-hook) for nightly export upload;
Colab notebook / PC script for the trainer side; gate + calibration logs
mirrored; README section "Active-learning loop"; pytest coverage for
every module above; final metric readout vs SPEC 12 targets.

## Sequencing

```
WS1 uncertainty ──> WS2 BADGE ─────────────┐
                                            ├──> WS5 curve ──> WS6 wrap
(export: shipped) ─> WS3 train+gate+adapter┘
WS4 calibration (independent, after 30+ verdicts exist)
```

Suggested order of execution: WS1 -> WS3 (manual first run) -> WS4 ->
WS2 -> WS5 -> WS6. Rationale: a single promoted adapter proves the whole
loop before sampler sophistication matters.

## Risk register (delta from SPEC 11)

| Risk | Mitigation |
|---|---|
| Head-only tuning underfits camera quirks | acceptable v1 tradeoff; gate simply won't promote; revisit deeper unfreeze on a paid host only |
| Uncertainty heuristic mis-ranks | BADGE falls back to naive on missing fields; curve (WS5) exposes it empirically |
| CPU training too slow on operator PC | ~500 imgs head-only ~= 20-60 min CPU; Colab free path documented in WS6 |
| VM regression of round time | every WS lands behind a flag; measured round >30s = revert flag |
| Firestore quota creep | only WS5 writes (1 doc/run); everything else disk/Storage |

## Definition of Done (plan level)

1. Operator reviews N frames; nightly (or manual) run trains, gates and
   promotes a head; dashboard curve gains a point; collector serves the
   new head - all with zero paid resources and zero manual VM edits.
2. The three known mis-fire classes (lamp-post person, signage person,
   edge-cropped car as bicycle) drop to zero on reviewed frames within
   three review batches per camera.
3. Rollback drill executed once: `--rollback` restores the previous
   head and the dashboard shows it.

## Kickoff inputs still owed by the operator

* Trainer host choice, sampler-flip timing, adapter retention (D9 -
  asked as multiple-choice at WS3 kickoff).
* The metro/tram photo for `src/docs/images/` regression fixtures.
