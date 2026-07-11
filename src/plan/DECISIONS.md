# Design-Spec Validation Verdicts

Line-by-line engineering review of `src/docs/SPEC_ActiveLearning_LoRA.md`
against the codebase and the production environment, performed before this
execution plan was written. Each verdict says what survives, what is
replaced, and why. The SPEC stays as the design reference; THIS file is the
list of deviations the execution plan builds on.

---

## D1 - MC-Dropout uncertainty (SPEC 4.1): REJECTED, replaced

**Spec said:** per-box uncertainty from T=10 stochastic passes with all
`nn.Dropout` modules kept in train() mode.

**Finding:** YOLOv8 *detection* models contain **zero `nn.Dropout`
modules** (Conv-BN-SiLU blocks, C2f, SPPF, Detect head - none carry
dropout). Enabling train() on dropouts is a no-op; the T=10 pass variance
would be exactly 0. Independently, 10 extra full passes on the e2-micro
would multiply burst cost far past the round budget (measured round is
12-20s; the spec's "<=10% overhead" claim did not survive arithmetic).

**Replacement (WS1):** two-component uncertainty, both nearly free:
1. *margin*: distance of the box conf from its class gate,
   `1 - |conf - gate| / gate_span`, high when the model itself was on the
   fence;
2. *flip-variance* (optional, sampled bursts only): ONE extra pass on the
   horizontally-flipped frame; per-box IoU-matched conf delta. Costs a
   single extra inference on ~1-in-5 bursts, not 10 on all.

Aggregate: `uncertainty = 0.6*margin + 0.4*flip_delta` (flip term 0 when
the extra pass is disabled). Same downstream contract as the spec
(metadata field `boxes[i].uncertainty` in [0,1]).

## D2 - LoRA-via-peft on the Detect head (SPEC 4.5, 4.7): REPLACED

**Spec said:** wrap `model.model.model[-1]` with peft LoRA layers, rank 8.

**Finding:** peft-wrapping Ultralytics' Detect module breaks the
trainer's direct attribute access (`stride`, `nc`, `reg_max`, EMA deep-
copies, checkpoint pickling). It is fightable, but everything LoRA buys
here (small artifact, frozen backbone) is available natively:

**Replacement (WS3):** `yolo detect train ... freeze=<all-but-head>` -
Ultralytics' own backbone-freeze - then save ONLY the head tensors as the
artifact (`adapter` = head-only state dict, ~4-6 MB). Loading = load base
yolov8s, overlay the head tensors. Identical promotion-gate / symlink /
rollback semantics as the spec; zero exotic dependencies. The word
"adapter" is kept for continuity.

## D3 - COCO export (SPEC 4.4): SUPERSEDED by shipped work

`tools/export_labels.py` already exports reviewed frames as a YOLO-format
dataset (images/ labels/ dataset.yaml, chronological 90/10 split, verdict
mapping incl. relabel + operator-added misses). Ultralytics trains from
this format natively, so the COCO step is dropped. A COCO converter can
be added later if an external tool ever needs it.

## D4 - BADGE embeddings (SPEC 4.2): UPGRADED input

Spec assumed HSV-histogram embeddings with OSNet "when configured".
OSNet ONNX now ships in-repo and is the default embedder everywhere
(auto-detected). BADGE gets 512-d identity-grade vectors from day one;
k-means++ init is hand-rolled (~30 lines) - sklearn stays OFF the VM.

## D5 - Architecture Option B (SPEC 6): CONFIRMED, transport exists

Split VM (inference + sampling + export) / external trainer (Colab or
operator PC) confirmed. The spec's open transport question is closed:
`app/pool_sync.py` already moves artifacts VM<->Storage<->operator with
manifests, batching and public URLs - the training round-trip reuses it
(new prefix `training/`).

## D6 - Bit-identical fallback (SPEC 4.7): TRIVIALLY SATISFIED

With head-overlay loading, "no adapter file" means the base model is
loaded untouched - byte-identical behavior, no identity-LoRA gymnastics.

## D7 - VM resource envelope: TIGHTENED after production incidents

The e2-micro has 1 GB physical RAM; two kernel oom-kill loops were
diagnosed live (696M peak + upload burst -> killed). Standing envelope
for ALL new VM-side work in this plan:
* 2 GB swapfile present; `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=2`;
* any new per-round compute must keep the measured round under ~30s
  (currently 12-20s at imgsz 512, burst 2);
* any new upload path must batch (<=40 objects/pass, pool_sync pattern);
* Firestore stays under 20,000 writes/day (currently ~19,008) - new
  collections must be write-throttled by design.

## D8 - Success-metric wording (SPEC 12): ADJUSTED

"mAP" targets are measured with Ultralytics `val` on the exporter's
chronological val split. The 40%-fewer-labels headline is measured
naive-vs-BADGE on label-count-matched checkpoints (chronological
comparison; a two-camera A/B is optional stretch, not a gate).

## D9 - Open decisions still requiring the operator (multiple-choice at kickoff)

1. Trainer host: Colab free notebook vs operator PC (CPU torch is
   installed and works; head-only fine-tune of ~500 images is ~20-60 min
   on CPU, minutes on any GPU).
2. (revised 2026-07-11) The frame queue is uncertainty-first by default
   and no naive mode exists; the remaining decision is whether to layer
   BADGE DIVERSITY (k-means++ spread) on top of pure uncertainty, and
   when. D8's efficiency comparison runs as an offline replay.
3. Adapter retention: keep last 7 + every ever-promoted (spec default) or
   keep all.
