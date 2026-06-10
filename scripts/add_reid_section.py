"""Insert a re-identification section into turkey_business_activity.ipynb.

Idempotent: if a cell already starts with the marker text we skip the insert.
The new section is placed right after Section 5 (dwell-time analysis), i.e. before
'Section 6 — Is it worth opening a business here'.
"""
import json
import sys
from pathlib import Path

import nbformat

sys.stdout.reconfigure(encoding="utf-8")

NB = Path(__file__).resolve().parent.parent / "notebooks" / "turkey_business_activity.ipynb"
MARKER = "## 5b. Re-identification — \"have I seen this person before?\""

nb = nbformat.read(NB, as_version=4)
if any(MARKER in (c.source if isinstance(c.source, str) else "".join(c.source))
       for c in nb.cells):
    print("Re-ID section already present — skipping.")
    sys.exit(0)

md_intro = f"""{MARKER}

The detection counts above tell you *how many* people are visible at any moment, but they
double-count anyone who lingers in front of the camera. To answer questions like *"how many
unique customers walked by today?"* or *"is that the same delivery van I saw yesterday?"*
we need **re-identification**: a persistent identity attached to each person/vehicle that
survives across frames, bursts and days.

The implementation is in `app/reid.py`:

1. For each YOLO detection, crop the bounding box.
2. Build a *masked* HSV color histogram (8x8x8 bins, V<30 pixels ignored — kills the
   sodium-yellow night cast on the Konya square) plus aspect ratio + normalized area.
3. L2-normalize -> 514-dim appearance vector.
4. Compare to every entity of the same class already in `data/reid.db` via cosine
   similarity. If the best match is >= `threshold` (default 0.92) we update its
   `sightings` and `last_seen`; otherwise we register a new entity.

This is a **demo-grade signature**. It works well in daylight (different clothing colors
give clearly different histograms). It produces false matches at night when the whole
scene is yellow-tinted — swap `embed_crop()` for an OSNet/torchreid forward pass for
production-grade re-ID; the SQLite registry around it stays the same."""

code_init = """from app.detect_core import load_model, grab_frame, detect_with_boxes, annotate
from app.reid import ReidStore
import cv2, time
import matplotlib.pyplot as plt

REID_DB = '../data/reid_notebook.db'
Path(REID_DB).parent.mkdir(parents=True, exist_ok=True)
# fresh registry for the notebook demo so re-runs are reproducible
Path(REID_DB).unlink(missing_ok=True)
reid = ReidStore(REID_DB, threshold=0.92)

# use the model we already loaded above; lower conf so we catch the small/distant people
# the Konya wide-angle camera shows.
CAM_ID = 'konya_hukumet'
cam = CAMERAS[CAM_ID]
stream_url = cam['url']
print('feeding re-ID from', cam['name'])"""

code_loop = """# Sample N frames every `interval_s` seconds, run YOLO on each, push every detection
# through the re-ID registry. ~2 minutes is enough to see returning IDs appear.
N_SAMPLES, INTERVAL_S, CONF = 20, 6, 0.25

rows = []
for i in range(N_SAMPLES):
    f = grab_frame(stream_url)
    if f is None:
        print(f'[{i:02d}] miss'); time.sleep(INTERVAL_S); continue
    counts, boxes = detect_with_boxes(model, f, conf=CONF)
    results = reid.update_from_frame(CAM_ID, f, boxes)
    new = sum(r.is_new for r in results)
    seen_again = len(results) - new
    rows.append({'sample': i, 'person': counts['person'], 'vehicles': counts['vehicles'],
                 'detections': len(boxes), 'new_ids': new, 'seen_again': seen_again})
    print(f'[{i:02d}] person={counts[\"person\"]} vehicles={counts[\"vehicles\"]} '
          f'-> new={new} seen_again={seen_again}')
    time.sleep(INTERVAL_S)

reid_df = pd.DataFrame(rows)
reid_df"""

code_stats = """# Roll-up: how many unique entities did we see? how many came back >=3 times?
stats = reid.stats(CAM_ID)
print('Total unique entities (this camera):', stats['total_unique'])
print('Total sightings:', stats['total_sightings'])
for cls, s in stats['per_class'].items():
    print(f\"  {cls:10s}  unique={s['unique']}  sightings={s['total_sightings']}  \"
          f\"regulars(>=3)={s['regulars']}\")

print('\\nTop returning entities:')
for r in reid.top_regulars(CAM_ID, n=10):
    print(f\"  #{r['entity_id']:4d}  {r['cls']:8s}  sightings={r['sightings']}  \"
          f\"first={r['first_seen']}  last={r['last_seen']}\")"""

code_plot = """# Visual: returning-visitor curve — what fraction of detections are 'seen again' over time?
if len(reid_df) >= 3:
    reid_df = reid_df.copy()
    reid_df['returning_rate'] = (reid_df['seen_again'] /
                                 reid_df['detections'].replace(0, np.nan))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(reid_df['sample'], reid_df['new_ids'], marker='o', label='new IDs')
    ax[0].plot(reid_df['sample'], reid_df['seen_again'], marker='s', label='seen again')
    ax[0].set_title('Re-ID activity per sample')
    ax[0].set_xlabel('sample #'); ax[0].set_ylabel('count'); ax[0].legend()

    ax[1].plot(reid_df['sample'], reid_df['returning_rate'].fillna(0), marker='o',
               color='#36d399')
    ax[1].set_title('Returning-visitor rate (seen_again / detections)')
    ax[1].set_xlabel('sample #'); ax[1].set_ylim(0, 1)
    plt.tight_layout(); plt.show()
else:
    print('Not enough samples for the returning-visitor plot.')"""

code_caveat = """# IMPORTANT — re-ID quality depends on the scene.
#
# At Konya Hukumet Meydani at night the whole scene is uniform sodium yellow.
# Color-histogram re-ID will over-merge IDs there. To validate the *concept*, point
# the camera at the daylight Grand Bazaar / Spice Bazaar (different clothing colors)
# or set `threshold=0.97` to be very conservative about matches.
#
# Production path:
#   pip install torchreid
#   from torchreid.utils import FeatureExtractor
#   extractor = FeatureExtractor(model_name='osnet_ain_x1_0', model_path='', device='cpu')
#   def embed_crop(crop, cls): return extractor([crop])[0].cpu().numpy()
# Then keep the rest of app/reid.py exactly as-is. The 2,048-dim OSNet embedding
# survives lighting changes, pose changes, and partial occlusion much better than
# a color histogram."""

new_cells = [
    nbformat.v4.new_markdown_cell(md_intro),
    nbformat.v4.new_code_cell(code_init),
    nbformat.v4.new_code_cell(code_loop),
    nbformat.v4.new_code_cell(code_stats),
    nbformat.v4.new_code_cell(code_plot),
    nbformat.v4.new_markdown_cell(code_caveat.replace("# ", "")),  # render as text
]

# find the cell that starts Section 6 ("## 6. ...") and insert before it
def cell_source(c) -> str:
    return c.source if isinstance(c.source, str) else "".join(c.source)

insert_at = None
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and cell_source(c).startswith("## 6."):
        insert_at = i; break
if insert_at is None:
    insert_at = len(nb.cells)

nb.cells[insert_at:insert_at] = new_cells
nbformat.write(nb, NB)
print(f"Inserted {len(new_cells)} cells at position {insert_at}.")
print(f"Notebook now has {len(nb.cells)} cells.")
