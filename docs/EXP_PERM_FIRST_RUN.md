# EXP-PERM first run on a recorded clip (2026-10-14)

**Clip:** OpenCV sample `vtest.avi` (fixed camera, 768x576, 10 fps; a parked van and
car, pedestrians crossing a forecourt). SHA-256
`45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`.
**Detector:** `yolov8n.pt` (ultralytics 8.3, CPU). **Frames:** 300. **Seed:** 0.
**Code:** `8c6fa46` plus the frozen-YOLO encoder from this PR.

Run as `experiments/run.py --source vtest.avi --max-frames 300` once with the
default 10-d encoder and once with `--encoder yolo`. Both logs replay with zero
mismatches. Attention depends only on the BBPs, so both runs attended exactly the
same boxes; only the embeddings differ, which makes the comparison direct.

Ground truth is the **unrepaired** bootstrap (`truth_bootstrap.py bootstrap
--min-confidence 0.4`): 62 provisional ids that break at every missed detection.
That makes two endpoints valid and one not:

- **valid:** ID switches inside a provisional track (a continuous detection chain
  bound to more than one object file is a genuine identity failure) and purity;
- **not interpretable:** merges, because an object file that correctly bridges
  two fragments of the same person is counted as a merge.

Reacquisition needs hand-repaired absences and is not reported here.

## 1. The 10-d crop embedding cannot carry identity on real video

| Endpoint (same 300 attended boxes) | simple (10-d) | yolo (256-d) |
|---|---|---|
| appearance similarity on matches, mean (min) | 0.993 (0.909) | 0.952 (0.682) |
| object files created | 7 | 15 |
| object files mixing YOLO classes | 3 of 7 | 2 of 15 |
| prototypes formed | 1 | 35 |
| novelty bands on matches | 299 known_instance | 265 known_instance, 30 known_category, 4 boundary |
| mean appearance prediction error | 0.007 | 0.048 |
| purity against provisional truth | 0.606 | 0.730 |
| ID switches against provisional truth | 19 | 18 |

With the colour/geometry vector every crop in this grey forecourt scores cosine
0.99 against every other, so appearance contributes a constant 0.6 to the binder
score and binding is decided by position and class alone. One object file
absorbed truck, person and bench detections; another truck, car, person and
"skis". One prototype formed for the whole clip. This is the **encoder
insufficient** branch of the protocol's decision rule.

DINOv2 weights are not reachable from this environment (403 on every host). The
frozen detector's own features are: `YOLO.embed` on the attended crop at layer 21,
input size 160, global-average-pooled to 256-d, costs 10 ms per crop on CPU and
separates people from vehicles (cosine ~0.5) and most people from each other
(0.77 to 0.96). It is now `--encoder yolo` behind the same `PerceptEncoder`
interface, with space id `yolov8n_embed_l21_160_v1`. The two remaining mixed
object files are the van and car parked side by side (3:18) and a person passing
a bench (5:2).

## 2. Re-identification after LOST is the open identity problem

With YOLO features, within-track switches did not fall (18 vs 19). The per-object
listing shows why: person object files absorb *different* pedestrians. One file
with 29 bound frames is spread over six provisional tracks whose mean centres are
hundreds of pixels apart. The binder gives a LOST or DORMANT object a fixed
spatial prior of 0.5 and the class hint 0.1, so re-identification needs only
appearance cosine 0.83, and at 40x80 px pedestrians clear that easily.

The sweep on the YOLO log shows the trade-off the protocol anticipated:

| binder match / reid | object files | switches | purity |
|---|---|---|---|
| 0.65 / 0.75 (logged) | 9 | 18 | 0.730 |
| 0.65 / 0.85 | 30 | 26 | 0.830 |
| 0.75 / 0.85 | 43 | 23 | 0.900 |
| `no_permanence` baseline | 49 | 32 | 0.892 |

Raising the re-id threshold trades merges for fragmentation; no threshold gives
both. The mechanism that is missing is the one the architecture map §6.2
describes for Stage 9: an absent object should carry a **predicted position with
uncertainty that grows while it is hidden**, not a flat prior, so a person who
left at the right edge is not re-bound to a newcomer at the left edge.

## 3. The Stage 2 attention proxy is degenerate on a real scene

Attention priority is `(1 - confidence)^2 x area / max_area` with a one-frame
spatial inhibition of return. On this clip that selects the large, low-confidence
static objects: 184 of 300 frames went to the parked van, the parked car and a
bench (133 frames on the van alone), and the inhibition histogram shows exactly
one box inhibited on 251 frames, so attention ping-pongs between the two highest
proxies. Pedestrians were attended on 113 frames, in fragments, which is also why
their velocity estimates and glimpses are poor and they fall to LOST so often (69
VISIBLE to OCCLUDED transitions, 25 to LOST).

The docs deferred replacing the proxies "until novelty is validated on recorded
clips". This run is that validation: the proxies do not scan the scene. Stage 4
novelty, Stage 5 prediction error and the object files' expectations are all
available now, so the next mechanism on the ladder is top-down attention
priority (unexpected BBPs first, then violated expectations, with object-based
inhibition of return), after which the identity endpoints should be re-measured.

## 4. Decisions

1. `--encoder yolo` is the encoder for real video; the 10-d vector stays the
   deterministic default for synthetic tests and CI.
2. Category learning stays gated: the pass rule (zero switches, purity 1.0 on a
   similar pair) is not met.
3. Next PRs in order: top-down attention priority with object-based IOR; absent
   objects with growing positional uncertainty in the binder; then re-run this
   protocol with hand-repaired truth on a scripted-occlusion clip.
4. Negative result retained: logs, sweep and eval JSON for both runs were kept
   with the clip hash; the numbers above are copied from them.
