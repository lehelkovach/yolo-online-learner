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

## 5. Follow-up: top-down attention on the same clip

`--attention top_down --ior-frames 3` (YOLO encoder, velocity fix from the
previous PR applied in both rows) replaces the proxy with the object files'
expectations: BBPs no object explains come first, then violated expectations,
then objects not attended for a while; inhibition of return is per object.

| | proxy (default) | top_down |
|---|---|---|
| frames attending a person | 113 | 259 |
| frames on static van / car / bench | 184 | 39 |
| most frames on one object | 133 | 58 |
| objects attended on 10 or more frames | 7 | 11 |
| glimpses | 584 | 1380 |
| OCCLUDED / LOST transitions | 86 / 53 | 190 / 91 |
| ID switches on provisional truth | 20 | 82 |
| purity | 0.73 | 0.34 |
| mean position error | 0.08 | 0.37 |

Attention now does what it should: it scans the scene and spends its budget on
the moving people. Identity gets worse, and the reason is not attention. With
pedestrians attended in turn, each one is kept alive between attentions by
glimpses, and a glimpse is many-to-many (any unattended box overlapping any
predicted box) and never moves the object. In a group, neighbours keep each
other VISIBLE at stale positions, the predicted boxes drift, and when attention
returns the binder has several plausible candidates with similar appearance and
picks the wrong one. Under the proxy this was hidden because attention sat on
parked vehicles, where glimpses are trivially right.

So `top_down` stays opt-in (the proxy remains the default and the baseline), and
the next mechanism is the one the ladder already names for Stage 9: a one-to-one
spatial assignment of unattended BBPs to VISIBLE and OCCLUDED objects that
updates position and velocity (never appearance), replacing the glimpse. The
identity endpoints are re-measured after it, with the same clip and truth.

## 6. Follow-up: one-to-one spatial track update

The glimpse is replaced by a track update (`ObjectMemory.track`): unattended
BBPs are assigned one-to-one to VISIBLE and OCCLUDED objects by spatial
similarity to the predicted box, greedily by similarity, and the assigned object
moves to the box and re-estimates its velocity. Appearance is never touched (no
embedding was made), and attention staleness counts attended frames only. Same
clip, same truth, YOLO encoder:

| | proxy, glimpse | proxy, track | top_down, glimpse | top_down, track |
|---|---|---|---|---|
| OCCLUDED / LOST transitions | 86 / 53 | 48 / 26 | 190 / 91 | 70 / 45 |
| objects created | 15 | 13 | 17 | 18 |
| ID switches on provisional truth | 20 | 10 | 82 | 55 |
| purity | 0.73 | 0.74 | 0.34 | 0.51 |
| mean position error | 0.08 | 0.10 | 0.37 | 0.23 |
| `no_permanence` baseline switches | 32 | | | |

Under the default proxy attention within-track switches halve and the
`no_permanence` baseline is now beaten by a factor of three. Under top-down
attention, which spends its budget on the crowd, switches fall by a third and
purity rises from 0.34 to 0.51, still short of the proxy because the proxy
mostly looks at parked vehicles. The remaining failures are in the group of
people standing together at the top right, where boxes overlap and the
appearance of 40x80 px pedestrians is close: that is the re-identification
problem of §2 (absent objects with a flat prior) plus true association
ambiguity, which the one-attended-crop design resolves only when attention
lands on the object. Both are what a hand-annotated scripted-occlusion clip
should measure next; `top_down` stays opt-in until then.

## 7. Negative result: positional uncertainty for absent objects

**Correction (see §8):** the `top_down` columns in this section were computed
offline and are not valid. Top-down attention is closed-loop, so a sweep that
replays the logged selections against a diverged memory cascades for reasons
unrelated to the variant. Only the `proxy` columns stand. The negative result
for the absent-object kernel holds under proxy attention (10 switches to 13-22)
and is unmeasured under top-down attention.

§2 proposed that a LOST object should predict its last box with uncertainty that
grows while it is hidden, instead of the flat spatial prior. That is now
`BinderConfig.absent_uncertainty_growth` (default 0, which keeps the flat prior
exactly): the absent object's spatial evidence becomes the larger of the prior and
a centre kernel around its last box whose scale widens by the growth factor per
missed frame. Because binder settings replay offline, the sweep answers directly
(track-update logs, YOLO encoder, same provisional truth):

| prior / growth | proxy: files, switches, purity | top_down: files, switches, purity |
|---|---|---|
| 0.5 / 0 (logged) | 8, 10, 0.74 | 16, 55, 0.51 |
| 0.5 / 0.5 | 9, 17, 0.76 | 13, 170, 0.33 |
| 0.1 / 0.5 | 11, 22, 0.74 | 13, 176, 0.29 |
| 0.1 / 1.0 | 10, 13, 0.77 | 13, 181, 0.30 |
| 0.1 / 0 | 25, 24, 0.82 | 52, 187, 0.42 |

Every variant is worse than the flat prior on this clip. The reason is the scene:
pedestrians walk through the same corridors, so a newcomer passing the spot where
someone vanished is now *more* likely to inherit that identity, and under
top-down attention, which looks at the corridors, identities swap. The far-away
reappearance the kernel guards against was not the dominant failure here. The
option stays, default off, for the scripted-occlusion protocol where an object
returns where it left; it is not a fix for crowds.

The threshold sweep on the same logs completes the picture:

| match / reid | proxy: files, switches, purity | top_down: files, switches, purity |
|---|---|---|
| 0.65 / 0.75 (logged) | 8, 10, 0.74 | 16, 55, 0.51 |
| 0.75 / 0.75 | 10, 8, 0.78 | 17, 164, 0.36 |
| 0.75 / 0.85 | 17, 8, 0.85 | 33, 192, 0.35 |
| 0.65 / 0.85 | 25, 24, 0.82 | 52, 187, 0.42 |

Under proxy attention a modest tightening helps a little. Under top-down
attention any tightening is harmful for a structural reason: a stricter binder
creates a second object file for a person it fails to re-match, and the
one-to-one track update then lets the old (OCCLUDED) and the new file fight over
the same boxes frame after frame. That is the "merge near-duplicates" hardening
the architecture map §4 lists, and it is the next identity mechanism: when a
newly created object and an existing absent one repeatedly claim the same track,
merge them and keep the older id. Thresholds cannot substitute for it.

## 8. The spatial gate, and why top-down logs cannot be swept offline

Reading the top-down track log frame by frame showed the real failure: at frame
1 a VISIBLE object was bound to a box with spatial similarity **0.02**, on the
other side of the frame, because appearance 0.94 plus the class hint cleared the
additive 0.65 threshold on their own; the same at frames 4, 5 and 9. With YOLO
features every pedestrian scores 0.9 or more against every other, so a present
object could teleport onto any person. A present object knows roughly where it
is, so position must gate the match rather than merely weigh in:
`BinderConfig.present_spatial_gate` rejects a VISIBLE or OCCLUDED candidate whose
spatial similarity to the predicted box is below the gate, whatever its
appearance; absent objects are untouched (the prior applies). It is now the
default (0.5); 0 restores the additive baseline.

Measured **live** (YOLO encoder, track update, same provisional truth):

| | proxy | top_down |
|---|---|---|
| switches, gate 0 (before) | 10 | 55 |
| switches, gate 0.3 | | 22 |
| switches, gate 0.5 | **7** | **12** |
| purity, gate 0 to 0.5 | 0.74 to 0.78 | 0.51 to 0.56 |
| mean position error, gate 0 to 0.5 | 0.10 to 0.055 | 0.23 to 0.07 |
| frames attending a person (gate 0.5) | 113 | 241 |

Under top-down attention, which attends the crowd, within-track switches fall
from 55 to 12 and the position error of fulfilled predictions drops to 0.07
diagonals. The `no_permanence` baseline sits at 32 (proxy). The remaining errors
are in the standing group at the top right, where boxes overlap; a
hand-annotated scripted-occlusion clip is still the proper measurement.

**Methodology.** The same gate measured offline with `sweep.py` on the top-down
log gave 171 switches (gate 0.3) and 166 (gate 0.5). The live proxy run
reproduces its sweep exactly (7 switches, purity 0.776), so the tool is right
for open-loop attention; the top-down selections depend on memory, and once a
variant's memory diverges the frozen selections stop refreshing the right
objects and identity cascades. `sweep.py` now refuses a log whose attention
mode is not `proxy` unless `--allow-frozen-attention` is passed, and every
top-down comparison in this document from here on is a live re-run.
