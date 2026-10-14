# EXP-PERM: validating object identity on a recorded clip

**Status:** harness delivered; the recorded-clip run is the open gate before
category learning (`docs/OBJECT_MEMORY_HANDOFF_RECONCILIATION.md` §7, step 1).

**Exit gate** (`docs/COGNITIVE_ARCHITECTURE_MAP.md` §9): controlled occlusion beats
the no-memory baseline on reacquisition, ID switches and position error.

## 1. Endpoints

All endpoints are computed by `experiments/identity_eval.py` from the session
JSONL plus a ground-truth file. Only the attended BBP of each frame is bound to an
object file, so only those bindings are scored.

| Endpoint | Definition |
|---|---|
| ID switch | consecutive bindings of the same physical object to different object-file ids |
| fragmentation | distinct object-file ids bound to one physical object (ideal 1) |
| merge | one object-file id bound to more than one physical object |
| purity | share of each object file's bindings that belong to its majority physical object, weighted over bindings |
| reacquisition | a ground-truth absence (frames where the object is not annotated) bounded by a binding before and a binding after; success iff both bindings share the object-file id |
| position error | at a successful reacquisition, pixel distance between the centre of the object file's predicted box (logged as `object_file.predicted_bbox`) and the attended box |

The log also records which memory state the object was in when it was re-bound
(`previous_visibility`: `occluded`, `lost` or `dormant`), so short and long absences
can be reported separately.

## 2. Protocol

1. **Record the clip** (locally, see `docs/HANDOFF.md`). Static camera, steady
   light, 10-20 s at 30 fps. Two or three objects that YOLO detects reliably,
   including one similar pair (for example two mugs of the same shape in different
   colours). Script the events: a brief occlusion (a hand passes, under 0.5 s), a
   long occlusion (3 s or more), one object leaving the frame and returning
   somewhere else, and two objects visible together. Keep clips out of git
   (`clips/` is ignored).
2. **Hash it** and keep the hash with the annotation:

   ```bash
   sha256sum clips/mugs_occlusion.mp4
   ```

3. **Run a session** on the clip:

   ```bash
   poetry run python experiments/run.py --source clips/mugs_occlusion.mp4 \
       --max-frames 600 --output-dir outputs
   ```

4. **Annotate ground truth** as one JSON document. Truth ids are your labels for
   physical objects, never YOLO classes. Annotate contiguous windows and every
   frame inside them, including the frames where an object is absent: absences
   are what define reacquisition attempts. Annotate every object in the frame, not
   only the attended one, so a change in attention does not need re-annotation.

   ```json
   {
     "source": {"uri": "clips/mugs_occlusion.mp4", "sha256": "<hash>"},
     "iou_threshold": 0.5,
     "frames": [
       {"frame_idx": 0, "boxes": [{"truth_id": "red_mug", "bbox": [412, 220, 530, 360]},
                                  {"truth_id": "blue_mug", "bbox": [700, 230, 815, 365]}]},
       {"frame_idx": 1, "boxes": [{"truth_id": "blue_mug", "bbox": [700, 230, 815, 365]}]}
     ]
   }
   ```

   A binding is labelled with the annotated box of highest IoU at or above
   `iou_threshold`; below it the binding counts as `unmatched`, and bindings on
   frames with no annotation count as `unannotated`.
5. **Score the logged run**:

   ```bash
   python experiments/identity_eval.py outputs/session_seed0_<time>.jsonl \
       --truth clips/mugs_occlusion.truth.json --output outputs/mugs_eval.json
   ```

6. **Sweep thresholds offline** on the same log. The sweep replays the logged BBPs,
   attended embeddings and attention selections through the memories under each
   variant, so it isolates the memory decisions; it always includes the logged
   configuration (and checks that it reproduces the log) and the `no_permanence`
   baseline, which forgets an object after two consecutive missed frames.

   ```bash
   python experiments/sweep.py outputs/session_seed0_<time>.jsonl \
       --truth clips/mugs_occlusion.truth.json \
       --match-threshold 0.55 0.65 0.75 --reid-threshold 0.7 0.75 0.85 \
       --lost-after 5 15 --output outputs/mugs_sweep.json
   ```

   Arbitrary variants (any field of the binder, permanence or prototype configs)
   can be given with `--variants variants.json`, a list of
   `{"name": ..., "overrides": {"binder": {...}, "permanence": {...}}}`.

## 3. Pre-declared decision rule

- **Pass:** with the logged default configuration, zero ID switches across the
  scripted occlusions, a reacquisition rate strictly above the `no_permanence`
  baseline, mean fragmentation strictly below it, and no merged object files for
  the similar pair (purity 1.0).
- **Thresholds only:** if some sweep variant passes while the default does not,
  change the defaults in a configuration-only PR and re-run the protocol.
- **Encoder insufficient:** if no variant separates the similar pair without
  fragmenting (merges and switches trade off across the sweep), the simple 10-d
  crop embedding cannot carry identity on real video. The next PR is a stronger
  `PerceptEncoder` (DINOv2-small) behind the same interface, then this protocol
  again. Category learning stays gated until a pass.

Record the commit hash, clip hash, seed, the eval and sweep JSON, and any negative
result alongside the recording (`docs/OBS_SETUP.md`).

## 4. Synthetic reference

`tests/identity_scenario.py` runs this protocol end to end on deterministic
synthetic frames (`tests/test_identity_eval.py`): the logged configuration scores
zero switches and two of two reacquisitions, while the `no_permanence` baseline
scores two switches and zero of two. A real clip must reproduce that shape of
result before the next layer is built on object identity.

## 5. Limits

- Attention is replayed from the log; the sweep never re-selects BBPs.
- Position error is defined only for successful reacquisitions. After LOST the
  stored velocity is zeroed, so the number reports the error of the stale box,
  not of a tracker; the Stage 9 tracker is still deferred.
- Gaps are counted over annotated frames in order, so annotate contiguous windows.
