# Object-memory handoff reconciliation

**Source reviewed:** `YOLO_ONLINE_LEARNER_HANDOFF.md` ("Development Handoff",
continuous perceptual learning system)

**Review date:** 2026-09-13

**Purpose:** map the newest explicit handoff onto the repository's canonical stage
plan, record what it changes, and describe what the first sprint implemented.

## 1. What the handoff asks for

The handoff extends the current pipeline into a memory architecture:

```text
BBP -> attention -> embedding -> ObjectFile binder -> prototype/instance memory
    -> category memory -> episodic memory -> percept graph -> consolidation
    -> declarative proposals -> (later) NoShogo/KSG
```

Its non-negotiable distinctions are `BBP != ObjectFile != Category !=
declarative fact`, and its first-class requirements are determinism, an
injectable ID factory, configuration-only thresholds, TDD per layer, and a JSONL
log that is a replayable cognitive trace.

Its recommended immediate sprint (section 26) is:

1. `PerceptEncoder` interface wrapping the simple embedding
2. `ObjectFile`
3. `ObjectBinder`
4. visibility/permanence state machine
5. `PrototypeMemory`
6. memory-relative novelty
7. JSONL logging of those results
8. TDD plus a deterministic replay test

It explicitly defers category learning until object identity is reliable.

## 2. Precedence decision

`docs/RESEARCH_HANDOFF_RECONCILIATION.md` §1 ranks "the current explicit user
handoff" above `docs/PHASED_PLAN.md`. This handoff is that document now, so its
sprint order governs, with two consequences:

- **Object files move before working memory.** The earlier order was
  `Stage 4 prototypes -> Stage 5 prediction -> Stage 8 WM -> thin KSG -> Stage 9
  tracking`. The handoff makes persistent object identity the foundation on which
  prototypes, categories and consolidation are built, so canonical Stage 9's
  object-file half lands now (without a neural tracker, as the handoff asks) and
  canonical Stage 4 (bounded prototype bank plus genuine novelty) lands with it.
- **Stage 5, Stage 8 and the thin KSG writer are unchanged in content** and still
  follow; they now run on top of object files rather than before them. The handoff
  routes KSG through consolidation proposals (`adapters/ksg.py`) instead of the
  thin visual writer, which is a narrowing of the same boundary, not a conflict:
  the thin writer's contract (immutable visual space ID, idempotent async
  submission, receipts) becomes the sink behind `DeclarativeMemorySink`.

Everything the earlier reconciliation retained as research context (registers,
BCF, active vision, robotics) is untouched by this handoff.

## 3. Handoff-stage crosswalk

The handoff numbers its own stages 1-12. They are not the canonical
`PHASED_PLAN.md` numbers. Use this table when naming PRs.

| Handoff stage | Canonical home | Module(s) | Status |
|---|---|---|---|
| 1 encoder interface | Stage 3 (extension) | `features/encoder.py` | done |
| 2 object files | Stage 9 (identity half) | `objects/object_file.py` | done |
| 3 binder | Stage 9 | `objects/binder.py`, `objects/ids.py` | done |
| 4 permanence | Stage 9 | `objects/memory.py` | done |
| 5 prototype memory | Stage 4 | `memory/prototypes.py` | done |
| 6 real novelty | Stage 4 | `memory/prototypes.py` (`score_novelty`) | done; attention feedback deferred |
| 7 episodic memory | new, after Stage 9 | `memory/episodes.py` | done |
| 8-9 categories | "categories after tracking" gate | `categories/` | after identity is validated on real clips |
| 10 percept graph | Stage 7 | `graph/percept_graph.py` | next, reuse existing graph |
| 11 consolidation | new | `consolidation/` | after 7-10 |
| 12 KSG adapter stub | thin KSG writer | `adapters/ksg.py` | last |
| run.py integration | Stage 0 harness | `experiments/cognition.py`, `experiments/replay.py` | done for 1-7 |

Canonical Stage 5 (top-down expected embedding and prediction error) and Stage 8
(K-slot working memory) are not in the handoff's sprint list but remain planned;
they should consume `ObjectFile.prototype_embedding` and `PrototypeMemory` rather
than raw BBPs.

## 4. What the sprint implemented

### Encoder interface (`features/encoder.py`)

`EmbeddingResult(vector, space_id)` and the `PerceptEncoder` protocol.
`SimpleCropEncoder` wraps `embed_attended_crop` unchanged; the legacy function and
its metrics helpers remain public. `ensure_same_space` raises
`EmbeddingSpaceMismatchError` so two spaces can never be compared silently.
Every memory layer below takes `EmbeddingResult`, never a bare vector.

### Object files and binder (`objects/`)

- `ObjectFile` is mutable belief state with an immutable `object_id`,
  `VisibilityState`, running-mean or learning-rate appearance update (explicit,
  never implicit), a velocity estimate for reappearance prediction, and a
  `to_dict`/`from_dict` round trip.
- `ObjectBinder` scores `appearance_weight * cosine + spatial_weight * spatial +
  class_weight * class_compatibility`. Spatial similarity is IoU against the
  predicted box, softened by an exponential centre-distance kernel. LOST/DORMANT
  objects use a fixed `absent_spatial_prior` and the stricter `reid_threshold`,
  so re-identification is appearance-led. YOLO class is a weak hint only; the
  tests prove class identity cannot force a match and class flips cannot break one.
- Ties keep insertion order (older object wins), so decisions are deterministic.
- `ObjectMemory` owns the ladder `VISIBLE -> OCCLUDED -> LOST -> DORMANT` with
  frame thresholds in `PermanenceConfig`, plus optional forgetting. Because
  attention embeds one BBP per frame, unattended BBPs that overlap a VISIBLE
  object's predicted box register as a *glimpse*: the miss counter is spared but
  no appearance or position update occurs. Without this, two visible objects would
  flicker OCCLUDED on alternate frames purely because of inhibition-of-return.
- IDs come from an injectable factory. `seeded_uuid_factory(seed)` yields
  UUID4-shaped identifiers that replay exactly; `sequential_id_factory` serves
  tests.

### Prototype memory and novelty (`memory/prototypes.py`)

`PrototypeMemoryConfig` selects the update rule by name (`running_mean` or
`learning_rate`), the spawn threshold, capacity, optional strength decay, and the
novelty band edges. Capacity eviction removes the weakest prototype
deterministically. `score_novelty` never mutates and returns `1 - max cosine`
with the nearest prototype ID and similarity; empty memory returns novelty `1.0`
with `memory_empty: true` and no nearest prototype.

Novelty is not yet fed back into `AttentionScheduler`; the scheduler keeps its
labelled confidence proxy until novelty is validated on recorded clips.

### Episodic memory (`memory/episodes.py`)

`EpisodicMemory` is an append-only, time-indexed store of `ObservationEvent`s,
one per attended observation that was bound to an object file. Events are frozen
dataclasses and carry the object id, frame, timestamp, box, embedding with its
space id, detector confidence and the object's category id at the time (always
`None` until categories exist). `last_seen`, `observations_for` and
`objects_seen_in_range` answer the provenance questions consolidation will ask;
range queries return objects ordered by first appearance so answers are
deterministic. Retention is bounded by `max_events` (oldest dropped, totals
kept), and an object's history stays queryable after it goes LOST or DORMANT.
Observation ids come from a third seeded UUID stream, so replay reproduces them.

### Trace and replay (`experiments/`)

`PerceptualLearner.step` sequences observe -> record episode -> glimpse ->
advance -> prototype update for one frame and returns the blocks appended to
every `frame` event (plus a top-level `observation_id`, `null` when nothing was
bound):

```json
"object_file": {"status": "ok", "object_id": "obj-…", "created": false,
                "reidentified": true, "match_score": 0.85, "candidate_count": 3,
                "previous_visibility": "lost", "visibility": "visible", "...": "..."},
"object_memory": {"counts": {"visible": 3, "occluded": 0, "lost": 0, "dormant": 0,
                  "total": 3}, "glimpsed": ["obj-…"], "transitions": []},
"learning": {"status": "ok", "novelty": 0.0, "nearest_prototype_id": "proto-…",
             "band": "known_instance", "prototype_updated": true, "...": "..."}
```

`session_start.config` now nests the binder, permanence and prototype configs and
`cognition_schema` records the ID seeds. `experiments/replay.py` rebuilds the
learner from those, feeds the logged BBPs and embeddings back through it and
diffs every decision (floats within 1e-9). A `cognition_summary` event precedes
`session_end`.

### Tests

`tests/test_encoder_interface.py`, `test_object_file.py`, `test_object_binder.py`,
`test_object_permanence.py`, `test_prototype_memory.py`, `test_novelty.py`,
`test_episodic_memory.py`, and `test_cognitive_replay.py`. The last one is the sprint acceptance scenario: red
mug A appears, moves, is occluded, returns, blue mug B and headphones C appear, A
leaves for ten frames and returns elsewhere. A keeps one UUID through OCCLUDED
and LOST, B and C get distinct UUIDs, three prototypes form, and the trace replays
from the log with zero mismatches. All pre-existing tests are unchanged.

## 5. Known limits of this baseline

- The simple 10-d embedding separates the synthetic objects cleanly; on real
  video, appearance similarity between different objects of the same colour
  and shape will be high. Thresholds are configuration, and the replay tool
  exists so threshold sweeps can be run offline on one recorded log.
- Only the attended BBP updates appearance. A long-visible but never-attended
  object stays VISIBLE via glimpses but its embedding does not drift with it.
- Glimpses use IoU only; a different object passing through the predicted box
  can keep a departed object VISIBLE for a few frames.
- `identity_confidence` currently stores the last match score; calibration is
  future work.

## 6. Related unmerged branches

`origin/agent/stage-4-prototype-bank` (unmerged, based on an older `main`)
carries an alternative prototype bank (`objects/prototype_bank.py`) with a
spawn cooldown, novelty hysteresis, utility-based eviction, a `learning_enabled`
disable switch, and a GitHub Actions CI workflow. `memory/prototypes.py` is the
canonical implementation because it consumes the `PerceptEncoder` contract and is
replayable from the log; the cooldown, hysteresis and disable switch are worth
porting as separate, individually tested config options, and the CI workflow
should be revived as its own PR. `origin/cursor/episodic-object-permanence-56eb`
only contains a plan note; its Kalman ghost-buffer idea is the Stage 9 tracker
that the handoff intentionally defers.

## 7. Next steps in handoff order

1. Percept graph integration: `object`, `prototype`, `observation` nodes and
   `OBSERVATION_OF`, `SIMILAR_TO`, `TRANSITIONS_TO` edges on the existing
   `PerceptGraph`, with save/load round trip.
2. Validate object identity on a hashed recorded clip with controlled occlusion
   (EXP-PERM endpoints: reacquisition, ID switches) before category learning.
3. Category memory and learner, then consolidation proposals, then the KSG sink.
