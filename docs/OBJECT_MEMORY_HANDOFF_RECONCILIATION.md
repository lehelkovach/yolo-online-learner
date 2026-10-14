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
| 10 percept graph | Stage 7 | `graph/memory_graph.py` on `graph/percept_graph.py` | done (object, observation, prototype nodes) |
| 11 consolidation | new | `consolidation/` | after 7-10 |
| 12 KSG adapter stub | thin KSG writer | `adapters/ksg.py` | last |
| run.py integration | Stage 0 harness | `experiments/cognition.py`, `experiments/replay.py` | done for 1-7 and 10 |
| EXP-PERM harness | "categories after tracking" gate | `experiments/identity_eval.py`, `experiments/sweep.py` | done; recorded-clip run pending |
| (not in handoff) prediction error | Stage 5 | `memory/prediction.py` | done; typed error, no plasticity change |
| (not in handoff) working memory | Stage 8 | `memory/working_memory.py` | done; register only, nothing reads it yet |

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
  attention embeds one BBP per frame, unattended BBPs are assigned one-to-one to
  VISIBLE and OCCLUDED objects by spatial similarity to their predicted boxes
  (`ObjectMemory.track`, logged as `tracked`): the object moves to the box and
  re-estimates its velocity, but its appearance, observation count and attended
  frame are untouched because no embedding was made. Without this, two visible
  objects would flicker OCCLUDED on alternate frames purely because of
  inhibition-of-return, and walking objects would drift away from their stale
  boxes. `track_update_position: false` restores the original glimpse (miss
  counter spared, nothing moves) as the baseline. `ObjectFile.last_frame_idx` is
  the last position evidence; `last_attended_frame_idx` the last appearance
  observation, which is what attention staleness counts.
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

Spawning into a non-empty memory is gated by two configuration options that
default to off: `novelty_hysteresis` (consecutive unmatched observations required
before a spawn, default 1) and `spawn_cooldown` (observations that must pass
after a spawn before the next, default 0). The hysteresis streak is keyed by the
attended object file, because inhibition of return alternates attention between
objects and a single shared streak would be reset by every other object before
it could grow; the cooldown is shared. A deferred observation changes
nothing and the `learning` block says so in `decision`
(`created`, `updated`, `deferred_hysteresis`, `deferred_cooldown` or
`disabled`); `prototype_id` is `null` for it. `learning_enabled: false` is the
non-learning baseline the experiment invariants require: novelty is still
scored, nothing is ever created or updated. All three are CLI flags on
`experiments/run.py` and sweepable through `experiments/sweep.py`.

The scheduler keeps its labelled confidence proxy as the default. The
recorded-clip run (`docs/EXP_PERM_FIRST_RUN.md`) validated that the proxy does
not scan a real scene, so `AttentionConfig.mode = "top_down"` now ranks BBPs by
the object files' expectations (unexpectedness, last prediction error,
staleness, confidence) with object-based inhibition of return; it is opt-in
until the Stage 9 spatial track update lets identity hold up under it.

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

### Percept graph integration (`graph/memory_graph.py`)

`MemoryGraph` is a typed layer over the existing NetworkX `PerceptGraph`, kept
deliberately small: `object`, `observation` and `prototype` nodes; `OBSERVATION_OF`
(observation -> object), `SIMILAR_TO` (observation -> nearest prototype, weight =
similarity) and `SEEN_WITH` (object -> object in canonical id order, weight =
number of frames both were VISIBLE and bound). Object nodes carry first/last seen,
observation count and current visibility, updated from permanence transitions.
Re-inserting an observation id raises, so an episodic event can never be attached
to two objects. Observation nodes are bounded by `max_observation_nodes` (oldest
evicted, counts kept); object and prototype nodes persist. Category nodes,
`INSTANCE_OF` and `DERIVED_FROM` arrive with the category learner.

Every frame event carries graph counts, and at session end the runner writes a
deterministic, sorted snapshot next to the log as `<session>_graph.json`. The
replay tool rebuilds the graph and diffs it against that snapshot, so the
relational state is covered by the same reproducibility contract as the
per-frame decisions.

### Trace and replay (`experiments/`)

`PerceptualLearner.step` sequences observe -> record episode -> prototype update
-> graph observation -> glimpse -> advance -> graph co-occurrence for one frame
and returns the blocks appended to every `frame` event (plus a top-level
`observation_id`, `null` when nothing was bound, and a `graph` block of counts):

```json
"object_file": {"status": "ok", "object_id": "obj-…", "created": false,
                "reidentified": true, "match_score": 0.85, "candidate_count": 3,
                "previous_visibility": "lost", "visibility": "visible", "...": "..."},
"object_memory": {"counts": {"visible": 3, "occluded": 0, "lost": 0, "dormant": 0,
                  "total": 3}, "tracked": [{"object_id": "obj-…", "bbp_index": 1,
                  "similarity": 0.9, "previous_visibility": "visible", "moved": true}],
                  "transitions": []},
"learning": {"status": "ok", "novelty": 0.0, "nearest_prototype_id": "proto-…",
             "band": "known_instance", "prototype_updated": true, "...": "..."}
```

`session_start.config` now nests the binder, permanence and prototype configs and
`cognition_schema` records the ID seeds. `experiments/replay.py` rebuilds the
learner from those, feeds the logged BBPs and embeddings back through it and
diffs every decision (floats within 1e-9). A `cognition_summary` event precedes
`session_end`.

### Identity evaluation and threshold sweep (`experiments/identity_eval.py`, `experiments/sweep.py`)

`identity_eval.py` scores a session log against a ground-truth JSON (per-frame
boxes with physical-object ids, matched to the attended BBP by IoU): ID switches,
fragmentation, merges, purity, reacquisition after each annotated absence, and
position error of the object file's predicted box at reacquisition. To make the
last endpoint computable offline, every binding now logs `predicted_bbox`, the
box the matched object file predicted for that frame before the update (`null`
when a file is created); the fixed-key schema for unbound frames carries the same
key. `sweep.py` replays one log through `PerceptualLearner` under configuration
variants (`replay_events_with_learner(events, cfg=...)`), always including the
logged configuration, which it checks reproduces the log, and a `no_permanence`
baseline that forgets an object after two consecutive misses.
`truth_bootstrap.py` makes the annotation cheap: it chains the logged BBPs into
provisional truth ids that break at every absence, prints a segment table, and
repairs ids with `relabel --merge/--name/--drop` (collisions keep the
higher-confidence box and are reported). `session_start` now records the git
commit as `code_revision`, which the bootstrap copies into the truth file's
provenance. The protocol, the annotation format and the pre-declared pass/fail
rule are in `docs/EXP_PERM_PROTOCOL.md`.

### Prediction error (`memory/prediction.py`, canonical Stage 5)

Each object file's accumulated appearance and predicted box are a top-down
expectation for the next frame. When attention binds a percept to an object
file, the binder has already scored it against that expectation *before* the
update, so the typed error costs nothing extra: `appearance_error` (`1 - cosine`),
`position_error` (centre distance over the predicted-box diagonal, clipped;
`null` when the object was LOST or DORMANT and so predicted no position) and a
`kind`: `fulfilled`, `violated` (appearance error at or above
`PredictionConfig.violation_threshold`) or `unexpected` (a new object file,
surprise `1.0`). The `prediction` block is part of the replay contract and the
session summary carries the EXP-PREDICT endpoints (counts per kind, mean errors).
Prototype-level error is the Stage 4 novelty. Nothing gates plasticity or
attention on it yet: that is Stage 6 and the deferred attention feedback.

### Working memory (`memory/working_memory.py`, canonical Stage 8)

A bounded register of the few object files the system currently has in mind.
An attention cue (a percept bound to an object file) loads that object or
refreshes its slot; when all `capacity` slots are taken the lowest-utility slot
is evicted, with `utility = recency + evidence_weight * log1p(cues) -
error_weight * last_error` (recency halving every `recency_half_life_frames`),
or by recency alone. An object that goes LOST releases its slot; OCCLUDED
objects stay in mind. Ties fall to the earliest load, so eviction is
deterministic and the `working_memory` block is part of the replay contract;
the summary carries occupancy, loads, reloads, refreshes, evictions and
releases. Slots hold references only; the object file remains the source of
truth. Nothing reads working memory yet: narrowing Stage 5 expectations and
attention to its contents follows the recorded-clip validation.

### Tests

`tests/test_encoder_interface.py`, `test_object_file.py`, `test_object_binder.py`,
`test_object_permanence.py`, `test_prototype_memory.py`, `test_novelty.py`,
`test_episodic_memory.py`, `test_percept_graph_memory.py`, and
`test_cognitive_replay.py`, and `test_identity_eval.py`. The replay test is the
sprint acceptance scenario, now shared from `tests/identity_scenario.py` so the
EXP-PERM tests can reuse it: red mug A appears, moves, is occluded, returns, blue
mug B and headphones C appear, A leaves for ten frames and returns elsewhere. A
keeps one UUID through OCCLUDED and LOST, B and C get distinct UUIDs, three
prototypes form, and the trace replays from the log with zero mismatches. Its
assertions are unchanged; `test_identity_eval.py` scores the same scenario with
the evaluator (zero switches, two of two reacquisitions) and shows the
`no_permanence` baseline failing it (two switches, zero of two).

## 5. Known limits of this baseline

- The simple 10-d embedding separates the synthetic objects cleanly; on real
  video it does not separate anything (cosine 0.99 between every pair of crops
  on the first recorded clip, `docs/EXP_PERM_FIRST_RUN.md`). `--encoder yolo`
  (`features/yolo_encoder.py`, frozen detector features behind the same
  interface) is the encoder for real video. Thresholds are configuration;
  `experiments/sweep.py` runs threshold sweeps offline on one recorded log and
  `identity_eval.py` scores them, so the question is answerable from a single
  recording.
- Only the attended BBP updates appearance. A long-visible but never-attended
  object stays VISIBLE and moves via track updates, but its embedding does not
  drift with it.
- Track updates are spatial only; a different object passing through the
  predicted box can take over a departed object's track until attention lands
  on it and the appearance check corrects the binding.
- `identity_confidence` currently stores the last match score; calibration is
  future work.
- Re-identification of absent objects uses the flat `absent_spatial_prior` by
  default. `absent_uncertainty_growth` adds a centre kernel around the last box
  that widens per missed frame; on the pedestrian clip it made identity worse
  (`docs/EXP_PERM_FIRST_RUN.md` §7), so it stays off and is meant for scripted
  occlusions where an object returns where it left. The open identity mechanism
  is merging near-duplicate object files that compete for one track.

## 6. Related unmerged branches

`origin/agent/stage-4-prototype-bank` (unmerged, based on an older `main`)
carries an alternative prototype bank (`objects/prototype_bank.py`) with a
spawn cooldown, novelty hysteresis, utility-based eviction, a `learning_enabled`
disable switch, and a GitHub Actions CI workflow. `memory/prototypes.py` is the
canonical implementation because it consumes the `PerceptEncoder` contract and is
replayable from the log. Its cooldown, hysteresis and disable switch have since
been ported into `PrototypeMemoryConfig` as `spawn_cooldown`,
`novelty_hysteresis` and `learning_enabled` (observation-counted rather than
frame-counted, since one observation is bound per frame), and the CI workflow
was revived; utility-based eviction remains an open alternative to the
strength-based eviction used here. `origin/cursor/episodic-object-permanence-56eb`
only contains a plan note; its Kalman ghost-buffer idea is the Stage 9 tracker
that the handoff intentionally defers.

## 7. Next steps in handoff order

1. Validate object identity on a hashed recorded clip with controlled occlusion
   (EXP-PERM endpoints: reacquisition, ID switches, position error) before
   category learning. The harness, annotation format and decision rule are in
   `docs/EXP_PERM_PROTOCOL.md`; what remains is recording and annotating the clip
   locally and running the two commands there. A first pass on a public
   pedestrian clip (`docs/EXP_PERM_FIRST_RUN.md`) showed the simple crop
   embedding cannot separate real objects, added the frozen-YOLO encoder, and
   found two further blockers: re-identification after LOST uses a flat spatial
   prior, and the Stage 2 attention proxy fixates on large static objects. Those
   two mechanisms come before the hand-annotated re-run.
2. Category memory and learner (`categories/`), adding `category` nodes and
   `INSTANCE_OF` edges to the graph.
3. Consolidation proposals with evidence ids drawn from episodic memory, then the
   `DeclarativeMemorySink` interface with null, JSONL and KSG sinks.
