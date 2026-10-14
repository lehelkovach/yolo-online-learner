## Minimal staged plan (least dependency first)

Design goal: add **one mechanism at a time**, keep interfaces stable, and make each stage publishable via logged metrics.

Active work-order note: the current explicit handoff (see
`docs/OBJECT_MEMORY_HANDOFF_RECONCILIATION.md`) makes persistent object identity
the foundation. Delivered so far: Stage 2 attention, Stage 3 embeddings behind a
`PerceptEncoder` interface, the object-file/permanence half of Stage 9 (no neural
tracker), Stage 4 prototype memory with memory-relative novelty, episodic memory
of attended observations, and a typed object/observation/prototype layer on the
Stage 7 percept graph, all logged and replayable. The EXP-PERM harness
(`experiments/identity_eval.py`, `experiments/sweep.py`, protocol in
`docs/EXP_PERM_PROTOCOL.md`) is in place; the recorded-clip run itself is the open
gate. Next in order: that real-clip identity check, categories, consolidation
proposals, then the KSG sink. Stage 5 and Stage 8 remain
planned and build on object files. Stages 6 and 7 retain their numbers below;
Stage 7's graph substrate is reused by the percept-graph integration step.

### Stage 0 — Experiment harness (reproducibility)

- Add a session runner that emits JSONL.
- Seed everything; keep config small and stable.
- Tests: deterministic log schema, stable output structure.

### Stage 1 — BBP generator (YOLO front-end)

- YOLO is frozen. It proposes BBPs.
- Tests: determinism and sanity checks on BBP schema.

### Stage 2 — Attention scheduler (serial conscious stream)

- Select 1 BBP per frame (WTA) with inhibition-of-return.
- Tests: exactly one selection; no “stuck” fixation.
- Status: the confidence/area proxy is the default. `AttentionConfig.mode =
  "top_down"` (opt-in, `--attention top_down`) ranks BBPs by the object files'
  expectations (unexpected first, then violated expectations, then staleness,
  confidence as a weak tie-break) with object-based inhibition of return; every
  component is logged and the decision is regenerated and checked by replay.
  On the first recorded clip it moves attention from parked vehicles to the
  pedestrians; it becomes the default once the Stage 9 spatial track update
  makes identity hold up under it (`docs/EXP_PERM_FIRST_RUN.md` §5).

### Stage 3 — Simple embeddings (no deep nets yet)

- Start with cheap embeddings (bbox geometry + basic crop statistics).
- Tests: stability of embedding distribution; bounded norms.

### Stage 4 — Prototype bank + novelty

- Online prototype matching + spawn-on-surprise + bounded counts.
- Tests: prototype count bounded; novelty spikes on distribution shift.

### Stage 5 — Dual processing predictive coding

- Top-down prototypes predict expected embedding; select winner(s) by min error.
- Tests: error decreases for repeated percepts; spikes for novel ones.
- Status: delivered as typed prediction error (`memory/prediction.py`). Each object
  file's accumulated appearance and predicted box are the top-down expectation; the
  attended observation's gap to it is logged per frame as `prediction`
  (`appearance_error`, `position_error`, kind `fulfilled`/`violated`/`unexpected`)
  and summarised at session end. Winner selection by minimum error is the Stage 4
  nearest-prototype match. Error-gated gain is Stage 6; error-driven attention
  waits for the recorded-clip validation.

### Stage 6 — Habituation / sensitization (gain gating)

- Repeated low error reduces gain; surprise increases gain and learning.
- Tests: habituation curves and sensitization spikes.

### Stage 7 — Graph memory + decay/pruning

- Store nodes/edges; decay + prune keeps it sparse.
- Tests: edge count remains bounded over long runs.

### Stage 8 — Working memory (few expected concepts)

- K-slot working memory of expected prototype/concept references; cue-based
  loading; eviction by utility/recency/error. Physical object files begin only
  after Stage 9 tracking.
- Tests: WM capacity never exceeded; cueing reloads prior objects.
- Status: delivered (`memory/working_memory.py`). Because the object-file half of
  Stage 9 landed first, slots hold object-file references: an attention cue loads
  or refreshes the bound object, the lowest-utility slot (recency, evidence, last
  prediction error) is evicted when full, LOST objects release their slot and
  OCCLUDED ones stay in mind. Logged per frame as `working_memory`, replayable.
  Nothing reads it yet: top-down use of WM contents follows the clip validation.

### Stage 9 — Tracking + motion

- Conventional tracker first; then learn motion prototypes.
- Tests: reduced drift / fewer ID switches; motion class separation.

### Stage 10 — Spiking recurrence + RL label association

- Add SNN recurrence once non-spiking recurrence plateaus.
- Add reward-modulated learning for label/action association.
- Tests: reward improves stable associations without blowing up prototypes/graph.

