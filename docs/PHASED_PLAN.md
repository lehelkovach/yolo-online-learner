## Minimal staged plan (least dependency first)

Design goal: add **one mechanism at a time**, keep interfaces stable, and make each stage publishable via logged metrics.

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

### Stage 3 — Simple embeddings (no deep nets yet)

- Start with cheap embeddings (bbox geometry + basic crop statistics).
- Tests: stability of embedding distribution; bounded norms.

### Stage 4 — Prototype bank + novelty

- Online prototype matching + spawn-on-surprise + bounded counts.
- Tests: prototype count bounded; novelty spikes on distribution shift.

### Stage 5 — Dual processing predictive coding

- Top-down prototypes predict expected embedding; select winner(s) by min error.
- Tests: error decreases for repeated percepts; spikes for novel ones.

### Stage 6 — Habituation / sensitization (gain gating)

- Repeated low error reduces gain; surprise increases gain and learning.
- Tests: habituation curves and sensitization spikes.

### Stage 7 — Graph memory + decay/pruning

- Store nodes/edges; decay + prune keeps it sparse.
- Tests: edge count remains bounded over long runs.

### Stage 8 — Working memory (few active objects)

- K-slot working memory; cue-based loading; eviction by utility/recency/error.
- Tests: WM capacity never exceeded; cueing reloads prior objects.

### Stage 9 — Tracking + motion

- Conventional tracker first; then learn motion prototypes.
- Tests: reduced drift / fewer ID switches; motion class separation.

### Stage 10 — Spiking recurrence + RL label association

- Add SNN recurrence once non-spiking recurrence plateaus.
- Add reward-modulated learning for label/action association.
- Tests: reward improves stable associations without blowing up prototypes/graph.

