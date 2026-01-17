## Consolidated staged plan (canonical)

**Last updated:** 2026-01-17

Design goal: add **one mechanism at a time**, keep interfaces stable, and make each stage publishable via logged metrics. This file merges prior plans from other branches (attention + episodic object permanence) into a single stage order. Update this file instead of creating new plan docs. Long-range ideas that are not yet committed to the core path live in `docs/FUTURE_IDEAS.md`.

### Current state

- **Stage 0 — Experiment harness (reproducibility):** Done
- **Stage 1 — BBP generator (YOLO front-end):** Done
- **Stage 2 — Attention scheduler (serial conscious stream):** Done (WTA + inhibition-of-return)
- **Stage 3 — Object permanence (tracking/world model):** Done (IoU tracker + ghost buffer)
- **Stage 4 — Sensory buffer + foveated sampling + gaze control:** Done (retina + gaze + buffer)
- **Stage 5 — Sparse embeddings + WTA lateral inhibition:** Done (WTA + Hebbian/decay)

### Core stages (pragmatic, iterative)

1. **Stage 4 — Sensory buffer + foveated sampling + gaze control**
   - Add sensory buffer; Gaussian fovea sampling; gaze center from attention/tracking.
   - Optional jitter and saccade policy behind flags.
   - Tests: crop bounds; deterministic sampling; gaze target updates.
2. **Stage 5 — Sparse embeddings + WTA lateral inhibition**
   - Use fovea crop + bbox geometry; WTA selects winners; Hebbian + anti-Hebbian decay.
   - Tests: sparsity target met; bounded weights; no collapse.
3. **Stage 6 — Object tokens + prototype bank + novelty**
   - Tokenize with `track_uuid`; online prototype matching; spawn-on-surprise.
   - Tests: stable token IDs across occlusion; prototype count bounded; novelty spikes on shift.
4. **Stage 7 — Predictive coding + error-gated attention**
   - Top-down prediction; attention dwell vs saccade by error.
   - Tests: error decreases for repeats; spikes for novel.
5. **Stage 8 — Habituation/sensitization with recovery**
   - Gain control for repeated low error; recovery after absence.
   - Tests: habituation curves; recovery; sensitization spikes.
6. **Stage 9 — Graph memory + category formation**
   - Persist token/prototype/part nodes; decay + prune for sparsity.
   - Tests: edge count bounded; category metrics stable.

### Optional stages (post-core)

7. **Stage 10 — Working memory + recurrence/attractors**
   - K-slot WM; recurrent attention register; attractor dynamics for focus.
   - Tests: WM capacity bounded; convergence under stable input.
8. **Stage 11 — Motion prototypes + temporal encoding**
   - Cluster motion vectors; optional spike-train encoding.
   - Tests: motion class separation; stable spike rate mapping.
9. **Stage 12 — Declarative tags + associative recall**
   - Language tags and recall links (graph or external memory store).
   - Tests: association precision/recall; stability under decay.

### Release and merge criteria

- Each stage must include: a unit test or integration test, JSONL metrics, and a minimal runnable example in `experiments/` or `scripts/`.
- Merge to `main` only after tests pass and docs updated.
- Tag stage completions as `v0.<stage>.0` (pre-1.0), patch fixes as `v0.<stage>.1+`.

