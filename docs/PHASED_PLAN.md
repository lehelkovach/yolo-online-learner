## Consolidated staged plan (canonical)

**Last updated:** 2026-01-17

Design goal: add **one mechanism at a time**, keep interfaces stable, and make each stage publishable via logged metrics. This file merges prior plans from other branches (attention + episodic object permanence) into a single stage order. Update this file instead of creating new plan docs.

### Current state

- **Stage 0 — Experiment harness (reproducibility):** Done
- **Stage 1 — BBP generator (YOLO front-end):** Done
- **Stage 2 — Attention scheduler (serial conscious stream):** Done (WTA + inhibition-of-return)

### Next stages (priority order)

1. **Stage 3 — Object permanence (tracking/world model)**
   - Kalman + IoU/Hungarian association, ghost buffer on occlusion.
   - Tests: ID stability across occlusion; ghost TTL bounded.
2. **Stage 4 — Simple embeddings (no deep nets yet)**
   - Geometry + crop statistics embeddings.
   - Tests: bounded norms; distribution stability.
3. **Stage 5 — Prototype bank + novelty**
   - Online matching + spawn-on-surprise + bounded counts.
   - Tests: prototype count bounded; novelty spikes on shift.
4. **Stage 6 — Dual processing predictive coding**
   - Top-down prediction + error-minimizing selection.
   - Tests: error decreases for repeats; spikes for novel.
5. **Stage 7 — Habituation / sensitization (gain gating)**
   - Repeated low error reduces gain; surprise increases gain.
   - Tests: habituation curves; sensitization spikes.
6. **Stage 8 — Graph memory + decay/pruning**
   - Store nodes/edges; decay + prune for sparsity.
   - Tests: edge count bounded over long runs.
7. **Stage 9 — Working memory (few active objects)**
   - K-slot WM; cue-based loading; eviction by utility/recency/error.
   - Tests: WM capacity never exceeded; cueing reloads prior objects.
8. **Stage 10 — Motion prototypes**
   - Cluster motion vectors; associate to objects.
   - Tests: motion classes separate; generalize across objects.
9. **Stage 11 — Recurrence + RL association (optional)**
   - Add recurrence only if non-spiking baselines plateau.
   - Tests: reward improves stable associations without instability.

### Release and merge criteria

- Each stage must include: a unit test or integration test, JSONL metrics, and a minimal runnable example in `experiments/` or `scripts/`.
- Merge to `main` only after tests pass and docs updated.
- Tag stage completions as `v0.<stage>.0` (pre-1.0), patch fixes as `v0.<stage>.1+`.

