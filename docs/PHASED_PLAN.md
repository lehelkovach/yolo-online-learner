## Consolidated staged plan (canonical)

**Last updated:** 2026-01-17

Design goal: add **one mechanism at a time**, keep interfaces stable, and make each stage publishable via logged metrics. This file merges prior plans into a single stage order that aligns with the Triune architecture. Update this file instead of creating new plan docs. Long-range ideas that are not yet committed to the core path live in `docs/FUTURE_IDEAS.md`.

### Scope and alignment

- Execution track: this file defines the stage-gated implementation order.
- Design context: `docs/REFRACTOR_PLAN.md` and `docs/TRIUNE_ARCHITECTURE.md`.
- Stage numbering is refactored to cover the full Triune scope while preserving
  completed milestones.

### Current state

- **Stage 0 — Experiment harness (reproducibility):** Done
- **Stage 1 — BBP generator (YOLO front-end):** Done
- **Stage 2 — Attention scheduler (serial conscious stream):** Done (WTA + inhibition-of-return)
- **Stage 3 — Object permanence (tracking/world model):** Done (IoU tracker + ghost buffer)
- **Stage 4 — Sensory buffer + foveated sampling + gaze control:** Done (retina + gaze + buffer)
- **Stage 5 — Sparse embeddings + WTA lateral inhibition:** Done (WTA + Hebbian/decay)

### Core stages (pragmatic, iterative)

1. **Stage 4 — Sensory buffer + foveated sampling + gaze control** (done)
   - Add sensory buffer; Gaussian fovea sampling; gaze center from attention/tracking.
   - Optional jitter and saccade policy behind flags.
   - Tests: crop bounds; deterministic sampling; gaze target updates.
2. **Stage 5 — Sparse embeddings + WTA lateral inhibition** (done)
   - Use fovea crop + bbox geometry; WTA selects winners; Hebbian + anti-Hebbian decay.
   - Tests: sparsity target met; bounded weights; no collapse.
3. **Stage 6 — Boundary Confidence Field (BCF)**
   - Per-token boundary belief grid; strengthen on evidence, decay elsewhere.
   - Tests: bbox jitter decreases; re-acquisition improves after occlusion.
4. **Stage 7 — Hebbian graph memory (token/feature/event)**
   - Sparse updates + lazy decay; top-k pruning in maintenance lane.
   - Tests: graph size bounded; stability improves under repeats.
5. **Stage 8 — Object tokens + prototype bank + novelty**
   - Tokenize with `track_uuid`; online prototype matching; spawn-on-surprise.
   - Tests: stable token IDs across occlusion; prototype count bounded; novelty spikes.
6. **Stage 9 — Predictive coding + error-gated attention**
   - Top-down prediction; attention dwell vs saccade by error.
   - Tests: error decreases for repeats; spikes for novel.
7. **Stage 10 — Habituation/sensitization with recovery**
   - Gain control for repeated low error; recovery after absence.
   - Tests: habituation curves; recovery; sensitization spikes.
8. **Stage 11 — Category formation + declarative tagging**
   - Graph clustering of tokens/prototypes; optional label hooks.
   - Tests: category metrics stable; memory remains bounded.

### Optional stages (post-core)

9. **Stage 12 — Working memory + recurrence/attractors**
   - K-slot WM; recurrent attention register; attractor dynamics for focus.
   - Tests: WM capacity bounded; convergence under stable input.
10. **Stage 13 — Motion prototypes + temporal encoding**
   - Cluster motion vectors; optional spike-train encoding.
   - Tests: motion class separation; stable spike rate mapping.
11. **Stage 14 — RL / conditioning / mirroring**
   - Reward-modulated association, classical conditioning, imitation learning.
   - Tests: reward improves stability; associations persist with decay.
12. **Stage 15 — Robotics integration**
   - Eye control + appendage control loops tied to attention and tokens.
   - Tests: closed-loop stability; reproducible behaviors in sim.

### Release and merge criteria

- Each stage must include: a unit test or integration test, JSONL metrics, and a minimal runnable example in `experiments/` or `scripts/`.
- Merge to `main` only after tests pass and docs updated.
- Tag stage completions as `v0.<stage>.0` (pre-1.0), patch fixes as `v0.<stage>.1+`.

