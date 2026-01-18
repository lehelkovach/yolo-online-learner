## Staged and branch development plan

Purpose: distill the full roadmap into a staged, branch-ordered execution plan
with test coverage expectations and benchmarking guidance. This plan aligns
with `docs/PHASED_PLAN.md`, `docs/REFRACTOR_PLAN.md`, `docs/TRIUNE_ARCHITECTURE.md`,
and `docs/PROTOYOLO_SPEC.md`.

---

### 1) Branching model (dependency-ordered)

Use stacked branches that depend on the previous stage. Each stage has one PR.

- Base integration branch: `cursor/project-development-strategy-38c9`
- Branch naming: `stage-<N>-<feature>`
- Create each stage from the previous stage branch:
  - `stage-6-object-tokens`
  - `stage-7-predictive-coding`
  - `stage-8-habituation`
  - `stage-9-graph-memory`
  - `stage-10-working-memory` (optional)
  - `stage-11-motion` (optional)
  - `stage-12-language-tags` (optional)
  - `stage-13-rl-conditioning` (optional)
  - `stage-14-robotics` (optional)

Gate for merge to integration:

- unit/integration tests added
- JSONL metrics present
- minimal runnable example documented

---

### 2) Stage map (core path)

Stages 0–5 are complete in the integration branch. Stage numbers align with
`docs/PHASED_PLAN.md`.

#### Stage 6 — Object tokens + prototype bank + novelty

- Build token store keyed by `track_uuid`
- Prototype matching with bounded growth
- Novelty spikes on distribution shift

Tests:
- token ID stability across occlusion
- prototype count bounded
- novelty spikes under shift

Expected logs:
- token ID, prototype ID, novelty score

#### Stage 7 — Predictive coding + error-gated attention

- Top-down prediction for fovea embeddings
- Error gates attention dwell vs saccade

Tests:
- error decreases on repeats
- error spikes on change

Expected logs:
- prediction error per frame
- attention dwell vs saccade decisions

#### Stage 8 — Habituation/sensitization with recovery

- Gain control for repeated low error
- Recovery after absence

Tests:
- monotonic habituation curve
- recovery to baseline after silence

Expected logs:
- gain/novelty signal

#### Stage 9 — Graph memory + category formation

- Hebbian graph nodes for tokens/features/events
- Lazy decay and top-k pruning

Tests:
- edge count bounded
- category clustering stability

Expected logs:
- graph node/edge counts
- prune events

---

### 3) Optional stages (post-core)

#### Stage 10 — Working memory + recurrence/attractors

- K-slot WM for attention focus
- Recurrent attention register

Tests:
- WM capacity bounded
- convergence under stable input

#### Stage 11 — Motion prototypes + temporal encoding

- Motion vectors from tracking
- Optional spike-train encoding

Tests:
- motion class separation
- stable spike rate mapping

#### Stage 12 — Declarative tags + associative recall

- Language tags linked to tokens/prototypes
- Retrieval + decay

Tests:
- precision/recall under decay

#### Stage 13 — RL / conditioning / mirroring

- Reward-modulated association
- Classical conditioning
- Imitation policies

Tests:
- reward improves stability
- associations persist with decay

#### Stage 14 — Robotics integration

- Eye servo loop driven by gaze
- Appendage control loop driven by tokens

Tests:
- closed-loop stability in sim
- reproducible behaviors

---

### 4) Test coverage expectations (per stage)

Each stage must include:

- Unit tests for core invariants
- At least one integration test (synthetic or short run)
- JSONL schema fields for the stage

Minimum unit test categories:

- determinism under seed
- bounded memory growth
- stability (no collapse)

Integration test categories:

- synthetic clip or small recorded clip
- regression check against previous stage

---

### 5) Benchmarking and regression suite

Use a consistent benchmark kit for each stage:

- 20–50 "golden clips" (synthetic + real)
- baseline YOLO run vs online run on the same clips
- record metrics in JSONL for offline plots

Baseline comparisons:

- YOLO-only detections vs online tracks
- track continuity / ghost ratio
- attention dwell stability
- embedding norm distribution

Performance checks:

- frames per second (CPU)
- memory footprint (graph size bounded)

---

### 6) Commands (reference)

- Unit tests: `pytest`
- Lint: `ruff check .`
- Online run: `python experiments/run.py --source <VIDEO_OR_CAM> --max-frames 300 --output-dir outputs`
- Tracking overlay: `python scripts/run_tracker_stream.py --source <VIDEO_OR_CAM> --max-frames 300`

See `CLOUD.md` for cloud vs local environment notes.

---

### 7) Design references

- `docs/PHASED_PLAN.md`: canonical stage order
- `docs/REFRACTOR_PLAN.md`: core refactor spec
- `docs/TRIUNE_ARCHITECTURE.md`: Triune architecture blueprint
- `docs/PROTOYOLO_SPEC.md`: ProtoYolo phylogenetic design
