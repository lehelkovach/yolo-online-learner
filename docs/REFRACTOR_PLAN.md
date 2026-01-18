## YOLO Hebbian Graph Perceptual Learner

Tiered refactor document for staged development

This doc is the single "north star" spec for refactoring the codebase into a
stage-gated, tiered online perceptual learning system: start with the smallest
reliable core (tracking + prediction + boundary belief), then progressively add
Hebbian graph memory, spatiotemporal hierarchy with feedback, and eventually
infant-like developmental scaffolding.

---

### 0) What this system is modeling

This system is not "a new YOLO." It is:

- YOLO (per frame): proposes object hypotheses (bboxes + confidence + type distribution).
- Perception over time (core): turns per-frame hypotheses into persistent tokens with
  prediction, boundary learning, and selective consolidation.
- Hebbian graph (memory): links tokens <-> features <-> events with fast local updates and
  cheap decay.
- Hierarchical spatiotemporal cascade (later): builds a cognitive architecture where
  higher levels predict lower levels and temporal structure is encoded via recurrence/spiking.

A short "branding" name that matches the model:

YOLO-POT-HG = You Only Look Once (per frame), Perceive Over Time, with Hebbian Graph.

---

### 1) Hard constraints (design principles)

1. Online: no batch retraining loops required for core learning.
2. Stage-gated: every feature is activated by --stage=N config.
3. Sparse-first: computation scales with active tokens/features, not total graph size.
4. Local learning: Hebbian/Oja/STDP-style updates are local; no global backprop dependency.
5. Deterministic core loop: real-time loop never stalls; heavy maintenance runs asynchronously.
6. Measured progress: each stage has acceptance metrics + regression clips.

---

### 2) System contract (data model)

#### 2.1 TrackToken (object file-lite)

Represents "one thing over time," regardless of class semantics.

- id
- state: bbox center/size + velocities (Kalman state)
- cov: uncertainty
- bbox: current output bbox (derived from state + boundary belief)
- boundary_belief (BCF): a small local grid/mask around predicted bbox
- confidence: scalar
- age, miss_count, last_seen
- type_hypothesis: optional distribution (from YOLO, not ground truth)
- appearance: optional embedding (only if you add it later)
- graph_node_id: pointer into Hebbian graph memory

#### 2.2 AttendedToken (attention register)

Each tick, pick one TrackToken to receive:

- higher boundary learning gain
- stronger graph updates
- prioritized logging/analysis

This is the "instruction pointer" analog in minimal form.

#### 2.3 Hebbian Graph Memory (online association store)

Nodes:

- TokenNode (TrackTokens)
- FeatureNode (shape primitive hits, motion signatures, boundary patterns,
  optional YOLO type bins)
- EventNode (optional later: occlusion, near-collision, "surprise boundary")

Edges (weighted):

- Token <-> Feature
- Token <-> Token (interaction/proximity/occlusion)
- Token <-> Event

All updates are sparse and local.

---

### 3) Core pipeline (per frame tick)

This is the stable backbone across all stages:

1. Measurements
   - YOLO detections (and/or primitive detectors early)
2. Predict
   - Kalman predicts each TrackToken state and uncertainty
3. Associate
   - Hungarian assignment on IoU between predicted boxes and detections
   - gating threshold prevents nonsense matches
4. Update
   - matched tracks: Kalman measurement update
   - unmatched tracks: increase uncertainty, increment miss count
5. Boundary belief
   - update BCF (strengthen where confirmed; decay elsewhere)
6. Attention selection
   - choose AttendedToken by priority score
7. Graph update (if enabled)
   - activate Token/Feature nodes for this tick
   - apply Hebbian updates on active edges + lazy decay
8. Output
   - stable tokens, predicted next states, "highlighted" attended token

---

### 4) Stage-gated development plan (baseline -> cognitive architecture)

#### Stage 0 -- Geometric prior curriculum (optional starter mode)

Goal: stable tracking on "drone survival geometry" before semantics.
Enabled detectors:

- closed convex shapes (circle/ellipse, rectangle)
- closure/edge consistency constraints

Outputs are still bboxes + boundary evidence.

Accept: stable detection/tracking in synthetic scenes with moving circles/rectangles.

#### Stage 1 -- Baseline tracking-by-detection (minimal viable perception)

Goal: persistent tokens + prediction.
Implement:

- Kalman prediction
- IoU cost matrix
- Hungarian assignment
- track lifecycle (init/confirm/lost/delete)

Accept: survives short dropouts, low fragmentation.

#### Stage 2 -- Boundary Confidence Field (BCF) online learning

Goal: "perceive over time" beyond YOLO: boundaries stabilize and persist.
Per token:

- keep a small BCF grid centered on predicted bbox
- update rule: strengthen with evidence; decay otherwise
- AttendedToken gets higher gain / lower decay

Evidence sources (start cheap):

- bbox edges / detected contour
- gradient magnitude around predicted boundary

Optional later:

- sparse optical flow around boundary (only if needed)

Accept: bbox jitter decreases over time; re-acquisition improves after short occlusions.

#### Stage 3 -- Hebbian Graph Memory (online associations, no backprop)

Goal: build memory linking tokens <-> features/events.
On each tick:

- activate nodes for current tokens + features
- update only edges in active subgraph (sparse)

Use the graph to:

- stabilize type hypotheses (consistency over time)
- bias attention (recently reinforced features raise priority)
- support re-identification (lightweight memory before adding embeddings)

Accept: graph converges for stable tokens; improves continuity under repeated exposures.

#### Stage 4 -- Robustness upgrade (choose exactly one based on failure mode)

Only add what solves the observed dominant failure:

- If ID switches dominate -> add appearance metric association (DeepSORT-like)
- If fragmentation from low confidence dominates -> add low-score association logic (ByteTrack-like)
- If camera motion dominates -> add sparse flow as extra association/boundary evidence

Accept: reduce the dominant failure metric without regressing earlier stages.

---

### 5) Fast parallel Hebbian updates + cheap decay (no global sweeps)

#### 5.1 Active-set update (sparse propagation)

Per tick you only update:

- edges touching the active nodes (tokens + features + events)
- optionally a 1-hop neighborhood for limited message passing

This keeps computation bounded by K active nodes, not total graph size.

#### 5.2 Efficient weight decay (lazy decay)

Do not "decay the whole graph" every tick.
Store per edge:

- w_raw
- t_last

When an edge is touched:

1. apply w = w_raw * exp(-lambda * (t - t_last))
2. apply Hebbian update (or Oja-stabilized)
3. set t_last = t, store new w_raw

#### 5.3 Maintenance lane (separate process/thread)

Run asynchronously:

- prune edges below epsilon
- enforce top-k outgoing edges per node
- rebuild adjacency structures (CSR/edge lists)
- occasional normalization

This keeps the real-time loop deterministic.

---

### 6) Hierarchical cascading DAG with feedback + temporal learning (later branch)

This is how you evolve from "tracking system" to "cognitive architecture."

#### 6.1 Spatial DAG + recurrent time

Within a single timestep, compute is staged like a DAG:

- bottom-up pass produces representations
- top-down pass produces predictions
- errors adjust state (optional iterative refinement)

Across time, each layer maintains state -> recurrence emerges when unrolled.

#### 6.2 Layer stack (pragmatic)

L0: measurements (YOLO boxes, primitive evidence, optional flow)  
L1: spatiotemporal feature field (can be recurrent/spiking)  
L2: tokens + BCF (object files + boundary belief)  
L3: Hebbian graph memory (Token/Feature/Event associations)

Feedback:

- L3 -> L2 biases attention and type stability
- L2 -> L1 predicts expected local boundary/motion evidence ("what should I see next?")
- L1 -> L0 gates evidence (optional)

#### 6.3 Where spiking fits (lowest-friction path)

Don't "spike YOLO." Instead:

- keep YOLO as the front-end sensor
- implement spiking/recurrent dynamics in L1 (temporal evidence encoding)
- keep token + graph layers sparse and event-driven naturally

Start non-spiking (leaky integrators), then optionally swap L1 for a spiking/reservoir module.

---

### 7) Developmental branching toward infant-like learning

This is the roadmap from "object tracking" -> "infant-like perceptual development."
Each branch attaches to the same token + BCF + graph spine.

Branch A -- Object permanence & occlusion bridging

- keep tokens alive through occlusion using prediction + continuity cues
- add rules for "unity" (common motion/continuity > appearance early)

Branch B -- Statistical learning over stable tokens

- once tokens are stable, learn feature co-occurrence and prototypes
- staged plasticity schedule:
  - early: high plasticity at low-level boundary features
  - later: freeze low-level; learn higher abstractions

Branch C -- Serial attention / binding constraints

- enforce AttendedToken bottleneck for consolidation
- add a binding pool linking token <-> type hypothesis <-> feature clusters
- measurable constraints: only a few tokens can be consolidated per window

Branch D -- Temporal organization & event boundaries

- track a context vector over time
- detect "event boundaries" via prediction error spikes
- gate memory writes and learning rates at boundaries

Branch E -- Visual routines / relational operators

- routines operating on BCF/masks:
  - trace contour, closure, inside/outside, near/far, between, containment
- bridges perception to proto-symbolic structure without full scene graphs

Branch F -- Agency/survival templates (separate branch; optional)

- looming/TTC, reafference cancellation, small-target salience
- consumes TrackTokens; does not replace the core

---

### 8) Iteration protocol (how to refactor safely)

The golden loop:

1. implement smallest viable change
2. run benchmark suite
3. inspect failure clips + token logs
4. add exactly one improvement
5. rerun + record delta

Regression kit:

- 20-50 "golden clips" per stage (synthetic + real)
- stage metrics thresholds must not regress

Always log:

- association decisions (IoU, matched/unmatched)
- Kalman state + covariance
- BCF sharpness/area/jitter metrics
- AttendedToken id + priority score
- graph update deltas + pruning stats (if enabled)

---

### 9) Repo structure (stage-friendly)

- yolo_hgp/core/ (dataclasses, configs)
- yolo_hgp/detectors/ (yolo.py, primitives.py)
- yolo_hgp/tracking/ (kalman.py, association.py, lifecycle.py)
- yolo_hgp/learning/bcf/ (bcf.py, evidence.py)
- yolo_hgp/attention/ (priority.py, register.py)
- yolo_hgp/memory/hebb_graph/ (graph.py, updates.py, decay.py, maintenance.py)
- yolo_hgp/hierarchy/ (layer_l1.py, feedback.py, spiking_optional.py)
- yolo_hgp/benchmarks/ (synthetic generators + eval harness)
- docs/ (this doc + stage checklists)

---

### 10) Minimal "stage checklists" (what must be true to move on)

Advance from Stage 1 -> 2 when:

- tracking survives dropouts
- prediction uncertainty behaves sensibly
- low fragmentation on baseline clips

Advance from Stage 2 -> 3 when:

- BCF reduces jitter
- occlusion re-acquisition improves measurably

Advance from Stage 3 -> 4 when:

- graph improves attention stability or re-ID in repeated exposures
- maintenance lane keeps graph bounded (top-k + pruning)

Only begin hierarchy/spiking branch when:

- the Stage 2/3 core is stable and fast (no architectural churn needed).
