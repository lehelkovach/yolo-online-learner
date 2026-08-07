# YOPL cognitive architecture and experiment map

**Status:** design map, not an implementation claim  
**Updated:** 2026-08-07  
**Canonical build order:** `docs/PHASED_PLAN.md`

This document consolidates the YOPL ideas recovered from the repository and the
relevant project conversations. It separates the architecture we have committed
to build from hypotheses that still need experiments and from the longer-term
Helper-style robotics program. For the 2026-08-07 ST² / SNN A-B / attention-register
coding-agent directive, see
`docs/DEVELOPMENTAL_COGNITION_CODING_AGENT_HANDOFF.md` (does not replace this map
or PHASED_PLAN numbering).

The governing engineering rule is unchanged: **one mechanism per PR, with tests
and JSONL metrics**. An idea appearing here does not mean it exists on `main`.

## 1. Fixed architecture

The canonical information flow is:

```text
frozen YOLO -> BBP hypotheses -> sensory buffers
            -> WTA attention with inhibition-of-return
            -> one attended-crop embedding
            -> working memory: stable expected concepts
            -> bottom-up recognition + top-down prediction
            -> prediction error / surprise-gated plasticity
            -> bounded local prototype updates
            -> NetworkX percept graph (short-term relational memory)
            -> later, stable evidence-backed commits to KnowShowGo (long-term memory)
```

This is the completed runtime flow, not the implementation order. The staged
dependency order is defined below and in `docs/PHASED_PLAN.md`.

The system is deliberately hybrid:

- YOLO supplies frozen, fallible object proposals. A YOLO class is evidence, not
  a canonical label.
- BBPs are transient, time-indexed hypotheses. They are not objects or concepts.
- Sensory buffers are bounded, short-lived per-frame or ring buffers. They are
  distinct from the attended-percept register and from working memory.
- Attention admits exactly one BBP to the expensive learning path when candidates
  exist.
- Local perceptual memory remains online, sparse, plastic, and bounded.
- Working memory holds a few currently expected concepts; it is not the entire
  NetworkX graph and it is not KnowShowGo.
- The NetworkX percept graph is a short-term relational workspace. It is not the
  durable source of truth.
- KnowShowGo is outside the frame loop and later stores only mature, versioned,
  provenance-bearing exemplars, concepts, and procedures.

## 2. Identity and naming

YOPL must keep these levels distinct:

| Level | Meaning | Lifetime |
|---|---|---|
| BBP | YOLO's localized hypothesis | one frame |
| Attended embedding | geometry and appearance of one selected crop | one attention tick |
| Prototype | recurring perceptual pattern | local sessions, bounded and revisable |
| Object file | tracked physical individual through time/occlusion | track plus permanence TTL |
| Category | abstraction shared by multiple instances | learned and revisable |
| Name | human-provided alias bound to an identity | versioned symbolic binding |
| KSG concept/exemplar | durable declarative identity and evidence | cross-session LTM |

Names never replace UUIDs. Interactive teaching should bind an alias such as
`red mug` to a stable prototype, object file, or category UUID and record who
supplied it and when. A detector class may be offered as a suggestion, but a
human-confirmed name is the first authoritative label.

This reflects the broader programming/semantic idea recovered from the sessions:
a name is a binding, while the referred-to object retains its own identity,
state, type, and history.

## 3. Attention, expectation, and a stream of cognition

The current attention scheduler is the first operational attention register. It
creates a serial stream on top of parallel BBPs:

1. Sensory buffers contain all current BBPs.
2. A priority calculation produces one WTA winner.
3. Inhibition-of-return prevents a single location from monopolizing the stream.
4. The winner enters the attended-percept register and receives the embedding and
   learning budget.
5. Later, working-memory contents generate top-down expected embeddings.
6. Mismatch creates prediction error, which redirects attention and gates
   bottom-up plasticity.

Stage 2 uses confidence and area only as temporary priority **proxies**. They must
not be reported as learned novelty, true prediction error, or actual motion.
Stage 4 replaces the novelty proxy, Stage 5 replaces the prediction-error proxy,
and actual motion remains unavailable until temporal state/tracking in Stage 9.

The prior stream-of-consciousness and global-workspace ideas are retained as an
experimental interpretation, not a claim of machine consciousness. A future
register schema may expose sensory candidates, the current attended percept,
working-memory expectations, surprise, selected goal, and motor intention. The
first scientific question is whether this sparse broadcast improves learning,
recall, and control compared with non-serial baselines.

## 4. Online plasticity without learning explosion

The initial prototype learner should favor simple, inspectable stability rules:

- L2-normalize embeddings and prototype directions.
- Use winner-only Oja or normalized running-mean updates.
- In the Stage 4 baseline, gate updates on valid crops, attention, and match
  quality. Prediction-error gating belongs to Stage 5.
- Use a bounded, declining learning rate as evidence accumulates.
- Decay separate strength, confidence, and utility values; do not decay a unit
  prototype vector and immediately renormalize it, which cancels directional
  decay.
- Bound the bank with `Kmax`.
- Require novelty hysteresis and a spawn cooldown before creating a prototype.
- Assert finite values, dimensional consistency, bounded counts, and expected
  norms at every update boundary.

Later hardening mechanisms must each remain separately testable: quarantine
uncertain observations, merge near-duplicates, evict weak/stale/low-utility
prototypes, keep a bounded exemplar reservoir, and slow or lock user-confirmed
prototypes while retaining a correction path. Explicit loser anti-Hebbian
updates are also deferred. If WTA, bounded capacity, merging, and utility decay
do not prevent collapse, decorrelation can be added as its own ablation and PR.

## 5. Developmental curriculum

The developmental theme is implemented as gated capabilities and plasticity, not
as a claim that the software reproduces child or animal development.

| Curriculum gate | Capability and experiment | Dependency |
|---|---|---|
| Sensorimotor input | frozen YOLO produces stable BBPs and JSONL | Stages 0-1 |
| Selective attention | serial WTA stream plus IOR | Stage 2 |
| Percept encoding | cheap attended-crop embeddings | Stage 3 |
| Familiarity | bounded match/spawn prototype memory | Stage 4 |
| Expectation | prototypes predict embeddings; error gates learning | Stage 5 |
| Stable focus | habituation/sensitization gain tests | Stage 6, later |
| Relational STM | sparse percept graph with decay/pruning | Stage 7, later |
| Expected concepts | K-slot working memory | Stage 8 |
| Thin durable visual memory | match/generalize/exemplar writer using mature Stage 4 prototypes | after Stage 8 |
| Object permanence | tracking, object files, occlusion prediction/rebinding | Stage 9 |
| Category learning | prototypes over multiple stable instances | after tracking |
| Event concepts | trajectory and relation templates such as approach/enter | after tracking |
| Social learning | teacher observation, correction, imitation | after stable objects/events |
| Rich durable teaching | tracked identities, categories, events, and procedures in KSG | after their local stages |

Plasticity may start high at a new curriculum gate and slow when representations
stabilize. Changes in learning rate must be driven by logged evidence such as
observation count, dispersion, prediction error, and correction history—not by a
hard-coded story about chronological age. Teacher feedback may reopen plasticity
for a corrected concept.

## 6. Research hypotheses retained for later experiments

### 6.1 Dynamic fovea and exploratory jitter

The proposed eye-like sampler uses a dense central attention region, lower-density
peripheral sampling, and a movable iris radius. Top-down expectation holds the
fovea on a predicted target; peripheral surprise can trigger a saccade. Small,
seeded spatial jitter can deliberately sample nearby pixels and viewpoints.

This should be tested only after the baseline crop embedding and predictive loop
are stable. Required ablations are:

- uniform crop versus foveated sampling;
- no jitter versus seeded jitter;
- bottom-up salience only versus expectation-plus-surprise saccades;
- accuracy/stability benefit versus added compute and representation noise.

Jitter must be reproducible from the session seed. It must never be hidden inside
the Stage 3 embedding baseline.

### 6.2 Object permanence and violation of expectation

An object file should survive brief disappearance, propagate a predicted state,
increase uncertainty while hidden, and rebind a compatible detection after
occlusion. A violation-of-expectation harness should measure reacquisition rate,
location error, ID switches, and surprise when expected reappearance fails.

Tracking identity and perceptual category are separate. A red mug and a blue mug
may share a category while retaining different object-file UUIDs.

### 6.3 Motion to event and verb-like concepts

After stable tracks exist, sliding windows can encode displacement, speed,
heading, acceleration, curvature, and pairwise relations such as distance change,
overlap, and containment. Bounded online clustering can form unsupervised event
templates. Human terms such as `approach`, `wipe`, `enter`, or `spill` are later
aliases bound to demonstrated templates, not labels inferred from one frame.

### 6.4 Social and imitation learning

The mirror-neuron analogy becomes an engineering hypothesis: observation and
execution should share task-relative representations of objects, relations,
goals, and action primitives. A teacher demonstration is transformed into an
inspectable skill trace:

```text
observe teacher -> track objects/hands -> segment changes -> infer subgoals
                -> map to safe robot primitives -> simulate/replay
                -> human correction -> versioned procedure evidence
```

The system should learn `move cloth across dirty surface until clean`, not copy
human joint angles. Teleoperation, dynamic movement primitives, or an existing
robot policy can supply motor primitives. YOPL supplies perception, expectation,
and high-level intent.

### 6.5 Mess and surface-state perception

A mess, spill, or food stain is usually a state/region of a surface rather than a
standalone YOLO object. The later cleaning experiment therefore needs
segmentation or surface-anomaly evidence, before/after state comparison, and a
confidence-aware `clean enough` stopping rule. A learned bounding box may localize
the region, but the box alone is not the concept.

## 7. KnowShowGo boundary: individual plasticity plus shared memory

The proposed hive-mind service is best represented as a versioned distribution
layer with local overlays:

- Each agent keeps private, fast, plastic prototypes and working memory.
- KSG stores mature declarative identities, confirmed aliases, exemplar
  relations, provenance, corrections, and procedures.
- A group may publish validated prototype priors or cognitive-policy recipes with
  compatibility metadata and evaluation results.
- Other agents may opt in, download a version, evaluate it locally, and retain
  individual or group-specific adaptations as overlays.
- Shared winners never silently overwrite local experience. Promotion requires
  reproducible evidence, validation policy, versioning, rollback, and trust scope.

This gives a concrete meaning to “genetic winners” and “memetic plasticity”:
the inherited artifact is a tested prior or recipe; local learning remains
plastic; accepted corrections can later compete for a new shared version.

KSG is never part of the frame, attention, motor-control, or safety loop. Images
are stored externally and referenced by URI plus content hash. Visual embeddings
use an immutable embedding-space identifier and must never be averaged or matched
with Gemini/text embeddings.

The thin writer after Stage 8 must use the real `knowshowgo-client` with
client-supplied visual embeddings through the verified `/api2.0` prototype
match, generalize, and exemplar operations. Contract tests must confirm the
client and server agree. The writer should be asynchronous, idempotent,
retryable, version-checked, and private by default. It may commit only after:

- the crop and embedding pass quality checks;
- the prototype is mature across repeated observations, preferably across tracks
  and sessions;
- dispersion is low and no merge/split decision is pending;
- any persisted name is user-confirmed;
- provenance, ownership, visibility, consent, and embedding-space metadata exist.

Immediate read-after-write, process-restart, and cross-day recall are integration
and rollout acceptance tests for the writer. They are not prerequisites that an
individual record can satisfy before its first commit. No text/Gemini vector may
enter this visual match/generalization path.

LLMs or curator agents may propose durable graph changes. Deterministic validators,
publication policy, and where needed a human decide whether those proposals become
canonical KSG revisions.

## 8. Helper-style robot boundary

The robotics program starts in simulation and preserves a hard safety boundary:

```text
YOPL perception / WM / intent
        -> task planner and approved skill primitives
        -> deterministic safety supervisor
        -> conventional robot controller
        -> simulated robot, then limited real hardware
```

Online perceptual learning never emits raw torques. Emergency stop, speed/force
limits, workspace constraints, collision checks, watchdogs, and human approval
remain below the cognitive architecture. Learned procedures may select only
allowlisted, parameter-bounded primitives. During a trial, online learning cannot
modify the safety supervisor, constraints, allowlist, or low-level controller.
KSG failure or network loss must not affect safe stopping.

The first useful robotics result is a tabletop cleaning demonstration in
simulation with a known tool, bounded surface, nonhazardous synthetic mess, and
human approval—not autonomous kitchen operation. Hardware begins only after the
simulation exit gates pass, with supervised low-speed commissioning and an
independently tested stop path.

## 9. Experiment and input ladders

Experiment IDs below are labels, not canonical stage numbers.

| Experiment | Primary metric | Exit gate before next dependency |
|---|---|---|
| EXP-ATTN | selection count, fixation run, IOR hits | exactly one winner when candidates exist |
| EXP-EMBED | dimension, norm, crop validity, repeated-crop drift | finite bounded deterministic vectors |
| EXP-PROTO | count, match/spawn, dispersion | bounded count; repeated inputs stabilize |
| EXP-PREDICT | error and learning gate over repetitions/shift | familiar error falls; shift error rises |
| EXP-WM | occupancy, eviction reason, cue hit | never exceeds K; deterministic eviction |
| EXP-KSG-THIN | match/create, restart/cross-day recall | stable UUID and visual embedding space survive reload |
| EXP-PERM | reacquisition, ID switches, position error | controlled occlusion beats no-memory baseline |
| EXP-CAT | assignment stability, novelty, optional purity | instances group without prototype explosion |
| EXP-EVENT | cluster repeatability, early prediction | templates transfer across objects/clips |
| EXP-FOVEA | stability and compute versus baseline | benefit survives seeded ablation |
| EXP-IMITATE | subgoal accuracy, intervention rate | safe replay succeeds in simulation |
| EXP-CLEAN | coverage, residue reduction, false-clean stop | bounded simulated task with safety supervisor |

Inputs advance separately from mechanisms:

```text
deterministic synthetic frames
  -> hashed recorded video or virtual-camera replay
  -> controlled webcam scene
  -> simulated robot camera
  -> supervised, low-speed hardware
```

Every experiment records the code revision, seed, detector/model version, input
URI/hash, configuration, platform, and metrics. Negative-result metrics,
provenance, and rejected prototype/KSG mappings are retained by default. Raw crops
or sensitive image content are retained only with consent and remain subject to
the applicable deletion policy.

## 10. Reconciled decisions and open questions

Decisions:

- Build Stages `2 -> 3 -> 4 -> 5 -> 8` in that order, then add the thin KSG
  prototype match/generalize/exemplar writer. Richer KSG schemas follow the local
  tracking, category, event, and procedure stages they describe.
- Tracking/object permanence follows a stable WM baseline rather than being
  smuggled into prototype learning.
- User labels are UUID aliases, not object identity.
- Decay targets confidence/strength/utility first; prototype direction uses
  normalized updates.
- Robotics reuses safe motor-control and policy stacks instead of placing motor
  learning inside YOPL.
- NetworkX remains an STM/workspace representation, not KSG LTM.

Explicit non-goals for this sequence are RedisGraph, YOLO fine-tuning, merging
with OSL Slack/job-application work, multimodal Gemini box generation, and a broad
Phase 1 refactor.

Open questions to settle with dedicated experiments or an architecture PR:

- `readme.md` calls the percept graph a DAG, while reciprocal prediction and
  transition relations may be cyclic. Do not silently change this invariant.
- The exact relationship among previously named YHGP, YOLO-POT-HG, BCF,
  PDS-ST2-GWB, CPMS, and OSL artifacts requires their source documents before
  their details can be treated as canonical.
- Whether foveated sampling improves this detector-based pipeline enough to
  justify its complexity is unknown.
- Whether event clusters support useful human verb bindings must be measured,
  not assumed.
- Shared KSG priors need compatibility, privacy, poisoning-resistance, and
  rollback policies before multi-agent distribution.

## 11. Provenance ledger

This consolidation distinguishes user-originated ideas from earlier assistant
recommendations.

| Source available during this audit | Contribution retained here |
|---|---|
| Current YOPL/Codex conversation | fixed architecture; stage order; BBP semantics; labels as aliases; Hebbian/Oja safeguards; webcam/video/robot path |
| “Cognitive Architecture for Robo Maid” | foveated iris, attention jitter, top-down saccades, developmental plasticity, social imitation, hive-mind sharing, module-by-module experiments |
| `YOLO_ONLINE_LEARNER_REVISED_ARCHITECTURE_RESEARCH_HANDOFF.md` (`085E5FC0…719`) | YHGP/YOLO-POT-HG, three rate lanes, PDS-ST2-GWB registers, BCF, category baselines, typed Hebbian graph, KSG consolidation, active perception, Helper-lite gates |
| “KSG Vision and Potential” | KSG as stable semantic/procedural memory and model-independent agent substrate |
| “KnowShowGo and Foundation Models” | evidence/provenance/versioning, curator proposal boundary, deterministic ingestion first, durable model-independent LTM |
| “Python Learning Approach” | names as bindings distinct from referent identity |
| “Tech Design Doc Draft” | consolidate with provenance; trim obsolete ideas; integrate against the real public client API |
| “YOLO-Online-Learner Status” | implementation-status evidence only; no new user mechanism |
| “Dog vs Horse Intelligence” | reviewed; no user-originated YOPL mechanism |
| Repository canon and legacy research notes | hierarchy, dual processing, Hebbian graph, habituation/sensitization, object permanence, motion/events, optional SNN/STDP |

The Codex app exposed the 50 most recent non-pinned sessions plus pinned sessions
for the initial audit. The dedicated Robo Maid session reported a larger handoff
covering YHGP, YOLO-POT-HG, BCF, PDS-ST2-GWB, attention registers, SNN/STDP, CPMS,
OSL, and KSG. The user subsequently supplied that source artifact; it was read in
full and reconciled in `docs/RESEARCH_HANDOFF_RECONCILIATION.md`. Its repository
state and milestone order are historical, so they do not silently override the
current canonical stage plan.

