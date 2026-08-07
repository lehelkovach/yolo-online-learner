# Revised research handoff reconciliation

**Source reviewed:** `YOLO_ONLINE_LEARNER_REVISED_ARCHITECTURE_RESEARCH_HANDOFF.md`

**Source SHA-256:** `085E5FC078D902075D688F939DA11E82AB0D46636B38E175D3BC641F545CB719`

**Review date:** 2026-07-31

**Purpose:** preserve useful research directions without silently replacing the
canonical YOPL architecture or stage order.

The source handoff was read in full. It is a rich research directive covering
YHGP/YOLO-POT-HG, Triune/ProtoYolo, Boundary Confidence Fields, PDS-ST2-GWB
attention registers, online categories, typed Hebbian graphs, complementary
memory, KSG consolidation, active perception, and a Helper-lite robot path.

It is not the executable build plan. Repository state and stage numbering in the
source predate the merged Stage 2 scheduler and the current user handoff.

## 1. Canonical precedence

When documents disagree, use this order for implementation decisions:

1. The current explicit user handoff and non-goals.
2. `docs/PHASED_PLAN.md` for PR stage numbers and dependency order.
3. `docs/HANDOFF.md` for local reproducibility and PR requirements.
4. `docs/COGNITIVE_ARCHITECTURE_MAP.md` for the reconciled runtime design,
   research hypotheses, and longer experiment ladder.
5. This document for the mapping from the revised research handoff.
6. `docs/DEVELOPMENTAL_COGNITION_CODING_AGENT_HANDOFF.md` for the 2026-08-07
   ST² / SNN A-B / attention-register developmental program (research directive;
   does not reorder PHASED_PLAN).
7. The older conceptual phase outline in `readme.md` and legacy branches as
   research context.

The current implementation sequence remains (see `docs/PHASED_PLAN.md` for full
text; revised 2026-08-07):

```text
Stage 4 pattern prototypes
  -> Stage 5 prediction error
  -> Stage 5b attention-gated plasticity
  -> Stage 8 K-slot working memory
  -> thin KSG visual writer
  -> Stage 9 tracking / object files / permanence
  -> Stage 6 habituation / sensitization
  -> Stage 7 typed Hebbian graph
  -> Stage 9b instance / category prototypes
  -> Stage 10a ConvGRU reference
  -> Stage 10b SNN A/B (not SNN-only)
  -> events / active vision / event-camera ladder
```

Each mechanism still requires its own PR, tests, deterministic replay where
applicable, and JSONL metrics. Stages 6 and 7 stay numbered but follow Stage 9
in the default revised track so permanence lands before rich graph growth.

## 2. Repository-state correction

The source describes an older `main` at `b8dfbd7` and treats Stage 2 attention as
unmerged. Current audited state is:

- Stage 2 was squash-merged in PR #3 at `29e6d081`.
- The cognitive architecture and experiment map was squash-merged in PR #4 at
  `149a2168`.
- Stage 2 now has deterministic top-1 WTA, spatial inhibition-of-return, proxy
  metrics, empty-frame schema coverage, and JSONL integration.
- The Stage 2 confidence/area fields remain explicitly named proxies. Stage 4
  supplies real novelty, Stage 5 supplies real prediction error, and Stage 9
  supplies temporal motion.

The handoff's remote-branch inventory remains useful. The reported
`cursor/fix-cli-imports-c0bd` branch requires an immediate, narrow disposition:
reproduce direct-entrypoint imports, inspect its fix/tests, and land a separate
bugfix PR if the current README commands fail. Before Stage 9, create a separate
branch-reconciliation PR for the object-permanence, project-strategy, and
environment branches. The two alternative attention branches are reference
implementations now; they must not displace the tested Stage 2 baseline without
an explicit behavioral comparison.

## 3. Apparent conflict: track-first versus attention-first

The source recommends turning detector proposals into persistent `TrackToken`s
before attention and category learning. The current staged plan intentionally
builds a track-free perceptual baseline first.

Both ideas are retained by defining the scope precisely:

- Stages 2-5 operate on transient BBPs to establish attention, crop encoding,
  bounded pattern prototypes, and prediction-error behavior with minimal
  dependencies.
- A Stage 4 prototype is a local recurring **perceptual pattern**. It must not be
  called a persistent object, physical instance, or mature category.
- Stage 9 introduces `TrackToken`/object-file identity, uncertainty, occlusion,
  ghosts, and rebinding.
- Multi-view object categories and object permanence require Stage 9 tracks.
- A later ablation may compare BBP-first and track-first attention under equal
  compute. It does not belong in Stage 3.

This preserves the cheap falsifiable baseline while accepting the source's
scientific warning: category metrics are misleading when identity switches are
not measured separately.

## 4. Decisions adopted into the research program

### 4.1 Three rate-separated lanes

The mature system should expose three explicit lanes:

- **Fast safety/control:** emergency stop, limits, collision avoidance, servo
  control, and reflexes; independent of attention, language, and KSG.
- **Cognitive:** object state, attention, recognition/prediction, WM, action
  proposals, verification, and auditable commits.
- **Consolidation:** replay, graph maintenance, merge/split proposals, KSG
  writes, reporting, and model updates over immutable snapshots.

The handoff's suggested cycle rates are experimental configuration ranges, not
biological constants or current implementation targets.

### 4.2 Detector-neutral boundaries

YOLO remains frozen and is the current adapter. The long-term domain contract
should not import Ultralytics, ROS, KSG, or a robot policy. RT-DETR, motion/depth
proposals, masks, open-vocabulary queries, audio, touch, and proprioception are
possible later adapters.

Do not refactor Phase 1 merely to anticipate them. Introduce an adapter-neutral
observation envelope only when the first alternative backend or modality has a
tested use case. Audit model and deployment licensing before distribution.

### 4.3 Explicit registers and meaningful commits

The following source vocabulary is retained for experiments after WM exists:

| Register | Operational role |
|---|---|
| `SB` | bounded modality-specific sensory/ring buffers |
| `CP` | scored percept, recall, goal, outcome, action, language, and interrupt candidates |
| `WM` | bounded goals, bindings, expected concepts, and procedure state |
| `APR` | current attention pointer, gains, context, and tie-break state |
| `BP` | provisional token/type/role/relation bindings |
| `FTR` | exactly one foreground token or compact chunk plus operation |
| `GWB` | immutable foreground snapshot delivered to subscribers |
| `SoC` | append-only log of meaningful foreground commits |

`PDS-ST2-GWB` is a research label for precision-weighted difference selection,
serial type/token binding, and operational broadcast. It is not a consciousness
claim.

The attention score matures by stage rather than appearing in one patch:

| Priority evidence | First real source |
|---|---|
| observation quality | validated BBP/crop quality |
| novelty | Stage 4 prototype memory |
| prediction error | Stage 5 top-down expectation |
| habituation/sensitization | deferred Stage 6 |
| goal/context relevance | Stage 8 WM and later executive |
| motion and identity uncertainty | Stage 9 tracking |
| interrupts/outcomes | later broadcast/action loop |
| IOR and deterministic tie-breaking | Stage 2 baseline |

Log each available component, the stable-ID tie-break, and eventually the
winner/runner-up margin. Unavailable evidence remains explicitly absent rather
than replaced by an unlabeled proxy.

A `BroadcastCommit` should occur on a typed, calibrated state delta, stable
dwell, operation boundary, action outcome, interrupt, detected event boundary,
or explicit audit checkpoint. It must not rebrand every frame as a thought.

Action outcomes, recalled memories, goals, and language may re-enter the candidate
pool. That input-output recurrence must use the same versioned, replayable
selection contract as sensory candidates.

After the WM/binding/broadcast baseline exists, add a rapid-serial-visual
presentation/attentional-blink protocol as a falsification test of finite binding
and broadcast capacity. Do not claim an ST2-like mechanism merely because a
capacity-one register exists.

### 4.4 Category and correction baselines

After tracking, evaluate a prototype/exemplar hybrid against transparent
baselines:

- DP-means for threshold-created clusters;
- Fuzzy ART for vigilance and stability-plasticity;
- SUSTAIN-inspired surprise-driven recruitment;
- a maintained online prototype learner;
- prototype-only and exemplar-only variants.

The lifecycle to test is:

```text
unknown episode -> draft cluster -> stable perceptual category
                -> named concept candidate -> KSG concept -> superseded/retired
```

An unmatched observation should normally enter quarantine/unknown evidence
rather than immediately create a permanent category. Preserve positive evidence,
hard negatives, contrast sets, corrections, provenance, and supersession. The
source phrase “unga, not bunga” becomes a structured positive assertion plus a
rejected candidate and, where known, the discriminating evidence.

Post-tracking category experiments should retain multiple close hypotheses,
aggregate multi-view track summaries with uncertainty and outlier rejection,
scope reversible corrections to token/episode/future matches, and never reuse a
superseded category ID.

### 4.5 Typed bounded graph

The future NetworkX STM should have versioned node/edge types, confidence,
support count, timestamps, provenance, and update-rule versions. Candidate edges
include co-attention, prediction/precedence, spatial/part relations,
instance-of hypotheses, action-outcome, context activation, contradiction, and
recall.

Required stability techniques are bounded eligibility neighborhoods, lazy
timestamp decay, normalization/homeostasis, per-type top-k caps, asynchronous
maintenance, and logged graph growth/churn/latency. Co-occurrence alone is not a
passing result; the graph needs a predeclared prediction target and baselines.

### 4.6 Complementary memory and KSG receipts

Keep fast local observations/episodes separate from slow semantic/procedural
consolidation. The thin writer after Stage 8 remains narrower than the source's
full KSG milestone:

- submit mature visual prototype/exemplar evidence through the verified public
  client and `/api2.0` visual match/generalize/exemplar contract;
- require an immutable visual embedding-space ID and never mix text/Gemini
  vectors with visual vectors;
- write asynchronously and idempotently;
- retain the local ID when KSG rejects or conflicts;
- store receipt, durable ID, version, disposition, and conflict information;
- verify immediate, restart, and cross-day durability.

Tracked identities, signed evidence, category revisions, relations, events, and
procedures receive richer proposal schemas only after their local mechanisms are
tested.

### 4.7 Active perception and BCF are later ablations

The Boundary Confidence Field is preserved as a hypothesis after a conventional
tracker baseline. Compare it with exponential box smoothing and a maintained
mask tracker. Retain it only if it improves declared localization,
re-identification, containment, or occlusion endpoints.

The Stage 9 association contract may abstain and retain uncertainty rather than
force an ambiguous match. Log tentative/active/occluded/ghost/retired lifecycle
events plus identity switches, merge/split hypotheses, loss, and rebinding.
Predeclare HOTA/IDF1 or appropriate equivalents, ID-switch count, and occlusion
recovery as tracking endpoints.

The foveated/jitter idea grows into active perception only after the passive
predictive system is stable. Camera pan/tilt/zoom, closer inspection, lighting,
or base repositioning are action proposals subject to cost and safety. Compare
passive view, scripted gaze, seeded jitter, and learned information-gain policies.

### 4.8 Robot policy reuse

YOPL supplies persistent context, expectation, attention, high-level intent,
verification, and recovery. Existing ROS 2 control/planning and maintained robot
policies supply low-level execution and dexterity behind interfaces. Language
models may propose goals, labels, explanations, or recovery candidates but never
bypass allowlists, approval, or the safety supervisor.

An independent outcome verifier must judge task success; never trust the action
policy's self-report alone. Recording must be visible and controllable, stored
household data must have explicit retention/encryption/deletion policy, and
action logs should be tamper-evident. Biometric identity remains out of scope
without explicit consent, legal review, and a separate design.

The robot ladder begins with simulated `inspect -> pick/place -> verify ->
recover/ask` before any surface-cleaning experiment.

### 4.9 Triune, CPMS, and OSL boundaries

Triune/ProtoYolo is retained only as a switchable ablation curriculum:
reactive primitives -> predictive object cognition -> semantic/procedural
cognition. It is not a literal or current theory of brain evolution.

CPMS contributes a pattern—normalized observations, allowlisted deterministic
signals, versioned prototypes, explanations, and reversible corrections. CPMS
itself does not enter the perception loop. OSL is a possible later executive
above perception for goals, procedure choice, recovery, and information requests;
it remains separate from OSL Slack/job-application work and is not a dependency
of YOPL tests.

## 5. Stage crosswalk

| Canonical gate | Source idea used | What is explicitly deferred |
|---|---|---|
| Stage 3 embedding | geometry/color/texture baseline, versioned schema, cached model identity | DINOv2, adapted encoder, masks, tracking |
| Stage 4 prototypes | bounded match/spawn, vigilance comparison plan, pattern IDs distinct from names | final categories, merge/split automation, KSG |
| Stage 5 prediction | expected embedding and typed error, oddball/repetition protocols | temporal event graph, action outcomes |
| Stage 8 WM | expected prototype/concept refs, configurable capacity, utility/recency/error eviction, register-compatible context | physical object files and full GWB/SoC cycle |
| Thin KSG | async client contract, receipt, provenance, idempotency, visual space ID | relations, procedures, public/shared publication |
| Stage 9 tracking | `TrackToken`, uncertainty, lifecycle, ghosts, occlusion/rebinding | BCF until tracker baseline passes |
| Categories | prototype/exemplar hybrid; ART/SUSTAIN/DP-means comparisons | semantic labels defining clusters |
| Graph/events | typed edges, traces, lazy decay, prediction objective, delta commits | SNN/STDP until deterministic reference passes |
| Active vision | fovea/jitter, information-gain view actions | unsupervised physical exploration |
| Helper-lite | inspect-act-verify-recover/ask loop | autonomous occupied-home deployment |

## 6. Experimental discipline adopted

For each cumulative mechanism report its benefit and cost against the previous
baseline. Important future ablations include:

- track versus no track and tracker alternatives;
- box versus mask tokenization;
- hand-engineered versus frozen versus adapted features;
- category baselines listed above;
- all-token versus top-1 versus top-k versus matched-budget random learning;
- running mean versus Oja/BCM-style normalized updates;
- no/eager/lazy decay;
- positive-only versus typed/signed evidence;
- replay and broadcast policies;
- WM capacity one, four, and seven;
- disconnected, mock, and real KSG;
- passive, scripted, and learned active vision.

Predeclare the primary endpoint for a mechanism. Graph growth, appealing
visualizations, biological language, or a polished demo are not sufficient. Log
accuracy/stability, calibration, memory growth, latency, compute, interventions,
false success, and recovery as appropriate. Retain negative results.

Global experiment invariants:

- use monotonic timestamps for durations and wall-clock time only for provenance;
- key embedding caches by content hash, model/encoder ID, preprocessing, and
  version;
- provide a disable switch and a non-learning baseline for every learned
  mechanism;
- use stable tie-breakers, seeded randomness, versioned schemas, and hashed
  replay inputs.

Maintain a small hashed challenge-clip matrix spanning lighting, blur, camera
motion, temporary/prolonged occlusion, similar identities, appearance/role
changes, novelty among familiar items, and user contradiction/correction.

The input ladder remains:

```text
synthetic deterministic frames
  -> hashed recorded/virtual-camera replay
  -> controlled webcam scene
  -> simulated robot camera
  -> supervised low-speed tabletop hardware
  -> controlled-room mobile manipulation
```

## 7. Ideas not adopted as immediate requirements

- Tracker-first reordering does not replace the current Stage 2-5 BBP baseline.
- Frozen DINOv2 does not replace the mandated cheap Stage 3 vector.
- Branch reconciliation does not reopen or rewrite merged Stage 2.
- The BCF, GWB/SoC, typed graph, SNN/STDP, open-vocabulary detectors, VLA policy,
  and active camera do not enter Stage 3.
- A synthetic prediction-error heatmap is not an RGB detector input. It may later
  select an ROI whose original sensor crop is analyzed.
- Cognitive-cycle timing ranges are configuration hypotheses, not constants.
- “Triune” is an engineering curriculum label, not a current evolutionary-brain
  theory claim.
- No mechanism is described as consciousness, AGI, or original merely because
  it is biologically inspired.

## 8. Next actions

Related 2026-08-07 consolidation:
`docs/DEVELOPMENTAL_COGNITION_CODING_AGENT_HANDOFF.md` (SNN A/B after ConvGRU
reference; PDS–ST²–GWB register evolution; track/object files remain Stage 9).

1. Reproduce the direct CLI entrypoint issue named above and, if confirmed, land
   the narrow import fix as its own PR without reopening Phase 1. **Done** (PR #6).
2. Complete Stage 3 as the smallest deterministic embedding experiment. **Done**
   (PR #7).
3. In Stage 4, keep the first prototype bank transparent and bounded; specify
   ART/SUSTAIN/DP-means comparisons without combining them into the same PR.
   **In progress on main path:** `objects/prototype_bank.py` implements the
   bounded match/spawn baseline (`normalized_running_mean_v1`); ART/SUSTAIN/
   DP-means remain separate later experiments.
4. In Stage 5, run repetition and distribution-shift tests for genuine prediction
   error.
5. Add K-slot WM at Stage 8, then the thin visual KSG writer with durability tests.
6. Before Stage 9, audit and reconcile the remote object-permanence and
   project-strategy branches against the now-stable interfaces.
7. Add tracking/object files and controlled occlusion tests before true
   multi-view categories, event templates, and interactive instance naming.
8. Only after those gates, schedule BCF, broadcast/commit, active-vision,
   imitation, simulation, and Helper-lite work as separate experiments.

