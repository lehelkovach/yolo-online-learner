# YOLO Online Learner — Developmental Cognition Coding-Agent Handoff

**Research/design consolidation date:** 2026-08-07  
**Audience:** coding agent working on `yolo-online-learner`  
**Status:** implementation directive and experimental roadmap  
**Role:** extend and reconcile prior architecture docs; **do not** silently replace
canonical stage numbering or the active build order.

Companion sources consolidated here:

- SNN vs ConvGRU A/B program, STDP, latency coding, event-camera ladder
- Developmental cognition north star (object files, permanence, Hebbian hierarchy,
  attention-gated plasticity, KSG consolidation, active vision)
- PDS–ST²–GWB attention-register model and reference implementations
- Ablation matrix, metrics, package sketch, and immediate work order

---

## 0. Canonical precedence (read first)

When this document and other docs disagree, use this order for **implementation**:

1. Current explicit user handoff / non-goals in the live conversation.
2. [`docs/PHASED_PLAN.md`](PHASED_PLAN.md) — PR stage numbers and dependency order.
3. [`docs/HANDOFF.md`](HANDOFF.md) — local reproducibility and PR requirements.
4. [`docs/COGNITIVE_ARCHITECTURE_MAP.md`](COGNITIVE_ARCHITECTURE_MAP.md) — reconciled
   runtime design and experiment ladder.
5. [`docs/RESEARCH_HANDOFF_RECONCILIATION.md`](RESEARCH_HANDOFF_RECONCILIATION.md) —
   prior research-handoff crosswalk.
6. **This document** — developmental / ST² / SNN research program and code sketches.
7. Legacy notes in `readme.md` and unmerged remote branches — research context only.

### Active executable build order (do not reorder without an explicit user gate)

Source of truth: [`docs/PHASED_PLAN.md`](PHASED_PLAN.md) (revised 2026-08-07).

```text
Stage 4 pattern prototypes + novelty
  -> Stage 5 prediction error
  -> Stage 5b attention-gated plasticity ablations
  -> Stage 8 K-slot working memory
  -> thin KSG visual writer
  -> Stage 9 TrackToken / object files / permanence
  -> Stage 6 habituation / sensitization
  -> Stage 7 typed Hebbian graph (Hebb ⊥ anti-Hebb ⊥ decay ⊥ eligibility)
  -> Stage 9b instance then category prototypes
  -> Stage 10a ConvGRU temporal reference
  -> Stage 10b SNN A/B (rate → latency → STDP → +adapt/+attention)
  -> events / affordances / active vision / event-camera ladder
```

Stages **0–3** are done. Stage 2 may **evolve** toward PDS–ST²–GWB registers in
optional PRs after Stage 5; do not rewrite the tested WTA baseline wholesale.

**Critical reconciliation with older track-first / SNN-first instincts:**

| Instinct | Decision in PHASED_PLAN |
|---|---|
| Track / object files early | **Stage 9** after track-free Stage 2–5 baseline |
| Full PDS–ST²–GWB register stack next | Evolve **from** current `AttentionScheduler` after Stage 5 |
| Hebbian graph soon | **Stage 7** after permanence (default) |
| SNN as early branch | **Stage 10b** only after **10a** ConvGRU reference |
| Habituation first | **Stage 6** after Stage 9 (default); feeds SNN thresholds |

One mechanism per PR. Tests + JSONL metrics required. Aggressive ablation required.

---

## 1. Executive directive

Build a **modular developmental cognition research platform**, not a monolithic
“brain” and not “YOLO plus a graph.”

**Core research hypothesis:**

> Stable semantic knowledge emerges through progressively stabilized perceptual
> representations, guided by selective attention, temporal continuity, local
> associative learning, and hierarchical abstraction.

**Engineering rules:**

- Everything measurable; every module independently benchmarkable before retention.
- Prefer developmental emergence over hard-coded symbolic rules.
- Attention allocates scarce computation and plasticity.
- Preserve provenance through every representational layer.
- YOLO / RT-DETR / DINO-style models are **scaffolding**, teachers, proposal
  sources, and baselines — not the ontology.
- Track identity ≠ perceptual category ≠ episodic token ≠ semantic label ≠ KSG UUID.
- Low-level perception may be massively parallel; cognitively consequential
  selection is serialized through **one** focused token register.
- Safety / reflex / motor stabilization stays **outside** the serial cognitive
  bottleneck.
- Spiking networks are an **A/B track** after a deterministic non-spiking reference
  behavior exists — SNNs do **not** automatically “compress more data in time.”
- Long-term KnowShowGo consolidation only after stability / evidence thresholds.

---

## 2. Repository truth (audited 2026-08-07)

Do **not** trust older handoffs that freeze `main` at `b8dfbd7` or treat Stage 2 as
unmerged.

| Area | State on current path |
|---|---|
| Stage 0 harness | `experiments/run.py` JSONL + seed |
| Stage 1 BBP | `perception/*` YOLO → BBP |
| Stage 2 attention | `attention/scheduler.py` WTA + IOR; proxies named `*_proxy` |
| Stage 3 embeddings | `features/simple_embedding.py` `simple_crop_v1` |
| Stage 4 prototypes | `objects/prototype_bank.py` match/spawn + novelty (land via PR) |
| Preview / OBS | `experiments/preview.py`; [`docs/OBS_SETUP.md`](OBS_SETUP.md) |
| CI | `.github/workflows/ci.yml` pytest + ruff |
| Graph stub | `graph/percept_graph.py` (not Stage-7 complete) |
| Tracking package | empty / remote salvage only |

### Remote salvage map (not Stage-4/5 prerequisites)

| Branch | Content | Action |
|---|---|---|
| `origin/agent/*` (stage 2–3, preview, docs) | Squash-landed leftovers | Ignore / delete after merge |
| `origin/cursor/attention-scheduler-2b25` | Alternate WTA | Reference only |
| `origin/cursor/fix-cli-imports-c0bd` | Superseded by PR #6 | Ignore |
| `origin/cursor/episodic-object-permanence-2576` | Kalman tracker + tests | **Stage 9** candidate |
| `origin/cursor/episodic-object-permanence-56eb` | Plans only | Mine decisions |
| `origin/cursor/episodic-object-permanence-704e` | `OPUS_PLAN.md` | Mine roadmap |
| `origin/cursor/project-development-strategy-38c9` | ProtoYolo / Triune / world_model / retina | Selective mine; draft PR #2 |
| `origin/cursor/dev-env-setup-2b25` | `AGENTS.md` / fixtures | Optional hygiene |

Before Stage 9: produce `docs/BRANCH_RECONCILIATION.md` against **then-stable**
interfaces (post Stage 5/8/KSG as applicable).

---

## 3. Target system at a glance

```text
SENSORS / INTERNAL FEEDBACK
        |
        v
 Sensory ring buffers          short-lived modality windows
        |
        v
 Adaptive sensory layer        habituation / sensitization / prediction error
        |
        +-------------------------------+
        |                               |
        v                               v
 Developmental path              Pretrained scaffold
 sparse / recurrent /            YOLO / DINO / SAM2 etc.
 Hebbian hierarchy
        |                               |
        +---------------+---------------+
                        v
                 TEMPORAL BINDING
                        |
                 OBJECT FILES (Stage 9+)
                        |
                 candidate pool
                        |
                        v
              ATTENTION ARBITER / APR
                        |
                        v
         FOCUSED TOKEN REGISTER (capacity = 1)
                        |   ^
                        |   | recurrent maintenance / context
                        v   |
               FOCUSED PROCESSOR
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       learning       action        memory
          |                           |
          v                           v
    Hebbian graph              working / episodic
          |                           |
          +-------------+-------------+
                        v
                 CONSOLIDATION
                        |
                        v
                    KnowShowGo
```

Completed runtime flow ≠ implementation order. Implementation order is §0 / PHASED_PLAN.

---

## 4. Representational ladder and time scales

| Layer | Typical lifetime | Persistent ID? | Notes |
|---|---|---|---|
| Sensory sample / BBP | ms–frame | observation ID | Transient hypothesis |
| Attended embedding | one attention tick | no | Stage 3 vector |
| Pattern prototype | session-bounded | pattern ID | Stage 4; **not** an object |
| Object file / TrackToken | 100 ms–minutes | TrackToken UUID | Stage 9+ |
| Episodic token | seconds–long | episode UUID | Later |
| Instance prototype | minutes–days+ | instance UUID | After tracking |
| Perceptual category | episodes–LTM | category UUID | After multi-instance tracks |
| Semantic concept | long term | KSG UUID | Consolidation only |

Learning rates should fall with consistent evidence; novelty / prediction error may
temporarily raise plasticity. Do not invent stage durations as biological ages.

---

## 5. Core data contracts

Use immutable dataclasses (or schema-validated records) with `schema_version`.
Store refs to bulky frames/embeddings rather than duplicating blobs.

```python
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class Observation:
    schema_version: int
    observation_id: str
    timestamp_ns: int
    source_id: str
    modality: str
    frame_ref: Optional[str]
    region: Optional["Region"]
    embedding_ref: Optional[str]
    detector_hypotheses: Tuple["LabelScore", ...]
    quality: float
    uncertainty: float
    provenance: "Provenance"


@dataclass(frozen=True)
class TrackToken:
    schema_version: int
    token_id: str
    created_ns: int
    updated_ns: int
    kinematic_state: Tuple[float, ...]
    covariance: Tuple[float, ...]
    region: "Region"
    embedding_summary_ref: Optional[str]
    feature_summary: Tuple[float, ...]
    type_hypotheses: Tuple["TypeScore", ...]
    observation_refs: Tuple[str, ...]
    confidence: float
    age_frames: int
    miss_count: int
    lifecycle: str  # tentative | active | occluded | ghost | retired


@dataclass(frozen=True)
class CognitiveCandidate:
    candidate_id: str
    kind: str  # percept | memory | goal | action | language | interrupt
    source_ref: str
    feature_ref: Optional[str]
    type_hypotheses: Tuple["TypeScore", ...]
    quality: float
    score_components: Mapping[str, float]
    created_ns: int


@dataclass(frozen=True)
class ForegroundToken:
    cycle_id: int
    selected_ref: str
    selected_kind: str
    feature_ref: Optional[str]
    type_hypotheses: Tuple["TypeScore", ...]
    context_ref: str
    operation: str
    priority_components: Mapping[str, float]
    confidence: float
    selected_ns: int
    dwell_cycles: int


@dataclass(frozen=True)
class BroadcastCommit:
    commit_id: str
    cycle_id: int
    foreground_ref: str
    prior_commit_ref: Optional[str]
    delta: float
    reason: str
    recipients: Tuple[str, ...]
    action_proposals: Tuple[str, ...]
    memory_write_proposals: Tuple[str, ...]
    timestamp_ns: int
```

**Today’s live contracts** remain `BBP` / `BoundingBox` / attention metrics /
`simple_crop_v1` / Stage-4 bank metrics. Evolve toward the above without breaking
JSONL keys mid-stage; version schemas when fields change.

---

## 6. ST²-inspired attention model: PDS–ST²–GWB

**Label:** PDS–ST²–GWB = precision/difference-weighted selection; serial
type/token binding informed by Simultaneous Type, Serial Token (ST²); operational
global broadcast.

This is a **computational reference architecture**, not a consciousness claim.

### 6.1 Register model

| Register | Capacity / role | Write semantics |
|---|---|---|
| Sensory Buffer (SB) | Short modality ring buffers | Overwritten/aged; does not act |
| Candidate Pool (CP) | Many perceptual + internal candidates | Scoring only |
| Working Memory (WM) | Bounded active/latent tokens | Committed ops only (Stage 8) |
| Attention Priority Register (APR) | Pointer, gain/precision, task state | Selects foreground |
| Binding Pool (BP) | Temporary type–token–role–episode bindings | Versioned provisional |
| Focused Token Register (FTR) | **Exactly one** token + operation | May trigger learning/actions/memory |
| Global Workspace Broadcast (GWB) | Immutable selected snapshot | Subscribers read/propose |
| Stream-of-Cognition log (SoC) | Append-only meaningful commits | Audit/replay, not a memory buffer |

**Current code:** Stage 2 `AttentionScheduler` is the first operational APR/FTR
slice (top-1 WTA + spatial IOR). Expand toward PDS–ST²–GWB **incrementally**; do
not displace the tested baseline without an equal-compute behavioral comparison.

### 6.2 Candidate scoring

```python
# Start interpretable; learn weights later only if justified.
score_i = q_i * (
    w_delta * prediction_error_i
  + w_novel * novelty_i
  + w_motion * motion_or_looming_i
  + w_goal * goal_relevance_i
  + w_uncert * capped_information_value_i
  + w_interrupt * interrupt_i
  + w_sens * sensitization_i
  - w_hab * habituation_i
  - w_ior * inhibition_of_return_i
)

winner = deterministic_argmax(candidates, key=(score, stable_candidate_id))
```

Retain compact historical baseline
`activation * (1 + sensitization) * exp(-alpha * habituation)` as an ablation.

Stage 2 today uses named **proxies** (`novelty_proxy`, `error_proxy`,
`motion_proxy`). Stage 4 supplies real novelty; Stage 5 supplies real prediction
error; Stage 9 supplies temporal motion. Never report proxies as learned signals.

### 6.3 Attention register reference implementation

```python
@dataclass
class AttentionRegisterState:
    token_id: str | None = None
    active_feature: "np.ndarray | None" = None
    operation: str = "observe"
    entered_ns: int = 0
    dwell_cycles: int = 0
    confidence: float = 0.0
    recurrent_gain: float = 1.0


class AttentionRegister:
    def __init__(self, switch_margin: float, min_dwell_cycles: int):
        self.state = AttentionRegisterState()
        self.switch_margin = switch_margin
        self.min_dwell_cycles = min_dwell_cycles

    def should_switch(self, current_score, challenger_score, interrupt=False):
        if interrupt:
            return True
        if self.state.token_id is None:
            return True
        if self.state.dwell_cycles < self.min_dwell_cycles:
            return False
        return challenger_score > current_score + self.switch_margin

    def load(self, candidate, now_ns):
        self.state = AttentionRegisterState(
            token_id=candidate.candidate_id,
            active_feature=resolve_feature(candidate.feature_ref),
            operation="observe",
            entered_ns=now_ns,
            dwell_cycles=0,
            confidence=candidate.quality,
            recurrent_gain=1.0,
        )

    def tick(self):
        if self.state.token_id is not None:
            self.state.dwell_cycles += 1

    def release(self):
        self.state = AttentionRegisterState()
```

### 6.4 Recurrent maintenance (FTR ≠ perceptual GRU ≠ WM)

The FTR’s recurrence **maintains** the selected token long enough for serial
processing. Keep three recurrence roles separate until ablations justify collapse:

1. Perceptual temporal integration (ConvGRU / later SNN)
2. Object / WM maintenance
3. Serial processor / FTR state

```python
# Minimal leaky recurrent maintenance
h_t = tanh(W_rec @ h_prev + W_in @ x_selected + W_ctx @ context)

# Optional attractor-style stabilization
h_t = (1 - leak) * h_prev + leak * f(W_rec @ h_prev + input_drive)

# Release/switch when: margin exceeded after min dwell, interrupt,
# operation complete, IOR/habituation rises, confidence collapses, token retires
```

### 6.5 ST² experiments (later)

- RSVP lag-dependent second-target deficit
- FTR capacity **1** invariant; WM capacity 1/4/7 and broadcast thresholds
- Binding stress: similar targets close in time → token swaps / binding errors
- Parallel/serial dissociation: perception/motor parallel; central token serial

Reference: Bowman & Wyble (2007) ST²; Yue, Newton & Marois (2025) ultrafast-fMRI
serial queuing.

---

## 7. Sensory buffers and non-associative learning

### 7.1 Ring buffer

```python
from collections import deque


class SensoryRing:
    def __init__(self, max_items: int):
        self.buf = deque(maxlen=max_items)

    def push(self, observation):
        self.buf.append(observation)

    def recent(self):
        return tuple(self.buf)
```

Static frames are smoke tests only. Primary learning units are short **replayable**
video/event sequences.

### 7.2 Habituation / sensitization (canonical Stage 6 when opened)

```python
@dataclass
class AdaptiveUnit:
    mean: float = 0.0
    variance: float = 1.0
    habituation: float = 0.0
    sensitization: float = 0.0


def update_unit(u, x, alpha=0.01, hab_rate=0.002, sens_decay=0.01):
    err = x - u.mean
    u.mean += alpha * err
    u.variance = (1 - alpha) * u.variance + alpha * (err * err)
    surprise = abs(err) / (u.variance**0.5 + 1e-6)

    u.habituation = min(1.0, u.habituation + hab_rate * max(0.0, 1.0 - surprise))
    u.sensitization = max(0.0, (1 - sens_decay) * u.sensitization)
    if surprise > SURPRISE_THRESHOLD:
        u.sensitization = min(1.0, u.sensitization + SENS_GAIN * surprise)

    gain = (1.0 + u.sensitization) * math.exp(-HAB_ALPHA * u.habituation)
    return gain * err, surprise
```

Compare patch/channel/feature adapters behind one interface. Test local adaptation
**and** model-based cancellation of expected input (top-down predictive filtering).

---

## 8. Feature providers (switchable)

```text
FeatureProvider.encode(frame_or_roi, history, context) -> FeaturePacket

FrozenDINOProvider          # later comparison; not Stage 3 baseline
YOLOFeatureProvider         # scaffold features
HandEngineeredProvider      # Stage 3 simple_crop_v1 lives here today
SparseAutoencoderProvider   # controlled comparison only
ConvGRUProvider             # non-spiking temporal reference
SpikingProvider             # SNN A/B later
HebbianGraphProvider        # developmental pathway
```

Autoencoders / DINO are **branches**, not magic substrates. Reconstruction is not
the main endpoint — measure temporal stability, re-ID, clustering, continual learning.

---

## 9. Hierarchical Hebbian / anti-Hebbian graph

Primary **long-horizon** experimental substrate (canonical Stage 7 when opened;
typed thin writer may appear earlier only as an explicit PR).

Keep **three mechanisms separate**:

1. Hebbian — strengthen recurring co-activation  
2. Anti-Hebbian — decorrelate competitors (not passive decay)  
3. Decay — forget unsupported structure  

```python
hebb = eta_pos * modulator * pre * post
anti = eta_neg * competitor_activity(pre, post)
decayed_w = old_w * math.exp(-decay_rate * delta_t)
new_w = clip(decayed_w + hebb - anti + homeostasis, W_MIN, W_MAX)
```

Eligibility / directional temporal association:

```python
eligibility[j] = lambda_e * eligibility[j] + x_j_t
w_ij = lazy_decay(w_ij, now - last_update)
w_ij += eta * attention_modulator * x_i_t * eligibility[j]
# A at t predicting B at t+dt => A --precedes/predicts--> B
```

Homeostasis baselines: Oja and/or BCM sliding threshold. Raw unbounded
`w += eta * x * y` is **unacceptable** for long online runs.

Growth constraints: only FTR token + bounded eligibility neighborhood update
synchronously; lazy decay on access; top-k edges by type; async prune/merge.

---

## 10. Object files, permanence, prototypes

### Tracking before multi-view categories (Stage 9+)

```python
for frame in replay:
    predicted = tracker.predict(active_tracks)
    observations = perception.propose(frame)
    matches, unmatched_obs, unmatched_tracks = associate(
        predicted, observations,
        cost=["geometry", "appearance", "mask_iou", "time"],
    )
    tracker.update(matches)
    tracker.create_tentative(unmatched_obs)
    tracker.mark_occluded_or_ghost(unmatched_tracks)
    emit_track_events()
```

Baseline: transparent Kalman + Hungarian; then maintained trackers. Category
metrics are meaningless without separate ID-switch measurement.

### Object file sketch

```python
@dataclass
class ObjectFile:
    track_token_id: str
    lifecycle: str
    bbox_or_mask: "Region"
    velocity: tuple[float, ...]
    covariance: tuple[float, ...]
    appearance_ema: "np.ndarray"
    feature_uncertainty: float
    class_distribution: dict[str, float]
    last_visible_ns: int
    predicted_hidden_state: "np.ndarray | None"
    prototype_ref: str | None
    attention_history: tuple[str, ...]
```

Permanence = persistence + hidden-state prediction + re-binding under uncertainty —
not a symbolic axiom. Test impossible reappearances via prediction error.

### Instance vs category prototypes

Do **not** merge “this specific object” and “objects like this.”

Stage 4 today: **pattern** prototypes over attended embeddings (no tracks).  
After Stage 9: instance prototypes across views; then category lifecycle:

```text
UNKNOWN_EPISODE -> DRAFT_CLUSTER -> STABLE_PERCEPTUAL_CATEGORY
  -> NAMED_CONCEPT_CANDIDATE -> KSG_PROMOTED_CONCEPT -> SUPERSEDED/RETIRED
```

Compare ART vigilance / DP-means / Fuzzy ART / SUSTAIN-inspired recruitment as
**separate** experiments — never one mega-PR.

---

## 11. Attention-gated plasticity

```python
def plasticity_modulator(attended, quality, novelty, prediction_error, outcome=0.0):
    a = 1.0 if attended else UNATTENDED_GAIN
    return clip(
        a
        * quality
        * (1 + NOVELTY_GAIN * novelty)
        * (1 + ERROR_GAIN * prediction_error)
        * (1 + OUTCOME_GAIN * outcome),
        0.0,
        MAX_PLASTICITY_GAIN,
    )
```

Core comparison: all-token vs equal-compute random vs top-1 vs top-k. Measure
sample efficiency, prototype contamination, forgetting, graph growth, identity
stability, compute.

---

## 12. Working memory and associative retrieval (Stage 8+)

```python
@dataclass
class WorkingMemorySlot:
    token_id: str | None
    active_state: "np.ndarray | None"
    latent_state_ref: str | None
    activation: float
    priority: float
    last_refreshed_ns: int
    uncertainty: float


class WorkingMemory:
    def __init__(self, capacity: int):
        self.slots = [
            WorkingMemorySlot(None, None, None, 0, 0, 0, 1) for _ in range(capacity)
        ]
```

Compare persistent recurrence vs latent/activity-silent vs hybrid refresh.

Autoassociation: partial → complete percept.  
Heteroassociation: visual → instance → category → KSG ref.

Retrieved memories enter the **same** candidate pool as percepts/goals/interrupts.

---

## 13. Spiking / STDP A–B track (not the default substrate)

### Position

Do **not** start with an SNN as the only implementation. Make SNNs an A/B branch
against a conventional recurrent baseline **after** deterministic non-spiking
reference behavior exists.

SNNs can represent timing naturally and can be energy-efficient on neuromorphic
hardware with sparse events. Whether they are more **information-efficient** than
ANN/RNN depends on coding scheme, spike rate, task, and hardware. Training and
hardware-mapping challenges remain real.

### First comparison topology

```text
same video / event stream
        │
        ├───────────────┐
        ▼               ▼
A. ConvGRU/RNN      B. SNN
   dense state         LIF/AdLIF
   continuous          spikes through time
        │               │
        └───────┬───────┘
                ▼
       SAME Hebbian graph
       SAME object files
       SAME attention logic
       SAME benchmark
```

Only the temporal encoding / recurrent substrate changes.

### Modest SNN start (LIF)

```text
τ dV/dt = -V + I(t)
V > θ  ⇒  spike, then reset
```

Adaptive threshold path (ties to habituation/sensitization):

```text
repeated stimulation → threshold ↑ → fewer spikes → habituation
surprising stimulation → threshold ↓ / gain ↑ → sensitization
```

Discrete reference:

```python
V_t = beta * V_prev + input_current
spike = 1 if V_t >= threshold else 0
if spike:
    V_t = reset_voltage
threshold_t = base_threshold + k_hab * habituation - k_sens * sensitization
```

### STDP (temporal causality, not mere correlation)

```python
dt = t_post - t_pre
if dt > 0:
    dw = A_plus * math.exp(-dt / tau_plus)
else:
    dw = -A_minus * math.exp(dt / tau_minus)
w = clip(lazy_decay(w) + attention_modulator * dw, W_MIN, W_MAX)
```

Directional sequences (edge → contour → hand → ball) become typed
`precedes/predicts` structure.

### Coding schemes

| Scheme | Idea |
|---|---|
| Rate | strong → many spikes |
| Latency | strong/important → spike **earlier** |

Formulate the first SNN hypothesis narrowly:

> Does sparse temporal spike coding + local timing-dependent plasticity produce
> more stable and sample-efficient online perceptual representations than
> conventional recurrent activation under identical sensory experience?

If **no**: keep Hebbian graph, attention register, object files, habituation; keep
conventional recurrence.  
If **yes**: justify event-camera / neuromorphic progression.

### A/B matrix

| Branch | Substrate | Primary measurements |
|---|---|---|
| A | ConvGRU / conventional recurrence | accuracy, latency, compute, stability |
| B | LIF/AdLIF rate coding | spikes/event, latency, accuracy, energy proxy |
| C | Latency coding | time-to-decision, temporal info efficiency |
| D | SNN + STDP | online sequence learning, interference |
| E | STDP + habituation + attention gating | full developmental temporal test |

### Hardware / sensor ladder

```text
PHASE A  RGB + ConvGRU vs simple SNN
PHASE B  Hebbian vs STDP temporal Hebbian
PHASE C  Adaptive neurons (habituation / sensitization / thresholds)
PHASE D  RGB + simulated event stream
PHASE E  Real event camera
PHASE F  Neuromorphic hardware — only if results justify it
```

### Flagship occlusion experiment

```text
moving ball → disappears → reappears
```

Compare: RNN | SNN rate | SNN latency | SNN+STDP | +habituation | +attention gating.  
Ask whether spike timing improves trajectory, persistence, and identity with fewer
active units/events.

Extra metrics beyond accuracy: bits/events, spikes/s, temporal prediction error,
latency to motion/change, ID stability, occlusion recovery, sequence learning,
sparsity, compute/energy proxy, catastrophic interference, sample efficiency.

Contemporary pointers: SDTrack (2025), SpikeFET (2025), hybrid spiking ViT for
event detection (ICML 2025). Use as **precedent**, not dependencies.

---

## 14. Events, affordances, KSG, active vision

- Events are first-class tokens (subject/predicate/object + evidence), separate
  from objects.
- Affordances are evidence-backed outcome predictions, not hard-coded labels.
- KSG writes only after stability/evidence gates; retain local IDs + receipts;
  provenance to frame/observation.
- Active vision / servo only after attention + identity work in replay; cognition
  emits `GazeRequest`, device layer owns PID/PWM.

---

## 15. Developmental curriculum (research D-ladder)

Map to **canonical PR stages**; do not renumber PHASED_PLAN.

| Research gate | Capability | Canonical hook |
|---|---|---|
| D0 | Static feature sanity | Stages 0–3 |
| D1 | Motion / temporal continuity | After Stage 5; recurrent encoder later |
| D2 | Persistent object files | Stage 9 |
| D3 | Object permanence | Stage 9+ |
| D4 | Instance prototypes across episodes | Post-tracking |
| D5 | Category formation | Post-tracking |
| D6 | Attention-gated plasticity | Post Stage 5; ablations |
| D7 | WM / retrieval | Stage 8 |
| D8 | Events / affordances | Post Stage 9 |
| D9 | Active vision | Late |
| D10 | KSG consolidation | Thin writer after Stage 8; rich after maturity |

---

## 16. Suggested package boundaries (target; evolve, don’t big-bang rename)

```text
# Near-term (respect existing packages)
perception/   attention/   features/   objects/   graph/   tracking/
experiments/  scripts/     tests/      docs/

# Longer-term modules (add when a stage opens)
domain/ schemas, ids, events
sensors/ sensory_ring, event_stream
adaptation/ habituation, sensitization, predictive_filter
learning/ hebbian_graph, anti_hebbian, decay, homeostasis, eligibility,
          prototypes, categories, plasticity
attention/ candidates, scoring, priority_register, binding_pool,
           focused_register, broadcast, inhibition_of_return
memory/ working_memory, recurrent_maintenance, associative, episodic,
        consolidation
cognition/ token_processor, event_detector, relations, affordances, goals
active_vision/ gaze_policy, servo_controller, simulator
integrations/ ksg_adapter, detector_adapter, sam2_adapter
visualization/ overlay, graph_view, cognitive_trace
```

---

## 17. Test strategy

### Unit

Score components; decay/Hebbian/anti-Hebbian signs; normalization; LR schedules;
ring boundaries; FTR capacity-one + deterministic ties; object lifecycle;
prototype bounds; WM capacity; KSG proposal idempotency.

### Golden replay

Tiny local fixtures; fixed seed → stable JSONL. Core tests must not require GPU,
network, or model downloads.

### Behavioral

Habituation / oddball; sensitization; permanence; RSVP blink; attention gating
budgets; memory completion; graph next-event MRR; active-vision information gain.

### Ablation matrix (every major knob independently toggleable)

```text
FEATURES: frozen_dino | yolo | hand | sparse_ae | recurrent | hebbian | spiking
TEMPORAL: none | optical_flow | kalman | GRU | SNN
HEBBIAN: off | raw | Oja | BCM
ANTI_HEBBIAN: off | lateral_competition
DECAY: off | eager | lazy
HABITUATION: off | local | predictive
SENSITIZATION: off | surprise_only | outcome_modulated
ATTENTION: off | random_budget | top1 | topk
FTR_CAPACITY: 1   # invariant; >1 only explicit challenge
WM_CAPACITY: 1 | 4 | 7
FTR_RECURRENCE: off | leaky | attractor
WM_MAINTENANCE: persistent | latent | hybrid
MEMORY: none | autoassoc | heteroassoc | both
SNN: off | rate | latency | STDP
KSG: disconnected | mock | live
ACTIVE_VISION: passive | scripted | information_gain
```

---

## 18. Metrics cheat sheet

Tracking: HOTA, IDF1, ID switches, occlusion recovery, ghost P/R.  
Representation: temporal stability, re-ID, clustering ARI/NMI.  
Continual: few-shot, forgetting, transfer, category growth.  
Graph: MRR, Recall@K, sparsity, churn, latency.  
Attention: margin, dwell, switch entropy, revisit, interrupt latency, blink curve.  
Memory: retrieval, completion, interference, active/latent occupancy.  
SNN: spikes/event, time-to-decision, sparsity, energy proxy.  
Systems: FPS, p50/p95/p99, memory, log growth.  
Active vision: reacquisition, info gain per motion/cost.

---

## 19. Exact research-stage track vs what to code next

The source “Stage 0–13” list below is a **research integration track**. Prefer
canonical PHASED_PLAN numbers when opening PRs.

| Research stage (source) | Meaning | When relative to canon |
|---|---|---|
| R0 | Branch reconcile + golden replay | Ongoing hygiene; formalize before Stage 9 |
| R1 | Sensor ring + Observation + tracker baseline | Stage 9 lead-in |
| R2 | Occlusion / permanence lifecycle | Stage 9 |
| R3 | Full CP/APR/BP/FTR/GWB/SoC without learning | Evolve Stage 2; optional PRs after Stage 5 |
| R4 | Frozen features + prototypes/categories | Stage 3–4 done for cheap path; categories later |
| R5 | Habituation/sensitization oddball | Stage 6 when opened |
| R6 | Hebbian graph + anti-Hebbian + decay + eligibility | Stage 7 when opened |
| R7 | Attention-gated plasticity + serial-token experiments | After Stage 5 novelty/error real |
| R8 | Recurrent encoder + WM + associative retrieval | Stage 8 + later temporal encoder |
| R9 | Sparse AE branch | Controlled comparison only |
| R10 | SNN/STDP A/B | After ConvGRU reference |
| R11 | Events / relations / affordances | Post Stage 9 |
| R12 | Episodic/semantic consolidation + KSG | Thin KSG after Stage 8 |
| R13 | Servo / simulated active vision | Late |

### Immediate coding-agent work order (executable now)

Follow [`docs/PHASED_PLAN.md`](PHASED_PLAN.md) active work order. Short form:

1. Land / stabilize **Stage 4** prototype bank (tests + CI green).
2. **Stage 5** prediction error (replace `error_proxy`); repetition + oddball.
3. **Stage 5b** attention-gated plasticity ablations (equal-compute).
4. **Stage 8** K-slot WM; then **thin KSG** writer.
5. Pre-Stage-9 `docs/BRANCH_RECONCILIATION.md`; then **Stage 9** tracker /
   permanence (salvage `2576` behind an interface).
6. **Stage 6** then **Stage 7** (habituation; typed Hebbian ⊥ anti-Hebbian ⊥ decay).
7. **Stage 9b** instance/category; **Stage 10a** ConvGRU; **only then Stage 10b**
   SNN/STDP A/B; then events / active vision / event-camera ladder.

Do **not** connect live KSG or servo hardware until offline replay tests pass;
mocks first. Do **not** start SNN before a ConvGRU (or equivalent) reference on
the same replay contract.

---

## 20. Success criteria

The program succeeds if hierarchical local associative learning, anti-Hebbian
competition, adaptive sensory processing, persistent object files, serial
attention, attention-gated plasticity, prototype learning, and progressive
consolidation produce **more stable, sample-efficient, interpretable, continually
learnable** perceptual representations than conventional end-to-end baselines —
**shown by deterministic replay, behavioral benchmarks, and ablations**, not by
biological language alone.

If SNN/STDP loses the A/B, retain the cognitive architecture on a conventional
recurrent substrate. That is a successful experiment.

---

## 21. Research references (implementation precedents)

1. Bowman & Wyble (2007), ST² model — https://pubmed.ncbi.nlm.nih.gov/17227181/
2. Yue, Newton & Marois (2025), ultrafast fMRI serial queuing —
   https://doi.org/10.1038/s41467-025-58228-0
3. Panichello et al. (2024), intermittent WM coding —
   https://doi.org/10.1038/s41586-024-08139-9
4. Tsukano et al. (2026), OFC predictive sensory filtering —
   https://doi.org/10.1038/s41593-026-02217-z
5. Oja (1982), normalized Hebbian PCA rule
6. Bienenstock, Cooper & Munro (1982), BCM sliding threshold
7. Fritzke (1994), Growing Neural Gas
8. Furao & Hasegawa (2006), ESOINN
9. Love, Medin & Gureckis (2004), SUSTAIN
10. Carpenter, Grossberg & Rosen (1991), Fuzzy ART
11. Locatello et al. (2020), Slot Attention
12. Patil et al. (2025), SAMP object-centric priors
13. Nguyen et al. (2026), Temporal Slot Activation
14. Ravi et al. (2024), SAM 2
15. ByteTrack — https://github.com/FoundationVision/ByteTrack
16. Shan et al. (2025), SDTrack — https://arxiv.org/abs/2503.08703
17. Xu et al. (2025), Hybrid Spiking ViT for event cameras —
   https://proceedings.mlr.press/v267/xu25e.html
18. Yang et al. (2025), SpikeFET — https://arxiv.org/abs/2505.20834
19. McClelland, McNaughton & O’Reilly (1995), Complementary Learning Systems
20. Anderson et al. (2004), ACT-R modular buffers

---

## 22. Final design position

The system should eventually: watch a changing world; adapt away predictable
irrelevant input; discover sparse recurring structure; bind observations into
persistent object files; maintain and recover hidden objects; learn instance and
category prototypes; allocate attention and plasticity selectively; retrieve
long-term associations into WM; learn temporal/event relations; move its camera
to reduce uncertainty; and only then consolidate evidence-backed knowledge into
KnowShowGo.

The research value is the **measured synthesis** and the ability to explain what
each mechanism contributes — with SNNs earning their place against ConvGRU under
identical experience, not assuming spike timing is magic.
