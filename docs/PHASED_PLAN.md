## Phased development track (least dependency first)

**Updated:** 2026-08-07  
**Design goal:** add **one mechanism at a time**, keep interfaces stable, make each
stage publishable via logged metrics and ablations.

Research detail, ST² register sketches, SNN/STDP A/B design, and code references:
[`docs/DEVELOPMENTAL_COGNITION_CODING_AGENT_HANDOFF.md`](DEVELOPMENTAL_COGNITION_CODING_AGENT_HANDOFF.md).
That document **must not** silently reorder this file; change the track here when
the user gate changes.

### Guiding rules (from the developmental handoff)

- YOLO is frozen scaffolding, not ontology.
- Pattern prototypes ≠ object files ≠ categories ≠ KSG concepts ≠ names.
- Cognitively consequential selection stays **capacity-one** (FTR); perception may
  stay parallel.
- Every learning mechanism is independently switchable.
- SNNs are an **A/B branch** against a conventional recurrent baseline — never the
  sole substrate until they earn it.
- Track / permanence after a track-free perceptual baseline (Stages 2–5).

---

## Active work order

```text
[done]  Stage 0 harness
[done]  Stage 1 BBP / frozen YOLO
[done]  Stage 2 attention (WTA + IOR)          → evolve toward PDS–ST²–GWB later
[done]  Stage 3 cheap attended embeddings
[now]   Stage 4 bounded pattern prototypes + novelty
[next]  Stage 5 top-down prediction error
[next]  Stage 5b attention-gated plasticity ablations (top-1 vs random vs all-token)
[next]  Stage 8 K-slot working memory
[next]  Thin KSG visual writer (match / generalize / exemplar)
[next]  Stage 9 TrackToken / object files / permanence
[then]  Stage 6 habituation / sensitization (adaptive gain; feeds later SNN thresholds)
[then]  Stage 7 typed Hebbian graph (Hebbian ⊥ anti-Hebbian ⊥ decay ⊥ eligibility)
[then]  Stage 9b instance prototypes across tracks; category baselines (separate PRs)
[then]  Stage 10a ConvGRU / conventional temporal reference (same replay contract)
[then]  Stage 10b SNN A/B (LIF rate → latency → STDP → +habituation/+attention)
[later] Events / relations / affordances
[later] Active vision / servo (GazeRequest; device PID isolated)
[later] RGB+simulated events → event camera → neuromorphic (only if metrics justify)
```

Stages keep stable numbers. Letters (`5b`, `9b`, `10a/b`) are **sub-PRs** under
that stage family, still one mechanism each.

Deferred relative to Stage 8 / thin KSG: nothing blocks opening Stage 6 or 7
*after* Stage 9 if experiments need adaptive gain or graph structure sooner —
but default order above prefers permanence before rich graph growth.

---

## Stage status and scope

### Stage 0 — Experiment harness (reproducibility) · DONE

- Session runner emits JSONL; seed everything; small stable config.
- Tests: deterministic log schema; stable output structure.
- Now also: CI (pytest + ruff); OBS / Virtual Camera laptop runbook.

### Stage 1 — BBP generator (YOLO front-end) · DONE

- Frozen YOLO proposes BBPs (transient hypotheses, not labels).
- Tests: determinism and BBP schema sanity.

### Stage 2 — Attention scheduler (serial stream) · DONE (evolve, don’t rewrite)

- Select 1 BBP per frame (WTA) with spatial inhibition-of-return.
- Metrics still use named **proxies** until later stages replace them.
- **Evolution track (optional PRs after Stage 5):** grow toward PDS–ST²–GWB
  registers — Candidate Pool, APR, Binding Pool, **FTR capacity = 1**, optional
  GWB/SoC commit log — without displacing the tested Stage 2 baseline unless an
  equal-compute comparison wins.
- ST² experiments (RSVP blink, binding stress) are **later** behavioral suites,
  not Stage 2 merge criteria.
- Tests: exactly one selection when candidates exist; no stuck fixation;
  deterministic tie-break.

### Stage 3 — Simple embeddings · DONE

- Cheap attended-crop embeddings (`simple_crop_v1`: geometry + RGB moments).
- Scope: WTA winner only; L2 unit vectors; versioned schema.
- DINOv2 / SAE / masks stay comparison branches, not replacements.
- Tests: bounded norms; deterministic features; invalid-crop schema.

### Stage 4 — Pattern prototype bank + novelty · IN PROGRESS / LAND NEXT

- Online match/spawn over attended embeddings; `Kmax`; novelty hysteresis;
  spawn cooldown; normalized running-mean (or documented update-rule id).
- Pattern IDs ≠ YOLO class ≠ name ≠ track ≠ category.
- JSONL: `prototype_bank` + `prototype_bank_schema`.
- ART / SUSTAIN / DP-means = **separate later comparison PRs**.
- Tests: count ≤ `Kmax`; novelty spikes on distribution shift; learning-disable
  freezes bank.

### Stage 5 — Dual processing / prediction error · NEXT

- Top-down expected embedding from matched (or WM-cued) prototypes; winner by
  min error; genuine prediction error replaces `error_proxy`.
- Tests: error falls on repetition; spikes on novel / oddball / shift.
- Primary endpoint: calibrated surprise for learning and attention — not a demo
  heatmap as detector input.

### Stage 5b — Attention-gated plasticity · AFTER STAGE 5

- Plasticity modulator from attention × quality × novelty × prediction error
  (outcome term later).
- Ablations under **equal compute**: all-token vs random-budget vs top-1 vs top-k.
- Tests: declared sample-efficiency / contamination / forgetting endpoints beat
  matched-budget baseline or document a negative result.

### Stage 6 — Habituation / sensitization (gain gating) · AFTER STAGE 9 DEFAULT

- Repeated low surprise reduces gain; surprise / consequence sensitizes.
- Implement as switchable adapters (patch/channel/feature); compare local vs
  predictive cancellation of expected input.
- Prepares adaptive thresholds for later SNN branch.
- Tests: habituation curves; oddball rebound; sensitization spikes.

### Stage 7 — Typed Hebbian graph · AFTER STAGE 6 DEFAULT

- Sparse typed multi-timescale graph; **separate** Hebbian, anti-Hebbian, decay,
  eligibility / precedes edges; Oja or BCM homeostasis (no raw unbounded Hebb).
- Lazy decay on access; bounded neighborhood updates from FTR token.
- STDP lands only under Stage **10b**, against this non-spiking reference.
- Tests: edge count bounded; predictive association above frequency baseline;
  each mechanism toggleable.

### Stage 8 — Working memory (few expected concepts) · AFTER 5 / 5b

- K-slot WM of expected prototype/concept refs; cue-based load; eviction by
  utility / recency / error.
- Distinct from FTR (one focused token) and from the full graph / KSG.
- Compare maintenance modes later: persistent vs latent vs hybrid refresh.
- Physical object files **do not** begin here — Stage 9.
- Tests: capacity never exceeded; cueing reloads prior refs.

### Thin KSG writer · AFTER STAGE 8

- Async client: visual match / generalize / exemplar only; receipt; provenance;
  idempotency; durable UUID mapping.
- Promote only after stability / evidence gates.
- Not a dump of every frame or edge.
- Tests: mock KSG durability; reject path keeps local IDs; no silent overwrite.

### Stage 9 — Tracking, object files, permanence · AFTER THIN KSG

- Conventional tracker first (Kalman/Hungarian baseline; salvage remote `2576`
  behind an interface after `docs/BRANCH_RECONCILIATION.md`).
- `TrackToken` lifecycle: tentative → active → occluded → ghost → retired.
- Hidden-state prediction + probabilistic re-bind; measure ID switches separately
  from category metrics.
- Then motion prototypes / short trajectory features.
- Flagship behavioral test: moving target → occlude → reappear.
- Tests: fewer ID switches / fragmentation; occlusion recovery; ghost precision.

### Stage 9b — Instance then category prototypes · AFTER STAGE 9

- Instance prototype: “this physical individual” across views/episodes.
- Category prototype: “things like this” with vigilance / unknown handling.
- Never merge the two concepts. Compare ART / DP-means / SUSTAIN in **separate**
  experiments.
- Tests: re-ID across leave/return; cluster separation; unknown rejection.

### Stage 10a — Conventional temporal recurrence · BEFORE ANY SNN

- ConvGRU / RNN temporal encoder on the **same** replay events and downstream
  contracts (attention, prototypes, object files, graph).
- Distinct from FTR recurrence and WM maintenance recurrence.
- Endpoints: next-feature prediction, change detection, identity-preserving
  temporal state, occlusion aids.
- Tests: beats no-temporal baseline on declared temporal endpoints; deterministic
  replay.

### Stage 10b — SNN / STDP A/B track · ONLY AFTER 10a

Narrow first hypothesis:

> Sparse temporal spike coding + local timing-dependent plasticity vs conventional
> recurrent activation under **identical** sensory experience.

Ladder inside 10b (separate PRs):

1. LIF/AdLIF **rate** coding  
2. **Latency** coding  
3. STDP (directional `precedes/predicts`)  
4. STDP + habituation/sensitization thresholds  
5. + attention-gated plasticity  

Same Hebbian graph / object files / attention / benchmarks as branch A.
SNNs do **not** automatically compress more information in time — measure it.

Metrics: spikes/event, time-to-decision, temporal prediction error, ID stability,
occlusion recovery, sparsity, energy/compute proxy, interference, sample
efficiency — not accuracy alone.

Hardware ladder **after** positive results: RGB → simulated events → event camera
→ neuromorphic. Do not buy hardware to “complete” Stage 10.

### Later program (post Stage 10 family)

| Track | Content |
|---|---|
| Events / relations | First-class event tokens; prediction-error boundaries |
| Affordances | Evidence-backed outcome predictions (rollable, graspable, …) |
| Associative memory | Auto- and hetero-associative retrieval into candidate pool |
| Active vision | `GazeRequest` → isolated servo/PID; info-gain policies |
| Rich KSG | Categories, procedures, multi-agent priors with provenance |
| Helper-lite / robot | inspect → act → verify → recover/ask; safety outside FTR |

---

## Crosswalk: research D-gates → this track

| D-gate | Capability | Lands in |
|---|---|---|
| D0 | Static feature sanity | Stages 0–3 |
| D1 | Motion / temporal continuity | Stage 10a (preview signals earlier via proxies only) |
| D2–D3 | Object files / permanence | Stage 9 |
| D4–D5 | Instance / category | Stage 9b |
| D6 | Attention-gated plasticity | Stage 5b |
| D7 | WM / retrieval | Stage 8 |
| D8 | Events / affordances | Post-10 later program |
| D9 | Active vision | Later program |
| D10 | KSG consolidation | Thin KSG after 8; rich later |

---

## PR discipline

Each PR must:

1. Touch **one** mechanism (or one documented ablation axis).
2. Add/extend tests (unit and/or golden replay).
3. Emit stable JSONL metrics (versioned schema keys).
4. Provide a disable / non-learning baseline where learning is introduced.
5. Avoid renaming “prototype” → “object” or “category” without Stage 9b evidence.

Remote branch salvage (object-permanence, ProtoYolo, alternate attention) happens
only at the gated stage above — never as drive-by merges into Stage 4/5.
