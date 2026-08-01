# Online Perceptual Learning with YOLO BBPs

## Full Development Plan & Engineering Handoff Document

**Status:** Consolidated, endâ€‘toâ€‘end design and build roadmap
**Audience:** Research engineers / systems engineers
**Goal:** Implement an online, continual, biologically inspired perceptual learning system using YOLO Bounding Box Percepts (BBPs), Hebbian learning, habituation/sensitization, WTA competition, topâ€‘down prediction, and a cascading perceptual DAG analogous to the human visual pathway.

---

## Quickstart (Phase 1: Video â†’ YOLO â†’ BBPs)

This repo now includes a minimal **Phase-1 scaffold**:

- `perception/bbp.py`: `BBP` + `BoundingBox` data model
- `perception/video.py`: video/camera frame iterator (OpenCV)
- `perception/yolo_adapter.py`: Ultralytics YOLO adapter â†’ BBPs
- `scripts/run_bbp_stream.py`: CLI to stream BBPs and optionally write JSONL

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -r requirements-vision.txt
```

### Install (Poetry, reproducible)

```bash
python3 -m pip install --user poetry
poetry install
poetry install --with vision
```

### Run on a video file

```bash
python scripts/run_bbp_stream.py --source path/to/video.mp4 --max-frames 200
```

### Run on webcam

```bash
python scripts/run_bbp_stream.py --source 0 --max-frames 200
```

### Save BBPs to JSONL

```bash
python scripts/run_bbp_stream.py --source 0 --save-jsonl outputs/bbps.jsonl --max-frames 200
```

### Run a session log (JSONL)

```bash
python experiments/run.py --source 0 --max-frames 300 --output-dir outputs
```

## Docs

- `docs/HANDOFF.md`: local handoff checklist + reproducibility notes
- `docs/PHASED_PLAN.md`: minimal staged plan (least dependency first)
- `docs/COGNITIVE_ARCHITECTURE_MAP.md`: consolidated architecture, research hypotheses, and experiment gates
- `docs/DEBUGGING.md`: debugging + refactor guidance
- `docs/OBS_SETUP.md`: OBS recording setup for studies
- `docs/REFERENCE_REPOS.md`: reference repos/libraries to fork or borrow from

## 0. Executive Summary

This system treats **YOLO detections as attentional percepts**, not labels. Each detection becomes a **Bounding Box Percept (BBP)**â€”a transient sensory hypothesis that feeds an **online learning pipeline**. Over time, BBPs are bound into tracks, tracks form object prototypes, prototypes organize into categories, and associations form a **dynamic percept graph**.

Learning is:

* **Online** (no offline retraining loops)
* **Local** (Hebbian / predictionâ€‘error gated updates)
* **Sparse** (WTA + inhibition)
* **Continual** (decay + metaplasticity prevent catastrophic forgetting)

The system implements **dual processing**:

* **Bottomâ€‘up recognition** (matching BBPs to learned prototypes)
* **Topâ€‘down prediction** (prototypes predict expected lowerâ€‘level features)

Attention acts as a **winnerâ€‘takeâ€‘most scheduler**, enforcing a single (or very small) conscious processing stream while allowing background stabilization.

---

## 1. Core Concepts & Definitions

### 1.1 Bounding Box Percept (BBP)

A BBP is the atomic perceptual unit emitted per frame.

**BBP = localized, timeâ€‘indexed percept hypothesis**

Properties:

* Spatial: bounding box
* Appearance: latent embedding
* Temporal: motion deltas
* Cognitive: salience, novelty, prediction error

BBPs are *not* objects, labels, or concepts.

---

### 1.2 Representational Hierarchy

```
Pixels
  â†“
BBPs (YOLO)
  â†“
Lowâ€‘level features (V1â€‘like)
  â†“
Parts (coâ€‘activated features)
  â†“
Object prototypes (identity)
  â†“
Categories / scenes / concepts
```

Each layer:

* Competes internally (WTA)
* Learns via local Hebbian / errorâ€‘gated updates
* Predicts the layer below (topâ€‘down)

---

### 1.3 Dual Processing

Each prototype acts as:

1. **Recognizer** â€“ explains incoming percepts
2. **Generator** â€“ predicts expected features

Prediction error drives:

* Plasticity
* Attention
* Novelty detection

---

## 2. System Architecture (Modules)

### 2.1 Perception I/O

* Video ingestion (OpenCV / PyAV)
* YOLO inference (frozen weights)
* BBP extraction

### 2.2 Attention & Routing

* Salience computation (Î” appearance, Î” motion, prediction error)
* WTA selection (1 object per attention tick)
* Inhibitionâ€‘ofâ€‘return (prevents fixation)

### 2.3 Representation Stack

| Layer    | Function           | Learning            | Output            |
| -------- | ------------------ | ------------------- | ----------------- |
| Feature  | Edges / textures   | WTA + Hebbian       | Sparse codes      |
| Part     | Feature assemblies | Hebbian             | Part prototypes   |
| Object   | Identity           | Prototype averaging | Object prototypes |
| Category | Similar objects    | Graph clustering    | Category nodes    |

### 2.4 Temporal Stack

* Tracking (object permanence)
* Slowness / trace rules
* Motion prototypes
* Optional recurrence (GRU/LSTM â†’ SNN later)

### 2.5 Graph Memory

* Dynamic DAG (NetworkX initially)
* Nodes: features, parts, objects, motions, categories
* Edges: coâ€‘occurs, partâ€‘of, predicts, transitions

---

## 3. Development Phases (Strict Order)

> **Stage-numbering note:** this older conceptual phase outline is retained as
> research context. PR order and executable stage numbers are defined by
> `docs/PHASED_PLAN.md`; in particular, current Stage 2 is attention, Stage 3 is
> cheap attended-crop embeddings, Stage 8 is K-slot working memory, and Stage 9
> is tracking. Do not use the legacy phase numbers below to name implementation
> PRs.

### Phase 1 â€” Frame Pipeline & BBP Generator

**Goal:** Deterministic, stable percept stream

**Deliverables**

* Video â†’ frames
* YOLO â†’ boxes
* BBP data structure

**Tests**

* Determinism
* Throughput
* Memory bounds

---

### Phase 2 â€” Static Feature Learning (No Tracking)

**Goal:** Learn sparse, stable visual features

**Mechanisms**

* Patch sampling from BBPs
* WTA competition
* Hebbian / predictiveâ€‘coding updates
* Weight normalization + homeostasis

**Tests**

* Feature diversity
* Sparsity targets
* No collapse / no divergence

---

### Phase 3 â€” Part Formation

**Goal:** Build midâ€‘level structure

**Mechanisms**

* Coâ€‘activation Hebbian learning
* Partâ€‘of edges

**Tests**

* Parts recur across exemplars
* Parts generalize across viewpoints

---

### Phase 4 â€” Object Prototype Formation (Static Identity)

**Goal:** Stable object identity without motion learning

**Mechanisms**

* Online prototype clustering
* Errorâ€‘gated averaging
* Noveltyâ€‘based spawning
* Habituation

**Tests**

* Same object â†’ same prototype
* Prototype count bounded
* Familiarity reduces error

---

### Phase 5 â€” Tracking & Slowness (First Temporal Learning)

**Goal:** Object permanence and invariance

**Mechanisms**

* SORT / DeepSORT
* Trace/slowness rules
* Temporal Hebbian edges

**Tests**

* Reduced representation drift
* Occlusion tolerance

---

### Phase 6 â€” Motion Prototypes

**Goal:** Learn motion as firstâ€‘class percepts

**Mechanisms**

* Motion vectors (Î”x, Î”y, Î”scale)
* Motion prototype clustering
* Object â†” motion associations

**Tests**

* Distinct motion classes
* Motion generalizes across objects

---

### Phase 7 â€” Recurrence (Optional, Only If Needed)

#### 7a. GRU / LSTM

* Perâ€‘track temporal buffers
* Predict next embedding / motion

#### 7b. Spiking / STDP (Advanced)

* Eventâ€‘driven temporal binding
* WTA + homeostatic STDP

**Entry condition:** Trace/slowness insufficient

---

### Phase 8 â€” Category & Concept Formation

**Goal:** Higherâ€‘level abstraction

**Mechanisms**

* Graph clustering
* Shared parts + motion patterns
* Category nodes

**Tests**

* Category purity
* Graph sparsity preserved

---

## 4. Learning Rules & Plasticity Control

### 4.1 Hebbian Core

```
Î”w_ij = Î· Â· a_i Â· a_j Â· gate
```

### 4.2 Decay (Forgetting)

* Node strength decay
* Edge weight decay
* Multiâ€‘timescale decay

### 4.3 Homeostasis

* Weight norm constraints
* Activity targets

### 4.4 Habituation & Sensitization

| Condition                    | Effect                       |
| ---------------------------- | ---------------------------- |
| Repeated accurate prediction | â†“ learning rate, â†“ salience  |
| Sudden error spike           | â†‘ learning rate, â†‘ attention |

---

## 5. Dual Processing: Topâ€‘Down Prediction Loop

### Bottomâ€‘Up

* BBP embedding â†’ nearest prototype

### Topâ€‘Down

* Prototype predicts expected features
* Decoder or identity mapping

### Error

* Prediction error drives:

  * Learning rate
  * Attention dwell time
  * Prototype updates

---

## 6. Attention Model (Consciousness Analogue)

* Compute priority = novelty Ã— error Ã— motion
* WTA selection (topâ€‘1)
* Inhibitionâ€‘ofâ€‘return
* Only attended item gets full plasticity

This enforces **serial symbolic binding** atop parallel perception.

---

## 7. NetworkX Integration

### Purpose

* Reference implementation
* Debugging
* Visualization

### Node Types

* Feature
* Part
* ObjectPrototype
* MotionPrototype
* Category

### Edge Types

* co_occurs
* part_of
* predicts
* transitions

### Update Cycle

1. Decay
2. Assignment
3. Hebbian update
4. Prune

---

## 8. Minimal Success Criteria (Milestones)

1. Prototypes form and stabilize
2. Old prototypes decay
3. Graph remains sparse
4. Attention selects one object
5. Prediction error decreases for familiar objects

---

## 9. Engineering Handoff Notes

### Recommended Stack

* Python
* PyTorch
* Ultralytics YOLO
* NetworkX
* OpenCV

### Repo Layout (Suggested)

```
perception/
  bbp.py
  yolo_adapter.py
features/
  wta_layer.py
  hebbian.py
objects/
  prototypes.py
tracking/
  tracker.py
attention/
  scheduler.py
graph/
  percept_graph.py
tests/
```

---

## 10. What This System Is / Is Not

**Is**

* Continual
* Online
* Preâ€‘symbolic â†’ symbolic
* Biologically inspired but engineered

**Is Not**

* Endâ€‘toâ€‘end supervised retraining
* Labelâ€‘centric
* Datasetâ€‘bound

---

**End of document â€” ready for implementation and iterative refinement.**

