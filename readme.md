# Online Perceptual Learning with YOLO BBPs

## Full Development Plan & Engineering Handoff Document

**Status:** Consolidated, end‑to‑end design and build roadmap
**Audience:** Research engineers / systems engineers
**Goal:** Implement an online, continual, biologically inspired perceptual learning system using YOLO Bounding Box Percepts (BBPs), Hebbian learning, habituation/sensitization, WTA competition, top‑down prediction, and a cascading perceptual DAG analogous to the human visual pathway.

---

## Quickstart (Phase 1-5: Video → YOLO → BBPs + Attention + Tracking + Sparse Codes)

This repo now includes a minimal **Phase-1 scaffold**:

- `perception/bbp.py`: `BBP` + `BoundingBox` data model
- `perception/video.py`: video/camera frame iterator (OpenCV)
- `perception/yolo_adapter.py`: Ultralytics YOLO adapter → BBPs
- `attention/scheduler.py`: WTA attention + inhibition-of-return (Phase 2)
- `tracking/world_model.py`: lightweight object permanence tracker (Stage 3)
- `vision/retina.py`: foveated sampling + sensory buffer (Stage 4)
- `features/wta_layer.py`: WTA + Hebbian/decay sparse codes (Stage 5)
- `scripts/run_bbp_stream.py`: CLI to stream BBPs and optionally write JSONL
- `scripts/run_tracker_stream.py`: CLI to visualize tracked objects

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

### Visualize tracking (webcam or OBS virtual camera)

```bash
python scripts/run_tracker_stream.py --source 0 --max-frames 300
```

### Run a session log (JSONL)

```bash
python experiments/run.py --source 0 --max-frames 300 --output-dir outputs
```

## Docs

- `docs/HANDOFF.md`: local handoff checklist + reproducibility notes
- `docs/REFRACTOR_PLAN.md`: tiered refactor spec (north star)
- `docs/PHASED_PLAN.md`: canonical staged plan (consolidated)
- `docs/DEBUGGING.md`: debugging + refactor guidance
- `docs/TRIUNE_ARCHITECTURE.md`: consolidated Triune design spec
- `docs/OBS_SETUP.md`: OBS recording setup for studies
- `docs/REFERENCE_REPOS.md`: reference repos/libraries to fork or borrow from

## 0. Executive Summary

This system treats **YOLO detections as attentional percepts**, not labels. Each detection becomes a **Bounding Box Percept (BBP)**—a transient sensory hypothesis that feeds an **online learning pipeline**. Over time, BBPs are bound into tracks, tracks form object prototypes, prototypes organize into categories, and associations form a **dynamic percept graph**.

Learning is:

* **Online** (no offline retraining loops)
* **Local** (Hebbian / prediction‑error gated updates)
* **Sparse** (WTA + inhibition)
* **Continual** (decay + metaplasticity prevent catastrophic forgetting)

The system implements **dual processing**:

* **Bottom‑up recognition** (matching BBPs to learned prototypes)
* **Top‑down prediction** (prototypes predict expected lower‑level features)

Attention acts as a **winner‑take‑most scheduler**, enforcing a single (or very small) conscious processing stream while allowing background stabilization.

---

## 1. Core Concepts & Definitions

### 1.1 Bounding Box Percept (BBP)

A BBP is the atomic perceptual unit emitted per frame.

**BBP = localized, time‑indexed percept hypothesis**

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
  ↓
BBPs (YOLO)
  ↓
Low‑level features (V1‑like)
  ↓
Parts (co‑activated features)
  ↓
Object prototypes (identity)
  ↓
Categories / scenes / concepts
```

Each layer:

* Competes internally (WTA)
* Learns via local Hebbian / error‑gated updates
* Predicts the layer below (top‑down)

---

### 1.3 Dual Processing

Each prototype acts as:

1. **Recognizer** – explains incoming percepts
2. **Generator** – predicts expected features

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

* Salience computation (Δ appearance, Δ motion, prediction error)
* WTA selection (1 object per attention tick)
* Inhibition‑of‑return (prevents fixation)

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
* Optional recurrence (GRU/LSTM → SNN later)

### 2.5 Graph Memory

* Dynamic DAG (NetworkX initially)
* Nodes: features, parts, objects, motions, categories
* Edges: co‑occurs, part‑of, predicts, transitions

---

## 3. Development Phases (Strict Order)

### Phase 1 — Frame Pipeline & BBP Generator

**Goal:** Deterministic, stable percept stream

**Deliverables**

* Video → frames
* YOLO → boxes
* BBP data structure

**Tests**

* Determinism
* Throughput
* Memory bounds

---

### Phase 2 — Static Feature Learning (No Tracking)

**Goal:** Learn sparse, stable visual features

**Mechanisms**

* Patch sampling from BBPs
* WTA competition
* Hebbian / predictive‑coding updates
* Weight normalization + homeostasis

**Tests**

* Feature diversity
* Sparsity targets
* No collapse / no divergence

---

### Phase 3 — Part Formation

**Goal:** Build mid‑level structure

**Mechanisms**

* Co‑activation Hebbian learning
* Part‑of edges

**Tests**

* Parts recur across exemplars
* Parts generalize across viewpoints

---

### Phase 4 — Object Prototype Formation (Static Identity)

**Goal:** Stable object identity without motion learning

**Mechanisms**

* Online prototype clustering
* Error‑gated averaging
* Novelty‑based spawning
* Habituation

**Tests**

* Same object → same prototype
* Prototype count bounded
* Familiarity reduces error

---

### Phase 5 — Tracking & Slowness (First Temporal Learning)

**Goal:** Object permanence and invariance

**Mechanisms**

* SORT / DeepSORT
* Trace/slowness rules
* Temporal Hebbian edges

**Tests**

* Reduced representation drift
* Occlusion tolerance

---

### Phase 6 — Motion Prototypes

**Goal:** Learn motion as first‑class percepts

**Mechanisms**

* Motion vectors (Δx, Δy, Δscale)
* Motion prototype clustering
* Object ↔ motion associations

**Tests**

* Distinct motion classes
* Motion generalizes across objects

---

### Phase 7 — Recurrence (Optional, Only If Needed)

#### 7a. GRU / LSTM

* Per‑track temporal buffers
* Predict next embedding / motion

#### 7b. Spiking / STDP (Advanced)

* Event‑driven temporal binding
* WTA + homeostatic STDP

**Entry condition:** Trace/slowness insufficient

---

### Phase 8 — Category & Concept Formation

**Goal:** Higher‑level abstraction

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
Δw_ij = η · a_i · a_j · gate
```

### 4.2 Decay (Forgetting)

* Node strength decay
* Edge weight decay
* Multi‑timescale decay

### 4.3 Homeostasis

* Weight norm constraints
* Activity targets

### 4.4 Habituation & Sensitization

| Condition                    | Effect                       |
| ---------------------------- | ---------------------------- |
| Repeated accurate prediction | ↓ learning rate, ↓ salience  |
| Sudden error spike           | ↑ learning rate, ↑ attention |

---

## 5. Dual Processing: Top‑Down Prediction Loop

### Bottom‑Up

* BBP embedding → nearest prototype

### Top‑Down

* Prototype predicts expected features
* Decoder or identity mapping

### Error

* Prediction error drives:

  * Learning rate
  * Attention dwell time
  * Prototype updates

---

## 6. Attention Model (Consciousness Analogue)

* Compute priority = novelty × error × motion
* WTA selection (top‑1)
* Inhibition‑of‑return
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
* Pre‑symbolic → symbolic
* Biologically inspired but engineered

**Is Not**

* End‑to‑end supervised retraining
* Label‑centric
* Dataset‑bound

---

**End of document — ready for implementation and iterative refinement.**
