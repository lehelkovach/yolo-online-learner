# Comprehensive Development Plan: yolo-online-learner

**Author:** Claude Opus 4.5  
**Date:** January 14, 2026  
**Status:** Master Implementation Roadmap  
**Scope:** Complete development plan covering all phases from current state to research-ready system

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Architecture Vision](#3-architecture-vision)
4. [Priority-Ranked Implementation Roadmap](#4-priority-ranked-implementation-roadmap)
5. [Deep Dive: Object Permanence (Stage 3.5)](#5-deep-dive-object-permanence-stage-35)
6. [Deep Dive: Attention System (Stage 2)](#6-deep-dive-attention-system-stage-2)
7. [Deep Dive: Feature Learning (Stage 3)](#7-deep-dive-feature-learning-stage-3)
8. [Deep Dive: Prototype Formation (Stage 4)](#8-deep-dive-prototype-formation-stage-4)
9. [Deep Dive: Predictive Coding (Stage 5)](#9-deep-dive-predictive-coding-stage-5)
10. [Deep Dive: Habituation & Sensitization (Stage 6)](#10-deep-dive-habituation--sensitization-stage-6)
11. [Deep Dive: Graph Memory (Stage 7)](#11-deep-dive-graph-memory-stage-7)
12. [Deep Dive: Working Memory (Stage 8)](#12-deep-dive-working-memory-stage-8)
13. [Deep Dive: Motion Prototypes (Stage 9)](#13-deep-dive-motion-prototypes-stage-9)
14. [Deep Dive: SNN & RL (Stage 10)](#14-deep-dive-snn--rl-stage-10)
15. [Infrastructure & Code Quality](#15-infrastructure--code-quality)
16. [Research Milestones & Paper Targets](#16-research-milestones--paper-targets)
17. [Testing Strategy](#17-testing-strategy)
18. [Summary & Immediate Actions](#18-summary--immediate-actions)

---

## 1. Executive Summary

The `yolo-online-learner` project aims to build an **online, continual, biologically-inspired perceptual learning system**. The system treats YOLO detections as transient sensory hypotheses (BBPs) that feed into a hierarchy of learning mechanisms—from low-level feature extraction to high-level category formation.

### Current Reality vs. Vision

| Aspect | Current State | Target State |
|--------|--------------|--------------|
| **Perception** | YOLO → BBPs (working) | Multi-scale, multi-modal |
| **Tracking** | None | Kalman-based object permanence |
| **Attention** | None (scaffolded) | WTA with inhibition-of-return |
| **Features** | None (scaffolded) | WTA + Hebbian sparse codes |
| **Prototypes** | None (scaffolded) | Online clustering with decay |
| **Memory** | NetworkX graph (minimal) | Full percept graph with decay |
| **Learning** | None | Hebbian + predictive coding |

### Key Insight: The Amnesia Problem

The most critical gap is **Episodic Object Permanence**. Without tracking, the system cannot:
- Maintain object identity across frames
- Learn associations between the "same" object over time
- Handle occlusions (objects "disappear" when hidden)

This document provides a complete roadmap to address all gaps systematically.

---

## 2. Current State Assessment

### 2.1 What's Implemented (Working)

| Component | File(s) | Status | Notes |
|-----------|---------|--------|-------|
| BBP data model | `perception/bbp.py` | **Complete** | Frozen dataclass, JSON serializable |
| Bounding box | `perception/bbp.py` | **Complete** | IoU, clipping, area calculations |
| Video ingestion | `perception/video.py` | **Complete** | OpenCV, supports files + cameras |
| YOLO adapter | `perception/yolo_adapter.py` | **Complete** | Ultralytics wrapper, BBP generation |
| Experiment runner | `experiments/run.py` | **Complete** | JSONL session logging |
| Config | `experiments/config.py` | **Complete** | Minimal, stable schema |
| Graph scaffold | `graph/percept_graph.py` | **Partial** | Basic NetworkX wrapper, decay only |
| BBP tests | `tests/test_bbp.py` | **Complete** | IoU, serialization roundtrip |

### 2.2 What's Scaffolded (Empty)

| Component | Directory | Status | Priority |
|-----------|-----------|--------|----------|
| Tracking | `tracking/` | **Empty** | **Critical** |
| Attention | `attention/` | **Empty** | High |
| Features | `features/` | **Empty** | High |
| Objects/Prototypes | `objects/` | **Empty** | High |

### 2.3 Code Quality Assessment

**Strengths:**
- Clean dataclass design with `frozen=True` and `slots=True`
- Proper type hints throughout
- JSONL logging for reproducibility
- Poetry for dependency management
- Ruff linting configured

**Weaknesses:**
- Only 3 tests (all for BBP)
- No CI/CD pipeline
- No integration tests
- No benchmarks or performance tests
- Empty module directories
- No mypy strict mode

### 2.4 Dependency Analysis

```
Current dependencies:
├── numpy >= 1.26 (core)
├── networkx >= 3.2 (graph)
├── opencv-python >= 4.10 (vision, optional)
└── ultralytics >= 8.2 (vision, optional)

Missing for full implementation:
├── scipy (Kalman filter, Hungarian algorithm)
├── torch (embeddings, optional SNN)
└── filterpy (alternative Kalman implementation)
```

---

## 3. Architecture Vision

### 3.1 Target Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PERCEPTION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Camera/Video → Frames → YOLO → BBPs (raw detections)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRACKING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  BBPs → WorldModel → TrackedObjects (with Kalman state)                     │
│                   → Ghost Buffer (occluded predictions)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ATTENTION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  TrackedObjects → Salience Computation → WTA Selection → Attended Object    │
│                                       → Inhibition-of-Return                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FEATURE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Attended Object → Patch Sampling → WTA Sparse Coding → Feature Activations │
│                 → Hebbian Learning (weight updates)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PROTOTYPE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Feature Activations → Prototype Matching → Best Match OR New Prototype     │
│                     → Online Averaging (update prototype)                    │
│                     → Novelty Detection (spawn on surprise)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PREDICTIVE CODING LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Prototype → Predicts Expected Features → Prediction Error                  │
│           → Error Gates Learning Rate                                        │
│           → Error Modulates Attention                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Working Memory (K active objects) ←→ Percept Graph (long-term)             │
│  Habituation/Sensitization (gain modulation)                                 │
│  Decay + Pruning (prevents unbounded growth)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Interfaces

Each layer should implement a consistent `step()` interface:

```python
class LayerInterface(Protocol):
    """Common interface for all processing layers."""
    
    def step(self, inputs: Any, frame_idx: int, timestamp_s: float) -> Any:
        """Process one frame of inputs, return outputs."""
        ...
    
    def get_state(self) -> dict:
        """Return serializable state for logging."""
        ...
    
    def reset(self) -> None:
        """Reset layer state (e.g., for new session)."""
        ...
```

---

## 4. Priority-Ranked Implementation Roadmap

### 4.1 Revised Stage Order

The original PHASED_PLAN.md puts tracking at Stage 9. This is too late. **Object permanence is foundational** for all downstream learning.

**New Priority Order:**

| Priority | Stage | Name | Blocks | Est. Effort |
|----------|-------|------|--------|-------------|
| **P0** | 0 | Experiment harness | - | Done |
| **P0** | 1 | BBP generator | - | Done |
| **P1** | **3.5** | **Object Permanence** | Stages 4-10 | 2-3 days |
| **P1** | 2 | Attention scheduler | Stages 4-10 | 1-2 days |
| **P2** | 3 | Simple embeddings | Stage 4 | 1 day |
| **P2** | 4 | Prototype bank | Stage 5 | 2-3 days |
| **P3** | 5 | Predictive coding | Stage 6 | 2-3 days |
| **P3** | 6 | Habituation/sensitization | Stage 7 | 1-2 days |
| **P4** | 7 | Graph memory | - | 2-3 days |
| **P4** | 8 | Working memory | - | 2-3 days |
| **P5** | 9 | Motion prototypes | - | 2-3 days |
| **P5** | 10 | SNN + RL | - | 1-2 weeks |

### 4.2 Dependency Graph

```
Stage 0 (Harness) ─────────────────────────────────────────────┐
       │                                                        │
       ▼                                                        │
Stage 1 (BBPs) ────────────────────────────────────────────────┤
       │                                                        │
       ├─────────────────┬──────────────────────────────────────┤
       ▼                 ▼                                      │
Stage 3.5 (Tracking)   Stage 2 (Attention)                      │
       │                 │                                      │
       └────────┬────────┘                                      │
                ▼                                               │
          Stage 3 (Embeddings)                                  │
                │                                               │
                ▼                                               │
          Stage 4 (Prototypes)                                  │
                │                                               │
                ▼                                               │
          Stage 5 (Predictive Coding)                           │
                │                                               │
                ▼                                               │
          Stage 6 (Habituation) ────────────────────────────────┤
                │                                               │
                ├─────────────────┬─────────────────────────────┤
                ▼                 ▼                             │
          Stage 7 (Graph)   Stage 8 (Working Memory)            │
                │                 │                             │
                └────────┬────────┘                             │
                         ▼                                      │
                   Stage 9 (Motion) ────────────────────────────┤
                         │                                      │
                         ▼                                      │
                   Stage 10 (SNN/RL) ───────────────────────────┘
```

### 4.3 Implementation Sprints

**Sprint 1: Foundation (Week 1)**
- [ ] Stage 3.5: Object Permanence (WorldModel, Kalman)
- [ ] Stage 2: Attention Scheduler (WTA, IoR)
- [ ] Tests for both
- [ ] Integration with experiment runner

**Sprint 2: Learning Core (Week 2)**
- [ ] Stage 3: Simple embeddings
- [ ] Stage 4: Prototype bank
- [ ] Stage 5: Predictive coding loop
- [ ] Metrics: prototype count, prediction error

**Sprint 3: Memory & Dynamics (Week 3)**
- [ ] Stage 6: Habituation/sensitization
- [ ] Stage 7: Graph memory with decay
- [ ] Stage 8: Working memory
- [ ] Metrics: graph sparsity, WM utilization

**Sprint 4: Advanced (Week 4+)**
- [ ] Stage 9: Motion prototypes
- [ ] Stage 10: SNN exploration (optional)
- [ ] Paper-ready metrics and visualizations

---

## 5. Deep Dive: Object Permanence (Stage 3.5)

### 5.1 The Amnesia Problem

**Current behavior:** Each frame is processed independently. When YOLO fails to detect an object (occlusion, blur, false negative), the system has no mechanism to:
1. Predict where the object should be
2. Maintain the object's identity
3. Distinguish "temporarily occluded" from "truly gone"

**Cognitive parallel:** Human infants develop object permanence around 8-12 months. The current system operates at 4-month-old level—if hidden, it's gone.

### 5.2 Solution: Ghost Buffer Architecture

```
Frame → YOLO → BBPs → Association → WorldModel → TrackedObjects
                          ↑              ↓
                    Kalman Predict ← Ghost Buffer
```

### 5.3 Core Implementation

#### TrackedObject State Machine

```python
# tracking/tracked_object.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

class ObjectState(Enum):
    VISIBLE = "visible"      # Currently detected
    OCCLUDED = "occluded"    # Predicted to exist, not detected
    LOST = "lost"            # Exceeded max occlusion time

@dataclass
class TrackedObject:
    """Persistent object with Kalman state estimation."""
    
    object_id: str
    class_id: int
    
    # Kalman state: [x, y, scale, aspect, vx, vy, vs, va]
    state: np.ndarray
    covariance: np.ndarray
    
    # Temporal tracking
    frames_visible: int = 0
    frames_occluded: int = 0
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    
    # State machine
    visibility: ObjectState = ObjectState.VISIBLE
    
    # Associated detection
    last_bbp: Optional["BBP"] = None
    
    # Appearance embedding for re-ID
    embedding: Optional[np.ndarray] = None
    
    @property
    def predicted_bbox(self) -> tuple[float, float, float, float]:
        """Convert Kalman state to xyxy bounding box."""
        x, y, s, a = self.state[:4]
        w = np.sqrt(max(s, 1) * max(a, 0.1))
        h = max(s, 1) / w if w > 0 else 1
        return (x - w/2, y - h/2, x + w/2, y + h/2)
```

#### Kalman Filter

```python
# tracking/kalman.py

import numpy as np
from typing import Tuple

class BBoxKalmanFilter:
    """
    Kalman Filter for bounding box tracking.
    
    State: [x, y, s, a, vx, vy, vs, va]
    - (x, y): center position
    - s: scale (area)
    - a: aspect ratio (w/h)
    - (vx, vy, vs, va): velocities
    
    Based on SORT tracker formulation.
    """
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float]):
        x1, y1, x2, y2 = initial_bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        s = w * h
        a = w / h if h > 0 else 1.0
        
        self.state = np.array([x, y, s, a, 0, 0, 0, 0], dtype=np.float64)
        
        # Constant velocity transition
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        
        # Observe position only
        self.H = np.eye(4, 8)
        
        # Process noise
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01
        
        # Measurement noise (scale/aspect are noisier)
        self.R = np.diag([1, 1, 10, 10])
        
        # Initial covariance
        self.P = np.eye(8)
        self.P[4:, 4:] *= 1000
    
    def predict(self) -> np.ndarray:
        """Predict next state. Call every frame."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy()
    
    def update(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Update with measurement. Call only when detected."""
        x1, y1, x2, y2 = bbox
        z = np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            (x2 - x1) * (y2 - y1),
            (x2 - x1) / (y2 - y1) if y2 > y1 else 1.0
        ])
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z - self.H @ self.state)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        
        return self.state.copy()
    
    def state_to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert state to xyxy box."""
        x, y, s, a = self.state[:4]
        s, a = max(s, 1), max(a, 0.1)
        w = np.sqrt(s * a)
        h = s / w
        return (x - w/2, y - h/2, x + w/2, y + h/2)
    
    def mahalanobis(self, bbox: Tuple[float, float, float, float]) -> float:
        """Mahalanobis distance for gating."""
        x1, y1, x2, y2 = bbox
        z = np.array([
            (x1 + x2) / 2, (y1 + y2) / 2,
            (x2 - x1) * (y2 - y1),
            (x2 - x1) / (y2 - y1) if y2 > y1 else 1.0
        ])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        return float(np.sqrt(y @ np.linalg.inv(S) @ y))
```

#### WorldModel

```python
# tracking/world_model.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class WorldModelConfig:
    max_occlusion_frames: int = 30
    min_hits_to_confirm: int = 3
    iou_threshold: float = 0.3
    max_mahalanobis: float = 9.49  # Chi-squared 95%, 4 DOF

class WorldModel:
    """Central state manager for object permanence."""
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        self.config = config or WorldModelConfig()
        self.objects: Dict[str, TrackedObject] = {}
        self.filters: Dict[str, BBoxKalmanFilter] = {}
        self._next_id = 0
    
    def step(self, bbps: List[BBP], frame_idx: int, timestamp_s: float
            ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """Process frame. Returns (active, lost) objects."""
        
        # 1. Predict all existing objects
        for kf in self.filters.values():
            kf.predict()
        
        # 2. Associate BBPs to objects
        matched, unmatched_bbps, unmatched_objs = self._associate(bbps)
        
        # 3. Update matched
        for obj_id, bbp in matched:
            self._update(obj_id, bbp, frame_idx, timestamp_s)
        
        # 4. Mark unmatched as occluded
        for obj_id in unmatched_objs:
            self._mark_occluded(obj_id)
        
        # 5. Birth new objects
        for bbp in unmatched_bbps:
            self._birth(bbp, frame_idx, timestamp_s)
        
        # 6. Reap lost objects
        lost = self._reap()
        
        return (list(self.objects.values()), lost)
    
    def _associate(self, bbps):
        """Greedy IoU + Mahalanobis association."""
        # ... (full implementation in code)
        pass
    
    def get_ghosts(self) -> List[TrackedObject]:
        """Return occluded objects with predicted positions."""
        return [o for o in self.objects.values() 
                if o.visibility == ObjectState.OCCLUDED]
```

### 5.4 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| ID persistence through occlusion | 30 frames | Synthetic test |
| Prediction accuracy | < 20px error | Compare predicted vs actual |
| ID switch rate | < 2/100 frames | Count re-assignments |
| Ghost precision | > 70% | Verify ghosts actually exist |

---

## 6. Deep Dive: Attention System (Stage 2)

### 6.1 Purpose

The attention system implements a **serial bottleneck**—only one (or very few) objects get full processing per frame. This:
- Enforces sparsity
- Prevents catastrophic interference
- Models biological attention constraints
- Creates a "conscious" processing stream

### 6.2 Components

#### Salience Computation

```python
# attention/salience.py

import numpy as np
from tracking.tracked_object import TrackedObject, ObjectState

def compute_salience(obj: TrackedObject, frame_idx: int) -> float:
    """
    Compute attention priority for an object.
    
    Factors:
    - Motion: faster objects are more salient
    - Novelty: recently appeared objects are more salient
    - Size: larger objects are more salient
    - Occlusion curiosity: occluded objects get periodic "check"
    """
    # Motion salience
    vx, vy = obj.velocity
    motion = np.sqrt(vx**2 + vy**2) / 50.0  # Normalize
    
    # Novelty (inverse of familiarity)
    novelty = 1.0 / (1 + obj.frames_visible / 30.0)
    
    # Size salience (larger = more salient)
    bbox = obj.predicted_bbox
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    size_salience = np.sqrt(area) / 200.0  # Normalize
    
    # Occlusion curiosity (peaks then decays)
    curiosity = 0.0
    if obj.visibility == ObjectState.OCCLUDED:
        t = obj.frames_occluded
        curiosity = t * np.exp(-t / 10.0)  # Peak at ~10 frames
    
    return motion + novelty + size_salience + curiosity
```

#### Winner-Take-All Selection

```python
# attention/scheduler.py

from dataclasses import dataclass, field
from typing import List, Optional, Set
import numpy as np

@dataclass
class AttentionConfig:
    k: int = 1  # Number of winners
    ior_frames: int = 5  # Inhibition-of-return duration
    ior_decay: float = 0.8  # IoR strength decay per frame

class AttentionScheduler:
    """Winner-take-all attention with inhibition-of-return."""
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self.ior_map: Dict[str, float] = {}  # object_id -> inhibition strength
        self.last_attended: Optional[str] = None
    
    def step(self, objects: List[TrackedObject], frame_idx: int
            ) -> List[TrackedObject]:
        """Select top-k objects to attend."""
        
        # Decay IoR
        self._decay_ior()
        
        # Compute effective salience (raw - IoR)
        scores = []
        for obj in objects:
            raw = compute_salience(obj, frame_idx)
            ior = self.ior_map.get(obj.object_id, 0.0)
            effective = max(0, raw - ior)
            scores.append((effective, obj))
        
        # Sort and select top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        winners = [obj for _, obj in scores[:self.config.k]]
        
        # Apply IoR to winners
        for obj in winners:
            self.ior_map[obj.object_id] = 1.0
        
        self.last_attended = winners[0].object_id if winners else None
        return winners
    
    def _decay_ior(self):
        """Decay all inhibition values."""
        to_remove = []
        for obj_id, strength in self.ior_map.items():
            new_strength = strength * self.config.ior_decay
            if new_strength < 0.01:
                to_remove.append(obj_id)
            else:
                self.ior_map[obj_id] = new_strength
        for obj_id in to_remove:
            del self.ior_map[obj_id]
    
    def get_state(self) -> dict:
        return {
            "last_attended": self.last_attended,
            "ior_active": len(self.ior_map),
        }
```

### 6.3 Tests

```python
def test_wta_selects_one():
    """Exactly one object should be selected."""
    scheduler = AttentionScheduler(AttentionConfig(k=1))
    objects = [make_tracked_object(f"obj_{i}") for i in range(5)]
    winners = scheduler.step(objects, frame_idx=0)
    assert len(winners) == 1

def test_ior_prevents_fixation():
    """Same object shouldn't win consecutive frames."""
    scheduler = AttentionScheduler(AttentionConfig(k=1, ior_frames=3))
    objects = [make_tracked_object(f"obj_{i}") for i in range(3)]
    
    attended = []
    for frame in range(10):
        winners = scheduler.step(objects, frame)
        attended.append(winners[0].object_id)
    
    # Should cycle through objects, not fixate
    assert len(set(attended)) > 1
```

---

## 7. Deep Dive: Feature Learning (Stage 3)

### 7.1 Purpose

Learn sparse, reusable visual features from BBP crops using:
- Winner-take-all competition (sparsity)
- Hebbian learning (local updates)
- Weight normalization (stability)

### 7.2 Simple Embedding First

Before neural features, use geometric embeddings:

```python
# features/embeddings.py

import numpy as np
from perception.bbp import BBP

def bbox_embedding(bbp: BBP, frame_width: int, frame_height: int) -> np.ndarray:
    """
    Simple geometric embedding for a BBP.
    
    Features:
    - Normalized center (x, y) in [0, 1]
    - Normalized size (w, h) in [0, 1]
    - Aspect ratio
    - Area fraction
    - Class one-hot (if available)
    """
    bbox = bbp.bbox
    cx = (bbox.x1 + bbox.x2) / 2 / frame_width
    cy = (bbox.y1 + bbox.y2) / 2 / frame_height
    w = bbox.w / frame_width
    h = bbox.h / frame_height
    aspect = w / h if h > 0 else 1.0
    area_frac = bbox.area / (frame_width * frame_height)
    
    return np.array([cx, cy, w, h, aspect, area_frac], dtype=np.float32)
```

### 7.3 WTA Sparse Coding Layer

```python
# features/wta_layer.py

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class WTAConfig:
    input_dim: int = 6
    num_units: int = 64
    k_winners: int = 4
    learning_rate: float = 0.01
    weight_decay: float = 0.001

class WTALayer:
    """
    Winner-take-all sparse coding layer.
    
    Units compete; top-k winners update weights via Hebbian learning.
    """
    
    def __init__(self, config: WTAConfig):
        self.config = config
        # Initialize weights (normalized rows)
        self.W = np.random.randn(config.num_units, config.input_dim)
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)
        
        # Activity tracking for homeostasis
        self.activity_ema = np.ones(config.num_units) / config.num_units
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sparse activation.
        
        Returns: sparse vector with k non-zero entries.
        """
        # Normalize input
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        
        # Compute activations (dot product similarity)
        activations = self.W @ x_norm
        
        # Winner-take-all: keep top-k
        k = self.config.k_winners
        threshold = np.partition(activations, -k)[-k]
        sparse = np.where(activations >= threshold, activations, 0.0)
        
        return sparse
    
    def learn(self, x: np.ndarray, sparse: np.ndarray) -> None:
        """
        Hebbian update for winners.
        
        Δw_i = η * a_i * (x - w_i)  for winners
        """
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        lr = self.config.learning_rate
        
        winners = np.where(sparse > 0)[0]
        for i in winners:
            # Oja's rule: moves weight toward input, maintains normalization
            self.W[i] += lr * sparse[i] * (x_norm - sparse[i] * self.W[i])
        
        # Weight decay
        self.W *= (1 - self.config.weight_decay)
        
        # Re-normalize
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-8
        
        # Update activity EMA for homeostasis
        active = (sparse > 0).astype(float)
        self.activity_ema = 0.99 * self.activity_ema + 0.01 * active
    
    def get_state(self) -> dict:
        return {
            "weight_norms": np.linalg.norm(self.W, axis=1).tolist(),
            "activity_ema": self.activity_ema.tolist(),
            "sparsity": float(np.mean(self.activity_ema > 0.01)),
        }
```

### 7.4 Tests

```python
def test_wta_sparsity():
    """Output should be sparse (exactly k winners)."""
    layer = WTALayer(WTAConfig(input_dim=6, num_units=64, k_winners=4))
    x = np.random.randn(6)
    sparse = layer.forward(x)
    assert np.sum(sparse > 0) == 4

def test_wta_learning_converges():
    """Repeated inputs should strengthen their representations."""
    layer = WTALayer(WTAConfig(input_dim=6, num_units=32, k_winners=2))
    
    # Single input pattern
    x = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
    
    initial_response = layer.forward(x).max()
    for _ in range(100):
        sparse = layer.forward(x)
        layer.learn(x, sparse)
    final_response = layer.forward(x).max()
    
    assert final_response > initial_response
```

---

## 8. Deep Dive: Prototype Formation (Stage 4)

### 8.1 Purpose

Build stable **object identity** representations:
- Cluster similar feature activations
- Online update (no batching)
- Bound prototype count
- Novelty-based spawning

### 8.2 Implementation

```python
# objects/prototypes.py

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class Prototype:
    """A learned object prototype."""
    prototype_id: str
    centroid: np.ndarray
    count: int = 1
    last_seen: int = 0
    cumulative_error: float = 0.0

@dataclass
class PrototypeConfig:
    dim: int = 64
    max_prototypes: int = 100
    spawn_threshold: float = 0.5  # Min distance to spawn new
    merge_threshold: float = 0.1  # Max distance to merge
    decay_rate: float = 0.001
    learning_rate: float = 0.1

class PrototypeBank:
    """
    Online prototype learning with bounded count.
    """
    
    def __init__(self, config: PrototypeConfig):
        self.config = config
        self.prototypes: Dict[str, Prototype] = {}
        self._next_id = 0
    
    def step(self, embedding: np.ndarray, frame_idx: int
            ) -> Tuple[Prototype, float, bool]:
        """
        Match or spawn prototype.
        
        Returns: (matched_prototype, distance, is_novel)
        """
        if not self.prototypes:
            # First prototype
            proto = self._spawn(embedding, frame_idx)
            return (proto, 0.0, True)
        
        # Find nearest prototype
        best_proto, best_dist = self._find_nearest(embedding)
        
        if best_dist > self.config.spawn_threshold:
            # Too far from all prototypes -> spawn new
            if len(self.prototypes) < self.config.max_prototypes:
                proto = self._spawn(embedding, frame_idx)
                return (proto, best_dist, True)
            else:
                # At capacity -> force match to nearest
                self._update(best_proto, embedding, frame_idx)
                return (best_proto, best_dist, False)
        else:
            # Close enough -> update existing
            self._update(best_proto, embedding, frame_idx)
            return (best_proto, best_dist, False)
    
    def _find_nearest(self, embedding: np.ndarray) -> Tuple[Prototype, float]:
        best_proto = None
        best_dist = float('inf')
        for proto in self.prototypes.values():
            dist = np.linalg.norm(embedding - proto.centroid)
            if dist < best_dist:
                best_dist = dist
                best_proto = proto
        return (best_proto, best_dist)
    
    def _spawn(self, embedding: np.ndarray, frame_idx: int) -> Prototype:
        self._next_id += 1
        proto_id = f"proto_{self._next_id:04d}"
        proto = Prototype(
            prototype_id=proto_id,
            centroid=embedding.copy(),
            count=1,
            last_seen=frame_idx,
        )
        self.prototypes[proto_id] = proto
        return proto
    
    def _update(self, proto: Prototype, embedding: np.ndarray, frame_idx: int):
        """Online averaging update."""
        lr = self.config.learning_rate
        error = np.linalg.norm(embedding - proto.centroid)
        
        # Move centroid toward new embedding
        proto.centroid = (1 - lr) * proto.centroid + lr * embedding
        proto.count += 1
        proto.last_seen = frame_idx
        proto.cumulative_error += error
    
    def decay(self, frame_idx: int) -> List[str]:
        """Decay and prune old prototypes."""
        removed = []
        for proto_id, proto in list(self.prototypes.items()):
            age = frame_idx - proto.last_seen
            if age > 1000 and proto.count < 10:
                # Old and rarely seen -> remove
                removed.append(proto_id)
                del self.prototypes[proto_id]
        return removed
    
    def get_state(self) -> dict:
        return {
            "num_prototypes": len(self.prototypes),
            "prototype_ids": list(self.prototypes.keys()),
            "avg_count": np.mean([p.count for p in self.prototypes.values()]) if self.prototypes else 0,
        }
```

### 8.3 Tests

```python
def test_prototype_spawning():
    """Novel inputs should spawn new prototypes."""
    bank = PrototypeBank(PrototypeConfig(dim=4, spawn_threshold=0.5))
    
    # First input
    _, _, is_novel = bank.step(np.array([1, 0, 0, 0]), 0)
    assert is_novel
    
    # Similar input
    _, dist, is_novel = bank.step(np.array([1, 0.1, 0, 0]), 1)
    assert not is_novel  # Should match
    
    # Different input
    _, _, is_novel = bank.step(np.array([0, 0, 1, 0]), 2)
    assert is_novel  # Should spawn

def test_prototype_bounded():
    """Prototype count should not exceed max."""
    config = PrototypeConfig(dim=4, max_prototypes=5, spawn_threshold=0.1)
    bank = PrototypeBank(config)
    
    # Add many different inputs
    for i in range(100):
        vec = np.random.randn(4)
        bank.step(vec, i)
    
    assert len(bank.prototypes) <= 5
```

---

## 9. Deep Dive: Predictive Coding (Stage 5)

### 9.1 Purpose

Implement **dual processing**:
- **Bottom-up**: BBP → Features → Best-matching prototype
- **Top-down**: Prototype → Predicts expected features
- **Error**: Drives learning rate, attention, novelty

### 9.2 Implementation

```python
# features/predictive_coding.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PredictiveCodingConfig:
    error_ema_alpha: float = 0.1  # Smoothing for error tracking
    error_learning_gate: float = 0.5  # Error threshold to boost learning
    max_learning_boost: float = 3.0

class PredictiveCodingLoop:
    """
    Predictive coding wrapper around prototype matching.
    
    Prototypes predict expected features; error gates learning.
    """
    
    def __init__(self, config: PredictiveCodingConfig):
        self.config = config
        self.error_ema: float = 0.0
        self.surprise_history: list = []
    
    def step(
        self,
        observation: np.ndarray,
        prediction: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute prediction error and learning rate modifier.
        
        Args:
            observation: Actual feature embedding
            prediction: Prototype's predicted embedding
            
        Returns:
            (error, learning_rate_modifier)
        """
        # Raw prediction error
        error = float(np.linalg.norm(observation - prediction))
        
        # Update error EMA
        self.error_ema = (
            (1 - self.config.error_ema_alpha) * self.error_ema +
            self.config.error_ema_alpha * error
        )
        
        # Surprise: how much higher is this error than expected?
        surprise = error / (self.error_ema + 1e-6)
        self.surprise_history.append(surprise)
        
        # Learning rate modifier
        if error > self.config.error_learning_gate:
            # High error -> boost learning
            lr_mod = min(surprise, self.config.max_learning_boost)
        else:
            # Low error -> normal learning
            lr_mod = 1.0
        
        return (error, lr_mod)
    
    def get_state(self) -> dict:
        return {
            "error_ema": self.error_ema,
            "recent_surprise": self.surprise_history[-10:] if self.surprise_history else [],
        }
```

### 9.3 Integration with Prototypes

```python
# In the main loop:

# Get observation
observation = wta_layer.forward(embedding)

# Get prediction from matched prototype
prediction = matched_prototype.centroid

# Compute error
error, lr_mod = pc_loop.step(observation, prediction)

# Gated learning
effective_lr = base_lr * lr_mod
wta_layer.learn(embedding, observation, lr=effective_lr)
```

---

## 10. Deep Dive: Habituation & Sensitization (Stage 6)

### 10.1 Purpose

Modulate gain based on exposure:
- **Habituation**: Repeated familiar stimuli → reduced response
- **Sensitization**: Surprising stimuli → enhanced response

### 10.2 Implementation

```python
# attention/habituation.py

import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class HabituationConfig:
    habituation_rate: float = 0.05  # How fast to habituate
    sensitization_rate: float = 0.2  # How fast to sensitize
    recovery_rate: float = 0.01  # How fast habituation recovers
    min_gain: float = 0.1
    max_gain: float = 3.0

class HabituationModule:
    """
    Per-object habituation and sensitization.
    """
    
    def __init__(self, config: HabituationConfig):
        self.config = config
        self.gains: Dict[str, float] = {}  # object_id -> gain
    
    def step(self, object_id: str, prediction_error: float) -> float:
        """
        Update gain for object based on prediction error.
        
        Low error -> habituate (reduce gain)
        High error -> sensitize (increase gain)
        
        Returns current gain.
        """
        current_gain = self.gains.get(object_id, 1.0)
        
        # Threshold for "surprising"
        if prediction_error < 0.2:
            # Habituate
            delta = -self.config.habituation_rate * current_gain
        elif prediction_error > 0.5:
            # Sensitize
            delta = self.config.sensitization_rate * (self.config.max_gain - current_gain)
        else:
            # Neutral -> slow recovery toward baseline
            delta = self.config.recovery_rate * (1.0 - current_gain)
        
        new_gain = np.clip(
            current_gain + delta,
            self.config.min_gain,
            self.config.max_gain
        )
        self.gains[object_id] = new_gain
        
        return new_gain
    
    def decay_all(self):
        """Slowly recover all gains toward baseline."""
        for obj_id in self.gains:
            self.gains[obj_id] += self.config.recovery_rate * (1.0 - self.gains[obj_id])
    
    def get_state(self) -> dict:
        return {
            "num_tracked": len(self.gains),
            "mean_gain": np.mean(list(self.gains.values())) if self.gains else 1.0,
            "habituated_count": sum(1 for g in self.gains.values() if g < 0.5),
            "sensitized_count": sum(1 for g in self.gains.values() if g > 1.5),
        }
```

---

## 11. Deep Dive: Graph Memory (Stage 7)

### 11.1 Purpose

The percept graph stores:
- **Nodes**: Features, parts, prototypes, categories
- **Edges**: co_occurs, part_of, predicts, transitions

### 11.2 Enhanced Implementation

```python
# graph/percept_graph.py (enhanced)

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import networkx as nx

@dataclass
class GraphConfig:
    edge_decay_rate: float = 0.01
    min_edge_weight: float = 0.001
    max_edges_per_node: int = 50
    prune_interval: int = 100

class PerceptGraph:
    """
    Dynamic percept graph with decay and pruning.
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.g = nx.DiGraph()
        self.step_count = 0
    
    def add_node(self, node_id: str, node_type: str, **attrs) -> None:
        self.g.add_node(node_id, node_type=node_type, created=self.step_count, **attrs)
    
    def add_edge(self, src: str, dst: str, edge_type: str, weight: float = 1.0) -> None:
        if self.g.has_edge(src, dst):
            # Strengthen existing edge
            self.g[src][dst]['weight'] += weight
        else:
            self.g.add_edge(src, dst, edge_type=edge_type, weight=weight)
    
    def step(self) -> dict:
        """Called each frame. Handles decay and periodic pruning."""
        self.step_count += 1
        
        # Decay all edges
        decayed = self._decay_edges()
        
        # Periodic pruning
        pruned = 0
        if self.step_count % self.config.prune_interval == 0:
            pruned = self._prune()
        
        return {"decayed": decayed, "pruned": pruned}
    
    def _decay_edges(self) -> int:
        rate = self.config.edge_decay_rate
        min_w = self.config.min_edge_weight
        to_remove = []
        
        for u, v, data in self.g.edges(data=True):
            data['weight'] *= (1 - rate)
            if data['weight'] < min_w:
                to_remove.append((u, v))
        
        for u, v in to_remove:
            self.g.remove_edge(u, v)
        
        return len(to_remove)
    
    def _prune(self) -> int:
        """Remove low-degree, old nodes."""
        to_remove = []
        for node in self.g.nodes():
            degree = self.g.degree(node)
            age = self.step_count - self.g.nodes[node].get('created', 0)
            if degree == 0 and age > 500:
                to_remove.append(node)
        
        for node in to_remove:
            self.g.remove_node(node)
        
        return len(to_remove)
    
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """Get neighbors, optionally filtered by edge type."""
        neighbors = []
        for _, dst, data in self.g.out_edges(node_id, data=True):
            if edge_type is None or data.get('edge_type') == edge_type:
                neighbors.append(dst)
        return neighbors
    
    def get_state(self) -> dict:
        return {
            "num_nodes": self.g.number_of_nodes(),
            "num_edges": self.g.number_of_edges(),
            "step_count": self.step_count,
        }
```

---

## 12. Deep Dive: Working Memory (Stage 8)

### 12.1 Purpose

Maintain a **fixed-capacity buffer** of active objects:
- K slots (e.g., 4-7, like human working memory)
- Eviction by utility/recency
- Cue-based retrieval from graph

### 12.2 Implementation

```python
# attention/working_memory.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class WMSlot:
    object_id: str
    embedding: np.ndarray
    last_accessed: int
    access_count: int = 0
    utility: float = 1.0

@dataclass
class WorkingMemoryConfig:
    capacity: int = 5
    decay_rate: float = 0.1
    access_boost: float = 0.5

class WorkingMemory:
    """
    Fixed-capacity working memory with utility-based eviction.
    """
    
    def __init__(self, config: Optional[WorkingMemoryConfig] = None):
        self.config = config or WorkingMemoryConfig()
        self.slots: Dict[str, WMSlot] = {}
        self.step_count = 0
    
    def step(self) -> None:
        """Decay utilities each frame."""
        self.step_count += 1
        for slot in self.slots.values():
            slot.utility *= (1 - self.config.decay_rate)
    
    def access(self, object_id: str, embedding: np.ndarray) -> bool:
        """
        Access or load an object into WM.
        
        Returns True if object was already in WM.
        """
        if object_id in self.slots:
            # Already in WM -> boost utility
            slot = self.slots[object_id]
            slot.utility += self.config.access_boost
            slot.last_accessed = self.step_count
            slot.access_count += 1
            return True
        
        # Need to load
        if len(self.slots) >= self.config.capacity:
            self._evict()
        
        self.slots[object_id] = WMSlot(
            object_id=object_id,
            embedding=embedding.copy(),
            last_accessed=self.step_count,
        )
        return False
    
    def _evict(self) -> str:
        """Evict lowest-utility slot."""
        if not self.slots:
            return None
        
        min_util = float('inf')
        evict_id = None
        for obj_id, slot in self.slots.items():
            if slot.utility < min_util:
                min_util = slot.utility
                evict_id = obj_id
        
        if evict_id:
            del self.slots[evict_id]
        return evict_id
    
    def contains(self, object_id: str) -> bool:
        return object_id in self.slots
    
    def get_contents(self) -> List[str]:
        return list(self.slots.keys())
    
    def get_state(self) -> dict:
        return {
            "capacity": self.config.capacity,
            "occupancy": len(self.slots),
            "contents": [
                {"id": s.object_id, "utility": s.utility, "accesses": s.access_count}
                for s in self.slots.values()
            ],
        }
```

---

## 13. Deep Dive: Motion Prototypes (Stage 9)

### 13.1 Purpose

Learn motion as first-class percepts:
- Cluster motion vectors (Δx, Δy, Δscale)
- Associate motion patterns with objects
- Enable motion-based prediction

### 13.2 Implementation

```python
# tracking/motion_prototypes.py

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class MotionPrototype:
    motion_id: str
    centroid: np.ndarray  # [dx, dy, ds] normalized
    count: int = 0
    associated_objects: set = field(default_factory=set)

class MotionPrototypeBank:
    """
    Cluster motion patterns into reusable prototypes.
    """
    
    def __init__(self, max_prototypes: int = 20, spawn_threshold: float = 0.3):
        self.max_prototypes = max_prototypes
        self.spawn_threshold = spawn_threshold
        self.prototypes: Dict[str, MotionPrototype] = {}
        self._next_id = 0
    
    def classify(self, motion_vec: np.ndarray, object_id: str) -> Tuple[str, float]:
        """
        Classify motion vector, return (motion_id, distance).
        """
        # Normalize motion vector
        norm = np.linalg.norm(motion_vec)
        if norm < 1e-6:
            return ("stationary", 0.0)
        motion_norm = motion_vec / norm
        
        # Find nearest prototype
        best_id, best_dist = None, float('inf')
        for proto in self.prototypes.values():
            dist = np.linalg.norm(motion_norm - proto.centroid)
            if dist < best_dist:
                best_dist = dist
                best_id = proto.motion_id
        
        if best_dist > self.spawn_threshold and len(self.prototypes) < self.max_prototypes:
            # Spawn new
            proto = self._spawn(motion_norm)
            proto.associated_objects.add(object_id)
            return (proto.motion_id, 0.0)
        elif best_id:
            # Match existing
            proto = self.prototypes[best_id]
            proto.count += 1
            proto.associated_objects.add(object_id)
            # Update centroid
            lr = 0.1
            proto.centroid = (1 - lr) * proto.centroid + lr * motion_norm
            proto.centroid /= np.linalg.norm(proto.centroid) + 1e-8
            return (best_id, best_dist)
        else:
            return ("unknown", float('inf'))
    
    def _spawn(self, motion_norm: np.ndarray) -> MotionPrototype:
        self._next_id += 1
        proto = MotionPrototype(
            motion_id=f"motion_{self._next_id:03d}",
            centroid=motion_norm.copy(),
            count=1,
        )
        self.prototypes[proto.motion_id] = proto
        return proto
```

---

## 14. Deep Dive: SNN & RL (Stage 10)

### 14.1 Entry Conditions

Only pursue SNN/RL when:
- Trace/slowness rules are insufficient for temporal learning
- Non-spiking recurrence (GRU/LSTM) has been tried
- Clear metric showing ceiling effect

### 14.2 SNN Options

```python
# Example using snnTorch (if/when needed)

import snntorch as snn
import torch

class SNNRecurrence(torch.nn.Module):
    """
    Spiking recurrent layer for temporal binding.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, hidden_size)
        self.lif = snn.Leaky(beta=0.9)  # Leaky integrate-and-fire
    
    def forward(self, x, mem):
        cur = self.fc(x)
        spk, mem = self.lif(cur, mem)
        return spk, mem
```

### 14.3 RL for Label Association

```python
# Reward-modulated Hebbian learning

def reward_modulated_update(weight, pre, post, reward, lr=0.01):
    """
    Three-factor learning rule: pre * post * reward
    """
    delta = lr * pre * post * reward
    return weight + delta
```

---

## 15. Infrastructure & Code Quality

### 15.1 Testing Improvements

**Current:** 3 tests in `tests/test_bbp.py`

**Target:** Comprehensive test suite

```
tests/
├── test_bbp.py              # Existing
├── test_kalman.py           # Kalman filter unit tests
├── test_world_model.py      # Object permanence tests
├── test_attention.py        # WTA and IoR tests
├── test_wta_layer.py        # Feature learning tests
├── test_prototypes.py       # Prototype bank tests
├── test_habituation.py      # Gain modulation tests
├── test_graph.py            # Graph decay/prune tests
├── test_working_memory.py   # WM capacity tests
├── test_integration.py      # End-to-end pipeline tests
└── conftest.py              # Shared fixtures
```

### 15.2 CI/CD Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Lint
        run: poetry run ruff check .
      - name: Type check
        run: poetry run mypy . --ignore-missing-imports
      - name: Test
        run: poetry run pytest -v --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 15.3 Type Checking

Add mypy to dev dependencies and enable strict mode:

```toml
# pyproject.toml additions
[tool.poetry.group.dev.dependencies]
mypy = ">=1.8"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
```

### 15.4 Documentation

Add docstrings and generate API docs:

```
docs/
├── OPUS_PLAN.md       # This file
├── api/               # Auto-generated API docs
│   ├── perception.md
│   ├── tracking.md
│   └── ...
└── tutorials/
    ├── quickstart.md
    └── custom_embeddings.md
```

### 15.5 Logging Improvements

Structured logging with levels:

```python
# utils/logging.py

import logging
import json
from datetime import datetime

def setup_logging(level: str = "INFO", jsonl_path: str = None):
    """Configure structured logging."""
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
            }
            if hasattr(record, 'metrics'):
                log_entry['metrics'] = record.metrics
            return json.dumps(log_entry)
    
    # ... setup handlers
```

---

## 16. Research Milestones & Paper Targets

### 16.1 Milestone 1: Object Permanence Demo

**Claim:** System maintains object identity through occlusion

**Evidence needed:**
- [ ] ID persistence through 30-frame occlusion
- [ ] Video demo of spinner-covers-button scenario
- [ ] Quantitative: ID switch rate < 2/100 frames
- [ ] Comparison to baseline (no tracking)

### 16.2 Milestone 2: Online Prototype Learning

**Claim:** System learns stable object representations online

**Evidence needed:**
- [ ] Prototype count stabilizes over time
- [ ] Same object → same prototype (intra-object consistency)
- [ ] Different objects → different prototypes (inter-object separation)
- [ ] Forgetting curve for rarely-seen objects

### 16.3 Milestone 3: Predictive Coding Benefits

**Claim:** Prediction error drives efficient learning

**Evidence needed:**
- [ ] Learning rate correlates with prediction error
- [ ] Familiar objects processed faster (habituation)
- [ ] Novel objects get more attention (sensitization)
- [ ] Ablation: remove prediction → learning degrades

### 16.4 Milestone 4: Graph Memory Dynamics

**Claim:** Percept graph captures semantic relationships

**Evidence needed:**
- [ ] Graph sparsity maintained over long runs
- [ ] Meaningful clusters emerge (category formation)
- [ ] Edge weights reflect co-occurrence statistics
- [ ] Visualization of learned graph structure

### 16.5 Paper Structure (Sketch)

```
Title: Online Perceptual Learning with Episodic Object Permanence

1. Introduction
   - YOLO as sensory front-end
   - Gap: detection ≠ understanding
   - Our contribution: online learning with object permanence

2. Related Work
   - Tracking-by-detection
   - Online learning
   - Predictive coding
   - Cognitive architectures

3. Method
   - BBP representation
   - WorldModel + Kalman tracking
   - WTA feature learning
   - Prototype formation
   - Predictive coding loop

4. Experiments
   - Object permanence (occlusion recovery)
   - Prototype stability
   - Habituation curves
   - Graph dynamics

5. Discussion
   - Limitations
   - Future work (SNN, RL)

6. Conclusion
```

---

## 17. Testing Strategy

### 17.1 Unit Test Guidelines

Each module should have:
- **Happy path**: Normal operation
- **Edge cases**: Empty inputs, boundary conditions
- **Determinism**: Fixed seed → same output
- **Performance**: Bounded memory, reasonable time

### 17.2 Integration Test Scenarios

```python
# tests/test_integration.py

def test_full_pipeline_synthetic():
    """Run full pipeline on synthetic BBP sequence."""
    # Generate synthetic BBPs with known occlusion pattern
    bbps_by_frame = generate_synthetic_sequence(
        num_frames=100,
        num_objects=3,
        occlusion_frames=[30, 40],  # Object 0 occluded frames 30-40
    )
    
    # Run pipeline
    world = WorldModel()
    attention = AttentionScheduler()
    prototypes = PrototypeBank()
    
    for frame_idx, bbps in enumerate(bbps_by_frame):
        tracked, _ = world.step(bbps, frame_idx, frame_idx/30.0)
        attended = attention.step(tracked, frame_idx)
        for obj in attended:
            embedding = bbox_embedding(obj.last_bbp)
            proto, error, is_novel = prototypes.step(embedding, frame_idx)
    
    # Verify object permanence
    # Object 0 should have same ID before and after occlusion
    ...

def test_full_pipeline_real_video():
    """Run on real test video with ground truth annotations."""
    # Requires test video + annotation file
    ...
```

### 17.3 Benchmark Suite

```python
# benchmarks/benchmark_tracking.py

def benchmark_world_model_throughput():
    """Measure frames/second for WorldModel."""
    world = WorldModel()
    
    # Warm up
    for i in range(100):
        bbps = [make_random_bbp() for _ in range(10)]
        world.step(bbps, i, i/30.0)
    
    # Benchmark
    import time
    start = time.perf_counter()
    for i in range(1000):
        bbps = [make_random_bbp() for _ in range(10)]
        world.step(bbps, i+100, (i+100)/30.0)
    elapsed = time.perf_counter() - start
    
    fps = 1000 / elapsed
    print(f"WorldModel throughput: {fps:.1f} fps")
    assert fps > 100  # Should handle >100fps easily
```

---

## 18. Summary & Immediate Actions

### 18.1 The Core Transformation

**Before:** YOLO detection = Object existence (memoryless)  
**After:** YOLO detection = Observation; WorldModel = Belief state (persistent)

### 18.2 Immediate Actions (Next 48 Hours)

1. **Create** `tracking/` module:
   - [ ] `tracking/__init__.py`
   - [ ] `tracking/kalman.py`
   - [ ] `tracking/tracked_object.py`
   - [ ] `tracking/world_model.py`

2. **Add tests**:
   - [ ] `tests/test_kalman.py`
   - [ ] `tests/test_world_model.py`

3. **Integrate** into experiment runner:
   - [ ] Update `experiments/config.py` with tracking params
   - [ ] Update `experiments/run.py` to use WorldModel

4. **Validate** with synthetic occlusion test

### 18.3 Success Criteria Checklist

**Object Permanence (P1):**
- [ ] Object ID persists through 30-frame occlusion
- [ ] Predicted bbox within 20px of actual
- [ ] ID switch rate < 2/100 frames

**Attention (P1):**
- [ ] Exactly k objects selected per frame
- [ ] No fixation (IoR working)
- [ ] Logs show attention cycling

**Prototypes (P2):**
- [ ] Prototype count bounded
- [ ] Same object → same prototype
- [ ] Novelty triggers new prototype

**Memory (P3):**
- [ ] Graph edge count bounded
- [ ] Working memory never exceeds K
- [ ] Habituation curves visible in logs

### 18.4 File Creation Checklist

```
New files to create:
├── tracking/
│   ├── __init__.py
│   ├── kalman.py
│   ├── tracked_object.py
│   ├── world_model.py
│   └── association.py
├── attention/
│   ├── __init__.py (populate)
│   ├── salience.py
│   ├── scheduler.py
│   ├── habituation.py
│   └── working_memory.py
├── features/
│   ├── __init__.py (populate)
│   ├── embeddings.py
│   ├── wta_layer.py
│   └── predictive_coding.py
├── objects/
│   ├── __init__.py (populate)
│   └── prototypes.py
├── tests/
│   ├── test_kalman.py
│   ├── test_world_model.py
│   ├── test_attention.py
│   ├── test_wta_layer.py
│   ├── test_prototypes.py
│   └── test_integration.py
└── .github/
    └── workflows/
        └── ci.yml
```

---

**End of Comprehensive Plan**

*This document serves as the master roadmap for transforming `yolo-online-learner` from a detection pipeline into a cognitively-inspired perceptual learning system with true object permanence, attention, and online learning.*
