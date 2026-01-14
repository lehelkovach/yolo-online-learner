# Episodic Object Permanence Implementation Plan

**Author:** Claude Opus 4.5  
**Date:** January 14, 2026  
**Status:** Proposed Architecture Enhancement  
**Scope:** Add persistent object state tracking to address the "amnesia" problem in occluded/missing detections

---

## Executive Summary

The current `yolo-online-learner` system possesses **Semantic Memory** (learning *what* an object is via online prototype formation) but lacks **Episodic Object Permanence** (remembering *where* a specific object instance is when temporarily invisible). This document proposes a phased implementation of a **State Management Layer** that maintains object identity and predicted position during occlusions, using Bayesian state estimation (Kalman Filters) and a "Ghost Buffer" architecture.

---

## 1. Problem Analysis: The Amnesia Gap

### Current Architecture Flow

```
Frame → YOLO → BBPs → [Prototype Matching] → Output
                 ↓
            (No Memory)
```

**Each frame is processed independently.** When YOLO fails to detect an object (due to occlusion, motion blur, or false negatives), the system has no mechanism to:

1. **Predict** where the object should be
2. **Maintain** the object's identity across the gap
3. **Distinguish** between "temporarily occluded" and "truly gone"

### Cognitive Science Perspective

In developmental psychology, **object permanence** is the understanding that objects continue to exist even when they cannot be perceived. Human infants develop this around 8-12 months. The current system operates at the cognitive level of a 4-month-old infant—if the object is hidden, it ceases to exist.

### Engineering Symptoms

| Scenario | Current Behavior | Desired Behavior |
|----------|-----------------|------------------|
| Button covered by spinner | Object "deleted" | Maintain predicted position as "Ghost" |
| Object temporarily exits frame | New ID assigned on return | Same ID preserved |
| Brief detection failure | Identity lost | Kalman prediction fills gap |
| Slow occlusion (object behind another) | Abrupt disappearance | Gradual confidence decay |

---

## 2. Proposed Solution: The Ghost Buffer Architecture

### 2.1 High-Level Architecture

```
Frame → YOLO → BBPs → Association → WorldModel → Output
                          ↑              ↓
                    Kalman Predict ← Ghost Buffer
```

**Key Components:**

1. **WorldModel**: Central state manager holding all tracked objects
2. **TrackedObject**: Per-object state including Kalman filter, identity, and visibility
3. **GhostBuffer**: Maintains "hallucinated" objects not currently detected
4. **Association**: Hungarian algorithm or greedy matching between BBPs and predictions

### 2.2 Core Data Structures

```python
# tracking/tracked_object.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

class ObjectState(Enum):
    VISIBLE = "visible"      # Currently detected by YOLO
    OCCLUDED = "occluded"    # Predicted to exist but not detected
    LOST = "lost"            # Exceeded max occlusion time
    
@dataclass
class TrackedObject:
    """
    Persistent object representation with Kalman state estimation.
    
    State vector: [x_center, y_center, scale, aspect_ratio, vx, vy, vs, va]
    - Position (x, y) in pixel coordinates
    - Scale (s) = sqrt(width * height)  
    - Aspect ratio (a) = width / height
    - Velocities for each component
    """
    object_id: str
    class_id: int
    
    # Kalman filter state
    state: np.ndarray  # [8,] state vector
    covariance: np.ndarray  # [8, 8] covariance matrix
    
    # Temporal tracking
    frames_visible: int = 0
    frames_occluded: int = 0
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    
    # Visibility state machine
    visibility: ObjectState = ObjectState.VISIBLE
    
    # Associated BBP when visible
    last_bbp: Optional["BBP"] = None
    
    # Embedding for re-identification
    embedding: Optional[np.ndarray] = None
    
    # Confidence accumulator
    cumulative_confidence: float = 0.0
    
    @property
    def predicted_bbox(self) -> tuple[float, float, float, float]:
        """Convert Kalman state to xyxy bounding box."""
        x, y, s, a = self.state[:4]
        w = np.sqrt(s * a)
        h = s / w if w > 0 else 0
        x1, y1 = x - w/2, y - h/2
        x2, y2 = x + w/2, y + h/2
        return (x1, y1, x2, y2)
    
    @property
    def velocity(self) -> tuple[float, float]:
        """Return (vx, vy) in pixels/frame."""
        return (self.state[4], self.state[5])
    
    @property
    def time_since_seen(self) -> int:
        """Frames since last detection."""
        return self.frames_occluded
```

### 2.3 Kalman Filter Implementation

```python
# tracking/kalman.py

import numpy as np
from typing import Tuple

class BBoxKalmanFilter:
    """
    Kalman Filter for bounding box tracking.
    
    State: [x, y, s, a, vx, vy, vs, va]
    Measurement: [x, y, s, a]
    
    Based on SORT (Simple Online and Realtime Tracking) formulation.
    """
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float]):
        """
        Initialize filter from xyxy bounding box.
        """
        x1, y1, x2, y2 = initial_bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        s = w * h  # scale = area
        a = w / h if h > 0 else 1.0  # aspect ratio
        
        # State vector: position + velocity
        self.state = np.array([x, y, s, a, 0, 0, 0, 0], dtype=np.float64)
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # position += velocity
        
        # Measurement matrix (we observe position, not velocity)
        self.H = np.eye(4, 8)
        
        # Process noise covariance
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01  # Lower noise for velocities
        self.Q[:4, :4] *= 1.0
        
        # Measurement noise covariance
        self.R = np.eye(4)
        self.R[2, 2] *= 10  # Scale measurement is noisier
        self.R[3, 3] *= 10  # Aspect ratio too
        
        # Initial covariance (high uncertainty in velocities)
        self.P = np.eye(8)
        self.P[4:, 4:] *= 1000
        
    def predict(self) -> np.ndarray:
        """
        Predict next state (call once per frame, even when occluded).
        
        Returns predicted state vector.
        """
        # State prediction
        self.state = self.F @ self.state
        
        # Covariance prediction  
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state.copy()
    
    def update(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Update state with measurement (call only when detected).
        
        Returns updated state vector.
        """
        # Convert bbox to measurement
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        s = w * h
        a = w / h if h > 0 else 1.0
        z = np.array([x, y, s, a])
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        y_residual = z - self.H @ self.state
        self.state = self.state + K @ y_residual
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.state.copy()
    
    def state_to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert current state to xyxy bounding box."""
        x, y, s, a = self.state[:4]
        s = max(s, 1.0)  # Prevent negative scale
        a = max(a, 0.1)  # Prevent extreme aspect ratios
        w = np.sqrt(s * a)
        h = s / w if w > 0 else 1.0
        return (x - w/2, y - h/2, x + w/2, y + h/2)
    
    def get_innovation(self, bbox: Tuple[float, float, float, float]) -> float:
        """
        Compute Mahalanobis distance for association scoring.
        
        Lower = better match to this filter's prediction.
        """
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        s = w * h
        a = w / h if h > 0 else 1.0
        z = np.array([x, y, s, a])
        
        # Innovation (measurement residual)
        y_hat = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Mahalanobis distance
        try:
            d = y_hat @ np.linalg.inv(S) @ y_hat
            return float(np.sqrt(d))
        except np.linalg.LinAlgError:
            return float('inf')
```

### 2.4 World Model (Central State Manager)

```python
# tracking/world_model.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from uuid import uuid4

from perception.bbp import BBP, BoundingBox
from tracking.kalman import BBoxKalmanFilter
from tracking.tracked_object import TrackedObject, ObjectState

@dataclass
class WorldModelConfig:
    """Configuration for object permanence behavior."""
    max_occlusion_frames: int = 30  # ~1 second at 30fps
    min_hits_to_confirm: int = 3    # Frames before object is "real"
    iou_threshold: float = 0.3       # For association
    max_mahalanobis: float = 9.4877  # Chi-squared 95% with 4 DOF
    use_appearance: bool = False     # Enable embedding matching
    appearance_weight: float = 0.5   # Weight for embedding distance

class WorldModel:
    """
    Central state manager for episodic object permanence.
    
    Maintains TrackedObject instances across frames, handling:
    - Association of BBPs to existing objects
    - Kalman prediction for occluded objects  
    - Object birth (new detections)
    - Object death (max occlusion exceeded)
    """
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        self.config = config or WorldModelConfig()
        self.objects: Dict[str, TrackedObject] = {}
        self.kalman_filters: Dict[str, BBoxKalmanFilter] = {}
        self.frame_count: int = 0
        self._next_id: int = 0
        
    def _generate_id(self) -> str:
        """Generate unique object ID."""
        self._next_id += 1
        return f"obj_{self._next_id:06d}"
    
    def step(
        self, 
        bbps: List[BBP], 
        frame_idx: int, 
        timestamp_s: float
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """
        Process one frame of detections.
        
        Args:
            bbps: Current frame's BBP detections
            frame_idx: Frame index
            timestamp_s: Frame timestamp
            
        Returns:
            (active_objects, lost_objects): Objects still tracked and newly lost
        """
        self.frame_count = frame_idx
        
        # 1. Predict all existing objects forward
        for obj_id, kf in self.kalman_filters.items():
            kf.predict()
            
        # 2. Associate BBPs to existing objects
        matched, unmatched_bbps, unmatched_objs = self._associate(bbps)
        
        # 3. Update matched objects
        for obj_id, bbp in matched:
            self._update_object(obj_id, bbp, frame_idx, timestamp_s)
            
        # 4. Handle unmatched objects (mark occluded)
        for obj_id in unmatched_objs:
            self._mark_occluded(obj_id, frame_idx)
            
        # 5. Birth new objects from unmatched BBPs
        for bbp in unmatched_bbps:
            self._birth_object(bbp, frame_idx, timestamp_s)
            
        # 6. Remove dead objects
        lost = self._reap_lost_objects()
        
        # 7. Return active and lost objects
        active = list(self.objects.values())
        return (active, lost)
    
    def _associate(
        self, bbps: List[BBP]
    ) -> Tuple[List[Tuple[str, BBP]], List[BBP], List[str]]:
        """
        Associate BBPs to tracked objects using IoU + Mahalanobis distance.
        
        Returns: (matched_pairs, unmatched_bbps, unmatched_object_ids)
        """
        if not self.objects or not bbps:
            return ([], bbps, list(self.objects.keys()))
        
        # Build cost matrix
        obj_ids = list(self.objects.keys())
        n_objs = len(obj_ids)
        n_bbps = len(bbps)
        cost = np.full((n_objs, n_bbps), np.inf)
        
        for i, obj_id in enumerate(obj_ids):
            kf = self.kalman_filters[obj_id]
            pred_bbox = kf.state_to_bbox()
            
            for j, bbp in enumerate(bbps):
                # IoU cost (1 - IoU)
                det_bbox = bbp.bbox.as_xyxy()
                iou = self._compute_iou(pred_bbox, det_bbox)
                iou_cost = 1.0 - iou
                
                # Mahalanobis distance
                mahal = kf.get_innovation(det_bbox)
                
                # Gating: reject if IoU too low or Mahalanobis too high
                if iou < self.config.iou_threshold:
                    continue
                if mahal > self.config.max_mahalanobis:
                    continue
                    
                # Combined cost
                cost[i, j] = iou_cost + 0.1 * mahal
        
        # Greedy assignment (can upgrade to Hungarian later)
        matched = []
        used_objs = set()
        used_bbps = set()
        
        while True:
            # Find minimum cost
            if np.all(np.isinf(cost)):
                break
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            if np.isinf(cost[i, j]):
                break
                
            matched.append((obj_ids[i], bbps[j]))
            used_objs.add(obj_ids[i])
            used_bbps.add(j)
            
            # Mark row and column as used
            cost[i, :] = np.inf
            cost[:, j] = np.inf
        
        unmatched_bbps = [b for k, b in enumerate(bbps) if k not in used_bbps]
        unmatched_objs = [oid for oid in obj_ids if oid not in used_objs]
        
        return (matched, unmatched_bbps, unmatched_objs)
    
    def _compute_iou(
        self, 
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """Compute IoU between two xyxy boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def _update_object(
        self, obj_id: str, bbp: BBP, frame_idx: int, timestamp_s: float
    ) -> None:
        """Update a matched object with new detection."""
        obj = self.objects[obj_id]
        kf = self.kalman_filters[obj_id]
        
        # Kalman update
        kf.update(bbp.bbox.as_xyxy())
        
        # Update object state
        obj.state = kf.state.copy()
        obj.covariance = kf.P.copy()
        obj.frames_visible += 1
        obj.frames_occluded = 0
        obj.last_seen_frame = frame_idx
        obj.last_seen_timestamp = timestamp_s
        obj.visibility = ObjectState.VISIBLE
        obj.last_bbp = bbp
        obj.cumulative_confidence += bbp.confidence
        
    def _mark_occluded(self, obj_id: str, frame_idx: int) -> None:
        """Mark an unmatched object as occluded (ghost)."""
        obj = self.objects[obj_id]
        kf = self.kalman_filters[obj_id]
        
        # Update state from prediction (already called in step)
        obj.state = kf.state.copy()
        obj.covariance = kf.P.copy()
        obj.frames_occluded += 1
        obj.visibility = ObjectState.OCCLUDED
        
    def _birth_object(
        self, bbp: BBP, frame_idx: int, timestamp_s: float
    ) -> str:
        """Create a new tracked object from unmatched BBP."""
        obj_id = self._generate_id()
        
        # Initialize Kalman filter
        kf = BBoxKalmanFilter(bbp.bbox.as_xyxy())
        self.kalman_filters[obj_id] = kf
        
        # Create tracked object
        obj = TrackedObject(
            object_id=obj_id,
            class_id=bbp.class_id or -1,
            state=kf.state.copy(),
            covariance=kf.P.copy(),
            frames_visible=1,
            frames_occluded=0,
            last_seen_frame=frame_idx,
            last_seen_timestamp=timestamp_s,
            visibility=ObjectState.VISIBLE,
            last_bbp=bbp,
            cumulative_confidence=bbp.confidence,
        )
        self.objects[obj_id] = obj
        
        return obj_id
    
    def _reap_lost_objects(self) -> List[TrackedObject]:
        """Remove objects that exceeded max occlusion time."""
        lost = []
        to_remove = []
        
        for obj_id, obj in self.objects.items():
            if obj.frames_occluded > self.config.max_occlusion_frames:
                obj.visibility = ObjectState.LOST
                lost.append(obj)
                to_remove.append(obj_id)
                
        for obj_id in to_remove:
            del self.objects[obj_id]
            del self.kalman_filters[obj_id]
            
        return lost
    
    def get_visible_objects(self) -> List[TrackedObject]:
        """Return only currently visible objects."""
        return [o for o in self.objects.values() 
                if o.visibility == ObjectState.VISIBLE]
    
    def get_ghost_objects(self) -> List[TrackedObject]:
        """Return occluded (ghost) objects with predicted positions."""
        return [o for o in self.objects.values() 
                if o.visibility == ObjectState.OCCLUDED]
    
    def get_all_objects(self) -> List[TrackedObject]:
        """Return all tracked objects (visible + ghosts)."""
        return list(self.objects.values())
    
    def get_object_by_id(self, obj_id: str) -> Optional[TrackedObject]:
        """Retrieve specific object by ID."""
        return self.objects.get(obj_id)
    
    def to_dict(self) -> dict:
        """Serialize world state for logging."""
        return {
            "frame_count": self.frame_count,
            "num_visible": len(self.get_visible_objects()),
            "num_ghosts": len(self.get_ghost_objects()),
            "objects": [
                {
                    "id": o.object_id,
                    "class_id": o.class_id,
                    "visibility": o.visibility.value,
                    "frames_occluded": o.frames_occluded,
                    "predicted_bbox": o.predicted_bbox,
                }
                for o in self.objects.values()
            ]
        }
```

---

## 3. Integration with Existing Architecture

### 3.1 Modified Pipeline Flow

```
Before:
  Frame → YOLO → BBPs → [Attention] → [Prototypes] → Output

After:
  Frame → YOLO → BBPs → WorldModel.step() → TrackedObjects
                              ↓                    ↓
                        Ghost Objects      → [Attention] → [Prototypes]
                        (predicted BBPs)
```

### 3.2 Updated `experiments/run.py`

```python
# Key changes to run_session()

from tracking.world_model import WorldModel, WorldModelConfig

def run_session(cfg: ExperimentConfig) -> Path:
    # ... existing setup ...
    
    # Initialize world model for object permanence
    world_config = WorldModelConfig(
        max_occlusion_frames=cfg.max_occlusion_frames,  # Add to config
        iou_threshold=cfg.tracking_iou_threshold,
    )
    world = WorldModel(world_config)
    
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "session_start", "config": asdict(cfg)}) + "\n")
        
        for fr in iter_frames(cfg.source, stride=cfg.stride, max_frames=cfg.max_frames):
            # Get raw BBPs from YOLO
            bbps = gen.detect_bbps(
                frame_idx=fr.frame_idx, 
                timestamp_s=fr.timestamp_s, 
                frame_bgr=fr.image
            )
            
            # Step world model (this maintains object permanence)
            active_objects, lost_objects = world.step(
                bbps, 
                frame_idx=fr.frame_idx,
                timestamp_s=fr.timestamp_s
            )
            
            # Log frame event with both raw BBPs AND tracked objects
            f.write(json.dumps({
                "event": "frame",
                "frame_idx": fr.frame_idx,
                "timestamp_s": fr.timestamp_s,
                "bbps": [b.to_dict() for b in bbps],
                "tracked": world.to_dict(),
                "lost": [{"id": o.object_id, "class": o.class_id} for o in lost_objects],
            }) + "\n")
        
        f.write(json.dumps({"event": "session_end"}) + "\n")
```

### 3.3 Attention Module Integration

The attention scheduler should operate on **TrackedObjects**, not raw BBPs:

```python
# attention/scheduler.py (proposed)

class AttentionScheduler:
    """
    Winner-take-all attention over tracked objects.
    
    Ghost objects can still be attended (the system "thinks about"
    where the occluded object should be).
    """
    
    def compute_priority(self, obj: TrackedObject) -> float:
        """
        Compute attention priority for an object.
        
        Factors:
        - novelty: newer objects get higher priority
        - prediction_error: unexpected motion increases priority
        - occlusion_curiosity: occluded objects may get "check" attention
        """
        novelty = 1.0 / (1 + obj.frames_visible)
        
        # Occluded objects get curiosity boost
        occlusion_boost = 0.0
        if obj.visibility == ObjectState.OCCLUDED:
            # Curiosity increases then decays
            t = obj.frames_occluded
            occlusion_boost = t * np.exp(-t / 10)  # Peak at ~10 frames
        
        # Velocity indicates motion salience
        vx, vy = obj.velocity
        motion_salience = np.sqrt(vx**2 + vy**2) / 100
        
        return novelty + occlusion_boost + motion_salience
```

---

## 4. Revised Phased Development Plan

### Integration Point: Between Stage 4 and Stage 5

The current PHASED_PLAN.md has tracking at Stage 9. **This is too late.** Object permanence is foundational and should be introduced early:

**Proposed New Stage Order:**

| Stage | Name | Depends On | Adds |
|-------|------|------------|------|
| 0 | Experiment harness | - | JSONL logging |
| 1 | BBP generator | Stage 0 | YOLO → BBPs |
| 2 | Attention scheduler | Stage 1 | WTA selection |
| 3 | Simple embeddings | Stage 2 | Bbox geometry features |
| **3.5** | **Object Permanence (NEW)** | Stage 1 | WorldModel, Kalman |
| 4 | Prototype bank | Stage 3, 3.5 | Online prototypes |
| 5 | Dual processing | Stage 4 | Predictive coding |
| ... | (remaining stages) | | |

### Rationale

- Prototypes should track **objects**, not raw BBPs
- Habituation requires knowing "this is the same object" across frames
- Prediction error is more meaningful with position prediction

---

## 5. Implementation Phases

### Phase A: Core Infrastructure (1-2 days)

**Files to create:**
- `tracking/kalman.py` - Kalman filter implementation
- `tracking/tracked_object.py` - TrackedObject dataclass
- `tracking/world_model.py` - Central state manager
- `tracking/association.py` - IoU/Mahalanobis matching

**Tests:**
- Kalman filter prediction/update cycle
- Object birth/death lifecycle
- Association correctness with synthetic BBPs

### Phase B: Integration (1 day)

**Modify:**
- `experiments/config.py` - Add tracking config fields
- `experiments/run.py` - Integrate WorldModel into session loop

**Add metrics to JSONL:**
- `num_visible`, `num_ghosts`, `num_lost`
- Per-object: `frames_occluded`, `predicted_bbox`

### Phase C: Attention Integration (1 day)

**Modify:**
- `attention/scheduler.py` - Operate on TrackedObjects
- Add occlusion curiosity signal

### Phase D: Prototype Integration (1-2 days)

**Modify:**
- Prototype matching uses TrackedObject embeddings
- Object identity persists across occlusions
- Habituation tracks per-object history

---

## 6. Testing Strategy

### Unit Tests

```python
# tests/test_kalman.py

def test_kalman_constant_velocity():
    """Object moving at constant velocity should be well-tracked."""
    kf = BBoxKalmanFilter((0, 0, 10, 10))
    
    # Simulate object moving right at 5 px/frame
    for t in range(10):
        kf.predict()
        kf.update((t*5, 0, t*5+10, 10))
    
    # Predict without update (occlusion)
    for _ in range(5):
        kf.predict()
    
    # Check prediction is reasonable
    pred_box = kf.state_to_bbox()
    expected_x = 14 * 5 + 5  # 75
    assert abs(pred_box[0] - expected_x + 5) < 10  # Within 10px


def test_world_model_occlusion():
    """Object should persist through short occlusion."""
    world = WorldModel(WorldModelConfig(max_occlusion_frames=5))
    
    # Frame 1: Object appears
    bbp1 = make_bbp(frame_idx=0, bbox=(10, 10, 20, 20))
    active, _ = world.step([bbp1], 0, 0.0)
    assert len(active) == 1
    obj_id = active[0].object_id
    
    # Frames 2-4: Object occluded (no BBPs)
    for i in range(1, 4):
        active, _ = world.step([], i, i/30.0)
        assert len(active) == 1
        assert active[0].object_id == obj_id
        assert active[0].visibility == ObjectState.OCCLUDED
    
    # Frame 5: Object reappears
    bbp5 = make_bbp(frame_idx=4, bbox=(12, 10, 22, 20))  # Moved slightly
    active, _ = world.step([bbp5], 4, 4/30.0)
    assert len(active) == 1
    assert active[0].object_id == obj_id  # Same ID!
    assert active[0].visibility == ObjectState.VISIBLE


def test_world_model_death():
    """Object should be lost after max occlusion frames."""
    world = WorldModel(WorldModelConfig(max_occlusion_frames=3))
    
    # Object appears then disappears
    bbp = make_bbp(frame_idx=0, bbox=(10, 10, 20, 20))
    world.step([bbp], 0, 0.0)
    
    # Exceed max occlusion
    for i in range(1, 5):
        active, lost = world.step([], i, i/30.0)
    
    assert len(active) == 0
    assert len(lost) == 1
```

### Integration Tests

```python
# tests/test_permanence_integration.py

def test_spinner_occlusion_scenario():
    """
    Simulate the spinner-covers-button scenario.
    
    - Button appears at frame 0
    - Spinner appears and covers button at frame 10
    - Spinner moves away at frame 20
    - Button should retain same ID
    """
    world = WorldModel()
    
    # Button visible
    for i in range(10):
        bbp = make_bbp(frame_idx=i, bbox=(100, 100, 150, 130), class_id=0)
        world.step([bbp], i, i/30.0)
    
    button_id = world.get_all_objects()[0].object_id
    
    # Button occluded by spinner
    for i in range(10, 20):
        spinner = make_bbp(frame_idx=i, bbox=(90, 90, 160, 140), class_id=1)
        active, _ = world.step([spinner], i, i/30.0)
        
        # Button should be ghost
        button = world.get_object_by_id(button_id)
        assert button is not None
        assert button.visibility == ObjectState.OCCLUDED
    
    # Button reappears
    for i in range(20, 25):
        button_bbp = make_bbp(frame_idx=i, bbox=(100, 100, 150, 130), class_id=0)
        active, _ = world.step([button_bbp], i, i/30.0)
    
    # Same ID preserved!
    button = world.get_object_by_id(button_id)
    assert button is not None
    assert button.visibility == ObjectState.VISIBLE
```

---

## 7. Metrics & Observability

### New JSONL Events

```json
{"event": "object_birth", "object_id": "obj_000001", "class_id": 0, "frame_idx": 42}
{"event": "object_occluded", "object_id": "obj_000001", "predicted_bbox": [100, 100, 150, 130]}
{"event": "object_reacquired", "object_id": "obj_000001", "frames_occluded": 15}
{"event": "object_lost", "object_id": "obj_000001", "total_frames": 157}
```

### Dashboard Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `mean_track_length` | Avg frames per object lifetime | > 50 |
| `occlusion_recovery_rate` | % of occlusions that recover | > 80% |
| `id_switch_rate` | ID changes per 100 frames | < 2 |
| `ghost_precision` | % of ghosts that actually exist | > 70% |

---

## 8. Future Enhancements

### 8.1 Appearance-Based Re-ID

Add embedding matching for long-term re-identification:

```python
# When object reappears after long occlusion, match by embedding
def _match_by_appearance(self, bbp: BBP, candidate_ghosts: List[TrackedObject]) -> Optional[str]:
    if bbp.embedding is None:
        return None
    
    best_match = None
    best_dist = float('inf')
    
    for ghost in candidate_ghosts:
        if ghost.embedding is not None:
            dist = np.linalg.norm(bbp.embedding - ghost.embedding)
            if dist < best_dist and dist < self.config.appearance_threshold:
                best_dist = dist
                best_match = ghost.object_id
    
    return best_match
```

### 8.2 Scene Context

Use scene graph to improve predictions:

```python
# If object A is always near object B, and B is visible,
# use B's position to predict where A should be
```

### 8.3 Occlusion Reasoning

Explicit occlusion detection:

```python
# If ghost G's predicted bbox overlaps visible object V,
# G is likely behind V (not gone)
def is_occluded_by(self, ghost: TrackedObject, visible: TrackedObject) -> bool:
    iou = self._compute_iou(ghost.predicted_bbox, visible.predicted_bbox)
    return iou > 0.5
```

---

## 9. Summary & Next Steps

### The Core Fix

**Before:** YOLO detection = Object existence  
**After:** YOLO detection = Observation; WorldModel = Belief state

The system will now maintain a **belief** about object existence that persists through temporary detection failures, using Kalman filtering to predict where objects should be.

### Immediate Actions

1. **Create** `tracking/` module with Kalman filter and WorldModel
2. **Test** with synthetic occlusion scenarios
3. **Integrate** into experiment runner
4. **Validate** with real video containing occlusions
5. **Measure** ID stability and occlusion recovery rate

### Success Criteria

- [ ] Object ID persists through 30-frame occlusion
- [ ] Predicted bbox within 20px of actual on reappearance  
- [ ] ID switch rate < 2 per 100 frames on test videos
- [ ] System can "click" predicted position of occluded button

---

**Document End**

*This plan transforms the system from a "memoryless recognizer" to a "world model with persistent beliefs"—the first step toward true cognitive agency.*
