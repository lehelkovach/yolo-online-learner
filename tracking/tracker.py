from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

from perception.bbp import BBP, BoundingBox
from tracking.kalman import KalmanBoxTracker

@dataclass
class ActiveObject:
    """
    Represents an object currently being tracked in the world model.
    """
    id: int
    kf: KalmanBoxTracker
    class_id: int | None
    confidence: float
    time_since_last_seen: float = 0.0
    is_ghost: bool = False
    
    # Configuration
    max_ghost_time: float = 5.0  # seconds

    def update(self, bbp: BBP) -> None:
        """Update with a matched detection."""
        self.kf.update(bbp.bbox)
        self.class_id = bbp.class_id
        self.confidence = bbp.confidence
        self.time_since_last_seen = 0.0
        self.is_ghost = False

    def predict(self, dt: float) -> None:
        """Advance state."""
        self.kf.predict()
        self.time_since_last_seen += dt
        if self.time_since_last_seen > 0:
            self.is_ghost = True

    @property
    def bbox(self) -> BoundingBox:
        return self.kf.predicted_bbox

    @property
    def is_dead(self) -> bool:
        return self.time_since_last_seen > self.max_ghost_time


class WorldModel:
    """
    Maintains the state of the world, including object permanence.
    """
    def __init__(self, iou_threshold: float = 0.3):
        self.active_objects: list[ActiveObject] = []
        self.next_id = 0
        self.iou_threshold = iou_threshold

    def update(self, bbps: list[BBP], dt: float = 1.0/30.0) -> None:
        """
        Update the world model with new detections.
        
        Args:
            bbps: List of Bounding Box Percepts from the current frame.
            dt: Time delta since last update (seconds).
        """
        
        # 1. Predict new locations for all existing objects
        for obj in self.active_objects:
            obj.predict(dt)

        # 2. Associate detections to objects
        matched_indices = set()
        
        if self.active_objects and bbps:
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(self.active_objects), len(bbps)))
            for i, obj in enumerate(self.active_objects):
                for j, bbp in enumerate(bbps):
                    iou_matrix[i, j] = obj.bbox.iou(bbp.bbox)

            # Hungarian Algorithm (maximize IoU => minimize -IoU)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    self.active_objects[r].update(bbps[c])
                    matched_indices.add(c)

        # 3. Create new objects for unmatched detections
        # If no active objects, all bbps are new.
        # If no bbps, loop doesn't run.
        for j, bbp in enumerate(bbps):
            if j not in matched_indices:
                self._create_object(bbp)

        # 4. Prune dead objects
        self.active_objects = [obj for obj in self.active_objects if not obj.is_dead]

    def _create_object(self, bbp: BBP) -> None:
        new_obj = ActiveObject(
            id=self.next_id,
            kf=KalmanBoxTracker(bbp.bbox),
            class_id=bbp.class_id,
            confidence=bbp.confidence
        )
        self.active_objects.append(new_obj)
        self.next_id += 1
