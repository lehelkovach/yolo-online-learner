from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from perception.bbp import BoundingBox

class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes.
    
    State: [x, y, w, h, vx, vy, vw, vh]
    Observation: [x, y, w, h]
    """

    def __init__(self, bbox: BoundingBox, dt: float = 1.0/30.0):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (F)
        # x' = x + vx*dt
        # y' = y + vy*dt
        # ...
        self.kf.F = np.eye(8)
        self.kf.F[0, 4] = dt
        self.kf.F[1, 5] = dt
        self.kf.F[2, 6] = dt
        self.kf.F[3, 7] = dt

        # Measurement function (H)
        # We observe x, y, w, h
        self.kf.H = np.eye(4, 8)

        # Measurement noise (R)
        self.kf.R *= 1.0 # Lowered from 10.0 to trust measurement more
        self.kf.R[2:, 2:] *= 10.0  # Higher uncertainty for width/height

        # Process noise (Q)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initial state
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        w = bbox.w
        h = bbox.h
        
        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        self.kf.P *= 10.0

    def predict(self) -> None:
        """Advance the state vector."""
        self.kf.predict()

    def update(self, bbox: BoundingBox) -> None:
        """Update the state with a new observation."""
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        w = bbox.w
        h = bbox.h
        
        z = np.array([cx, cy, w, h])
        self.kf.update(z)

    @property
    def predicted_bbox(self) -> BoundingBox:
        """Get the current predicted bounding box."""
        cx, cy, w, h = self.kf.x[:4]
        
        # Ensure w/h are positive
        w = max(1.0, w)
        h = max(1.0, h)
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return BoundingBox(float(x1), float(y1), float(x2), float(y2))
