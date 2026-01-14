# Plan for Episodic Object Permanence

## Goal
Implement "Episodic Object Permanence" by separating Recognition (YOLO) from State (Tracker). The system should maintain object state even when occluded, using a "Ghost" buffer.

## Architecture Changes

### 1. New Dependency
- Add `filterpy` for Kalman Filter implementation.

### 2. Tracking Module (`tracking/`)
- **`KalmanTracker`**: A wrapper around `filterpy.kalman.KalmanFilter` to track position (x, y) and velocity (dx, dy) of a bounding box center, and potentially width/height.
- **`ActiveObject`**: A class representing a tracked object.
    - Properties:
        - `id`: Unique identifier.
        - `kf`: Instance of `KalmanTracker`.
        - `bbox`: Estimated bounding box.
        - `time_since_last_seen`: Counter for occlusion handling.
        - `history`: Optional history of positions.
        - `class_id`: Most recent or most frequent class ID.
        - `confidence`: Confidence of the track.
- **`WorldModel`** (or `ObjectTracker`): Manages the list of `ActiveObject`s.
    - Methods:
        - `predict()`: Updates all KF predictions.
        - `update(detections)`: Associates YOLO detections to existing tracks (using IoU or Hungarian algorithm) and updates KFs. Handles creation of new tracks and deletion of dead tracks.
        - `get_active_objects()`: Returns current state of all objects (including "ghosts").

### 3. Integration
- Create a new script or update `scripts/run_bbp_stream.py` to use the `WorldModel`.
- The main loop will:
    1. Get frame.
    2. Get YOLO detections (`BBP`s).
    3. `world_model.predict()`
    4. `world_model.update(detections)`
    5. Render/Log the active objects (distinguishing between "Visible" and "Ghost").

## Implementation Steps

1.  **Install Dependencies**: Add `filterpy` to `pyproject.toml`.
2.  **Implement Kalman Filter Wrapper**: `tracking/kalman.py`.
3.  **Implement World Model**: `tracking/tracker.py`.
4.  **Update Script**: Modify `scripts/run_bbp_stream.py` or create `scripts/run_tracker_stream.py`.
5.  **Verify**: Run with a video file (or mock data) to verify object persistence during occlusion.
