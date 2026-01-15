# Project Roadmap & Implementation Status

**Date:** January 14, 2026
**Current Status:** Phase 1 (BBP Generation) and Phase 5 (Basic Tracking/Object Permanence) are implemented.
**Goal:** Full Online Perceptual Learning System (as defined in `readme.md`).

This document consolidates the implementation of Episodic Object Permanence with the broader project roadmap.

## 1. Recently Completed: Episodic Object Permanence (The "Ghost" Buffer)

*Implemented in `tracking/` module.*

- **Problem Solved:** "Amnesia" where objects ceased to exist upon occlusion.
- **Solution:** `WorldModel` with `KalmanBoxTracker`.
- **Mechanism:**
    - **Recognition (YOLO)** matches to **State (ActiveObject)**.
    - **Kalman Filter** predicts position/velocity when occluded.
    - **Ghost Objects** persist for `max_ghost_time` (5s) without visual confirmation.
- **Key Files:**
    - `tracking/kalman.py`: State estimation (position + velocity).
    - `tracking/tracker.py`: Association (IoU/Hungarian) and lifecycle management.
    - `scripts/run_tracker_stream.py`: Demo integration showing "Visible" vs "Ghost" states.

## 2. Updated Phased Implementation Plan

This plan integrates the original `PHASED_PLAN.md` with the recent tracking implementation.

### Phase 1: Frame Pipeline & BBP Generator [COMPLETED]
- Video -> YOLO -> BBP Data Structure.
- *Status*: Done (`perception/bbp.py`, `perception/yolo_adapter.py`).

### Phase 2: Static Feature Learning (No Tracking) [NEXT STEP]
- **Goal:** Learn sparse, stable visual features from BBP crops.
- **Mechanism:**
    - **Patch Sampling:** Extract and normalize image patches from BBP bounding boxes.
    - **WTA Competition:** Winner-Take-All layer to enforce sparsity.
    - **Hebbian Learning:** Update weights based on co-activation of input and winner.
- **Deliverable:** A `features/` module that learns a "visual dictionary" from the video stream online.

### Phase 3: Part Formation
- **Goal:** Build mid-level structure (features co-activating).
- **Mechanism:**
    - Second layer of Hebbian learning taking Phase 2 outputs.
    - "Part-of" edges in the Percept Graph.

### Phase 4: Object Prototype Formation
- **Goal:** Stable object identity.
- **Mechanism:**
    - Online clustering of parts/features.
    - Error-gated averaging.
    - Top-down prediction to validate matches.

### Phase 5: Advanced Tracking & Slowness [PARTIALLY COMPLETED]
- **Done:** Basic Kalman Tracking & Object Permanence.
- **Remaining:**
    - **Appearance Matching:** Currently, tracking uses only spatial overlap (IoU). Future improvement: Use Phase 2 embeddings to re-identify objects after long occlusions (DeepSORT style).
    - **Slowness/Trace Rules:** Learn temporal invariance (objects change slowly).

### Phase 6: Motion Prototypes
- **Goal:** Learn motion as first-class percepts.
- **Mechanism:**
    - Cluster motion vectors (from Kalman Filter state: `vx`, `vy`).
    - Form "Motion Prototypes" (e.g., "moving left", "falling").

### Phase 7: Recurrence & Working Memory
- **Goal:** Temporal context and maintaining active state.
- **Mechanism:**
    - Recurrent connections or Working Memory buffer.
    - Inhibition-of-Return (IoR) to prevent stuck attention.

### Phase 8: Category & Concept Formation
- **Goal:** Higher-level abstraction.
- **Mechanism:** Graph clustering on the `PerceptGraph`.

## 3. Immediate Action Items

1.  **Validate Tracking:**
    - Run `scripts/run_tracker_stream.py` on diverse video data.
    - Tune `KalmanFilter` noise parameters (`R`, `Q`) and `max_ghost_time` if objects drift too much or vanish too quickly.

2.  **Start Phase 2 (Feature Learning):**
    - Create `features/` module.
    - Implement `FeatureExtractor`: Simple resizing/normalization of BBP crops (e.g., to 32x32 grayscale).
    - Implement `HebbianLayer`: Matrix of weights with local update rules.

3.  **Refine Documentation:**
    - Update `readme.md` to reflect the existence of the Tracker/WorldModel.
