from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Minimal experiment configuration.

    Keep this intentionally small and stable: it becomes part of your paper's method section.
    """

    seed: int = 0
    source: str | int = 0
    max_frames: int = 300
    stride: int = 1
    loop: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_device: str | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    # Output folder relative to repo root.
    output_dir: str = "outputs"
    # Tracking (Stage 3).
    tracking_iou_threshold: float = 0.3
    tracking_max_missed: int = 5
    tracking_min_confidence: float = 0.0
    tracking_bbox_smoothing: float = 0.7
    tracking_velocity_smoothing: float = 0.8
    # Stage 4: sensory buffer + fovea + gaze control.
    fovea_size: tuple[int, int] = (96, 96)
    periphery_stride: int = 8
    gaze_jitter_std: float = 1.5
    gaze_jitter_max: float = 6.0
    gaze_pull_strength: float = 0.7
    sensory_buffer_capacity: int = 1
    # Stage 5: sparse embeddings + WTA.
    wta_units: int = 16
    wta_k: int = 1
    wta_learning_rate: float = 0.15
    wta_decay: float = 0.01

