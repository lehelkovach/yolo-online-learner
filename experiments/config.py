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

