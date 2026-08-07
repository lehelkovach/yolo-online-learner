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
    preview: bool = False
    # Stage-4 prototype bank (perceptual patterns, not objects/tracks).
    prototype_kmax: int = 32
    prototype_match_threshold: float = 0.85
    prototype_spawn_cooldown_frames: int = 5
    prototype_novelty_hysteresis: int = 2
    prototype_learning: bool = True
