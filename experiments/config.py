from __future__ import annotations

from dataclasses import dataclass, field

from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID
from graph.memory_graph import MemoryGraphConfig
from memory.episodes import EpisodicMemoryConfig
from memory.prototypes import PrototypeMemoryConfig
from objects.binder import BinderConfig
from objects.memory import PermanenceConfig


def _default_prototype_config() -> PrototypeMemoryConfig:
    return PrototypeMemoryConfig(embedding_space_id=SIMPLE_EMBEDDING_SPACE_ID)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Minimal experiment configuration.

    Keep this intentionally small and stable: it becomes part of your paper's method section.
    All learning thresholds live in the nested configs so a replay can rebuild them exactly.
    """

    seed: int = 0
    source: str | int = 0
    max_frames: int = 300
    stride: int = 1
    yolo_model: str = "yolov8n.pt"
    yolo_device: str | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    # Recorded detections to replay instead of running YOLO (``None`` = live YOLO).
    detections: str | None = None
    # Output folder relative to repo root.
    output_dir: str = "outputs"
    preview: bool = False
    binder: BinderConfig = field(default_factory=BinderConfig)
    permanence: PermanenceConfig = field(default_factory=PermanenceConfig)
    prototypes: PrototypeMemoryConfig = field(default_factory=_default_prototype_config)
    episodes: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    graph: MemoryGraphConfig = field(default_factory=MemoryGraphConfig)
