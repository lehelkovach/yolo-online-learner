from __future__ import annotations

from dataclasses import dataclass, field

from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID
from graph.memory_graph import MemoryGraphConfig
from memory.episodes import EpisodicMemoryConfig
from memory.prediction import PredictionConfig
from memory.prototypes import PrototypeMemoryConfig
from memory.working_memory import WorkingMemoryConfig
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
    # Attended-crop encoder: "simple" (10-d geometry + colour) or "yolo" (frozen
    # detector features, see features/yolo_encoder.py). The prototype memory's
    # embedding_space_id must match the encoder's space id.
    encoder: str = "simple"
    encoder_layer: int = 21
    encoder_imgsz: int = 160
    # Output folder relative to repo root.
    output_dir: str = "outputs"
    preview: bool = False
    binder: BinderConfig = field(default_factory=BinderConfig)
    permanence: PermanenceConfig = field(default_factory=PermanenceConfig)
    prototypes: PrototypeMemoryConfig = field(default_factory=_default_prototype_config)
    episodes: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    graph: MemoryGraphConfig = field(default_factory=MemoryGraphConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
