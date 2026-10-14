"""Frozen-YOLO crop encoder: a stronger ``PerceptEncoder`` with no new weights.

The attended crop is run through the frozen YOLOv8 network up to one layer and
global-average-pooled (ultralytics' ``embed``), then L2-normalised. The detector
is already the sensory substrate, so its features are the cheapest discriminative
appearance descriptor available offline: on the recorded pedestrian clip the 10-d
colour vector scored cosine 0.99 for every pair of crops, while these features
separate people from vehicles and people from each other.

The space id names weights, layer and input size (``yolov8n_embed_l21_160_v1``), so
vectors from different settings can never be compared or averaged by mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from features.encoder import EmbeddingResult
from features.simple_embedding import crop_window
from perception.bbp import BoundingBox

YOLO_ENCODER_VERSION = "v1"
DEFAULT_LAYER = 21  # last C2f of the YOLOv8 head, 256 channels for the "n" variant
DEFAULT_IMGSZ = 160
MIN_CROP_SIDE = 4


@dataclass(frozen=True, slots=True)
class YoloCropEmbedding:
    """A pooled, normalised backbone feature and the integer crop it came from."""

    vector: tuple[float, ...]
    crop_xyxy: tuple[int, int, int, int]
    raw_norm: float


def yolo_space_id(model: str, layer: int, imgsz: int) -> str:
    return f"{Path(model).stem}_embed_l{layer}_{imgsz}_{YOLO_ENCODER_VERSION}"


class YoloCropEncoder:
    """Wrap ultralytics ``YOLO.embed`` behind the ``PerceptEncoder`` contract."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        *,
        layer: int = DEFAULT_LAYER,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | None = None,
        model_object: Any | None = None,
    ) -> None:
        if layer < 0:
            raise ValueError("layer must be non-negative")
        if imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        self.model_name = str(model)
        self.layer = int(layer)
        self.imgsz = int(imgsz)
        self.device = device
        if model_object is None:
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Ultralytics YOLO is not installed. Install with: pip install ultralytics"
                ) from e
            model_object = YOLO(model)
        self._model = model_object
        self._dimension: int | None = None
        probe = self._embed(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))
        self._dimension = int(probe.shape[0])

    @property
    def space_id(self) -> str:
        return yolo_space_id(self.model_name, self.layer, self.imgsz)

    @property
    def dimension(self) -> int:
        assert self._dimension is not None
        return self._dimension

    def schema(self) -> dict[str, object]:
        return {
            "embedding_space_id": self.space_id,
            "dimension": self.dimension,
            "encoder": "yolo_embed",
            "model": self.model_name,
            "layer": self.layer,
            "imgsz": self.imgsz,
            "pooling": "global_average",
            "normalization": "l2",
            "input_channel_order": "bgr",
            "crop_rasterization": "floor_start_ceil_stop",
            "min_crop_side": MIN_CROP_SIDE,
            "scope": "wta_winner_only",
        }

    def _embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        outputs = self._model.embed(
            [crop_bgr], imgsz=self.imgsz, device=self.device, verbose=False,
            embed=[self.layer],
        )
        vector = outputs[0]
        if hasattr(vector, "detach"):
            vector = vector.detach().cpu().numpy()
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if self._dimension is not None and vector.shape != (self._dimension,):
            raise RuntimeError(
                f"encoder returned {vector.shape[0]} values, expected {self._dimension}"
            )
        return vector

    def encode_crop(self, frame_bgr: object, bbox: BoundingBox) -> YoloCropEmbedding | None:
        window = crop_window(frame_bgr, bbox)
        if window is None:
            return None
        _, (x1, y1, x2, y2) = window
        if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
            return None
        crop = frame_bgr[y1:y2, x1:x2]  # type: ignore[index]
        raw = self._embed(np.ascontiguousarray(crop))
        if not np.isfinite(raw).all():
            return None
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm <= 0.0:
            return None
        unit = raw / raw_norm
        return YoloCropEmbedding(
            vector=tuple(float(v) for v in unit), crop_xyxy=(x1, y1, x2, y2), raw_norm=raw_norm
        )

    def to_result(self, crop: YoloCropEmbedding | None) -> EmbeddingResult | None:
        if crop is None:
            return None
        return EmbeddingResult(vector=crop.vector, space_id=self.space_id)

    def encode(self, frame_bgr: object, bbox: BoundingBox) -> EmbeddingResult | None:
        return self.to_result(self.encode_crop(frame_bgr, bbox))
