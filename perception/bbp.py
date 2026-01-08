from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def clip(self, width: int, height: int) -> BoundingBox:
        x1 = max(0.0, min(float(width), self.x1))
        y1 = max(0.0, min(float(height), self.y1))
        x2 = max(0.0, min(float(width), self.x2))
        y2 = max(0.0, min(float(height), self.y2))
        # Ensure proper ordering after clipping.
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return 0.0 if union <= 0.0 else float(inter / union)


@dataclass(frozen=True, slots=True)
class BBP:
    """
    Bounding Box Percept (BBP).

    Treat YOLO detections as transient percept hypotheses, not labels.
    """

    frame_idx: int
    timestamp_s: float
    bbox: BoundingBox
    confidence: float
    class_id: int | None = None
    # Placeholder for appearance representation; can be filled later (e.g., backbone embedding).
    embedding: tuple[float, ...] | None = None
    # Cognitive / routing signals (computed downstream).
    salience: float | None = None
    novelty: float | None = None
    prediction_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep bbox nested but JSON-friendly.
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> BBP:
        bbox = d.get("bbox")
        if isinstance(bbox, dict):
            bbox_obj = BoundingBox(**bbox)
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_obj = BoundingBox(*map(float, bbox))
        else:
            raise TypeError(f"Unsupported bbox format: {type(bbox)!r}")
        return BBP(
            frame_idx=int(d["frame_idx"]),
            timestamp_s=float(d["timestamp_s"]),
            bbox=bbox_obj,
            confidence=float(d["confidence"]),
            class_id=None if d.get("class_id") is None else int(d["class_id"]),
            embedding=None
            if d.get("embedding") is None
            else tuple(float(x) for x in d["embedding"]),
            salience=None if d.get("salience") is None else float(d["salience"]),
            novelty=None if d.get("novelty") is None else float(d["novelty"]),
            prediction_error=None
            if d.get("prediction_error") is None
            else float(d["prediction_error"]),
        )

