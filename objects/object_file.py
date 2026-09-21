from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from features.encoder import ensure_same_space
from perception.bbp import BoundingBox


class VisibilityState(StrEnum):
    """Belief about whether one physical entity is currently observable."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    LOST = "lost"
    DORMANT = "dormant"


ABSENT_STATES = frozenset({VisibilityState.LOST, VisibilityState.DORMANT})


@dataclass(slots=True)
class ObjectFile:
    """
    Persistent hypothesis about one physical entity.

    An ``ObjectFile`` is mutable belief state, not a percept. Its ``object_id`` never
    changes after construction; every other field describes the current belief.
    No semantic label is required and category membership is a separate, optional link.
    """

    object_id: str
    first_seen_s: float
    last_seen_s: float
    last_frame_idx: int
    last_bbox: BoundingBox
    prototype_embedding: tuple[float, ...]
    embedding_space_id: str
    observation_count: int = 1
    visibility: VisibilityState = VisibilityState.VISIBLE
    category_id: str | None = None
    identity_confidence: float = 1.0
    first_frame_idx: int = field(default=-1)
    missed_frames: int = 0
    last_class_id: int | None = None
    # Pixels per frame, estimated from the last two observations.
    velocity_px_per_frame: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.first_frame_idx < 0:
            self.first_frame_idx = self.last_frame_idx
        if self.observation_count < 1:
            raise ValueError("observation_count must be >= 1")
        vector = np.asarray(self.prototype_embedding, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("prototype_embedding must be a finite non-empty vector")
        self.prototype_embedding = tuple(float(v) for v in vector)

    @property
    def center(self) -> tuple[float, float]:
        box = self.last_bbox
        return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)

    def predicted_bbox(self, frame_idx: int) -> BoundingBox:
        """Extrapolate the last box by the stored velocity to ``frame_idx``."""
        elapsed = max(0, int(frame_idx) - int(self.last_frame_idx))
        dx = self.velocity_px_per_frame[0] * elapsed
        dy = self.velocity_px_per_frame[1] * elapsed
        box = self.last_bbox
        return BoundingBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)

    def update(
        self,
        *,
        timestamp_s: float,
        frame_idx: int,
        bbox: BoundingBox,
        embedding: tuple[float, ...],
        embedding_space_id: str,
        class_id: int | None = None,
        appearance_eta: float | None = None,
        identity_confidence: float | None = None,
    ) -> None:
        """
        Fold one new observation into this object file.

        ``appearance_eta`` selects the prototype update rule explicitly:
        ``None`` uses the exact running mean ``(n*p + x) / (n + 1)``; a float uses
        ``p + eta * (x - p)``.
        """
        ensure_same_space(self.embedding_space_id, embedding_space_id, context=self.object_id)
        new = np.asarray(embedding, dtype=np.float64)
        old = np.asarray(self.prototype_embedding, dtype=np.float64)
        if new.shape != old.shape:
            raise ValueError(
                f"{self.object_id}: embedding dimension {new.shape} != {old.shape}"
            )
        if not np.isfinite(new).all():
            raise ValueError(f"{self.object_id}: embedding must be finite")
        if int(frame_idx) < int(self.last_frame_idx):
            raise ValueError(
                f"{self.object_id}: frame {frame_idx} precedes last frame {self.last_frame_idx}"
            )

        elapsed = int(frame_idx) - int(self.last_frame_idx)
        if elapsed > 0 and self.visibility == VisibilityState.VISIBLE:
            old_cx, old_cy = self.center
            new_cx = (bbox.x1 + bbox.x2) / 2.0
            new_cy = (bbox.y1 + bbox.y2) / 2.0
            self.velocity_px_per_frame = (
                (new_cx - old_cx) / elapsed,
                (new_cy - old_cy) / elapsed,
            )
        elif elapsed > 0:
            # Motion while unobserved is unknown; do not extrapolate stale velocity.
            self.velocity_px_per_frame = (0.0, 0.0)

        n = self.observation_count
        if appearance_eta is None:
            updated = (n * old + new) / (n + 1)
        else:
            if not 0.0 <= appearance_eta <= 1.0:
                raise ValueError("appearance_eta must be in [0, 1]")
            updated = old + appearance_eta * (new - old)
        self.prototype_embedding = tuple(float(v) for v in updated)
        self.observation_count = n + 1
        self.last_seen_s = float(timestamp_s)
        self.last_frame_idx = int(frame_idx)
        self.last_bbox = bbox
        self.missed_frames = 0
        self.visibility = VisibilityState.VISIBLE
        if class_id is not None:
            self.last_class_id = int(class_id)
        if identity_confidence is not None:
            self.identity_confidence = float(identity_confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "first_seen_s": self.first_seen_s,
            "last_seen_s": self.last_seen_s,
            "first_frame_idx": self.first_frame_idx,
            "last_frame_idx": self.last_frame_idx,
            "last_bbox": list(self.last_bbox.as_xyxy()),
            "prototype_embedding": list(self.prototype_embedding),
            "embedding_space_id": self.embedding_space_id,
            "observation_count": self.observation_count,
            "visibility": self.visibility.value,
            "category_id": self.category_id,
            "identity_confidence": self.identity_confidence,
            "missed_frames": self.missed_frames,
            "last_class_id": self.last_class_id,
            "velocity_px_per_frame": list(self.velocity_px_per_frame),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ObjectFile:
        return ObjectFile(
            object_id=str(d["object_id"]),
            first_seen_s=float(d["first_seen_s"]),
            last_seen_s=float(d["last_seen_s"]),
            last_frame_idx=int(d["last_frame_idx"]),
            last_bbox=BoundingBox(*map(float, d["last_bbox"])),
            prototype_embedding=tuple(float(v) for v in d["prototype_embedding"]),
            embedding_space_id=str(d["embedding_space_id"]),
            observation_count=int(d.get("observation_count", 1)),
            visibility=VisibilityState(d.get("visibility", "visible")),
            category_id=d.get("category_id"),
            identity_confidence=float(d.get("identity_confidence", 1.0)),
            first_frame_idx=int(d.get("first_frame_idx", d["last_frame_idx"])),
            missed_frames=int(d.get("missed_frames", 0)),
            last_class_id=None if d.get("last_class_id") is None else int(d["last_class_id"]),
            velocity_px_per_frame=tuple(
                float(v) for v in d.get("velocity_px_per_frame", (0.0, 0.0))
            ),
        )
