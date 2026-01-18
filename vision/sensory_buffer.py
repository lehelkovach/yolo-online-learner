from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.bbp import BoundingBox


@dataclass(frozen=True, slots=True)
class SensorySnapshot:
    frame_idx: int
    timestamp_s: float
    fovea_center: tuple[float, float]
    fovea_bbox: BoundingBox
    fovea: np.ndarray
    periphery: np.ndarray
    gaze_jitter: tuple[float, float]
    gaze_target: tuple[float, float]
    meta: dict[str, Any]


class SensoryBuffer:
    """
    Small buffer for fovea + periphery samples and associated metadata.
    """

    def __init__(self, *, capacity: int = 1) -> None:
        self.capacity = max(1, int(capacity))
        self._buffer: list[SensorySnapshot] = []

    def update(self, snapshot: SensorySnapshot) -> None:
        self._buffer.append(snapshot)
        if len(self._buffer) > self.capacity:
            self._buffer = self._buffer[-self.capacity :]

    def latest(self) -> SensorySnapshot | None:
        return self._buffer[-1] if self._buffer else None
