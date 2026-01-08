from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from perception.bbp import BBP, BoundingBox


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    frame_idx: int
    bbp_index: int
    bbp: BBP
    score: float


class InhibitionOfReturn:
    """
    Simple inhibition-of-return (IOR) over recently attended locations.

    Stores a short memory of the last attended bounding boxes and applies a penalty to
    candidates that overlap too much (IoU threshold).
    """

    def __init__(
        self,
        *,
        max_memory: int = 5,
        iou_threshold: float = 0.3,
        penalty: float = 0.5,
    ) -> None:
        self.max_memory = int(max_memory)
        self.iou_threshold = float(iou_threshold)
        self.penalty = float(penalty)
        self._recent: list[BoundingBox] = []

    def score_multiplier(self, bbox: BoundingBox) -> float:
        if not self._recent:
            return 1.0
        for b in self._recent:
            if bbox.iou(b) >= self.iou_threshold:
                return max(0.0, 1.0 - self.penalty)
        return 1.0

    def update(self, bbox: BoundingBox) -> None:
        self._recent.append(bbox)
        if len(self._recent) > self.max_memory:
            self._recent = self._recent[-self.max_memory :]


class AttentionScheduler:
    """
    Winner-take-most attention selector.

    Phase-2 scope: pick exactly one BBP per frame based on a simple salience score.
    Later stages will add novelty, prediction error, WM cues, motion, etc.
    """

    def __init__(
        self,
        *,
        ior: InhibitionOfReturn | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        self.ior = ior if ior is not None else InhibitionOfReturn()
        self.min_confidence = float(min_confidence)

    def base_score(self, bbp: BBP) -> float:
        # Minimal, dependency-free salience. Expand later.
        return float(bbp.confidence)

    def select(self, bbps: Iterable[BBP]) -> AttentionSelection | None:
        best: AttentionSelection | None = None
        for idx, b in enumerate(bbps):
            if b.confidence < self.min_confidence:
                continue
            score = self.base_score(b) * self.ior.score_multiplier(b.bbox)
            if best is None or score > best.score:
                best = AttentionSelection(frame_idx=b.frame_idx, bbp_index=idx, bbp=b, score=score)

        if best is not None:
            self.ior.update(best.bbp.bbox)
        return best

