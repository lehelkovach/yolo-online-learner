from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from perception.bbp import BBP, BoundingBox


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Winner-take-all selection with inhibition-of-return (IOR)."""

    # Frames a attended region stays suppressed after selection.
    ior_ttl_frames: int = 8
    # IoU at or above this overlap triggers IOR penalty against a candidate.
    ior_iou_threshold: float = 0.25
    # Subtracted from the candidate score when overlapping an inhibited region.
    ior_penalty: float = 10.0


@dataclass(slots=True)
class _Inhibition:
    bbox: BoundingBox
    frames_left: int


class AttentionScheduler:
    """
    Serial conscious stream: exactly one BBP per frame (WTA) with IOR.

    Scores use ``salience`` when set, otherwise ``confidence``. Recently attended
    regions are suppressed so attention can shift to other candidates.
    """

    def __init__(self, config: AttentionConfig | None = None) -> None:
        self.config = config or AttentionConfig()
        self._inhibitions: list[_Inhibition] = []

    def select(self, bbps: Sequence[BBP]) -> BBP | None:
        """
        Choose at most one BBP for this frame.

        Returns ``None`` when ``bbps`` is empty; otherwise the WTA winner.
        """
        if not bbps:
            return None

        self._tick_inhibitions()

        best: BBP | None = None
        best_score = float("-inf")
        best_idx = len(bbps)

        for idx, bbp in enumerate(bbps):
            score = self._base_score(bbp) - self._ior_suppression(bbp.bbox)
            if score > best_score or (score == best_score and idx < best_idx):
                best_score = score
                best = bbp
                best_idx = idx

        if best is not None:
            self._register_inhibition(best.bbox)
        return best

    def _base_score(self, bbp: BBP) -> float:
        if bbp.salience is not None:
            return float(bbp.salience)
        return float(bbp.confidence)

    def _ior_suppression(self, bbox: BoundingBox) -> float:
        penalty = 0.0
        thr = self.config.ior_iou_threshold
        for inh in self._inhibitions:
            iou = bbox.iou(inh.bbox)
            if iou >= thr:
                penalty = max(penalty, self.config.ior_penalty * iou)
        return penalty

    def _tick_inhibitions(self) -> None:
        self._inhibitions = [
            _Inhibition(bbox=inh.bbox, frames_left=inh.frames_left - 1)
            for inh in self._inhibitions
            if inh.frames_left > 1
        ]

    def _register_inhibition(self, bbox: BoundingBox) -> None:
        self._inhibitions.append(
            _Inhibition(bbox=bbox, frames_left=self.config.ior_ttl_frames)
        )
