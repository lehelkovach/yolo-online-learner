from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from perception.bbp import BBP


@dataclass(frozen=True, slots=True)
class AttentionWeights:
    """Weights used to turn BBP signals into a WTA priority score."""

    salience: float = 1.0
    novelty: float = 1.0
    prediction_error: float = 1.0
    confidence: float = 0.25


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """Result of one attention tick."""

    bbp: BBP
    index: int
    raw_score: float
    inhibited_score: float
    inhibition: float


class AttentionScheduler:
    """
    Winner-take-all scheduler with inhibition-of-return.

    The scheduler is stateful: each selected BBP is stored briefly so overlapping
    candidates are down-weighted on subsequent ticks.
    """

    def __init__(
        self,
        *,
        weights: AttentionWeights | None = None,
        inhibition_strength: float = 1.0,
        inhibition_iou_threshold: float = 0.25,
        history_size: int = 8,
    ) -> None:
        if history_size < 0:
            raise ValueError("history_size must be non-negative")
        if inhibition_strength < 0.0:
            raise ValueError("inhibition_strength must be non-negative")
        if not (0.0 <= inhibition_iou_threshold <= 1.0):
            raise ValueError("inhibition_iou_threshold must be in [0, 1]")

        self.weights = weights or AttentionWeights()
        self.inhibition_strength = float(inhibition_strength)
        self.inhibition_iou_threshold = float(inhibition_iou_threshold)
        self._recent: deque[BBP] = deque(maxlen=history_size)

    def reset(self) -> None:
        """Clear inhibition history."""

        self._recent.clear()

    def select(self, bbps: Sequence[BBP]) -> AttentionSelection | None:
        """Select at most one BBP for the current attention tick."""

        if not bbps:
            return None

        best: AttentionSelection | None = None
        for index, bbp in enumerate(bbps):
            raw_score = self.score(bbp)
            inhibition = self.inhibition_for(bbp)
            selection = AttentionSelection(
                bbp=bbp,
                index=index,
                raw_score=raw_score,
                inhibited_score=raw_score - inhibition,
                inhibition=inhibition,
            )
            if best is None or self._is_better(selection, best):
                best = selection

        assert best is not None
        self._recent.append(best.bbp)
        return best

    def score(self, bbp: BBP) -> float:
        """Compute the bottom-up priority score before inhibition."""

        return (
            self.weights.salience * _signal(bbp.salience)
            + self.weights.novelty * _signal(bbp.novelty)
            + self.weights.prediction_error * _signal(bbp.prediction_error)
            + self.weights.confidence * bbp.confidence
        )

    def inhibition_for(self, bbp: BBP) -> float:
        """Compute overlap-based inhibition from recently selected BBPs."""

        if not self._recent:
            return 0.0

        max_overlap = 0.0
        for recent in self._recent:
            overlap = bbp.bbox.iou(recent.bbox)
            if _same_stream(bbp, recent) and overlap > max_overlap:
                max_overlap = overlap

        if max_overlap < self.inhibition_iou_threshold:
            return 0.0
        return self.inhibition_strength * max_overlap

    @staticmethod
    def _is_better(candidate: AttentionSelection, incumbent: AttentionSelection) -> bool:
        if candidate.inhibited_score != incumbent.inhibited_score:
            return candidate.inhibited_score > incumbent.inhibited_score
        if candidate.raw_score != incumbent.raw_score:
            return candidate.raw_score > incumbent.raw_score
        return candidate.index < incumbent.index


def _signal(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _same_stream(a: BBP, b: BBP) -> bool:
    """Treat matching known classes as the same attentional stream."""

    return a.class_id is None or b.class_id is None or a.class_id == b.class_id
