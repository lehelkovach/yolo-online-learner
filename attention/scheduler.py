from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from perception.bbp import BBP, BoundingBox

AttentionMetrics = dict[str, float | int | None]


def empty_attention_metrics() -> AttentionMetrics:
    """Return the stable JSONL schema for a frame without BBP candidates."""
    return {
        "selected_bbp_index": None,
        "priority": None,
        "novelty_proxy": None,
        "error_proxy": None,
        "motion_proxy": None,
        "candidate_count": 0,
        "inhibited_count": 0,
    }


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """The single BBP admitted to the serial attention stream."""

    bbp_index: int
    bbp: BBP
    priority: float
    novelty_proxy: float
    error_proxy: float
    motion_proxy: float
    inhibited_count: int

    def to_metrics(self, candidate_count: int) -> AttentionMetrics:
        return {
            "selected_bbp_index": self.bbp_index,
            "priority": self.priority,
            "novelty_proxy": self.novelty_proxy,
            "error_proxy": self.error_proxy,
            "motion_proxy": self.motion_proxy,
            "candidate_count": candidate_count,
            "inhibited_count": self.inhibited_count,
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    index: int
    bbp: BBP
    priority: float
    novelty_proxy: float
    error_proxy: float
    motion_proxy: float
    inhibited: bool


class AttentionScheduler:
    """Winner-take-all BBP selection with spatial inhibition-of-return."""

    def __init__(self, *, ior_frames: int = 1, ior_iou_threshold: float = 0.5) -> None:
        if ior_frames < 0:
            raise ValueError("ior_frames must be non-negative")
        if not 0.0 <= ior_iou_threshold <= 1.0:
            raise ValueError("ior_iou_threshold must be in [0, 1]")
        self.ior_frames = ior_frames
        self.ior_iou_threshold = ior_iou_threshold
        self._recent_winners: list[BoundingBox] = []

    def select(self, bbps: Sequence[BBP]) -> AttentionSelection | None:
        """Select exactly one candidate when BBPs are present, otherwise ``None``."""
        if not bbps:
            return None

        max_area = max((bbp.bbox.area for bbp in bbps), default=0.0)
        candidates = [
            self._score(index, bbp, max_area=max_area) for index, bbp in enumerate(bbps)
        ]
        available = [candidate for candidate in candidates if not candidate.inhibited]
        pool = available or candidates
        winner = max(pool, key=lambda candidate: (candidate.priority, -candidate.index))

        if self.ior_frames:
            self._recent_winners.append(winner.bbp.bbox)
            self._recent_winners = self._recent_winners[-self.ior_frames :]

        return AttentionSelection(
            bbp_index=winner.index,
            bbp=winner.bbp,
            priority=winner.priority,
            novelty_proxy=winner.novelty_proxy,
            error_proxy=winner.error_proxy,
            motion_proxy=winner.motion_proxy,
            inhibited_count=sum(candidate.inhibited for candidate in candidates),
        )

    def reset(self) -> None:
        """Clear inhibition state at a stream boundary."""
        self._recent_winners.clear()

    def _score(self, index: int, bbp: BBP, *, max_area: float) -> _Candidate:
        confidence = min(1.0, max(0.0, float(bbp.confidence)))
        novelty_proxy = 1.0 - confidence
        error_proxy = max(novelty_proxy, 1e-6)
        motion_proxy = bbp.bbox.area / max_area if max_area > 0.0 else 0.0
        priority = novelty_proxy * error_proxy * motion_proxy
        inhibited = any(
            bbp.bbox.iou(previous) >= self.ior_iou_threshold
            for previous in self._recent_winners
        )
        return _Candidate(
            index=index,
            bbp=bbp,
            priority=priority,
            novelty_proxy=novelty_proxy,
            error_proxy=error_proxy,
            motion_proxy=motion_proxy,
            inhibited=inhibited,
        )
