"""Winner-take-all attention with inhibition of return.

Two priority policies, selected by ``AttentionConfig.mode``:

``proxy`` (the Stage 2 baseline, ``AttentionConfig(mode="proxy", ior_frames=1)``):
``(1 - confidence) * max(1 - confidence, eps) * area / max_area`` with spatial
inhibition of return over the last winners' boxes. Confidence and area are
temporary proxies and are logged under their proxy names.

``top_down`` (the default): each object file in memory supplies an expectation (its predicted
box, when it was last attended, and how surprising its last observation was).
A BBP explained by an expectation (spatial similarity at or above
``explain_similarity``) inherits that object's staleness and surprise; a BBP no
expectation explains is unexpected. Priority is a weighted sum of
``unexpectedness``, ``surprise``, ``staleness`` and detector ``confidence``, every
component logged. Inhibition of return is object-based: an explained BBP is
inhibited while its object was attended fewer than ``ior_frames`` frames ago, so
the scheduler keeps no state of its own in this mode and the decision replays from
object memory. Unexplained BBPs fall back to the spatial inhibition.

Ties keep the lower BBP index, so every decision is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from objects.binder import spatial_similarity
from perception.bbp import BBP, BoundingBox

AttentionMode = Literal["proxy", "top_down"]
AttentionMetrics = dict[str, Any]

_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Priority policy, weights and inhibition windows. Nothing here is implicit."""

    mode: AttentionMode = "top_down"
    # proxy: frames a winner's box stays inhibited; top_down: frames an attended
    # object stays inhibited (and the spatial fallback window for unexplained BBPs).
    # The Stage 2 proxy baseline is AttentionConfig(mode="proxy", ior_frames=1).
    ior_frames: int = 3
    ior_iou_threshold: float = 0.5
    # Spatial similarity at or above which an expectation explains a BBP.
    explain_similarity: float = 0.3
    # Frames since an object was attended at which its staleness saturates at 1.
    stale_after_frames: int = 15
    unexpected_weight: float = 1.0
    surprise_weight: float = 1.0
    stale_weight: float = 0.5
    confidence_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.mode not in ("proxy", "top_down"):
            raise ValueError(f"unknown attention mode {self.mode!r}")
        if self.ior_frames < 0:
            raise ValueError("ior_frames must be non-negative")
        if not 0.0 <= self.ior_iou_threshold <= 1.0:
            raise ValueError("ior_iou_threshold must be in [0, 1]")
        if not 0.0 < self.explain_similarity <= 1.0:
            raise ValueError("explain_similarity must be in (0, 1]")
        if self.stale_after_frames < 1:
            raise ValueError("stale_after_frames must be >= 1")
        for name in ("unexpected_weight", "surprise_weight", "stale_weight", "confidence_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Expectation:
    """What one object file expects this frame; produced top-down before selection."""

    object_id: str
    predicted_bbox: BoundingBox
    last_attended_frame: int
    surprise: float
    visibility: str


def empty_attention_metrics(mode: AttentionMode = "top_down") -> AttentionMetrics:
    """Return the stable JSONL schema for a frame without BBP candidates."""
    return {
        "selected_bbp_index": None,
        "mode": mode,
        "priority": None,
        "runner_up_priority": None,
        "margin": None,
        "candidate_count": 0,
        "inhibited_count": 0,
        "novelty_proxy": None,
        "error_proxy": None,
        "motion_proxy": None,
        "unexpectedness": None,
        "surprise": None,
        "staleness": None,
        "confidence": None,
        "explained_by": None,
    }


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """The single BBP admitted to the serial attention stream, with its evidence."""

    bbp_index: int
    bbp: BBP
    mode: AttentionMode
    priority: float
    inhibited_count: int
    runner_up_priority: float | None
    novelty_proxy: float | None = None
    error_proxy: float | None = None
    motion_proxy: float | None = None
    unexpectedness: float | None = None
    surprise: float | None = None
    staleness: float | None = None
    confidence: float | None = None
    explained_by: str | None = None

    def to_metrics(self, candidate_count: int) -> AttentionMetrics:
        margin = (
            None if self.runner_up_priority is None else self.priority - self.runner_up_priority
        )
        return {
            "selected_bbp_index": self.bbp_index,
            "mode": self.mode,
            "priority": self.priority,
            "runner_up_priority": self.runner_up_priority,
            "margin": margin,
            "candidate_count": candidate_count,
            "inhibited_count": self.inhibited_count,
            "novelty_proxy": self.novelty_proxy,
            "error_proxy": self.error_proxy,
            "motion_proxy": self.motion_proxy,
            "unexpectedness": self.unexpectedness,
            "surprise": self.surprise,
            "staleness": self.staleness,
            "confidence": self.confidence,
            "explained_by": self.explained_by,
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    index: int
    bbp: BBP
    priority: float
    inhibited: bool
    novelty_proxy: float | None = None
    error_proxy: float | None = None
    motion_proxy: float | None = None
    unexpectedness: float | None = None
    surprise: float | None = None
    staleness: float | None = None
    confidence: float | None = None
    explained_by: str | None = None


class AttentionScheduler:
    """Winner-take-all BBP selection with inhibition of return."""

    def __init__(
        self,
        config: AttentionConfig | None = None,
        *,
        ior_frames: int | None = None,
        ior_iou_threshold: float | None = None,
    ) -> None:
        cfg = config or AttentionConfig()
        overrides: dict[str, Any] = {}
        if ior_frames is not None:
            overrides["ior_frames"] = ior_frames
        if ior_iou_threshold is not None:
            overrides["ior_iou_threshold"] = ior_iou_threshold
        if overrides:
            cfg = AttentionConfig(**{**cfg.to_dict(), **overrides})
        self.config = cfg
        self._recent_winners: list[BoundingBox] = []

    @property
    def ior_frames(self) -> int:
        return self.config.ior_frames

    @property
    def ior_iou_threshold(self) -> float:
        return self.config.ior_iou_threshold

    def select(
        self,
        bbps: Sequence[BBP],
        *,
        expectations: Sequence[Expectation] | None = None,
    ) -> AttentionSelection | None:
        """Select exactly one candidate when BBPs are present, otherwise ``None``."""
        if not bbps:
            return None

        if self.config.mode == "top_down":
            candidates = self._score_top_down(bbps, expectations or ())
        else:
            max_area = max((bbp.bbox.area for bbp in bbps), default=0.0)
            candidates = [
                self._score_proxy(index, bbp, max_area=max_area)
                for index, bbp in enumerate(bbps)
            ]
        available = [candidate for candidate in candidates if not candidate.inhibited]
        pool = available or candidates
        ranked = sorted(pool, key=lambda candidate: (-candidate.priority, candidate.index))
        winner = ranked[0]
        runner_up = ranked[1].priority if len(ranked) > 1 else None

        if self.config.ior_frames:
            self._recent_winners.append(winner.bbp.bbox)
            self._recent_winners = self._recent_winners[-self.config.ior_frames :]

        return AttentionSelection(
            bbp_index=winner.index,
            bbp=winner.bbp,
            mode=self.config.mode,
            priority=winner.priority,
            inhibited_count=sum(candidate.inhibited for candidate in candidates),
            runner_up_priority=runner_up,
            novelty_proxy=winner.novelty_proxy,
            error_proxy=winner.error_proxy,
            motion_proxy=winner.motion_proxy,
            unexpectedness=winner.unexpectedness,
            surprise=winner.surprise,
            staleness=winner.staleness,
            confidence=winner.confidence,
            explained_by=winner.explained_by,
        )

    def reset(self) -> None:
        """Clear inhibition state at a stream boundary."""
        self._recent_winners.clear()

    # ----- policies ----------------------------------------------------------------
    def _spatially_inhibited(self, bbox: BoundingBox) -> bool:
        return any(
            bbox.iou(previous) >= self.config.ior_iou_threshold
            for previous in self._recent_winners
        )

    def _score_proxy(self, index: int, bbp: BBP, *, max_area: float) -> _Candidate:
        confidence = min(1.0, max(0.0, float(bbp.confidence)))
        novelty_proxy = 1.0 - confidence
        error_proxy = max(novelty_proxy, _EPS)
        motion_proxy = bbp.bbox.area / max_area if max_area > 0.0 else 0.0
        return _Candidate(
            index=index,
            bbp=bbp,
            priority=novelty_proxy * error_proxy * motion_proxy,
            inhibited=self._spatially_inhibited(bbp.bbox),
            novelty_proxy=novelty_proxy,
            error_proxy=error_proxy,
            motion_proxy=motion_proxy,
        )

    def _score_top_down(
        self, bbps: Sequence[BBP], expectations: Sequence[Expectation]
    ) -> list[_Candidate]:
        cfg = self.config
        candidates: list[_Candidate] = []
        for index, bbp in enumerate(bbps):
            best: Expectation | None = None
            best_similarity = 0.0
            for expectation in expectations:  # input order breaks ties deterministically
                value = spatial_similarity(bbp.bbox, expectation.predicted_bbox, scale=1.0)
                if value > best_similarity:
                    best, best_similarity = expectation, value
            explained = best is not None and best_similarity >= cfg.explain_similarity
            confidence = min(1.0, max(0.0, float(bbp.confidence)))
            if explained:
                assert best is not None
                since = max(0, int(bbp.frame_idx) - int(best.last_attended_frame))
                unexpectedness = 1.0 - best_similarity
                surprise = min(1.0, max(0.0, float(best.surprise)))
                staleness = min(1.0, since / cfg.stale_after_frames)
                inhibited = since < cfg.ior_frames
                explained_by: str | None = best.object_id
            else:
                unexpectedness = 1.0
                surprise = 0.0
                staleness = 1.0
                inhibited = self._spatially_inhibited(bbp.bbox)
                explained_by = None
            priority = (
                cfg.unexpected_weight * unexpectedness
                + cfg.surprise_weight * surprise
                + cfg.stale_weight * staleness
                + cfg.confidence_weight * confidence
            )
            candidates.append(
                _Candidate(
                    index=index,
                    bbp=bbp,
                    priority=priority,
                    inhibited=inhibited,
                    unexpectedness=unexpectedness,
                    surprise=surprise,
                    staleness=staleness,
                    confidence=confidence,
                    explained_by=explained_by,
                )
            )
        return candidates
