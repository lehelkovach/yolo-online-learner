from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from features.encoder import EmbeddingResult, cosine_similarity, ensure_same_space
from objects.object_file import ABSENT_STATES, ObjectFile, VisibilityState
from perception.bbp import BBP, BoundingBox


@dataclass(frozen=True, slots=True)
class BinderConfig:
    """All identity-association weights and thresholds live here, never in code."""

    match_threshold: float = 0.65
    # Stricter threshold for objects that are LOST/DORMANT (re-identification).
    reid_threshold: float = 0.75
    appearance_weight: float = 0.6
    spatial_weight: float = 0.3
    class_weight: float = 0.1
    # Spatial similarity = max(IoU, exp(-center_distance / (scale * box_diagonal))).
    spatial_distance_scale: float = 1.0
    # Spatial evidence used for LOST/DORMANT objects whose position is unknown.
    absent_spatial_prior: float = 0.5
    # Class compatibility when both class ids are known and differ.
    class_mismatch_similarity: float = 0.0
    # Class compatibility when at least one class id is unknown.
    class_unknown_similarity: float = 1.0
    # ``None`` = exact running mean; float = learning-rate update of object appearance.
    appearance_eta: float | None = None
    # Continuity gates for objects believed present (VISIBLE/OCCLUDED). A present object
    # is near its predicted box and about its last size, whatever its appearance says;
    # a percept violating either gate is vetoed rather than scored.
    present_min_spatial: float = 0.2
    # Max allowed ratio between the percept's box area and the predicted box area,
    # in either direction. ``None`` disables the size gate.
    present_max_area_ratio: float | None = 3.0

    def __post_init__(self) -> None:
        for name in ("match_threshold", "reid_threshold", "absent_spatial_prior",
                     "present_min_spatial"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.present_max_area_ratio is not None and self.present_max_area_ratio < 1.0:
            raise ValueError("present_max_area_ratio must be >= 1 or None")
        for name in ("appearance_weight", "spatial_weight", "class_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.spatial_distance_scale <= 0.0:
            raise ValueError("spatial_distance_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    object_id: str
    score: float
    appearance_similarity: float
    spatial_similarity: float
    class_compatibility: float
    visibility: VisibilityState
    threshold: float
    # Name of the continuity gate that rejected this candidate outright, if any.
    veto: str | None = None

    @property
    def accepted(self) -> bool:
        return self.veto is None and self.score >= self.threshold


@dataclass(frozen=True, slots=True)
class BindingResult:
    """Explicit outcome of associating one percept with object memory."""

    object_id: str
    created: bool
    reidentified: bool
    match_score: float | None
    candidate_count: int
    previous_visibility: VisibilityState | None = None
    appearance_similarity: float | None = None
    spatial_similarity: float | None = None
    class_compatibility: float | None = None
    runner_up_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "created": self.created,
            "reidentified": self.reidentified,
            "match_score": self.match_score,
            "candidate_count": self.candidate_count,
            "previous_visibility": (
                None if self.previous_visibility is None else self.previous_visibility.value
            ),
            "appearance_similarity": self.appearance_similarity,
            "spatial_similarity": self.spatial_similarity,
            "class_compatibility": self.class_compatibility,
            "runner_up_score": self.runner_up_score,
        }


def spatial_similarity(candidate: BoundingBox, predicted: BoundingBox, *, scale: float) -> float:
    """IoU, softened by an exponential center-distance kernel for fast movers."""
    iou = candidate.iou(predicted)
    diag = math.hypot(predicted.w, predicted.h)
    if diag <= 0.0:
        return iou
    cx = (candidate.x1 + candidate.x2) / 2.0 - (predicted.x1 + predicted.x2) / 2.0
    cy = (candidate.y1 + candidate.y2) / 2.0 - (predicted.y1 + predicted.y2) / 2.0
    kernel = math.exp(-math.hypot(cx, cy) / (scale * diag))
    return max(iou, kernel)


def _area_ratio_ok(candidate: BoundingBox, predicted: BoundingBox, max_ratio: float) -> bool:
    """True when neither box is more than ``max_ratio`` times the area of the other."""
    a, b = candidate.area, predicted.area
    if a <= 0.0 or b <= 0.0:
        return False
    return max(a / b, b / a) <= max_ratio


class ObjectBinder:
    """Decide whether an attended percept belongs to an existing object file."""

    def __init__(self, config: BinderConfig | None = None) -> None:
        self.config = config or BinderConfig()

    def class_compatibility(self, class_id: int | None, other: int | None) -> float:
        if class_id is None or other is None:
            return self.config.class_unknown_similarity
        return 1.0 if int(class_id) == int(other) else self.config.class_mismatch_similarity

    def score(self, bbp: BBP, embedding: EmbeddingResult, obj: ObjectFile) -> CandidateScore:
        ensure_same_space(obj.embedding_space_id, embedding.space_id, context=obj.object_id)
        cfg = self.config
        appearance = cosine_similarity(embedding.vector, obj.prototype_embedding)
        appearance = max(0.0, appearance)
        veto: str | None = None
        if obj.visibility in ABSENT_STATES:
            spatial = cfg.absent_spatial_prior
            threshold = cfg.reid_threshold
        else:
            predicted = obj.predicted_bbox(bbp.frame_idx)
            spatial = spatial_similarity(bbp.bbox, predicted, scale=cfg.spatial_distance_scale)
            threshold = cfg.match_threshold
            if spatial < cfg.present_min_spatial:
                veto = "spatial"
            elif cfg.present_max_area_ratio is not None and not _area_ratio_ok(
                bbp.bbox, predicted, cfg.present_max_area_ratio
            ):
                veto = "size"
        klass = self.class_compatibility(bbp.class_id, obj.last_class_id)
        total = (
            cfg.appearance_weight * appearance
            + cfg.spatial_weight * spatial
            + cfg.class_weight * klass
        )
        return CandidateScore(
            object_id=obj.object_id,
            score=total,
            appearance_similarity=appearance,
            spatial_similarity=spatial,
            class_compatibility=klass,
            visibility=obj.visibility,
            threshold=threshold,
            veto=veto,
        )

    def rank(
        self,
        bbp: BBP,
        embedding: EmbeddingResult,
        objects: Iterable[ObjectFile],
    ) -> list[CandidateScore]:
        """Score every candidate; ties keep insertion order (older object first)."""
        scores = [self.score(bbp, embedding, obj) for obj in objects]
        # ``sorted`` is stable, so equal scores preserve the caller's ordering.
        return sorted(scores, key=lambda s: -s.score)

    def best_match(
        self,
        bbp: BBP,
        embedding: EmbeddingResult,
        objects: Iterable[ObjectFile],
    ) -> tuple[CandidateScore | None, list[CandidateScore]]:
        ranked = self.rank(bbp, embedding, objects)
        for candidate in ranked:
            if candidate.accepted:
                return candidate, ranked
        return None, ranked
