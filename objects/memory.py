from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from features.encoder import EmbeddingResult
from objects.binder import BinderConfig, BindingResult, ObjectBinder, spatial_similarity
from objects.ids import IdFactory, random_uuid_factory
from objects.object_file import ObjectFile, VisibilityState
from perception.bbp import BBP


@dataclass(frozen=True, slots=True)
class PermanenceConfig:
    """Frame thresholds for the VISIBLE -> OCCLUDED -> LOST -> DORMANT ladder."""

    occluded_after_frames: int = 1
    lost_after_frames: int = 5
    dormant_after_frames: int = 30
    # ``None`` keeps dormant objects forever; an int forgets them after that many misses.
    forget_after_frames: int | None = None
    # Unattended BBPs are assigned one-to-one to VISIBLE and OCCLUDED objects by
    # spatial similarity to the predicted box (IoU softened by the centre kernel);
    # an assignment at or above this threshold is a track update. ``None`` disables.
    glimpse_iou_threshold: float | None = 0.5
    # ``True``: a track update moves the object and re-estimates its velocity.
    # ``False``: the pre-Stage-9 glimpse, which only spares the miss counter.
    track_update_position: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.occluded_after_frames <= self.lost_after_frames:
            raise ValueError("require 1 <= occluded_after_frames <= lost_after_frames")
        if self.lost_after_frames > self.dormant_after_frames:
            raise ValueError("require lost_after_frames <= dormant_after_frames")
        if self.forget_after_frames is not None and (
            self.forget_after_frames <= self.dormant_after_frames
        ):
            raise ValueError("forget_after_frames must exceed dormant_after_frames")
        if self.glimpse_iou_threshold is not None and not (
            0.0 < self.glimpse_iou_threshold <= 1.0
        ):
            raise ValueError("glimpse_iou_threshold must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrackAssignment:
    """One unattended BBP assigned to one object this frame."""

    object_id: str
    bbp_index: int
    similarity: float
    previous_visibility: VisibilityState
    moved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "bbp_index": self.bbp_index,
            "similarity": self.similarity,
            "previous_visibility": self.previous_visibility.value,
            "moved": self.moved,
        }


@dataclass(frozen=True, slots=True)
class VisibilityTransition:
    object_id: str
    frame_idx: int
    from_state: VisibilityState
    to_state: VisibilityState | None  # ``None`` means the object was forgotten.
    missed_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "frame_idx": self.frame_idx,
            "from": self.from_state.value,
            "to": None if self.to_state is None else self.to_state.value,
            "missed_frames": self.missed_frames,
        }


@dataclass(slots=True)
class ObjectMemory:
    """
    Object files plus short-term permanence.

    Call ``observe`` for the attended percept, optionally ``glimpse`` for unattended
    BBPs, then ``advance`` exactly once per frame to apply miss counting.
    """

    binder_config: BinderConfig = field(default_factory=BinderConfig)
    permanence: PermanenceConfig = field(default_factory=PermanenceConfig)
    id_factory: IdFactory = field(default_factory=random_uuid_factory)
    _objects: dict[str, ObjectFile] = field(default_factory=dict, init=False)
    _seen_this_frame: dict[str, int] = field(default_factory=dict, init=False)
    _binder: ObjectBinder = field(init=False)

    def __post_init__(self) -> None:
        self._binder = ObjectBinder(self.binder_config)

    # ----- queries ---------------------------------------------------------------
    @property
    def objects(self) -> list[ObjectFile]:
        return list(self._objects.values())

    def get(self, object_id: str) -> ObjectFile:
        return self._objects[object_id]

    def __len__(self) -> int:
        return len(self._objects)

    def counts(self) -> dict[str, int]:
        counts = {state.value: 0 for state in VisibilityState}
        for obj in self._objects.values():
            counts[obj.visibility.value] += 1
        counts["total"] = len(self._objects)
        return counts

    # ----- updates ---------------------------------------------------------------
    def observe(self, bbp: BBP, embedding: EmbeddingResult) -> BindingResult:
        """Bind one attended percept to an existing or new object file."""
        candidates = [
            obj for obj in self._objects.values()
            if self._seen_this_frame.get(obj.object_id) != bbp.frame_idx
        ]
        best, ranked = self._binder.best_match(bbp, embedding, candidates)
        runner_up = None
        if best is not None:
            others = [s.score for s in ranked if s.object_id != best.object_id]
            runner_up = max(others) if others else None

        if best is None:
            object_id = self.id_factory()
            if object_id in self._objects:
                raise RuntimeError(f"id factory produced duplicate object id {object_id!r}")
            self._objects[object_id] = ObjectFile(
                object_id=object_id,
                first_seen_s=bbp.timestamp_s,
                last_seen_s=bbp.timestamp_s,
                last_frame_idx=bbp.frame_idx,
                first_frame_idx=bbp.frame_idx,
                last_bbox=bbp.bbox,
                prototype_embedding=embedding.vector,
                embedding_space_id=embedding.space_id,
                last_class_id=bbp.class_id,
            )
            self._seen_this_frame[object_id] = bbp.frame_idx
            return BindingResult(
                object_id=object_id,
                created=True,
                reidentified=False,
                match_score=None,
                candidate_count=len(candidates),
                runner_up_score=ranked[0].score if ranked else None,
            )

        obj = self._objects[best.object_id]
        previous = obj.visibility
        predicted = obj.predicted_bbox(bbp.frame_idx).as_xyxy()
        obj.update(
            timestamp_s=bbp.timestamp_s,
            frame_idx=bbp.frame_idx,
            bbox=bbp.bbox,
            embedding=embedding.vector,
            embedding_space_id=embedding.space_id,
            class_id=bbp.class_id,
            appearance_eta=self.binder_config.appearance_eta,
            identity_confidence=best.score,
        )
        self._seen_this_frame[obj.object_id] = bbp.frame_idx
        return BindingResult(
            object_id=obj.object_id,
            created=False,
            reidentified=previous is not VisibilityState.VISIBLE,
            match_score=best.score,
            candidate_count=len(candidates),
            previous_visibility=previous,
            appearance_similarity=best.appearance_similarity,
            spatial_similarity=best.spatial_similarity,
            class_compatibility=best.class_compatibility,
            runner_up_score=runner_up,
            predicted_bbox=predicted,
        )

    def track(self, bbps: Sequence[BBP], *, frame_idx: int) -> list[TrackAssignment]:
        """Assign unattended BBPs one-to-one to VISIBLE/OCCLUDED objects by position.

        Pairs are taken greedily by spatial similarity between the BBP and the
        object's predicted box (ties: object insertion order, then BBP order), each
        BBP and each object at most once, at or above ``glimpse_iou_threshold``.
        With ``track_update_position`` the object moves to the box and re-estimates
        its velocity; appearance is never touched because no embedding was made.
        Without it this is the old glimpse: the miss counter is spared, nothing moves.
        """
        threshold = self.permanence.glimpse_iou_threshold
        if threshold is None or not bbps:
            return []
        candidates = [
            obj for obj in self._objects.values()
            if obj.visibility in (VisibilityState.VISIBLE, VisibilityState.OCCLUDED)
            and self._seen_this_frame.get(obj.object_id) != frame_idx
        ]
        pairs: list[tuple[float, int, int]] = []
        for obj_rank, obj in enumerate(candidates):
            predicted = obj.predicted_bbox(frame_idx)
            for index, bbp in enumerate(bbps):
                value = spatial_similarity(
                    bbp.bbox, predicted, scale=self.binder_config.spatial_distance_scale
                )
                if value >= threshold:
                    pairs.append((value, obj_rank, index))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        taken_objects: set[int] = set()
        taken_bbps: set[int] = set()
        assignments: list[TrackAssignment] = []
        for value, obj_rank, index in pairs:
            if obj_rank in taken_objects or index in taken_bbps:
                continue
            taken_objects.add(obj_rank)
            taken_bbps.add(index)
            obj = candidates[obj_rank]
            previous = obj.visibility
            if self.permanence.track_update_position:
                obj.track(frame_idx=frame_idx, bbox=bbps[index].bbox)
            self._seen_this_frame[obj.object_id] = frame_idx
            assignments.append(
                TrackAssignment(
                    object_id=obj.object_id,
                    bbp_index=index,
                    similarity=float(value),
                    previous_visibility=previous,
                    moved=self.permanence.track_update_position,
                )
            )
        return assignments

    def glimpse(self, bbps: Sequence[BBP], *, frame_idx: int) -> list[str]:
        """Compatibility wrapper: object ids kept alive by ``track``."""
        return [assignment.object_id for assignment in self.track(bbps, frame_idx=frame_idx)]

    def advance(self, frame_idx: int) -> list[VisibilityTransition]:
        """Count a miss for every object not observed or glimpsed at ``frame_idx``."""
        cfg = self.permanence
        transitions: list[VisibilityTransition] = []
        forgotten: list[str] = []
        for obj in self._objects.values():
            if self._seen_this_frame.get(obj.object_id) == frame_idx:
                continue
            obj.missed_frames += 1
            missed = obj.missed_frames
            before = obj.visibility
            after: VisibilityState | None = before
            if cfg.forget_after_frames is not None and missed >= cfg.forget_after_frames:
                after = None
            elif missed >= cfg.dormant_after_frames:
                after = VisibilityState.DORMANT
            elif missed >= cfg.lost_after_frames:
                after = VisibilityState.LOST
            elif missed >= cfg.occluded_after_frames:
                after = VisibilityState.OCCLUDED
            if after is not before:
                transitions.append(
                    VisibilityTransition(
                        object_id=obj.object_id,
                        frame_idx=frame_idx,
                        from_state=before,
                        to_state=after,
                        missed_frames=missed,
                    )
                )
                if after is None:
                    forgotten.append(obj.object_id)
                else:
                    obj.visibility = after
        for object_id in forgotten:
            del self._objects[object_id]
        self._seen_this_frame = {
            k: v for k, v in self._seen_this_frame.items() if v >= frame_idx
        }
        return transitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "binder_config": self.binder_config.to_dict(),
            "permanence": self.permanence.to_dict(),
            "objects": [obj.to_dict() for obj in self._objects.values()],
        }
