from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from perception.bbp import BBP, BoundingBox
from tracking.object_permanence import ObjectPermanenceTracker, TrackedObject


@dataclass(frozen=True, slots=True)
class ActiveObject:
    track_id: int
    class_id: int | None
    bbox: BoundingBox
    confidence: float
    status: str
    last_seen_s: float
    time_since_last_seen_s: float
    age: int
    hits: int


class WorldModel:
    """
    Minimal state manager that preserves object identity over occlusion.

    This owns a tracker and exposes a stable "ActiveObjects" view for downstream
    components, keeping recognition (BBPs) separate from state.
    """

    def __init__(
        self,
        *,
        tracker: ObjectPermanenceTracker | None = None,
        iou_threshold: float = 0.3,
        max_age_s: float = 5.0,
        process_var: float = 1.0,
        measurement_var: float = 4.0,
    ) -> None:
        self.tracker = tracker or ObjectPermanenceTracker(
            iou_threshold=iou_threshold,
            max_age_s=max_age_s,
            process_var=process_var,
            measurement_var=measurement_var,
        )
        self.active_objects: list[ActiveObject] = []

    def reset(self) -> None:
        self.tracker.reset()
        self.active_objects = []

    def update(self, bbps: Sequence[BBP], *, timestamp_s: float) -> list[ActiveObject]:
        tracked = self.tracker.update(bbps, timestamp_s=timestamp_s)
        self.active_objects = [self._to_active(obj) for obj in tracked]
        return self.active_objects

    def visible_objects(self) -> list[ActiveObject]:
        return [obj for obj in self.active_objects if obj.status == "visible"]

    def ghost_objects(self) -> list[ActiveObject]:
        return [obj for obj in self.active_objects if obj.status == "ghost"]

    @staticmethod
    def _to_active(obj: TrackedObject) -> ActiveObject:
        status = "ghost" if obj.is_ghost else "visible"
        return ActiveObject(
            track_id=obj.track_id,
            class_id=obj.class_id,
            bbox=obj.bbox,
            confidence=obj.confidence,
            status=status,
            last_seen_s=obj.last_seen_s,
            time_since_last_seen_s=obj.time_since_last_seen_s,
            age=obj.age,
            hits=obj.hits,
        )
