from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import NAMESPACE_OID, uuid5

from perception.bbp import BBP, BoundingBox


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    track_uuid: str
    bbox: BoundingBox
    class_id: int | None
    confidence: float | None
    frame_idx: int
    timestamp_s: float
    age: int
    hits: int
    misses: int
    state: str
    stability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _TrackState:
    track_id: int
    track_uuid: str
    bbox: BoundingBox
    class_id: int | None
    confidence: float
    age: int
    hits: int
    misses: int
    last_seen_frame: int
    last_seen_s: float
    last_step_frame: int
    last_step_s: float
    vx: float
    vy: float

    def predict(self, frame_idx: int, timestamp_s: float) -> None:
        dt = max(0.0, float(timestamp_s - self.last_step_s))
        if dt > 0.0:
            self.bbox = _shift_bbox(self.bbox, dx=self.vx * dt, dy=self.vy * dt)
        frame_advance = max(0, frame_idx - self.last_step_frame)
        self.age += frame_advance
        self.last_step_frame = frame_idx
        self.last_step_s = float(timestamp_s)

    def update(
        self,
        bbp: BBP,
        *,
        frame_idx: int,
        timestamp_s: float,
        velocity_smoothing: float,
        bbox_smoothing: float,
    ) -> None:
        prev_bbox = self.bbox
        dt = max(1e-6, float(timestamp_s - self.last_seen_s))
        vx_new, vy_new = _velocity(prev_bbox, bbp.bbox, dt)
        self.vx = velocity_smoothing * self.vx + (1.0 - velocity_smoothing) * vx_new
        self.vy = velocity_smoothing * self.vy + (1.0 - velocity_smoothing) * vy_new
        self.bbox = _blend_bbox(prev_bbox, bbp.bbox, alpha=bbox_smoothing)
        if bbp.class_id is not None:
            self.class_id = int(bbp.class_id)
        self.confidence = float(bbp.confidence)
        self.hits += 1
        self.misses = 0
        self.last_seen_frame = frame_idx
        self.last_seen_s = float(timestamp_s)
        self.last_step_frame = frame_idx
        self.last_step_s = float(timestamp_s)

    def mark_missed(self) -> None:
        self.misses += 1

    def stability(self) -> float:
        return min(1.0, self.hits / max(3, self.age))

    def to_tracked_object(self, frame_idx: int, timestamp_s: float) -> TrackedObject:
        state = "visible" if self.misses == 0 else "ghost"
        return TrackedObject(
            track_id=self.track_id,
            track_uuid=self.track_uuid,
            bbox=self.bbox,
            class_id=self.class_id,
            confidence=self.confidence,
            frame_idx=frame_idx,
            timestamp_s=float(timestamp_s),
            age=self.age,
            hits=self.hits,
            misses=self.misses,
            state=state,
            stability=self.stability(),
        )


class WorldModel:
    """
    Simple object permanence tracker with IoU association and ghost buffer.

    This is a lightweight baseline (no external deps). It prioritizes stable track IDs
    and visibility status; future stages can swap in Kalman/Hungarian without changing
    the public interface.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        min_confidence: float = 0.0,
        bbox_smoothing: float = 0.7,
        velocity_smoothing: float = 0.8,
        max_tracks: int = 1000,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_confidence = float(min_confidence)
        self.bbox_smoothing = float(bbox_smoothing)
        self.velocity_smoothing = float(velocity_smoothing)
        self.max_tracks = int(max_tracks)
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def step(self, bbps: list[BBP], *, frame_idx: int, timestamp_s: float) -> list[TrackedObject]:
        detections = [b for b in bbps if b.confidence >= self.min_confidence]

        for track in self._tracks:
            track.predict(frame_idx, timestamp_s)

        matches, unmatched_tracks, unmatched_dets = _greedy_iou_match(
            self._tracks, detections, self.iou_threshold
        )

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            track.update(
                detections[det_idx],
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                velocity_smoothing=self.velocity_smoothing,
                bbox_smoothing=self.bbox_smoothing,
            )

        for track_idx in unmatched_tracks:
            self._tracks[track_idx].mark_missed()

        for det_idx in unmatched_dets:
            if len(self._tracks) >= self.max_tracks:
                break
            self._tracks.append(
                _new_track(
                    detections[det_idx],
                    track_id=self._next_id,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.max_missed]
        return sorted(
            (t.to_tracked_object(frame_idx, timestamp_s) for t in self._tracks),
            key=lambda tr: tr.track_id,
        )


def _new_track(bbp: BBP, *, track_id: int, frame_idx: int, timestamp_s: float) -> _TrackState:
    track_uuid = str(uuid5(NAMESPACE_OID, f"track-{track_id}"))
    return _TrackState(
        track_id=track_id,
        track_uuid=track_uuid,
        bbox=bbp.bbox,
        class_id=bbp.class_id,
        confidence=float(bbp.confidence),
        age=1,
        hits=1,
        misses=0,
        last_seen_frame=frame_idx,
        last_seen_s=float(timestamp_s),
        last_step_frame=frame_idx,
        last_step_s=float(timestamp_s),
        vx=0.0,
        vy=0.0,
    )


def _greedy_iou_match(
    tracks: list[_TrackState],
    dets: list[BBP],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(dets)))
    matches: list[tuple[int, int]] = []

    while True:
        best: tuple[int, int, float] | None = None
        for ti in unmatched_tracks:
            for di in unmatched_dets:
                iou = tracks[ti].bbox.iou(dets[di].bbox)
                if iou < iou_threshold:
                    continue
                if best is None or iou > best[2]:
                    best = (ti, di, iou)
        if best is None:
            break
        ti, di, _ = best
        matches.append((ti, di))
        unmatched_tracks.remove(ti)
        unmatched_dets.remove(di)

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


def _shift_bbox(bbox: BoundingBox, *, dx: float, dy: float) -> BoundingBox:
    return BoundingBox(
        x1=bbox.x1 + dx,
        y1=bbox.y1 + dy,
        x2=bbox.x2 + dx,
        y2=bbox.y2 + dy,
    )


def _center(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def _velocity(prev: BoundingBox, curr: BoundingBox, dt: float) -> tuple[float, float]:
    cx_prev, cy_prev = _center(prev)
    cx_curr, cy_curr = _center(curr)
    return ((cx_curr - cx_prev) / dt, (cy_curr - cy_prev) / dt)


def _blend_bbox(prev: BoundingBox, curr: BoundingBox, *, alpha: float) -> BoundingBox:
    a = float(alpha)
    b = 1.0 - a
    return BoundingBox(
        x1=(a * prev.x1) + (b * curr.x1),
        y1=(a * prev.y1) + (b * curr.y1),
        x2=(a * prev.x2) + (b * curr.x2),
        y2=(a * prev.y2) + (b * curr.y2),
    )
