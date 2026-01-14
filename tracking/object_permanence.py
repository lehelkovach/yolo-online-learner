from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from perception.bbp import BBP, BoundingBox


def _bbox_to_measurement(bbox: BoundingBox) -> np.ndarray:
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0
    w = max(1e-3, bbox.x2 - bbox.x1)
    h = max(1e-3, bbox.y2 - bbox.y1)
    return np.array([cx, cy, w, h], dtype=float)


def _measurement_to_bbox(meas: np.ndarray) -> BoundingBox:
    cx, cy, w, h = meas.tolist()
    w = max(1e-3, float(w))
    h = max(1e-3, float(h))
    x1 = float(cx - w / 2.0)
    y1 = float(cy - h / 2.0)
    x2 = float(cx + w / 2.0)
    y2 = float(cy + h / 2.0)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


class KalmanFilterCV:
    """Constant-velocity Kalman filter for bbox center/size state."""

    def __init__(
        self,
        initial_state: np.ndarray,
        *,
        initial_cov: float = 10.0,
        process_var: float = 1.0,
        measurement_var: float = 4.0,
    ) -> None:
        self.x = initial_state.astype(float)
        self.P = np.eye(8, dtype=float) * float(initial_cov)
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)

    def predict(self, dt: float) -> None:
        dt = float(max(0.0, dt))
        F = np.eye(8, dtype=float)
        for i in range(4):
            F[i, i + 4] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.eye(8, dtype=float) * self.process_var

    def update(self, z: np.ndarray) -> None:
        H = np.zeros((4, 8), dtype=float)
        for i in range(4):
            H[i, i] = 1.0
        R = np.eye(4, dtype=float) * self.measurement_var
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=float) - K @ H) @ self.P

    def measurement(self) -> np.ndarray:
        return self.x[:4].copy()


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    class_id: int | None
    bbox: BoundingBox
    confidence: float
    last_seen_s: float
    time_since_last_seen_s: float
    age: int
    hits: int
    is_ghost: bool


@dataclass
class _Track:
    track_id: int
    kf: KalmanFilterCV
    class_id: int | None
    confidence: float
    last_seen_s: float
    last_update_s: float
    age: int = 0
    hits: int = 0


class ObjectPermanenceTracker:
    """
    Lightweight tracker that keeps "ghost" objects alive during occlusion.

    This is a minimal, dependency-light wrapper to keep object state decaying
    rather than disappearing the moment YOLO misses a detection.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_age_s: float = 5.0,
        process_var: float = 1.0,
        measurement_var: float = 4.0,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age_s = float(max_age_s)
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(self, bbps: Sequence[BBP], *, timestamp_s: float) -> list[TrackedObject]:
        detections = list(bbps)

        for track in self._tracks:
            dt = timestamp_s - track.last_update_s
            track.kf.predict(dt)
            track.last_update_s = float(timestamp_s)
            track.age += 1

        predicted_boxes = [self._track_bbox(track) for track in self._tracks]
        matches, _, matched_dets = self._match_tracks(predicted_boxes, detections)

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            det = detections[det_idx]
            z = _bbox_to_measurement(det.bbox)
            track.kf.update(z)
            track.confidence = float(det.confidence)
            if track.class_id is None and det.class_id is not None:
                track.class_id = int(det.class_id)
            track.last_seen_s = float(timestamp_s)
            track.hits += 1

        survivors: list[_Track] = []
        for idx, track in enumerate(self._tracks):
            time_since = timestamp_s - track.last_seen_s
            if time_since <= self.max_age_s:
                survivors.append(track)
        self._tracks = survivors

        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            self._tracks.append(self._new_track(det, timestamp_s))

        return [self._to_tracked_object(track, timestamp_s) for track in self._tracks]

    def _new_track(self, det: BBP, timestamp_s: float) -> _Track:
        state = np.zeros(8, dtype=float)
        state[:4] = _bbox_to_measurement(det.bbox)
        kf = KalmanFilterCV(
            state, process_var=self.process_var, measurement_var=self.measurement_var
        )
        track = _Track(
            track_id=self._next_id,
            kf=kf,
            class_id=det.class_id,
            confidence=float(det.confidence),
            last_seen_s=float(timestamp_s),
            last_update_s=float(timestamp_s),
            age=1,
            hits=1,
        )
        self._next_id += 1
        return track

    def _track_bbox(self, track: _Track) -> BoundingBox:
        meas = track.kf.measurement()
        return _measurement_to_bbox(meas)

    def _to_tracked_object(self, track: _Track, timestamp_s: float) -> TrackedObject:
        time_since = float(max(0.0, timestamp_s - track.last_seen_s))
        return TrackedObject(
            track_id=track.track_id,
            class_id=track.class_id,
            bbox=self._track_bbox(track),
            confidence=track.confidence,
            last_seen_s=track.last_seen_s,
            time_since_last_seen_s=time_since,
            age=track.age,
            hits=track.hits,
            is_ghost=time_since > 0.0,
        )

    def _match_tracks(
        self, predicted_boxes: Sequence[BoundingBox], detections: Sequence[BBP]
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        candidates: list[tuple[float, int, int]] = []
        for t_idx, bbox in enumerate(predicted_boxes):
            t_class = self._tracks[t_idx].class_id
            for d_idx, det in enumerate(detections):
                if not self._class_compatible(t_class, det.class_id):
                    continue
                iou = bbox.iou(det.bbox)
                if iou < self.iou_threshold:
                    continue
                candidates.append((iou, t_idx, d_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])
        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _, t_idx, d_idx in candidates:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            matches.append((t_idx, d_idx))
        return matches, matched_tracks, matched_dets

    @staticmethod
    def _class_compatible(track_class: int | None, det_class: int | None) -> bool:
        if track_class is None or det_class is None:
            return True
        return int(track_class) == int(det_class)
