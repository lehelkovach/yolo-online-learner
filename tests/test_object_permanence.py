import pytest

from perception.bbp import BBP, BoundingBox
from tracking.object_permanence import ObjectPermanenceTracker


def _make_bbp(
    *,
    frame_idx: int,
    timestamp_s: float,
    bbox: BoundingBox,
    class_id: int = 1,
    confidence: float = 0.9,
) -> BBP:
    return BBP(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        embedding=None,
        salience=None,
        novelty=None,
        prediction_error=None,
    )


def test_tracker_keeps_ghost_until_max_age() -> None:
    tracker = ObjectPermanenceTracker(max_age_s=1.0, iou_threshold=0.1)
    t0 = 0.0
    bbp = _make_bbp(frame_idx=0, timestamp_s=t0, bbox=BoundingBox(10, 10, 20, 20))

    objects = tracker.update([bbp], timestamp_s=t0)
    assert len(objects) == 1
    assert objects[0].is_ghost is False

    t1 = 0.5
    objects = tracker.update([], timestamp_s=t1)
    assert len(objects) == 1
    assert objects[0].is_ghost is True
    assert objects[0].time_since_last_seen_s == pytest.approx(0.5)

    t2 = 1.6
    objects = tracker.update([], timestamp_s=t2)
    assert objects == []


def test_tracker_reuses_track_on_iou_match() -> None:
    tracker = ObjectPermanenceTracker(max_age_s=1.0, iou_threshold=0.1)
    t0 = 0.0
    bbp0 = _make_bbp(frame_idx=0, timestamp_s=t0, bbox=BoundingBox(10, 10, 20, 20))
    objects = tracker.update([bbp0], timestamp_s=t0)
    assert len(objects) == 1
    track_id = objects[0].track_id

    t1 = 0.1
    bbp1 = _make_bbp(frame_idx=1, timestamp_s=t1, bbox=BoundingBox(11, 10, 21, 20))
    objects = tracker.update([bbp1], timestamp_s=t1)
    assert len(objects) == 1
    assert objects[0].track_id == track_id
    assert objects[0].hits == 2
