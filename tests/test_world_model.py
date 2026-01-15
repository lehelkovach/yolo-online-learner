import pytest

from perception.bbp import BBP, BoundingBox
from tracking.world_model import WorldModel


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


def test_world_model_tracks_visible_and_ghost() -> None:
    world = WorldModel(max_age_s=1.0, iou_threshold=0.1)
    t0 = 0.0
    bbp = _make_bbp(frame_idx=0, timestamp_s=t0, bbox=BoundingBox(10, 10, 20, 20))

    active = world.update([bbp], timestamp_s=t0)
    assert len(active) == 1
    assert active[0].status == "visible"

    t1 = 0.4
    active = world.update([], timestamp_s=t1)
    assert len(active) == 1
    assert active[0].status == "ghost"
    assert active[0].time_since_last_seen_s == pytest.approx(0.4)

    t2 = 1.6
    active = world.update([], timestamp_s=t2)
    assert active == []
