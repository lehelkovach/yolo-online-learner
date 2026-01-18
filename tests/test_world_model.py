from perception.bbp import BBP, BoundingBox
from tracking.world_model import WorldModel


def _bbp(frame_idx: int, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> BBP:
    return BBP(
        frame_idx=frame_idx,
        timestamp_s=float(frame_idx),
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=conf,
        class_id=1,
    )


def test_world_model_keeps_id_stable() -> None:
    world = WorldModel(iou_threshold=0.3, max_missed=2)
    ids: list[int] = []
    for i in range(3):
        bbp = _bbp(i, 10 + i, 10, 30 + i, 30)
        tracks = world.step([bbp], frame_idx=i, timestamp_s=float(i))
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_world_model_ghost_then_expire() -> None:
    world = WorldModel(iou_threshold=0.3, max_missed=1)
    tracks = world.step([_bbp(0, 0, 0, 10, 10)], frame_idx=0, timestamp_s=0.0)
    assert tracks[0].state == "visible"
    tracks = world.step([], frame_idx=1, timestamp_s=1.0)
    assert len(tracks) == 1
    assert tracks[0].state == "ghost"
    tracks = world.step([], frame_idx=2, timestamp_s=2.0)
    assert tracks == []


def test_world_model_filters_low_confidence() -> None:
    world = WorldModel(min_confidence=0.5)
    tracks = world.step([_bbp(0, 0, 0, 10, 10, conf=0.1)], frame_idx=0, timestamp_s=0.0)
    assert tracks == []
