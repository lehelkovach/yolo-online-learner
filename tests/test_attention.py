from attention.scheduler import AttentionScheduler, InhibitionOfReturn
from perception.bbp import BBP, BoundingBox


def _bbp(frame_idx: int, x1: float, y1: float, x2: float, y2: float, conf: float) -> BBP:
    return BBP(
        frame_idx=frame_idx,
        timestamp_s=float(frame_idx),
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=conf,
        class_id=None,
    )


def test_attention_selects_highest_confidence() -> None:
    sched = AttentionScheduler(ior=InhibitionOfReturn(max_memory=0))
    bbps = [
        _bbp(0, 0, 0, 10, 10, 0.2),
        _bbp(0, 0, 0, 10, 10, 0.9),
        _bbp(0, 0, 0, 10, 10, 0.5),
    ]
    sel = sched.select(bbps)
    assert sel is not None
    assert sel.bbp_index == 1


def test_inhibition_of_return_penalizes_overlap() -> None:
    ior = InhibitionOfReturn(max_memory=5, iou_threshold=0.3, penalty=0.9)
    sched = AttentionScheduler(ior=ior)

    a = _bbp(0, 0, 0, 10, 10, 0.8)
    b = _bbp(1, 0, 0, 10, 10, 0.85)  # overlaps strongly, slightly higher conf
    c = _bbp(1, 100, 100, 110, 110, 0.7)  # no overlap, lower conf

    sel0 = sched.select([a])
    assert sel0 is not None
    assert sel0.bbp_index == 0

    # With strong penalty, the overlapping higher-confidence BBP should lose
    # to the non-overlapping one.
    sel1 = sched.select([b, c])
    assert sel1 is not None
    assert sel1.bbp_index == 1

