from __future__ import annotations

from attention.scheduler import AttentionConfig, AttentionScheduler
from perception.bbp import BBP, BoundingBox


def _bbp(
    frame_idx: int,
    bbox: BoundingBox,
    *,
    confidence: float = 0.5,
    salience: float | None = None,
) -> BBP:
    return BBP(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 30.0,
        bbox=bbox,
        confidence=confidence,
        class_id=0,
        salience=salience,
    )


def test_select_returns_none_when_no_candidates() -> None:
    scheduler = AttentionScheduler()
    assert scheduler.select([]) is None


def test_select_exactly_one_bbp_when_candidates_present() -> None:
    scheduler = AttentionScheduler()
    bbps = [
        _bbp(0, BoundingBox(0, 0, 10, 10), confidence=0.3),
        _bbp(0, BoundingBox(20, 20, 30, 30), confidence=0.9),
        _bbp(0, BoundingBox(40, 40, 50, 50), confidence=0.6),
    ]
    selected = scheduler.select(bbps)
    assert selected is not None
    assert selected in bbps
    assert selected.confidence == 0.9


def test_select_uses_salience_over_confidence() -> None:
    scheduler = AttentionScheduler()
    bbps = [
        _bbp(0, BoundingBox(0, 0, 10, 10), confidence=0.99, salience=0.1),
        _bbp(0, BoundingBox(20, 20, 30, 30), confidence=0.1, salience=0.95),
    ]
    selected = scheduler.select(bbps)
    assert selected is not None
    assert selected.bbox.as_xyxy() == (20.0, 20.0, 30.0, 30.0)


def test_ior_prevents_fixation_on_same_region() -> None:
    scheduler = AttentionScheduler(
        AttentionConfig(ior_ttl_frames=6, ior_iou_threshold=0.2, ior_penalty=20.0)
    )
    sticky = BoundingBox(50, 50, 100, 100)
    alt = BoundingBox(200, 200, 250, 250)
    choices: list[BoundingBox] = []

    for frame in range(8):
        bbps = [
            _bbp(frame, sticky, confidence=0.95),
            _bbp(frame, alt, confidence=0.85),
        ]
        selected = scheduler.select(bbps)
        assert selected is not None
        choices.append(selected.bbox)

    sticky_hits = sum(1 for b in choices if b.iou(sticky) > 0.5)
    alt_hits = sum(1 for b in choices if b.iou(alt) > 0.5)
    assert sticky_hits < len(choices)
    assert alt_hits >= 2
    assert sticky_hits >= 2


def test_one_winner_per_frame_in_sequence() -> None:
    scheduler = AttentionScheduler()
    regions = [
        BoundingBox(0, 0, 10, 10),
        BoundingBox(15, 15, 25, 25),
        BoundingBox(30, 30, 40, 40),
    ]
    for frame in range(12):
        bbps = [
            _bbp(frame, regions[i], confidence=0.4 + 0.2 * (i % 3)) for i in range(3)
        ]
        selected = scheduler.select(bbps)
        assert selected is not None
        assert selected.frame_idx == frame
