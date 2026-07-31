from attention.scheduler import AttentionScheduler
from perception.bbp import BBP, BoundingBox


def _bbp(x1: float, *, confidence: float = 0.5, size: float = 10.0) -> BBP:
    return BBP(
        frame_idx=0,
        timestamp_s=0.0,
        bbox=BoundingBox(x1, 0.0, x1 + size, size),
        confidence=confidence,
    )


def test_selects_exactly_one_winner_from_non_empty_frame() -> None:
    scheduler = AttentionScheduler()
    bbps = [_bbp(0.0, confidence=0.9), _bbp(20.0, confidence=0.2)]

    selection = scheduler.select(bbps)

    assert selection is not None
    assert selection.bbp_index == 1
    assert selection.bbp is bbps[1]
    assert selection.to_metrics(len(bbps))["candidate_count"] == 2


def test_returns_none_for_empty_frame() -> None:
    assert AttentionScheduler().select([]) is None


def test_inhibition_of_return_prevents_stuck_fixation() -> None:
    scheduler = AttentionScheduler(ior_frames=1, ior_iou_threshold=0.5)
    bbps = [_bbp(0.0, confidence=0.2), _bbp(20.0, confidence=0.4)]

    first = scheduler.select(bbps)
    second = scheduler.select(bbps)

    assert first is not None and second is not None
    assert first.bbp_index == 0
    assert second.bbp_index == 1
    assert second.inhibited_count == 1


def test_all_inhibited_candidates_still_produce_one_winner() -> None:
    scheduler = AttentionScheduler(ior_frames=1, ior_iou_threshold=0.5)
    only_bbp = [_bbp(0.0)]

    assert scheduler.select(only_bbp) is not None
    selection = scheduler.select(only_bbp)

    assert selection is not None
    assert selection.bbp_index == 0
    assert selection.inhibited_count == 1


def test_ties_are_deterministic_by_input_order() -> None:
    scheduler = AttentionScheduler(ior_frames=0)
    bbps = [_bbp(0.0), _bbp(20.0)]

    assert scheduler.select(bbps).bbp_index == 0  # type: ignore[union-attr]
