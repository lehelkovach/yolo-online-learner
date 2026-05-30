from __future__ import annotations

import pytest

from attention.scheduler import AttentionScheduler, AttentionWeights
from perception.bbp import BBP, BoundingBox


def _bbp(
    *,
    x1: float,
    y1: float,
    confidence: float = 0.5,
    class_id: int | None = 1,
    salience: float | None = None,
    novelty: float | None = None,
    prediction_error: float | None = None,
) -> BBP:
    return BBP(
        frame_idx=0,
        timestamp_s=0.0,
        bbox=BoundingBox(x1, y1, x1 + 10, y1 + 10),
        confidence=confidence,
        class_id=class_id,
        salience=salience,
        novelty=novelty,
        prediction_error=prediction_error,
    )


def test_select_returns_none_for_empty_candidates() -> None:
    scheduler = AttentionScheduler()

    assert scheduler.select([]) is None


def test_selects_single_highest_priority_bbp() -> None:
    scheduler = AttentionScheduler(
        weights=AttentionWeights(salience=1.0, novelty=1.0, prediction_error=1.0, confidence=0.0)
    )
    low = _bbp(x1=0, y1=0, salience=0.1, novelty=0.1, prediction_error=0.1)
    high = _bbp(x1=20, y1=0, salience=0.2, novelty=0.4, prediction_error=0.3)

    selection = scheduler.select([low, high])

    assert selection is not None
    assert selection.bbp == high
    assert selection.index == 1


def test_inhibition_of_return_prevents_stuck_fixation() -> None:
    scheduler = AttentionScheduler(
        weights=AttentionWeights(salience=1.0, novelty=0.0, prediction_error=0.0, confidence=0.0),
        inhibition_strength=1.0,
        inhibition_iou_threshold=0.25,
    )
    first_spot = _bbp(x1=0, y1=0, salience=0.8)
    second_spot = _bbp(x1=40, y1=0, salience=0.7)

    first = scheduler.select([first_spot, second_spot])
    second = scheduler.select([first_spot, second_spot])

    assert first is not None
    assert second is not None
    assert first.bbp == first_spot
    assert second.bbp == second_spot
    assert second.inhibition == 0.0


def test_reset_clears_inhibition_history() -> None:
    scheduler = AttentionScheduler(
        weights=AttentionWeights(salience=1.0, novelty=0.0, prediction_error=0.0, confidence=0.0)
    )
    candidate = _bbp(x1=0, y1=0, salience=0.8)

    scheduler.select([candidate])
    assert scheduler.inhibition_for(candidate) > 0.0

    scheduler.reset()

    assert scheduler.inhibition_for(candidate) == 0.0


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="history_size"):
        AttentionScheduler(history_size=-1)
    with pytest.raises(ValueError, match="inhibition_strength"):
        AttentionScheduler(inhibition_strength=-0.1)
    with pytest.raises(ValueError, match="inhibition_iou_threshold"):
        AttentionScheduler(inhibition_iou_threshold=1.1)
