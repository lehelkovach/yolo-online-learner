from __future__ import annotations

import pytest

from features.encoder import EmbeddingSpaceMismatchError
from objects.object_file import ObjectFile, VisibilityState
from perception.bbp import BoundingBox


def _object(**overrides: object) -> ObjectFile:
    kwargs: dict[str, object] = {
        "object_id": "obj-1",
        "first_seen_s": 0.0,
        "last_seen_s": 0.0,
        "last_frame_idx": 0,
        "last_bbox": BoundingBox(0.0, 0.0, 10.0, 10.0),
        "prototype_embedding": (1.0, 0.0),
        "embedding_space_id": "space_a",
    }
    kwargs.update(overrides)
    return ObjectFile(**kwargs)  # type: ignore[arg-type]


def test_new_object_file_defaults() -> None:
    obj = _object()

    assert obj.observation_count == 1
    assert obj.visibility is VisibilityState.VISIBLE
    assert obj.category_id is None
    assert obj.identity_confidence == 1.0
    assert obj.first_frame_idx == 0
    assert obj.missed_frames == 0
    assert obj.center == (5.0, 5.0)


def test_update_running_mean_matches_hand_computation_and_keeps_id() -> None:
    obj = _object()

    obj.update(
        timestamp_s=0.5,
        frame_idx=1,
        bbox=BoundingBox(2.0, 0.0, 12.0, 10.0),
        embedding=(0.0, 1.0),
        embedding_space_id="space_a",
        class_id=3,
    )

    assert obj.object_id == "obj-1"
    assert obj.observation_count == 2
    assert obj.prototype_embedding == pytest.approx((0.5, 0.5))
    assert obj.last_seen_s == 0.5
    assert obj.last_frame_idx == 1
    assert obj.last_bbox == BoundingBox(2.0, 0.0, 12.0, 10.0)
    assert obj.last_class_id == 3
    assert obj.velocity_px_per_frame == pytest.approx((2.0, 0.0))

    obj.update(
        timestamp_s=1.0,
        frame_idx=2,
        bbox=BoundingBox(4.0, 0.0, 14.0, 10.0),
        embedding=(0.0, 1.0),
        embedding_space_id="space_a",
    )
    assert obj.observation_count == 3
    assert obj.prototype_embedding == pytest.approx((1.0 / 3.0, 2.0 / 3.0))


def test_update_learning_rate_rule_is_explicit() -> None:
    obj = _object()

    obj.update(
        timestamp_s=1.0,
        frame_idx=1,
        bbox=obj.last_bbox,
        embedding=(0.0, 1.0),
        embedding_space_id="space_a",
        appearance_eta=0.25,
    )

    assert obj.prototype_embedding == pytest.approx((0.75, 0.25))
    with pytest.raises(ValueError):
        obj.update(
            timestamp_s=2.0,
            frame_idx=2,
            bbox=obj.last_bbox,
            embedding=(0.0, 1.0),
            embedding_space_id="space_a",
            appearance_eta=1.5,
        )


def test_update_restores_visibility_and_clears_misses() -> None:
    obj = _object(visibility=VisibilityState.LOST, missed_frames=7)

    obj.update(
        timestamp_s=3.0,
        frame_idx=9,
        bbox=BoundingBox(20.0, 20.0, 30.0, 30.0),
        embedding=(1.0, 0.0),
        embedding_space_id="space_a",
    )

    assert obj.visibility is VisibilityState.VISIBLE
    assert obj.missed_frames == 0
    # Motion during absence is unknown, so velocity is not extrapolated.
    assert obj.velocity_px_per_frame == (0.0, 0.0)


def test_predicted_bbox_extrapolates_velocity() -> None:
    obj = _object(velocity_px_per_frame=(1.5, -0.5))

    predicted = obj.predicted_bbox(4)

    assert predicted == BoundingBox(6.0, -2.0, 16.0, 8.0)
    assert obj.predicted_bbox(0) == obj.last_bbox


def test_update_rejects_space_dimension_and_time_regressions() -> None:
    obj = _object()
    common = {"timestamp_s": 1.0, "bbox": obj.last_bbox}

    with pytest.raises(EmbeddingSpaceMismatchError):
        obj.update(frame_idx=1, embedding=(1.0, 0.0), embedding_space_id="space_b", **common)
    with pytest.raises(ValueError):
        obj.update(frame_idx=1, embedding=(1.0, 0.0, 0.0), embedding_space_id="space_a", **common)
    with pytest.raises(ValueError):
        obj.update(frame_idx=-1, embedding=(1.0, 0.0), embedding_space_id="space_a", **common)
    assert obj.observation_count == 1


def test_serialization_round_trip() -> None:
    obj = _object(category_id="cat-1", identity_confidence=0.8, last_class_id=2)
    obj.update(
        timestamp_s=1.0,
        frame_idx=1,
        bbox=BoundingBox(1.0, 1.0, 11.0, 11.0),
        embedding=(0.0, 1.0),
        embedding_space_id="space_a",
    )
    obj.visibility = VisibilityState.OCCLUDED
    obj.missed_frames = 2

    restored = ObjectFile.from_dict(obj.to_dict())

    assert restored == obj
    assert restored.to_dict() == obj.to_dict()


@pytest.mark.parametrize(
    "overrides",
    [
        {"object_id": ""},
        {"observation_count": 0},
        {"prototype_embedding": ()},
        {"prototype_embedding": (float("nan"), 0.0)},
    ],
)
def test_invalid_construction_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _object(**overrides)
