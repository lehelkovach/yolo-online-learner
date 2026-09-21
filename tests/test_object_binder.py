from __future__ import annotations

import pytest

from features.encoder import EmbeddingResult, EmbeddingSpaceMismatchError
from objects.binder import BinderConfig, ObjectBinder, spatial_similarity
from objects.ids import seeded_uuid_factory, sequential_id_factory
from objects.memory import ObjectMemory
from objects.object_file import ObjectFile, VisibilityState
from perception.bbp import BBP, BoundingBox

SPACE = "test_space"


def _bbp(
    x1: float,
    *,
    frame_idx: int = 0,
    size: float = 10.0,
    class_id: int | None = None,
) -> BBP:
    return BBP(
        frame_idx=frame_idx,
        timestamp_s=float(frame_idx) / 10.0,
        bbox=BoundingBox(x1, 0.0, x1 + size, size),
        confidence=0.9,
        class_id=class_id,
    )


def _emb(*values: float) -> EmbeddingResult:
    return EmbeddingResult(vector=tuple(values), space_id=SPACE)


def _memory(**kwargs: object) -> ObjectMemory:
    return ObjectMemory(id_factory=sequential_id_factory(), **kwargs)  # type: ignore[arg-type]


def test_same_embedding_nearby_location_binds_to_same_object() -> None:
    memory = _memory()

    first = memory.observe(_bbp(0.0, frame_idx=0), _emb(1.0, 0.0))
    second = memory.observe(_bbp(2.0, frame_idx=1), _emb(1.0, 0.0))

    assert first.created is True and first.match_score is None
    assert second.created is False
    assert second.object_id == first.object_id == "obj-000001"
    assert second.match_score is not None and second.match_score >= 0.65
    assert second.candidate_count == 1
    assert second.reidentified is False
    assert memory.get(first.object_id).observation_count == 2


def test_clearly_different_embedding_creates_new_object() -> None:
    memory = _memory()

    first = memory.observe(_bbp(0.0, frame_idx=0), _emb(1.0, 0.0))
    second = memory.observe(_bbp(0.0, frame_idx=1), _emb(0.0, 1.0))

    assert second.created is True
    assert second.object_id != first.object_id
    assert len(memory) == 2
    assert second.runner_up_score is not None and second.runner_up_score < 0.65


def test_identical_class_id_alone_cannot_force_identity() -> None:
    memory = _memory()

    first = memory.observe(_bbp(0.0, frame_idx=0, class_id=41), _emb(1.0, 0.0))
    far_and_different = memory.observe(
        _bbp(500.0, frame_idx=1, class_id=41), _emb(0.0, 1.0)
    )

    assert far_and_different.created is True
    assert far_and_different.object_id != first.object_id


def test_class_mismatch_is_only_a_weak_hint() -> None:
    memory = _memory()

    first = memory.observe(_bbp(0.0, frame_idx=0, class_id=1), _emb(1.0, 0.0))
    # Same appearance, same place, but YOLO flipped its class: still the same object.
    second = memory.observe(_bbp(0.0, frame_idx=1, class_id=2), _emb(1.0, 0.0))

    assert second.created is False
    assert second.object_id == first.object_id
    assert second.class_compatibility == 0.0


def test_candidate_tie_breaking_is_deterministic_and_prefers_older_object() -> None:
    binder = ObjectBinder()
    a = ObjectFile("a", 0.0, 0.0, 0, BoundingBox(0, 0, 10, 10), (1.0, 0.0), SPACE)
    b = ObjectFile("b", 0.0, 0.0, 0, BoundingBox(0, 0, 10, 10), (1.0, 0.0), SPACE)

    best, ranked = binder.best_match(_bbp(0.0, frame_idx=1), _emb(1.0, 0.0), [a, b])
    best_reversed, _ = binder.best_match(_bbp(0.0, frame_idx=1), _emb(1.0, 0.0), [b, a])

    assert best is not None and best.object_id == "a"
    assert ranked[0].score == ranked[1].score
    assert best_reversed is not None and best_reversed.object_id == "b"
    assert best.score == best_reversed.score


def test_scores_use_configured_weights_exactly() -> None:
    config = BinderConfig(appearance_weight=0.5, spatial_weight=0.4, class_weight=0.1)
    binder = ObjectBinder(config)
    obj = ObjectFile("a", 0.0, 0.0, 0, BoundingBox(0, 0, 10, 10), (1.0, 0.0), SPACE)

    score = binder.score(_bbp(0.0, frame_idx=1, class_id=None), _emb(1.0, 0.0), obj)

    assert score.appearance_similarity == pytest.approx(1.0)
    assert score.spatial_similarity == pytest.approx(1.0)
    assert score.class_compatibility == pytest.approx(1.0)
    assert score.score == pytest.approx(1.0)
    assert score.threshold == config.match_threshold


def test_negative_cosine_is_clamped_to_zero_appearance() -> None:
    binder = ObjectBinder()
    obj = ObjectFile("a", 0.0, 0.0, 0, BoundingBox(0, 0, 10, 10), (1.0, 0.0), SPACE)

    score = binder.score(_bbp(0.0, frame_idx=1), _emb(-1.0, 0.0), obj)

    assert score.appearance_similarity == 0.0


def test_absent_objects_use_reid_threshold_and_spatial_prior() -> None:
    config = BinderConfig(absent_spatial_prior=0.2)
    binder = ObjectBinder(config)
    obj = ObjectFile(
        "a", 0.0, 0.0, 0, BoundingBox(0, 0, 10, 10), (1.0, 0.0), SPACE,
        visibility=VisibilityState.LOST,
    )

    score = binder.score(_bbp(900.0, frame_idx=50), _emb(1.0, 0.0), obj)

    assert score.spatial_similarity == 0.2
    assert score.threshold == config.reid_threshold


def test_space_mismatch_raises_instead_of_silently_comparing() -> None:
    memory = _memory()
    memory.observe(_bbp(0.0, frame_idx=0), _emb(1.0, 0.0))

    with pytest.raises(EmbeddingSpaceMismatchError):
        memory.observe(
            _bbp(0.0, frame_idx=1), EmbeddingResult(vector=(1.0, 0.0), space_id="other")
        )


def test_spatial_similarity_softens_iou_with_distance_kernel() -> None:
    predicted = BoundingBox(0.0, 0.0, 10.0, 10.0)

    assert spatial_similarity(predicted, predicted, scale=1.0) == pytest.approx(1.0)
    disjoint = BoundingBox(12.0, 0.0, 22.0, 10.0)
    assert disjoint.iou(predicted) == 0.0
    assert 0.0 < spatial_similarity(disjoint, predicted, scale=1.0) < 1.0
    far = BoundingBox(1000.0, 0.0, 1010.0, 10.0)
    assert spatial_similarity(far, predicted, scale=1.0) == pytest.approx(0.0, abs=1e-9)


def test_binder_config_validation() -> None:
    with pytest.raises(ValueError):
        BinderConfig(match_threshold=1.5)
    with pytest.raises(ValueError):
        BinderConfig(appearance_weight=-0.1)
    with pytest.raises(ValueError):
        BinderConfig(spatial_distance_scale=0.0)


def _run_scenario(memory: ObjectMemory) -> list[str]:
    ids = []
    ids.append(memory.observe(_bbp(0.0, frame_idx=0), _emb(1.0, 0.0)).object_id)
    memory.advance(0)
    ids.append(memory.observe(_bbp(100.0, frame_idx=1), _emb(0.0, 1.0)).object_id)
    memory.advance(1)
    ids.append(memory.observe(_bbp(1.0, frame_idx=2), _emb(1.0, 0.0)).object_id)
    memory.advance(2)
    ids.append(memory.observe(_bbp(50.0, frame_idx=3), _emb(0.7, 0.7)).object_id)
    memory.advance(3)
    return ids


def test_replay_with_seeded_uuid_factory_reproduces_ids() -> None:
    first = _run_scenario(ObjectMemory(id_factory=seeded_uuid_factory(7)))
    second = _run_scenario(ObjectMemory(id_factory=seeded_uuid_factory(7)))
    other_seed = _run_scenario(ObjectMemory(id_factory=seeded_uuid_factory(8)))

    assert first == second
    assert first[0] == first[2]
    assert len(set(first)) == 3
    assert all(len(value) == 36 for value in first)
    assert first != other_seed
