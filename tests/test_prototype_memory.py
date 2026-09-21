from __future__ import annotations

import pytest

from features.encoder import EmbeddingResult, EmbeddingSpaceMismatchError
from memory.prototypes import PrototypeMemory, PrototypeMemoryConfig
from objects.ids import seeded_uuid_factory, sequential_id_factory

SPACE = "test_space"


def _emb(*values: float) -> EmbeddingResult:
    return EmbeddingResult(vector=tuple(values), space_id=SPACE)


def _memory(**overrides: object) -> PrototypeMemory:
    config = PrototypeMemoryConfig(embedding_space_id=SPACE, **overrides)  # type: ignore[arg-type]
    return PrototypeMemory(config=config, id_factory=sequential_id_factory("proto"))


def test_first_observation_initializes_prototype() -> None:
    memory = _memory()

    update = memory.observe(_emb(1.0, 0.0), timestamp_s=0.5)

    assert update.created is True and update.updated is False
    assert update.prototype_id == "proto-000001"
    assert update.novelty.memory_empty is True
    assert update.novelty.novelty_score == 1.0
    assert update.prototype_count == 1
    prototype = memory.get("proto-000001")
    assert prototype.centroid == (1.0, 0.0)
    assert prototype.count == 1
    assert prototype.first_seen_s == prototype.last_seen_s == 0.5
    assert prototype.strength == 1.0


def test_repeated_identical_observations_leave_centroid_stable() -> None:
    memory = _memory()
    for step in range(5):
        update = memory.observe(_emb(0.6, 0.8), timestamp_s=float(step))

    assert update.created is False and update.updated is True
    assert len(memory) == 1
    prototype = memory.get("proto-000001")
    assert prototype.centroid == pytest.approx((0.6, 0.8))
    assert prototype.count == 5
    assert prototype.last_seen_s == 4.0


def test_running_mean_update_matches_hand_calculation() -> None:
    memory = _memory(match_threshold=0.5)
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)
    memory.observe(_emb(0.8, 0.6), timestamp_s=1.0)  # cos = 0.8 >= 0.5

    assert memory.get("proto-000001").centroid == pytest.approx((0.9, 0.3))

    memory.observe(_emb(0.6, 0.0), timestamp_s=2.0)
    expected = ((2 * 0.9 + 0.6) / 3, (2 * 0.3 + 0.0) / 3)
    assert memory.get("proto-000001").centroid == pytest.approx(expected)


def test_learning_rate_update_matches_hand_calculation() -> None:
    memory = _memory(update_rule="learning_rate", learning_rate=0.25, match_threshold=0.5)
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)
    memory.observe(_emb(0.8, 0.6), timestamp_s=1.0)

    expected = (1.0 + 0.25 * (0.8 - 1.0), 0.0 + 0.25 * (0.6 - 0.0))
    assert memory.get("proto-000001").centroid == pytest.approx(expected)


def test_update_rule_must_be_explicit() -> None:
    with pytest.raises(ValueError):
        PrototypeMemoryConfig(embedding_space_id=SPACE, update_rule="ema")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PrototypeMemoryConfig(embedding_space_id=SPACE, learning_rate=0.0)
    with pytest.raises(ValueError):
        PrototypeMemoryConfig(embedding_space_id="")


def test_dissimilar_observation_spawns_new_prototype() -> None:
    memory = _memory(match_threshold=0.9)
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)

    update = memory.observe(_emb(0.0, 1.0), timestamp_s=1.0)

    assert update.created is True
    assert update.prototype_id == "proto-000002"
    assert update.novelty.nearest_prototype_id == "proto-000001"
    assert update.novelty.novelty_score == pytest.approx(1.0)
    assert len(memory) == 2


def test_space_and_dimension_mismatch_raise() -> None:
    memory = _memory()
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)

    with pytest.raises(EmbeddingSpaceMismatchError):
        memory.observe(EmbeddingResult(vector=(1.0, 0.0), space_id="other"), timestamp_s=1.0)
    with pytest.raises(ValueError):
        memory.observe(_emb(1.0, 0.0, 0.0), timestamp_s=1.0)
    with pytest.raises(ValueError):
        memory.score_novelty(_emb(1.0, 0.0, 0.0))
    assert len(memory) == 1


def test_capacity_is_bounded_and_evicts_weakest_deterministically() -> None:
    memory = _memory(max_prototypes=2, match_threshold=0.99)
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)
    memory.observe(_emb(0.0, 1.0), timestamp_s=1.0)
    memory.observe(_emb(0.0, 1.0), timestamp_s=2.0)  # strengthens proto-2

    update = memory.observe(_emb(-1.0, 0.0), timestamp_s=3.0)

    assert update.created is True
    assert update.evicted_prototype_id == "proto-000001"
    assert len(memory) == 2
    assert [p.prototype_id for p in memory.prototypes] == ["proto-000002", "proto-000003"]


def test_strength_decay_is_opt_in() -> None:
    memory = _memory(match_threshold=0.99, strength_decay=0.5)
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)
    memory.observe(_emb(0.0, 1.0), timestamp_s=1.0)

    assert memory.get("proto-000001").strength == pytest.approx(0.5)
    assert memory.get("proto-000002").strength == pytest.approx(1.0)


def test_deterministic_replay_yields_identical_state() -> None:
    def scenario() -> dict[str, object]:
        config = PrototypeMemoryConfig(embedding_space_id=SPACE, match_threshold=0.8)
        memory = PrototypeMemory(config=config, id_factory=seeded_uuid_factory(3))
        vectors = [(1.0, 0.0), (0.9, 0.1), (0.0, 1.0), (0.1, 0.95), (1.0, 0.05)]
        for step, vector in enumerate(vectors):
            memory.observe(_emb(*vector), timestamp_s=float(step))
        return memory.to_dict()

    assert scenario() == scenario()
