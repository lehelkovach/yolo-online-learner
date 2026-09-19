from __future__ import annotations

import dataclasses

import pytest

from features.encoder import EmbeddingResult, EmbeddingSpaceMismatchError
from memory.episodes import EpisodicMemory, EpisodicMemoryConfig, ObservationEvent
from objects.ids import seeded_uuid_factory, sequential_id_factory
from perception.bbp import BBP, BoundingBox

SPACE = "test_space"


def _bbp(frame_idx: int, x1: float = 0.0, confidence: float = 0.9) -> BBP:
    return BBP(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 10.0,
        bbox=BoundingBox(x1, 0.0, x1 + 10.0, 10.0),
        confidence=confidence,
    )


def _emb(*values: float) -> EmbeddingResult:
    return EmbeddingResult(vector=tuple(values), space_id=SPACE)


def _memory(**config: object) -> EpisodicMemory:
    return EpisodicMemory(
        embedding_space_id=SPACE,
        config=EpisodicMemoryConfig(**config),  # type: ignore[arg-type]
        id_factory=sequential_id_factory("obs"),
    )


def test_events_append_in_order_and_are_immutable() -> None:
    memory = _memory()

    first = memory.record(bbp=_bbp(0), object_id="a", embedding=_emb(1.0, 0.0))
    second = memory.record(bbp=_bbp(1, x1=2.0), object_id="a", embedding=_emb(1.0, 0.0))

    assert first.observation_id == "obs-000001"
    assert second.observation_id == "obs-000002"
    assert memory.events == [first, second]
    assert len(memory) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.object_id = "b"  # type: ignore[misc]


def test_last_seen_and_observations_for_are_correct() -> None:
    memory = _memory()
    memory.record(bbp=_bbp(0), object_id="a", embedding=_emb(1.0, 0.0))
    memory.record(bbp=_bbp(1), object_id="b", embedding=_emb(0.0, 1.0))
    latest_a = memory.record(bbp=_bbp(2, x1=5.0), object_id="a", embedding=_emb(1.0, 0.0))

    assert memory.last_seen("a") == latest_a
    assert memory.last_seen("a").frame_idx == 2
    assert memory.last_seen("b").frame_idx == 1
    assert memory.last_seen("missing") is None
    assert [e.frame_idx for e in memory.observations_for("a")] == [0, 2]
    assert memory.observations_for("missing") == []


def test_history_remains_queryable_after_object_disappears() -> None:
    memory = _memory()
    memory.record(bbp=_bbp(0), object_id="a", embedding=_emb(1.0, 0.0))
    memory.record(bbp=_bbp(1), object_id="a", embedding=_emb(1.0, 0.0))
    for frame_idx in range(2, 40):  # a is never observed again
        memory.record(bbp=_bbp(frame_idx), object_id="b", embedding=_emb(0.0, 1.0))

    assert len(memory.observations_for("a")) == 2
    assert memory.last_seen("a").timestamp_s == pytest.approx(0.1)


def test_objects_seen_in_range_is_ordered_by_first_appearance() -> None:
    memory = _memory()
    memory.record(bbp=_bbp(0), object_id="a", embedding=_emb(1.0, 0.0))
    memory.record(bbp=_bbp(5), object_id="b", embedding=_emb(0.0, 1.0))
    memory.record(bbp=_bbp(6), object_id="a", embedding=_emb(1.0, 0.0))
    memory.record(bbp=_bbp(9), object_id="c", embedding=_emb(0.5, 0.5))

    assert memory.objects_seen_in_range(0.0, 0.9) == ["a", "b", "c"]
    assert memory.objects_seen_in_range(0.5, 0.6) == ["b", "a"]
    assert memory.objects_seen_in_range(0.7, 0.8) == []
    with pytest.raises(ValueError):
        memory.objects_seen_in_range(1.0, 0.0)


def test_capacity_drops_oldest_events_but_keeps_totals() -> None:
    memory = _memory(max_events=3)
    for frame_idx in range(5):
        memory.record(bbp=_bbp(frame_idx), object_id="a", embedding=_emb(1.0, 0.0))

    assert [e.frame_idx for e in memory.events] == [2, 3, 4]
    assert memory.counts() == {
        "retained": 3,
        "recorded_total": 5,
        "dropped_total": 2,
        "objects_with_history": 1,
    }
    assert memory.last_seen("a").frame_idx == 4


def test_capacity_eviction_forgets_objects_with_no_retained_events() -> None:
    memory = _memory(max_events=2)
    memory.record(bbp=_bbp(0), object_id="a", embedding=_emb(1.0, 0.0))
    memory.record(bbp=_bbp(1), object_id="b", embedding=_emb(0.0, 1.0))
    memory.record(bbp=_bbp(2), object_id="b", embedding=_emb(0.0, 1.0))

    assert memory.last_seen("a") is None
    assert memory.observations_for("a") == []
    assert memory.counts()["objects_with_history"] == 1


def test_rejects_wrong_space_time_regression_and_bad_config() -> None:
    memory = _memory()
    memory.record(bbp=_bbp(3), object_id="a", embedding=_emb(1.0, 0.0))

    with pytest.raises(EmbeddingSpaceMismatchError):
        memory.record(
            bbp=_bbp(4), object_id="a", embedding=EmbeddingResult((1.0,), "other")
        )
    with pytest.raises(ValueError):
        memory.record(bbp=_bbp(2), object_id="a", embedding=_emb(1.0, 0.0))
    with pytest.raises(ValueError):
        memory.record(bbp=_bbp(4), object_id="", embedding=_emb(1.0, 0.0))
    with pytest.raises(ValueError):
        EpisodicMemoryConfig(max_events=0)
    with pytest.raises(ValueError):
        EpisodicMemory(embedding_space_id="")
    assert len(memory) == 1


def test_event_dict_round_trip() -> None:
    memory = _memory()
    event = memory.record(
        bbp=_bbp(7, x1=3.5, confidence=0.42),
        object_id="a",
        embedding=_emb(0.6, 0.8),
        category_id="cat-1",
    )

    assert ObservationEvent.from_dict(event.to_dict()) == event
    assert memory.to_dict()["events"] == [event.to_dict()]


def test_replay_with_seeded_ids_is_deterministic() -> None:
    def scenario() -> dict[str, object]:
        memory = EpisodicMemory(
            embedding_space_id=SPACE, id_factory=seeded_uuid_factory(11, prefix="obs-")
        )
        for frame_idx, (obj, vec) in enumerate(
            [("a", (1.0, 0.0)), ("b", (0.0, 1.0)), ("a", (0.9, 0.1))]
        ):
            memory.record(bbp=_bbp(frame_idx), object_id=obj, embedding=_emb(*vec))
        return memory.to_dict()

    first = scenario()
    assert first == scenario()
    assert all(e["observation_id"].startswith("obs-") for e in first["events"])  # type: ignore[index]
