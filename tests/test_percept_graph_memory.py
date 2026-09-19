from __future__ import annotations

import pytest

from graph.memory_graph import (
    OBSERVATION_OF,
    SEEN_WITH,
    SIMILAR_TO,
    MemoryGraph,
    MemoryGraphConfig,
)
from memory.episodes import ObservationEvent
from perception.bbp import BoundingBox

SPACE = "test_space"


def _event(observation_id: str, object_id: str, frame_idx: int) -> ObservationEvent:
    return ObservationEvent(
        observation_id=observation_id,
        object_id=object_id,
        timestamp_s=frame_idx / 10.0,
        frame_idx=frame_idx,
        bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
        embedding_space_id=SPACE,
        embedding=(1.0, 0.0),
        confidence=0.9,
    )


def test_observation_creates_typed_nodes_and_edges() -> None:
    graph = MemoryGraph()

    graph.record_observation(
        _event("obs-1", "obj-a", 0),
        visibility="visible",
        nearest_prototype_id="proto-1",
        nearest_similarity=0.75,
    )

    assert graph.node_type("obs-1") == "observation"
    assert graph.node_type("obj-a") == "object"
    assert graph.node_type("proto-1") == "prototype"
    assert graph.g.edges["obs-1", "obj-a"]["edge_type"] == OBSERVATION_OF
    assert graph.g.edges["obs-1", "proto-1"]["edge_type"] == SIMILAR_TO
    assert graph.g.edges["obs-1", "proto-1"]["weight"] == pytest.approx(0.75)
    assert graph.g.nodes["obj-a"]["observation_count"] == 1
    assert graph.g.nodes["obj-a"]["visibility"] == "visible"
    assert graph.g.nodes["proto-1"]["match_count"] == 1
    assert graph.counts()["nodes"] == 3 and graph.counts()["edges"] == 2


def test_duplicate_observation_insertion_is_rejected() -> None:
    graph = MemoryGraph()
    graph.record_observation(_event("obs-1", "obj-a", 0))

    with pytest.raises(ValueError):
        graph.record_observation(_event("obs-1", "obj-b", 1))

    assert graph.observations_of("obj-a") == ["obs-1"]
    assert not graph.has_node("obj-b")


def test_observations_accumulate_on_object_in_frame_order() -> None:
    graph = MemoryGraph()
    graph.record_observation(_event("obs-2", "obj-a", 5), visibility="visible")
    graph.record_observation(_event("obs-1", "obj-a", 1), visibility="visible")
    graph.record_observation(_event("obs-3", "obj-b", 2))

    assert graph.observations_of("obj-a") == ["obs-1", "obs-2"]
    assert graph.g.nodes["obj-a"]["observation_count"] == 2
    assert graph.g.nodes["obj-a"]["last_seen_s"] == pytest.approx(0.1)
    assert graph.observations_of("missing") == []


def test_seen_with_is_canonical_and_counts_frames() -> None:
    graph = MemoryGraph()

    assert graph.record_cooccurrence(["obj-b", "obj-a", "obj-c"], timestamp_s=0.0) == 3
    assert graph.record_cooccurrence(["obj-a", "obj-b"], timestamp_s=0.1) == 1
    assert graph.record_cooccurrence(["obj-a"], timestamp_s=0.2) == 0

    assert graph.g.has_edge("obj-a", "obj-b") and not graph.g.has_edge("obj-b", "obj-a")
    assert graph.g.edges["obj-a", "obj-b"]["edge_type"] == SEEN_WITH
    assert graph.seen_with("obj-a") == {"obj-b": 2.0, "obj-c": 1.0}
    assert graph.seen_with("obj-c") == {"obj-a": 1.0, "obj-b": 1.0}
    assert graph.counts()["seen_with_edges"] == 3


def test_visibility_updates_do_not_create_nodes() -> None:
    graph = MemoryGraph()
    graph.record_observation(_event("obs-1", "obj-a", 0), visibility="visible")

    graph.set_visibility("obj-a", "lost")
    graph.set_visibility("obj-missing", "lost")

    assert graph.g.nodes["obj-a"]["visibility"] == "lost"
    assert not graph.has_node("obj-missing")


def test_observation_capacity_evicts_oldest_nodes_only() -> None:
    graph = MemoryGraph(MemoryGraphConfig(max_observation_nodes=2))
    for idx in range(4):
        graph.record_observation(_event(f"obs-{idx}", "obj-a", idx), nearest_prototype_id="p",
                                 nearest_similarity=1.0)

    counts = graph.counts()
    assert counts["observation_nodes"] == 2
    assert counts["evicted_observations"] == 2
    assert graph.observations_of("obj-a") == ["obs-2", "obs-3"]
    assert graph.has_node("obj-a") and graph.has_node("p")
    assert graph.g.nodes["obj-a"]["observation_count"] == 4  # history count survives
    with pytest.raises(ValueError):
        MemoryGraphConfig(max_observation_nodes=0)


def test_snapshot_round_trip_is_lossless_and_ordered() -> None:
    graph = MemoryGraph(MemoryGraphConfig(max_observation_nodes=3))
    graph.record_observation(_event("obs-b", "obj-2", 1), visibility="visible",
                             nearest_prototype_id="proto-1", nearest_similarity=0.9)
    graph.record_observation(_event("obs-a", "obj-1", 2), visibility="occluded")
    graph.record_cooccurrence(["obj-1", "obj-2"], timestamp_s=0.2)
    snapshot = graph.snapshot()

    restored = MemoryGraph(MemoryGraphConfig(max_observation_nodes=3))
    restored.load_snapshot(snapshot)

    assert [n["id"] for n in snapshot["nodes"]] == sorted(n["id"] for n in snapshot["nodes"])
    assert restored.snapshot() == snapshot
    assert restored.counts() == graph.counts()
    restored.record_observation(_event("obs-c", "obj-1", 3))
    restored.record_observation(_event("obs-d", "obj-1", 4))
    # obs-b (frame 1) was the oldest retained node, so it is evicted first.
    assert restored.observations_of("obj-1") == ["obs-a", "obs-c", "obs-d"]
    assert restored.observations_of("obj-2") == []
    assert restored.has_node("obj-2")


def test_replay_produces_identical_snapshots() -> None:
    def scenario() -> dict[str, object]:
        graph = MemoryGraph()
        graph.record_observation(_event("obs-1", "obj-a", 0), visibility="visible",
                                 nearest_prototype_id="proto-1", nearest_similarity=0.8)
        graph.record_cooccurrence(["obj-a", "obj-b"], timestamp_s=0.0)
        graph.record_observation(_event("obs-2", "obj-b", 1), visibility="visible")
        graph.set_visibility("obj-a", "occluded")
        return graph.snapshot()

    assert scenario() == scenario()
