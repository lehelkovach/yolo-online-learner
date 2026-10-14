"""End-to-end deterministic scenario for persistent object identity.

The scenario itself lives in ``tests/identity_scenario.py`` so the EXP-PERM harness
tests can reuse it. Identity must survive and the JSONL trace must replay exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import replay
from tests.identity_scenario import (
    attended_truth,
    identity_map,
    run_scripted_session,
    scenario,
)


def test_each_physical_object_keeps_one_stable_object_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    mapping = identity_map(events)

    assert set(mapping) == {"A", "B", "C"}
    assert all(len(ids) == 1 for ids in mapping.values()), mapping
    ids = {label: next(iter(values)) for label, values in mapping.items()}
    assert len(set(ids.values())) == 3
    assert all(value.startswith("obj-") for value in ids.values())

    summary = next(event for event in events if event["event"] == "cognition_summary")
    assert summary["object_counts"]["total"] == 3
    assert summary["prototype_count"] == 3


def test_occlusion_and_departure_produce_states_not_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = {event["frame_idx"]: event for event in events if event["event"] == "frame"}
    a_id = next(iter(identity_map(events)["A"]))

    # 3. Occlusion: one missed frame is OCCLUDED, never a deletion.
    occluded = [t for t in frames[5]["object_memory"]["transitions"] if t["object_id"] == a_id]
    assert occluded == [
        {"object_id": a_id, "frame_idx": 5, "from": "visible", "to": "occluded",
         "missed_frames": 1}
    ]
    assert frames[6]["object_memory"]["counts"]["total"] == 1

    # 4. Reappearance re-identifies the same object file.
    back = frames[7]["object_file"]
    assert back["object_id"] == a_id
    assert back["created"] is False
    assert back["reidentified"] is True
    assert back["previous_visibility"] == "occluded"
    assert back["visibility"] == "visible"

    # 7. Departure walks OCCLUDED -> LOST; the object remains in memory.
    lost = [
        t for idx in range(16, 26) for t in frames[idx]["object_memory"]["transitions"]
        if t["object_id"] == a_id
    ]
    assert [t["to"] for t in lost] == ["occluded", "lost"]
    assert frames[25]["object_memory"]["counts"] == {
        "visible": 2, "occluded": 0, "lost": 1, "dormant": 0, "total": 3
    }

    # 8. Return after the gap at a new location is a LOST -> VISIBLE re-identification.
    returned = frames[26]["object_file"]
    assert attended_truth(frames[26]) == "A"
    assert returned["object_id"] == a_id
    assert returned["reidentified"] is True
    assert returned["previous_visibility"] == "lost"
    assert returned["spatial_similarity"] == pytest.approx(0.5)
    assert returned["match_score"] >= 0.75


def test_unattended_objects_are_glimpsed_instead_of_flickering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = {event["frame_idx"]: event for event in events if event["event"] == "frame"}
    ids = {label: next(iter(v)) for label, v in identity_map(events).items()}

    # A and B are both present in frames 10-12; attention alternates, glimpses cover the rest.
    for idx in range(11, 13):
        attended = attended_truth(frames[idx])
        other = "B" if attended == "A" else "A"
        tracked = frames[idx]["object_memory"]["tracked"]
        assert [t["object_id"] for t in tracked] == [ids[other]]
        assert tracked[0]["moved"] is True and tracked[0]["previous_visibility"] == "visible"
        assert frames[idx]["object_memory"]["transitions"] == []


def test_novelty_is_memory_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = {event["frame_idx"]: event for event in events if event["event"] == "frame"}

    first = frames[0]["learning"]
    assert first["status"] == "ok"
    assert first["memory_empty"] is True and first["novelty"] == 1.0
    assert first["prototype_created"] is True

    again = frames[1]["learning"]
    assert again["novelty"] < 0.05 and again["band"] == "known_instance"
    assert again["prototype_updated"] is True
    assert again["nearest_prototype_id"] == first["prototype_id"]

    b_first = frames[10]["learning"] if attended_truth(frames[10]) == "B" else frames[11][
        "learning"
    ]
    assert b_first["prototype_created"] is True
    assert b_first["novelty"] > 0.5 and b_first["band"] == "novel"

    empty = frames[5]
    assert empty["object_file"]["status"] == "no_selection"
    assert empty["learning"]["status"] == "no_selection"


def test_trace_replays_exactly_from_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)

    replayed = replay.replay_events(events)

    assert len(replayed) == len(scenario())
    assert replay.compare_traces(events, replayed) == []


def test_graph_records_identity_provenance_and_cooccurrence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = {event["frame_idx"]: event for event in events if event["event"] == "frame"}
    ids = {label: next(iter(v)) for label, v in identity_map(events).items()}
    summary = next(event for event in events if event["event"] == "cognition_summary")

    session_path = next(tmp_path.glob("seed0/session_*.jsonl"))
    snapshot = json.loads(
        session_path.with_name(summary["graph_snapshot_path"]).read_text(encoding="utf-8")
    )
    nodes = {n["id"]: n for n in snapshot["nodes"]}
    edges = snapshot["edges"]

    # One object node per physical object, one observation node per binding.
    assert {n["node_type"] for n in nodes.values()} == {"object", "observation", "prototype"}
    assert summary["graph"]["object_nodes"] == 3
    assert summary["graph"]["prototype_nodes"] == 3
    assert summary["graph"]["observation_nodes"] == summary["episodes"]["retained"]
    assert summary["graph"]["observation_of_edges"] == summary["graph"]["observation_nodes"]
    assert nodes[ids["A"]]["visibility"] == "visible"

    # Every observation points at exactly the object it was bound to.
    for frame in frames.values():
        if frame["object_file"]["status"] != "ok":
            continue
        obs = frame["observation_id"]
        targets = [
            e["dst"] for e in edges
            if e["src"] == obs and e["edge_type"] == "OBSERVATION_OF"
        ]
        assert targets == [frame["object_file"]["object_id"]]

    # A, B and C were visible together, so SEEN_WITH links all three pairs.
    seen_with = {
        tuple(sorted((e["src"], e["dst"]))): e["weight"]
        for e in edges if e["edge_type"] == "SEEN_WITH"
    }
    def pair(x: str, y: str) -> tuple[str, str]:
        return tuple(sorted((ids[x], ids[y])))  # type: ignore[return-value]

    assert set(seen_with) == {pair("A", "B"), pair("A", "C"), pair("B", "C")}
    assert seen_with[pair("B", "C")] > seen_with[pair("A", "C")]
    # Only bound objects count: C is detected from frame 13 but not attended until A
    # leaves, so three visible pairs first occur once A returns at frame 26.
    assert frames[10]["graph"]["cooccurrence_pairs"] == 1
    assert frames[13]["graph"]["cooccurrence_pairs"] == 1
    assert frames[26]["graph"]["cooccurrence_pairs"] == 3


def test_graph_snapshot_round_trips_and_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from graph.memory_graph import MemoryGraph

    events = run_scripted_session(monkeypatch, tmp_path)
    session_path = next(tmp_path.glob("seed0/session_*.jsonl"))
    summary = next(event for event in events if event["event"] == "cognition_summary")
    snapshot_path = session_path.with_name(summary["graph_snapshot_path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    restored = MemoryGraph()
    restored.load_snapshot(snapshot)
    assert restored.snapshot() == snapshot

    _, learner = replay.replay_events_with_learner(events)
    assert learner.graph_snapshot() == snapshot
    assert replay.compare_graph_snapshot(session_path, events, learner) == []


def test_two_runs_with_same_seed_are_identical_and_seeds_change_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def trace(events: list[dict]) -> list[dict]:
        return [
            {key: event[key] for key in replay.TRACE_KEYS}
            for event in events
            if event["event"] == "frame"
        ]

    first = run_scripted_session(monkeypatch, tmp_path / "a")
    second = run_scripted_session(monkeypatch, tmp_path / "b")
    other = run_scripted_session(monkeypatch, tmp_path / "c", seed=1)

    assert trace(first) == trace(second)
    assert identity_map(first) != identity_map(other)
    assert first[0]["cognition_schema"] == {
        "object_id_prefix": "obj-",
        "prototype_id_prefix": "proto-",
        "observation_id_prefix": "obs-",
        "id_factory": "seeded_uuid4",
        "object_id_seed": 0,
        "prototype_id_seed": 1,
        "observation_id_seed": 2,
        "embedding_space_id": "simple_crop_v1",
    }


def test_every_binding_records_one_unique_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = [event for event in events if event["event"] == "frame"]

    bound = [f for f in frames if f["object_file"]["status"] == "ok"]
    unbound = [f for f in frames if f["object_file"]["status"] != "ok"]
    ids = [f["observation_id"] for f in bound]

    assert bound and unbound
    assert all(i is not None and i.startswith("obs-") for i in ids)
    assert len(set(ids)) == len(ids)
    assert all(f["observation_id"] is None for f in unbound)

    summary = next(event for event in events if event["event"] == "cognition_summary")
    assert summary["episodes"]["recorded_total"] == len(bound)
    assert summary["episodes"]["retained"] == len(bound)
    assert summary["episodes"]["objects_with_history"] == 3


def test_replay_cli_detects_a_tampered_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    # Keep the copy next to the original so the graph snapshot sidecar resolves.
    path = tmp_path / "seed0" / "session_copy.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    assert replay.main([str(path)]) == 0
    assert "0 mismatches" in capsys.readouterr().out

    tampered = [dict(e) for e in events]
    frame = next(e for e in tampered if e["event"] == "frame" and e["frame_idx"] == 7)
    frame["object_file"] = {**frame["object_file"], "reidentified": False}
    path.write_text("\n".join(json.dumps(e) for e in tampered) + "\n", encoding="utf-8")

    assert replay.main([str(path)]) == 1
    assert "frame[7].object_file.reidentified" in capsys.readouterr().out
