"""Stage 8 (EXP-WM): capacity never exceeded, deterministic eviction, cueing reloads."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments import replay
from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID
from memory.prototypes import PrototypeMemoryConfig
from memory.working_memory import (
    WorkingMemory,
    WorkingMemoryConfig,
    empty_working_memory_metrics,
)
from tests.identity_scenario import attended_truth, identity_map, run_scripted_session


def _wm(**overrides: object) -> WorkingMemory:
    return WorkingMemory(config=WorkingMemoryConfig(**overrides))  # type: ignore[arg-type]


def test_capacity_is_never_exceeded_and_eviction_is_deterministic() -> None:
    wm = _wm(capacity=3)
    history = []
    for frame in range(20):
        update = wm.cue(f"obj-{frame}", frame_idx=frame, error=0.0)
        history.append(update)
        assert len(wm) <= 3 and update.occupancy == len(wm)
    assert wm.contents() == ["obj-17", "obj-18", "obj-19"]
    # Each eviction removed the least recently cued object (all other terms equal).
    assert [u.evicted_object_id for u in history[3:]] == [f"obj-{i}" for i in range(17)]
    assert wm.counts() == {
        "capacity": 3, "occupancy": 3, "max_occupancy": 3, "loads": 20, "reloads": 0,
        "refreshes": 0, "evictions": 17, "releases": 0,
    }


def test_refresh_reload_and_release() -> None:
    wm = _wm(capacity=2)
    assert wm.cue("a", frame_idx=0, error=1.0).loaded is True
    assert wm.cue("b", frame_idx=1, error=1.0).loaded is True

    again = wm.cue("a", frame_idx=2, error=0.1)
    assert again.refreshed and not again.loaded and again.evicted_object_id is None
    assert wm.slot("a").cue_count == 2 and wm.slot("a").last_error == 0.1

    third = wm.cue("c", frame_idx=3, error=1.0)
    assert third.evicted_object_id == "b"  # least recently cued
    assert wm.contents() == ["a", "c"]

    back = wm.cue("b", frame_idx=4, error=0.0)
    # "a" was cued twice with low error; "c" once with a failed prediction: "c" goes.
    assert back.loaded and back.reloaded and back.evicted_object_id == "c"
    assert wm.contents() == ["a", "b"] and wm.counts()["reloads"] == 1

    assert wm.release("a") is True and wm.release("a") is False
    assert wm.contents() == ["b"] and wm.counts()["releases"] == 1
    with pytest.raises(ValueError):
        wm.cue("", frame_idx=5, error=0.0)
    with pytest.raises(ValueError):
        wm.cue("x", frame_idx=5, error=-1.0)


def test_utility_weighs_recency_evidence_and_error() -> None:
    cfg = WorkingMemoryConfig(capacity=2, recency_half_life_frames=10.0,
                              evidence_weight=0.1, error_weight=0.5)
    wm = WorkingMemory(config=cfg)
    wm.cue("surprising", frame_idx=0, error=1.0)
    wm.cue("familiar", frame_idx=0, error=0.0)
    utilities = wm.utilities(10)
    assert utilities["familiar"] == pytest.approx(0.5 + 0.1 * math.log1p(1))
    assert utilities["surprising"] == pytest.approx(0.5 + 0.1 * math.log1p(1) - 0.5)
    # Same recency: the object whose last prediction failed is the one to let go.
    evicted = wm.cue("new", frame_idx=10, error=0.0)
    assert evicted.evicted_object_id == "surprising"
    assert evicted.evicted_utility == pytest.approx(utilities["surprising"])

    recency = _wm(capacity=2, eviction="recency")
    recency.cue("surprising", frame_idx=0, error=1.0)
    recency.cue("familiar", frame_idx=1, error=0.0)
    assert recency.cue("new", frame_idx=2, error=0.0).evicted_object_id == "surprising"
    assert set(recency.utilities(2)) == {"familiar", "new"}


def test_config_is_validated_and_schema_is_fixed() -> None:
    for bad in ({"capacity": 0}, {"eviction": "random"}, {"recency_half_life_frames": 0.0},
                {"evidence_weight": -1.0}):
        with pytest.raises(ValueError):
            WorkingMemoryConfig(**bad)  # type: ignore[arg-type]
    assert set(empty_working_memory_metrics("no_selection")) == {
        "status", "object_id", "loaded", "reloaded", "refreshed", "evicted_object_id",
        "evicted_utility", "occupancy",
    }


def test_scenario_releases_lost_objects_reloads_on_return_and_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = run_scripted_session(monkeypatch, tmp_path)
    frames = {e["frame_idx"]: e for e in events if e["event"] == "frame"}
    ids = {label: next(iter(v)) for label, v in identity_map(events).items()}

    assert frames[0]["working_memory"]["loaded"] is True
    assert frames[0]["working_memory"]["slots"] == [ids["A"]]
    assert frames[5]["working_memory"]["status"] == "no_selection"
    assert frames[5]["working_memory"]["slots"] == [ids["A"]]  # occluded objects stay in mind
    # A goes LOST at frame 20 (five misses) and leaves working memory.
    lost_frame = next(
        idx for idx, f in frames.items()
        if any(t["object_id"] == ids["A"] and t["to"] == "lost"
               for t in f["object_memory"]["transitions"])
    )
    assert frames[lost_frame]["working_memory"]["released"] == [ids["A"]]
    assert ids["A"] not in frames[lost_frame]["working_memory"]["slots"]
    # Cueing it again on return reloads it.
    assert attended_truth(frames[26]) == "A"
    back = frames[26]["working_memory"]
    assert back["loaded"] is True and back["reloaded"] is True
    assert all(f["working_memory"]["occupancy"] is None or f["working_memory"]["occupancy"] <= 4
               for f in frames.values())

    summary = next(e for e in events if e["event"] == "cognition_summary")["working_memory"]
    assert summary["capacity"] == 4 and summary["max_occupancy"] == 3
    assert summary["reloads"] == 1 and summary["releases"] == 1 and summary["evictions"] == 0
    assert replay.compare_traces(events, replay.replay_events(events)) == []


def test_small_capacity_evicts_and_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    from experiments.config import ExperimentConfig

    tight = replace(ExperimentConfig().working_memory, capacity=2)
    events = run_scripted_session(
        monkeypatch, tmp_path, working_memory=tight,
        prototypes=PrototypeMemoryConfig(embedding_space_id=SIMPLE_EMBEDDING_SPACE_ID),
    )
    frames = [e for e in events if e["event"] == "frame"]
    assert max(f["working_memory"]["occupancy"] or 0 for f in frames) == 2
    evictions = [f["working_memory"]["evicted_object_id"] for f in frames
                 if f["working_memory"]["evicted_object_id"]]
    assert evictions  # three objects compete for two slots
    assert events[0]["config"]["working_memory"]["capacity"] == 2
    assert replay.compare_traces(events, replay.replay_events(events)) == []
