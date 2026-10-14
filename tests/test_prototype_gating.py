"""Spawn gating for prototype memory: hysteresis, cooldown and the non-learning baseline.

Defaults (hysteresis 1, cooldown 0, learning on) reproduce the previous behaviour
exactly; every existing prototype test runs unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import replay, sweep
from experiments.cognition import empty_learning_metrics
from features.encoder import EmbeddingResult
from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID
from memory.prototypes import PrototypeMemory, PrototypeMemoryConfig
from objects.ids import sequential_id_factory
from tests.identity_scenario import run_scripted_session

SPACE = "test_space"
AXES = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}


def _emb(axis: int) -> EmbeddingResult:
    return EmbeddingResult(vector=AXES[axis], space_id=SPACE)


def _memory(**overrides: object) -> PrototypeMemory:
    config = PrototypeMemoryConfig(embedding_space_id=SPACE, **overrides)  # type: ignore[arg-type]
    return PrototypeMemory(config=config, id_factory=sequential_id_factory("proto"))


def _decisions(memory: PrototypeMemory, axes: list[int]) -> list[str]:
    return [
        memory.observe(_emb(axis), timestamp_s=float(i)).decision
        for i, axis in enumerate(axes)
    ]


def test_defaults_name_the_decision_and_keep_the_old_behaviour() -> None:
    memory = _memory()
    assert _decisions(memory, [0, 0, 1, 1, 2]) == [
        "created", "updated", "created", "updated", "created"
    ]
    assert len(memory) == 3
    update = memory.observe(_emb(2), timestamp_s=9.0)
    assert update.to_dict()["decision"] == "updated" and update.prototype_id == "proto-000003"
    assert empty_learning_metrics("no_selection")["decision"] is None
    assert set(update.to_dict()) | {"status"} == set(empty_learning_metrics("no_selection"))


def test_hysteresis_requires_consecutive_unmatched_observations() -> None:
    memory = _memory(novelty_hysteresis=2)

    first = memory.observe(_emb(0), timestamp_s=0.0)
    assert first.decision == "created"  # the first prototype is never deferred

    deferred = memory.observe(_emb(1), timestamp_s=1.0)
    assert deferred.decision == "deferred_hysteresis"
    assert deferred.prototype_id is None and not deferred.created and not deferred.updated
    assert deferred.novelty.novelty_score == pytest.approx(1.0)
    assert deferred.novelty.nearest_prototype_id == "proto-000001"
    assert len(memory) == 1 and memory.get("proto-000001").centroid == AXES[0]

    assert memory.observe(_emb(1), timestamp_s=2.0).decision == "created"
    assert len(memory) == 2

    # A match in between resets the streak, so the novel axis has to persist again.
    assert _decisions(memory, [2, 0, 2, 2]) == [
        "deferred_hysteresis", "updated", "deferred_hysteresis", "created"
    ]
    assert len(memory) == 3


def test_cooldown_spaces_spawns_by_observation_count() -> None:
    memory = _memory(spawn_cooldown=2)

    assert _decisions(memory, [0, 1, 1, 1]) == [
        "created", "deferred_cooldown", "deferred_cooldown", "created"
    ]
    assert len(memory) == 2
    # Matches count as passed observations too.
    assert _decisions(memory, [2, 0, 0, 2]) == [
        "deferred_cooldown", "updated", "updated", "created"
    ]
    assert len(memory) == 3
    state = memory.to_dict()["state"]
    assert state == {"observations": 8, "unmatched_streaks": {}, "last_spawn_observation": 8}


def test_hysteresis_streaks_are_kept_per_key() -> None:
    """Attention alternates between objects, so each object carries its own streak."""
    memory = _memory(novelty_hysteresis=2)

    def observe(axis: int, key: str) -> str:
        return memory.observe(_emb(axis), timestamp_s=0.0, streak_key=key).decision

    assert observe(0, "objA") == "created"
    assert observe(1, "objB") == "deferred_hysteresis"
    assert observe(0, "objA") == "updated"  # A's match must not reset B's streak
    assert memory.to_dict()["state"]["unmatched_streaks"] == {"objB": 1}
    assert observe(1, "objB") == "created"
    assert memory.to_dict()["state"]["unmatched_streaks"] == {}
    # Two different keys never pool their evidence into one spawn.
    assert observe(2, "objC") == "deferred_hysteresis"
    assert observe(2, "objD") == "deferred_hysteresis"
    assert len(memory) == 2


def test_hysteresis_is_checked_before_cooldown() -> None:
    memory = _memory(novelty_hysteresis=2, spawn_cooldown=5)
    assert _decisions(memory, [0, 1, 1, 1]) == [
        "created", "deferred_hysteresis", "deferred_cooldown", "deferred_cooldown"
    ]


def test_disabled_learning_scores_novelty_but_never_changes_memory() -> None:
    memory = _memory(learning_enabled=False)

    updates = [memory.observe(_emb(axis), timestamp_s=float(i)) for i, axis in enumerate([0, 0, 1])]

    assert [u.decision for u in updates] == ["disabled"] * 3
    assert all(u.prototype_id is None and u.novelty.memory_empty for u in updates)
    assert all(u.novelty.novelty_score == 1.0 for u in updates)
    assert len(memory) == 0 and memory.to_dict()["state"]["observations"] == 3


def test_gating_config_is_validated() -> None:
    with pytest.raises(ValueError):
        PrototypeMemoryConfig(embedding_space_id=SPACE, novelty_hysteresis=0)
    with pytest.raises(ValueError):
        PrototypeMemoryConfig(embedding_space_id=SPACE, spawn_cooldown=-1)
    assert PrototypeMemoryConfig(embedding_space_id=SPACE).to_dict()["learning_enabled"] is True


def test_gated_sessions_log_decisions_and_replay_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gated = PrototypeMemoryConfig(
        embedding_space_id=SIMPLE_EMBEDDING_SPACE_ID, novelty_hysteresis=2, spawn_cooldown=3
    )
    events = run_scripted_session(monkeypatch, tmp_path / "gated", prototypes=gated)
    decisions = [
        e["learning"]["decision"] for e in events
        if e["event"] == "frame" and e["learning"]["status"] == "ok"
    ]
    assert decisions[0] == "created" and "deferred_hysteresis" in decisions
    assert events[0]["config"]["prototypes"]["novelty_hysteresis"] == 2
    assert replay.compare_traces(events, replay.replay_events(events)) == []

    summary = next(e for e in events if e["event"] == "cognition_summary")
    # B and C are attended only in alternation with other objects; their per-object
    # streaks still reach the hysteresis, so all three earn a prototype, later.
    assert summary["prototype_count"] == 3
    first_seen = {}
    for e in events:
        if e["event"] == "frame" and e["learning"].get("prototype_created"):
            first_seen[e["learning"]["prototype_id"]] = e["frame_idx"]
    assert sorted(first_seen.values())[1] > 11  # B's first attention (frame 11) was deferred

    disabled = PrototypeMemoryConfig(
        embedding_space_id=SIMPLE_EMBEDDING_SPACE_ID, learning_enabled=False
    )
    events = run_scripted_session(monkeypatch, tmp_path / "off", prototypes=disabled)
    frames = [e for e in events if e["event"] == "frame" and e["learning"]["status"] == "ok"]
    assert {f["learning"]["decision"] for f in frames} == {"disabled"}
    assert all(f["learning"]["prototype_id"] is None for f in frames)
    summary = next(e for e in events if e["event"] == "cognition_summary")
    assert summary["prototype_count"] == 0 and summary["graph"]["prototype_nodes"] == 0
    assert replay.compare_traces(events, replay.replay_events(events)) == []


def test_sweep_can_vary_the_gating(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments.identity_eval import GroundTruth
    from tests.identity_scenario import scenario_ground_truth

    events = run_scripted_session(monkeypatch, tmp_path)
    results = sweep.sweep(
        events, GroundTruth.from_dict(scenario_ground_truth()),
        [("logged", {}), ("no_learning", {"prototypes": {"learning_enabled": False}})],
    )
    assert results[0]["reproduces_log"] is True
    # Object identity does not depend on prototype memory, so the score is unchanged.
    assert results[1]["report"]["id_switches"] == 0
    assert results[1]["report"]["reacquisition"]["rate"] == 1.0
