from __future__ import annotations

import math

import numpy as np
import pytest

from features.simple_embedding import SIMPLE_EMBEDDING_DIM
from objects.prototype_bank import (
    PrototypeBank,
    PrototypeBankConfig,
    empty_prototype_bank_metrics,
    prototype_bank_schema,
)


def _unit(*values: float) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (SIMPLE_EMBEDDING_DIM,):
        raise AssertionError("test helper expects full embedding dim")
    return tuple(float(x) for x in vector / np.linalg.norm(vector))


def _axis(index: int, *, dim: int = SIMPLE_EMBEDDING_DIM) -> tuple[float, ...]:
    values = [0.0] * dim
    values[index] = 1.0
    return tuple(values)


def test_schema_records_locked_stage4_knobs() -> None:
    config = PrototypeBankConfig()
    assert prototype_bank_schema(config) == {
        "update_rule_id": "normalized_running_mean_v1",
        "embedding_space_id": "simple_crop_v1",
        "embedding_dim": 10,
        "kmax": 32,
        "match_threshold": 0.85,
        "spawn_cooldown_frames": 5,
        "novelty_hysteresis": 2,
        "learning_enabled": True,
        "identity": "perceptual_pattern",
        "match_score": "cosine_dot_l2",
    }


def test_empty_metrics_keep_fixed_keys() -> None:
    assert empty_prototype_bank_metrics(status="no_selection") == {
        "status": "no_selection",
        "prototype_count": 0,
        "match_id": None,
        "match_similarity": None,
        "novelty": None,
        "spawned": False,
        "evicted": False,
        "bank_full": False,
    }


def test_first_embedding_spawns_a_pattern() -> None:
    bank = PrototypeBank(PrototypeBankConfig(kmax=4, novelty_hysteresis=2))
    metrics = bank.observe(_axis(0), frame_idx=0)

    assert metrics["status"] == "ok"
    assert metrics["spawned"] is True
    assert metrics["prototype_count"] == 1
    assert metrics["match_id"] == "pattern-00000000"
    assert metrics["match_similarity"] == pytest.approx(1.0)
    assert metrics["novelty"] == pytest.approx(1.0)


def test_repeat_match_lowers_novelty_and_updates_mean() -> None:
    bank = PrototypeBank(PrototypeBankConfig(match_threshold=0.85))
    first = _axis(0)
    near = _unit(0.98, 0.1, *([0.0] * 8))

    spawn = bank.observe(first, frame_idx=0)
    match = bank.observe(near, frame_idx=1)

    assert spawn["spawned"] is True
    assert match["spawned"] is False
    assert match["match_id"] == spawn["match_id"]
    assert match["match_similarity"] is not None and match["match_similarity"] >= 0.85
    assert match["novelty"] == pytest.approx(1.0 - float(match["match_similarity"]))
    assert match["novelty"] < 0.2
    assert match["prototype_count"] == 1


def test_distribution_shift_spikes_novelty_then_spawns_after_hysteresis() -> None:
    bank = PrototypeBank(
        PrototypeBankConfig(
            match_threshold=0.85,
            novelty_hysteresis=2,
            spawn_cooldown_frames=0,
        )
    )
    bank.observe(_axis(0), frame_idx=0)

    first_novel = bank.observe(_axis(1), frame_idx=1)
    second_novel = bank.observe(_axis(1), frame_idx=2)

    assert first_novel["spawned"] is False
    assert first_novel["novelty"] == pytest.approx(1.0)
    assert second_novel["spawned"] is True
    assert second_novel["prototype_count"] == 2
    assert second_novel["novelty"] == pytest.approx(1.0)


def test_spawn_cooldown_blocks_immediate_second_spawn() -> None:
    bank = PrototypeBank(
        PrototypeBankConfig(
            match_threshold=0.85,
            novelty_hysteresis=1,
            spawn_cooldown_frames=5,
        )
    )
    bank.observe(_axis(0), frame_idx=0)
    blocked = bank.observe(_axis(1), frame_idx=1)
    allowed = bank.observe(_axis(1), frame_idx=5)

    assert blocked["spawned"] is False
    assert allowed["spawned"] is True


def test_kmax_bounds_count_with_eviction() -> None:
    bank = PrototypeBank(
        PrototypeBankConfig(
            kmax=2,
            match_threshold=0.99,
            novelty_hysteresis=1,
            spawn_cooldown_frames=0,
        )
    )
    metrics = []
    for frame_idx in range(20):
        metrics.append(bank.observe(_axis(frame_idx % SIMPLE_EMBEDDING_DIM), frame_idx=frame_idx))

    assert all(item["prototype_count"] <= 2 for item in metrics)
    assert bank.prototype_count == 2
    assert any(item["evicted"] for item in metrics)
    assert metrics[-1]["bank_full"] is True


def test_learning_disable_freezes_bank_and_keeps_schema() -> None:
    learning = PrototypeBank(PrototypeBankConfig())
    learning.observe(_axis(0), frame_idx=0)

    disabled = PrototypeBank(PrototypeBankConfig(learning_enabled=False))
    disabled._prototypes = list(learning._prototypes)
    disabled._next_id = learning._next_id

    before = disabled.prototype_count
    metrics = disabled.observe(_axis(1), frame_idx=1)

    assert metrics["status"] == "disabled"
    assert metrics["spawned"] is False
    assert metrics["evicted"] is False
    assert disabled.prototype_count == before
    assert set(metrics) == {
        "status",
        "prototype_count",
        "match_id",
        "match_similarity",
        "novelty",
        "spawned",
        "evicted",
        "bank_full",
    }


def test_disabled_bank_from_construction_never_spawns() -> None:
    bank = PrototypeBank(PrototypeBankConfig(learning_enabled=False))
    metrics = bank.observe(_axis(0), frame_idx=0)

    assert metrics["status"] == "disabled"
    assert metrics["prototype_count"] == 0
    assert metrics["spawned"] is False
    assert metrics["novelty"] == pytest.approx(1.0)


def test_invalid_and_missing_embeddings_are_rejected() -> None:
    bank = PrototypeBank()
    bank.observe(_axis(0), frame_idx=0)

    missing = bank.observe(None, frame_idx=1)
    invalid = bank.observe((1.0, 2.0), frame_idx=2)
    nonfinite = bank.observe(tuple([math.nan] + [0.0] * 9), frame_idx=3)

    assert missing["status"] == "no_embedding"
    assert missing["prototype_count"] == 1
    assert invalid["status"] == "invalid_embedding"
    assert nonfinite["status"] == "invalid_embedding"


def test_observe_is_deterministic_for_identical_streams() -> None:
    config = PrototypeBankConfig(kmax=4, novelty_hysteresis=2, spawn_cooldown_frames=1)
    stream = [_axis(0), _axis(0), _axis(1), _axis(1), _axis(2), _axis(2)]

    def run() -> list[dict[str, object]]:
        bank = PrototypeBank(config)
        return [bank.observe(vector, frame_idx=index) for index, vector in enumerate(stream)]

    assert run() == run()
