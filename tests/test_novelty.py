from __future__ import annotations

import pytest

from features.encoder import EmbeddingResult
from memory.prototypes import PrototypeMemory, PrototypeMemoryConfig, novelty_band
from objects.ids import sequential_id_factory

SPACE = "test_space"


def _emb(*values: float) -> EmbeddingResult:
    return EmbeddingResult(vector=tuple(values), space_id=SPACE)


def _memory(**overrides: object) -> PrototypeMemory:
    config = PrototypeMemoryConfig(embedding_space_id=SPACE, **overrides)  # type: ignore[arg-type]
    return PrototypeMemory(config=config, id_factory=sequential_id_factory("proto"))


def test_exact_known_prototype_has_zero_novelty() -> None:
    memory = _memory()
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)

    result = memory.score_novelty(_emb(2.0, 0.0))  # same direction, different norm

    assert result.novelty_score == pytest.approx(0.0)
    assert result.nearest_prototype_id == "proto-000001"
    assert result.nearest_similarity == pytest.approx(1.0)
    assert result.band == "known_instance"
    assert result.memory_empty is False


def test_distant_vector_has_high_novelty() -> None:
    memory = _memory()
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)

    orthogonal = memory.score_novelty(_emb(0.0, 1.0))
    opposite = memory.score_novelty(_emb(-1.0, 0.0))

    assert orthogonal.novelty_score == pytest.approx(1.0)
    assert orthogonal.band == "novel"
    assert opposite.novelty_score == 1.0  # clipped, never above 1


def test_empty_memory_contract_is_explicit() -> None:
    result = _memory().score_novelty(_emb(1.0, 0.0))

    assert result.to_dict() == {
        "novelty": 1.0,
        "nearest_prototype_id": None,
        "nearest_similarity": None,
        "band": "novel",
        "memory_empty": True,
    }


def test_novelty_uses_nearest_prototype_and_stable_tie_break() -> None:
    memory = _memory(match_threshold=0.999)
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)
    memory.observe(_emb(0.0, 1.0), timestamp_s=1.0)

    result = memory.score_novelty(_emb(1.0, 1.0))  # equidistant

    assert result.nearest_prototype_id == "proto-000001"
    assert result.nearest_similarity == pytest.approx(2**-0.5)
    assert result.novelty_score == pytest.approx(1.0 - 2**-0.5)


def test_bands_follow_configured_thresholds() -> None:
    config = PrototypeMemoryConfig(
        embedding_space_id=SPACE,
        known_instance_max=0.1,
        known_category_max=0.3,
        boundary_max=0.5,
    )

    assert novelty_band(0.0, config) == "known_instance"
    assert novelty_band(0.1, config) == "known_instance"
    assert novelty_band(0.2, config) == "known_category"
    assert novelty_band(0.4, config) == "boundary"
    assert novelty_band(0.6, config) == "novel"
    with pytest.raises(ValueError):
        PrototypeMemoryConfig(embedding_space_id=SPACE, known_instance_max=0.5, boundary_max=0.2)


def test_scoring_never_mutates_memory() -> None:
    memory = _memory()
    memory.observe(_emb(1.0, 0.0), timestamp_s=0.0)
    before = memory.to_dict()

    memory.score_novelty(_emb(0.0, 1.0))

    assert memory.to_dict() == before
