from __future__ import annotations

import numpy as np
import pytest

from features.encoder import (
    EmbeddingResult,
    EmbeddingSpaceMismatchError,
    PerceptEncoder,
    SimpleCropEncoder,
    cosine_similarity,
    ensure_same_space,
    similarity,
)
from features.simple_embedding import SIMPLE_EMBEDDING_SPACE_ID, embed_attended_crop
from perception.bbp import BoundingBox


def _frame() -> np.ndarray:
    return np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)


def test_simple_crop_encoder_satisfies_protocol_and_space_id_is_stable() -> None:
    encoder = SimpleCropEncoder()

    assert isinstance(encoder, PerceptEncoder)
    assert encoder.space_id == SIMPLE_EMBEDDING_SPACE_ID == "simple_crop_v1"
    assert SimpleCropEncoder().space_id == encoder.space_id


def test_wrapper_matches_legacy_function_exactly() -> None:
    frame = _frame()
    bbox = BoundingBox(2.0, 3.0, 12.0, 14.0)

    legacy = embed_attended_crop(frame, bbox)
    wrapped = SimpleCropEncoder().encode(frame, bbox)

    assert legacy is not None and wrapped is not None
    assert wrapped.vector == legacy.vector
    assert wrapped.space_id == SIMPLE_EMBEDDING_SPACE_ID
    assert SimpleCropEncoder().encode(frame, bbox) == wrapped


def test_wrapper_propagates_invalid_crop_as_none() -> None:
    encoder = SimpleCropEncoder()

    assert encoder.encode(None, BoundingBox(0.0, 0.0, 1.0, 1.0)) is None
    assert encoder.encode(_frame(), BoundingBox(5.0, 5.0, 5.0, 9.0)) is None
    assert encoder.to_result(None) is None


def test_embedding_result_rejects_empty_space_or_vector() -> None:
    with pytest.raises(ValueError):
        EmbeddingResult(vector=(1.0,), space_id="")
    with pytest.raises(ValueError):
        EmbeddingResult(vector=(), space_id="x")


def test_mismatched_spaces_cannot_be_compared_silently() -> None:
    a = EmbeddingResult(vector=(1.0, 0.0), space_id="space_a")
    b = EmbeddingResult(vector=(1.0, 0.0), space_id="space_b")

    with pytest.raises(EmbeddingSpaceMismatchError):
        similarity(a, b)
    with pytest.raises(EmbeddingSpaceMismatchError):
        ensure_same_space("space_a", "space_b")
    ensure_same_space("space_a", "space_a")


def test_cosine_similarity_is_bounded_and_rejects_degenerate_inputs() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)
    assert cosine_similarity((2.0, 0.0), (5.0, 0.0)) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        cosine_similarity((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError):
        cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))
