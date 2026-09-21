from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from features.simple_embedding import (
    SIMPLE_EMBEDDING_SPACE_ID,
    SimpleCropEmbedding,
    embed_attended_crop,
)
from perception.bbp import BoundingBox


class EmbeddingSpaceMismatchError(ValueError):
    """Raised when vectors from different embedding spaces would be compared."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """One embedding vector tagged with the immutable space it lives in."""

    vector: tuple[float, ...]
    space_id: str

    def __post_init__(self) -> None:
        if not self.space_id:
            raise ValueError("space_id must be a non-empty string")
        if len(self.vector) == 0:
            raise ValueError("vector must be non-empty")


@runtime_checkable
class PerceptEncoder(Protocol):
    """Common contract every attended-crop encoder must satisfy."""

    @property
    def space_id(self) -> str: ...

    def encode(self, frame_bgr: object, bbox: BoundingBox) -> EmbeddingResult | None: ...


class SimpleCropEncoder:
    """Wrap the deterministic Stage-3 crop embedding behind ``PerceptEncoder``."""

    @property
    def space_id(self) -> str:
        return SIMPLE_EMBEDDING_SPACE_ID

    def encode_crop(self, frame_bgr: object, bbox: BoundingBox) -> SimpleCropEmbedding | None:
        """Return the full Stage-3 result (vector plus crop provenance)."""
        return embed_attended_crop(frame_bgr, bbox)

    def to_result(self, crop: SimpleCropEmbedding | None) -> EmbeddingResult | None:
        if crop is None:
            return None
        return EmbeddingResult(vector=crop.vector, space_id=self.space_id)

    def encode(self, frame_bgr: object, bbox: BoundingBox) -> EmbeddingResult | None:
        return self.to_result(self.encode_crop(frame_bgr, bbox))


def ensure_same_space(expected: str, actual: str, *, context: str = "embedding") -> None:
    """Raise ``EmbeddingSpaceMismatchError`` unless both space identifiers agree."""
    if expected != actual:
        raise EmbeddingSpaceMismatchError(
            f"{context}: expected embedding space {expected!r}, got {actual!r}"
        )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``; zero vectors or dimension mismatch raise."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape or va.ndim != 1:
        raise ValueError(f"vector shapes differ: {va.shape} vs {vb.shape}")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0.0 or nb <= 0.0 or not np.isfinite(na) or not np.isfinite(nb):
        raise ValueError("cosine similarity is undefined for zero or non-finite vectors")
    value = float(np.dot(va, vb) / (na * nb))
    return max(-1.0, min(1.0, value))


def similarity(a: EmbeddingResult, b: EmbeddingResult) -> float:
    """Cosine similarity between two tagged embeddings; spaces must match."""
    ensure_same_space(a.space_id, b.space_id, context="similarity")
    return cosine_similarity(a.vector, b.vector)
