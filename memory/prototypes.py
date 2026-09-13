from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from features.encoder import EmbeddingResult, cosine_similarity, ensure_same_space
from objects.ids import IdFactory, random_uuid_factory

UpdateRule = Literal["running_mean", "learning_rate"]


@dataclass(frozen=True, slots=True)
class PrototypeMemoryConfig:
    """Prototype learning rule, thresholds and capacity. Nothing here is implicit."""

    embedding_space_id: str
    update_rule: UpdateRule = "running_mean"
    learning_rate: float = 0.1
    # Similarity at or above this updates the nearest prototype; below spawns a new one.
    match_threshold: float = 0.9
    max_prototypes: int = 64
    # Multiplicative strength decay applied to non-winning prototypes per observation.
    strength_decay: float = 0.0
    # Novelty band edges (inclusive upper bounds), lowest to highest novelty.
    known_instance_max: float = 0.1
    known_category_max: float = 0.3
    boundary_max: float = 0.5

    def __post_init__(self) -> None:
        if not self.embedding_space_id:
            raise ValueError("embedding_space_id must be non-empty")
        if self.update_rule not in ("running_mean", "learning_rate"):
            raise ValueError(f"unknown update_rule {self.update_rule!r}")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        if self.max_prototypes < 1:
            raise ValueError("max_prototypes must be >= 1")
        if not 0.0 <= self.strength_decay < 1.0:
            raise ValueError("strength_decay must be in [0, 1)")
        if not 0.0 <= self.known_instance_max <= self.known_category_max <= self.boundary_max:
            raise ValueError("novelty band edges must be non-decreasing and >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Prototype:
    """An online-learned representation accumulated from attended observations."""

    prototype_id: str
    embedding_space_id: str
    centroid: tuple[float, ...]
    count: int
    first_seen_s: float
    last_seen_s: float
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prototype_id": self.prototype_id,
            "embedding_space_id": self.embedding_space_id,
            "centroid": list(self.centroid),
            "count": self.count,
            "first_seen_s": self.first_seen_s,
            "last_seen_s": self.last_seen_s,
            "strength": self.strength,
        }


NoveltyBand = Literal["known_instance", "known_category", "boundary", "novel"]


@dataclass(frozen=True, slots=True)
class NoveltyResult:
    """Memory-relative novelty: ``1 - max_similarity`` to known prototypes.

    Contract for empty memory: ``novelty_score`` is ``1.0`` (maximal),
    ``nearest_prototype_id`` and ``nearest_similarity`` are ``None``, and
    ``memory_empty`` is ``True``.
    """

    novelty_score: float
    nearest_prototype_id: str | None
    nearest_similarity: float | None
    band: NoveltyBand
    memory_empty: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "novelty": self.novelty_score,
            "nearest_prototype_id": self.nearest_prototype_id,
            "nearest_similarity": self.nearest_similarity,
            "band": self.band,
            "memory_empty": self.memory_empty,
        }


@dataclass(frozen=True, slots=True)
class PrototypeUpdate:
    prototype_id: str
    created: bool
    updated: bool
    evicted_prototype_id: str | None
    novelty: NoveltyResult
    prototype_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.novelty.to_dict(),
            "prototype_id": self.prototype_id,
            "prototype_created": self.created,
            "prototype_updated": self.updated,
            "evicted_prototype_id": self.evicted_prototype_id,
            "prototype_count": self.prototype_count,
        }


def novelty_band(novelty: float, config: PrototypeMemoryConfig) -> NoveltyBand:
    if novelty <= config.known_instance_max:
        return "known_instance"
    if novelty <= config.known_category_max:
        return "known_category"
    if novelty <= config.boundary_max:
        return "boundary"
    return "novel"


@dataclass(slots=True)
class PrototypeMemory:
    """Bounded bank of prototypes with an explicitly configured update rule."""

    config: PrototypeMemoryConfig
    id_factory: IdFactory = field(default_factory=random_uuid_factory)
    _prototypes: dict[str, Prototype] = field(default_factory=dict, init=False)

    @property
    def prototypes(self) -> list[Prototype]:
        return list(self._prototypes.values())

    def get(self, prototype_id: str) -> Prototype:
        return self._prototypes[prototype_id]

    def __len__(self) -> int:
        return len(self._prototypes)

    def _check(self, embedding: EmbeddingResult) -> np.ndarray:
        ensure_same_space(self.config.embedding_space_id, embedding.space_id, context="prototype")
        vector = np.asarray(embedding.vector, dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("embedding must be a finite 1-D vector")
        if self._prototypes:
            dim = len(next(iter(self._prototypes.values())).centroid)
            if vector.shape != (dim,):
                raise ValueError(f"embedding dimension {vector.shape[0]} != prototype {dim}")
        return vector

    def score_novelty(self, embedding: EmbeddingResult) -> NoveltyResult:
        """Novelty relative to memory; never mutates state."""
        self._check(embedding)
        if not self._prototypes:
            return NoveltyResult(
                novelty_score=1.0,
                nearest_prototype_id=None,
                nearest_similarity=None,
                band="novel",
                memory_empty=True,
            )
        best_id: str | None = None
        best_similarity = -2.0
        for prototype in self._prototypes.values():  # insertion order = stable tie-break
            value = cosine_similarity(embedding.vector, prototype.centroid)
            if value > best_similarity:
                best_similarity = value
                best_id = prototype.prototype_id
        novelty = min(1.0, max(0.0, 1.0 - best_similarity))
        return NoveltyResult(
            novelty_score=novelty,
            nearest_prototype_id=best_id,
            nearest_similarity=best_similarity,
            band=novelty_band(novelty, self.config),
            memory_empty=False,
        )

    def observe(self, embedding: EmbeddingResult, *, timestamp_s: float) -> PrototypeUpdate:
        """Score novelty, then update the nearest prototype or spawn a new one."""
        vector = self._check(embedding)
        novelty = self.score_novelty(embedding)
        cfg = self.config

        if (
            novelty.nearest_prototype_id is not None
            and novelty.nearest_similarity is not None
            and novelty.nearest_similarity >= cfg.match_threshold
        ):
            prototype = self._prototypes[novelty.nearest_prototype_id]
            old = np.asarray(prototype.centroid, dtype=np.float64)
            if cfg.update_rule == "running_mean":
                n = prototype.count
                new = (n * old + vector) / (n + 1)
            else:
                new = old + cfg.learning_rate * (vector - old)
            prototype.centroid = tuple(float(v) for v in new)
            prototype.count += 1
            prototype.last_seen_s = float(timestamp_s)
            prototype.strength += 1.0
            self._decay_others(prototype.prototype_id)
            return PrototypeUpdate(
                prototype_id=prototype.prototype_id,
                created=False,
                updated=True,
                evicted_prototype_id=None,
                novelty=novelty,
                prototype_count=len(self._prototypes),
            )

        evicted = None
        if len(self._prototypes) >= cfg.max_prototypes:
            evicted = self._evict_weakest()
        prototype_id = self.id_factory()
        if prototype_id in self._prototypes:
            raise RuntimeError(f"id factory produced duplicate prototype id {prototype_id!r}")
        self._prototypes[prototype_id] = Prototype(
            prototype_id=prototype_id,
            embedding_space_id=cfg.embedding_space_id,
            centroid=tuple(float(v) for v in vector),
            count=1,
            first_seen_s=float(timestamp_s),
            last_seen_s=float(timestamp_s),
            strength=1.0,
        )
        self._decay_others(prototype_id)
        return PrototypeUpdate(
            prototype_id=prototype_id,
            created=True,
            updated=False,
            evicted_prototype_id=evicted,
            novelty=novelty,
            prototype_count=len(self._prototypes),
        )

    def _decay_others(self, winner_id: str) -> None:
        if self.config.strength_decay <= 0.0:
            return
        factor = 1.0 - self.config.strength_decay
        for prototype in self._prototypes.values():
            if prototype.prototype_id != winner_id:
                prototype.strength *= factor

    def _evict_weakest(self) -> str:
        # Lowest strength first; ties fall to the oldest first_seen, then insertion order.
        weakest = min(
            enumerate(self._prototypes.values()),
            key=lambda item: (item[1].strength, item[1].first_seen_s, item[0]),
        )[1]
        del self._prototypes[weakest.prototype_id]
        return weakest.prototype_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "prototypes": [p.to_dict() for p in self._prototypes.values()],
        }
