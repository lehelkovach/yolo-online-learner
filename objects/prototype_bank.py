from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from features.simple_embedding import SIMPLE_EMBEDDING_DIM, SIMPLE_EMBEDDING_SPACE_ID

UPDATE_RULE_ID = "normalized_running_mean_v1"
PrototypeBankMetrics = dict[str, object]


@dataclass(frozen=True, slots=True)
class PrototypeBankConfig:
    """Stage-4 bank knobs recorded in session_start."""

    kmax: int = 32
    match_threshold: float = 0.85
    spawn_cooldown_frames: int = 5
    novelty_hysteresis: int = 2
    learning_enabled: bool = True
    embedding_dim: int = SIMPLE_EMBEDDING_DIM
    embedding_space_id: str = SIMPLE_EMBEDDING_SPACE_ID
    update_rule_id: str = UPDATE_RULE_ID

    def __post_init__(self) -> None:
        if self.kmax < 1:
            raise ValueError("kmax must be >= 1")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        if self.spawn_cooldown_frames < 0:
            raise ValueError("spawn_cooldown_frames must be non-negative")
        if self.novelty_hysteresis < 1:
            raise ValueError("novelty_hysteresis must be >= 1")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")


@dataclass(slots=True)
class _Prototype:
    pattern_id: str
    vector: np.ndarray
    observation_count: int
    last_matched_frame: int
    created_frame: int

    def utility(self, frame_idx: int) -> float:
        age = max(0, int(frame_idx) - self.last_matched_frame)
        return float(self.observation_count) / (1.0 + float(age))


def prototype_bank_schema(config: PrototypeBankConfig) -> dict[str, object]:
    """Return the immutable Stage-4 config/schema block for session_start."""
    return {
        "update_rule_id": config.update_rule_id,
        "embedding_space_id": config.embedding_space_id,
        "embedding_dim": config.embedding_dim,
        "kmax": config.kmax,
        "match_threshold": config.match_threshold,
        "spawn_cooldown_frames": config.spawn_cooldown_frames,
        "novelty_hysteresis": config.novelty_hysteresis,
        "learning_enabled": config.learning_enabled,
        "identity": "perceptual_pattern",
        "match_score": "cosine_dot_l2",
    }


def empty_prototype_bank_metrics(*, status: str) -> PrototypeBankMetrics:
    """Return the stable JSONL schema when no bank update occurs."""
    return {
        "status": status,
        "prototype_count": 0,
        "match_id": None,
        "match_similarity": None,
        "novelty": None,
        "spawned": False,
        "evicted": False,
        "bank_full": False,
    }


class PrototypeBank:
    """Bounded online match/spawn memory over attended L2 embeddings.

    Prototypes are recurring perceptual patterns. They are not tracks, physical
    object files, categories, or human names.
    """

    def __init__(self, config: PrototypeBankConfig | None = None) -> None:
        self.config = config or PrototypeBankConfig()
        self._prototypes: list[_Prototype] = []
        self._next_id = 0
        self._novel_streak = 0
        self._last_spawn_frame: int | None = None

    @property
    def prototype_count(self) -> int:
        return len(self._prototypes)

    def observe(
        self,
        embedding: tuple[float, ...] | None,
        *,
        frame_idx: int,
    ) -> PrototypeBankMetrics:
        """Match or spawn against one attended embedding; return fixed-key metrics."""
        if embedding is None:
            return self._snapshot_metrics(status="no_embedding")

        vector = self._validate_embedding(embedding)
        if vector is None:
            return self._snapshot_metrics(status="invalid_embedding")

        best_id, best_sim, best_index = self._best_match(vector)
        novelty = 1.0 if best_sim is None else float(1.0 - best_sim)
        matched = best_sim is not None and best_sim >= self.config.match_threshold

        if not self.config.learning_enabled:
            return {
                "status": "disabled",
                "prototype_count": self.prototype_count,
                "match_id": best_id if matched else None,
                "match_similarity": None if best_sim is None else float(best_sim),
                "novelty": novelty,
                "spawned": False,
                "evicted": False,
                "bank_full": self.prototype_count >= self.config.kmax,
            }

        spawned = False
        evicted = False

        if matched:
            assert best_index is not None and best_id is not None and best_sim is not None
            self._update_prototype(best_index, vector, frame_idx=frame_idx)
            self._novel_streak = 0
            match_id = best_id
            match_similarity = float(best_sim)
        else:
            self._novel_streak += 1
            if self._can_spawn(frame_idx=frame_idx):
                if self.prototype_count >= self.config.kmax:
                    self._evict_lowest_utility(frame_idx=frame_idx)
                    evicted = True
                match_id = self._spawn(vector, frame_idx=frame_idx)
                match_similarity = 1.0
                novelty = 1.0
                spawned = True
                self._novel_streak = 0
                self._last_spawn_frame = frame_idx
            else:
                match_id = None
                match_similarity = None if best_sim is None else float(best_sim)

        return {
            "status": "ok",
            "prototype_count": self.prototype_count,
            "match_id": match_id,
            "match_similarity": match_similarity,
            "novelty": novelty,
            "spawned": spawned,
            "evicted": evicted,
            "bank_full": self.prototype_count >= self.config.kmax,
        }

    def reset(self) -> None:
        """Clear all patterns at a stream boundary."""
        self._prototypes.clear()
        self._next_id = 0
        self._novel_streak = 0
        self._last_spawn_frame = None

    def idle_metrics(self, *, status: str) -> PrototypeBankMetrics:
        """Fixed-key metrics when the frame has no attended embedding to learn from."""
        return self._snapshot_metrics(status=status)

    def _snapshot_metrics(self, *, status: str) -> PrototypeBankMetrics:
        base = empty_prototype_bank_metrics(status=status)
        base["prototype_count"] = self.prototype_count
        base["bank_full"] = self.prototype_count >= self.config.kmax
        return base

    def _validate_embedding(self, embedding: tuple[float, ...]) -> np.ndarray | None:
        if len(embedding) != self.config.embedding_dim:
            return None
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.shape != (self.config.embedding_dim,) or not np.isfinite(vector).all():
            return None
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            return None
        unit = vector / norm
        if not np.isfinite(unit).all():
            return None
        return unit

    def _best_match(
        self, vector: np.ndarray
    ) -> tuple[str | None, float | None, int | None]:
        if not self._prototypes:
            return None, None, None
        best_index = 0
        best_sim = float(np.dot(self._prototypes[0].vector, vector))
        for index, prototype in enumerate(self._prototypes[1:], start=1):
            similarity = float(np.dot(prototype.vector, vector))
            if similarity > best_sim or (
                math.isclose(similarity, best_sim)
                and prototype.pattern_id < self._prototypes[best_index].pattern_id
            ):
                best_sim = similarity
                best_index = index
        best = self._prototypes[best_index]
        return best.pattern_id, best_sim, best_index

    def _can_spawn(self, *, frame_idx: int) -> bool:
        if self.prototype_count == 0:
            return True
        if self._novel_streak < self.config.novelty_hysteresis:
            return False
        if self._last_spawn_frame is None:
            return True
        return (frame_idx - self._last_spawn_frame) >= self.config.spawn_cooldown_frames

    def _spawn(self, vector: np.ndarray, *, frame_idx: int) -> str:
        if self.prototype_count >= self.config.kmax:
            raise RuntimeError("refusing to spawn above kmax")
        pattern_id = f"pattern-{self._next_id:08d}"
        self._next_id += 1
        self._prototypes.append(
            _Prototype(
                pattern_id=pattern_id,
                vector=vector.copy(),
                observation_count=1,
                last_matched_frame=frame_idx,
                created_frame=frame_idx,
            )
        )
        return pattern_id

    def _update_prototype(
        self, index: int, vector: np.ndarray, *, frame_idx: int
    ) -> None:
        prototype = self._prototypes[index]
        count = prototype.observation_count
        blended = (count * prototype.vector + vector) / float(count + 1)
        norm = float(np.linalg.norm(blended))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("prototype update produced a non-finite vector")
        unit = blended / norm
        if not np.isfinite(unit).all():
            raise RuntimeError("prototype update produced non-finite components")
        assert abs(float(np.linalg.norm(unit)) - 1.0) < 1e-6
        prototype.vector = unit
        prototype.observation_count = count + 1
        prototype.last_matched_frame = frame_idx

    def _evict_lowest_utility(self, *, frame_idx: int) -> None:
        if not self._prototypes:
            return
        victim_index = min(
            range(len(self._prototypes)),
            key=lambda index: (
                self._prototypes[index].utility(frame_idx),
                self._prototypes[index].pattern_id,
            ),
        )
        del self._prototypes[victim_index]
