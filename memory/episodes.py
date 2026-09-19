"""Episodic perceptual memory: an append-only, time-indexed record of observations.

Object files hold current belief and prototypes average across time. Episodic memory
keeps the individual events, so later layers (consolidation, categories, the percept
graph) can cite exactly which observations support a claim.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from features.encoder import EmbeddingResult, ensure_same_space
from objects.ids import IdFactory, random_uuid_factory
from perception.bbp import BBP, BoundingBox


@dataclass(frozen=True, slots=True)
class EpisodicMemoryConfig:
    """Retention limits. Events are never mutated; only the oldest are dropped."""

    # ``None`` keeps every event; an int drops the oldest once the store is full.
    max_events: int | None = 10_000

    def __post_init__(self) -> None:
        if self.max_events is not None and self.max_events < 1:
            raise ValueError("max_events must be >= 1 or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """One attended observation bound to one object file at one moment."""

    observation_id: str
    object_id: str
    timestamp_s: float
    frame_idx: int
    bbox: BoundingBox
    embedding_space_id: str
    embedding: tuple[float, ...]
    confidence: float
    category_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "object_id": self.object_id,
            "timestamp_s": self.timestamp_s,
            "frame_idx": self.frame_idx,
            "bbox": list(self.bbox.as_xyxy()),
            "embedding_space_id": self.embedding_space_id,
            "embedding": list(self.embedding),
            "confidence": self.confidence,
            "category_id": self.category_id,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ObservationEvent:
        return ObservationEvent(
            observation_id=str(d["observation_id"]),
            object_id=str(d["object_id"]),
            timestamp_s=float(d["timestamp_s"]),
            frame_idx=int(d["frame_idx"]),
            bbox=BoundingBox(*map(float, d["bbox"])),
            embedding_space_id=str(d["embedding_space_id"]),
            embedding=tuple(float(v) for v in d["embedding"]),
            confidence=float(d["confidence"]),
            category_id=d.get("category_id"),
        )


@dataclass(slots=True)
class EpisodicMemory:
    """Bounded, ordered store of ``ObservationEvent``s with per-object indexes."""

    embedding_space_id: str
    config: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    id_factory: IdFactory = field(default_factory=random_uuid_factory)
    _events: deque[ObservationEvent] = field(default_factory=deque, init=False)
    _by_object: dict[str, list[ObservationEvent]] = field(default_factory=dict, init=False)
    _last_frame_idx: int | None = field(default=None, init=False)
    _recorded_total: int = field(default=0, init=False)
    _dropped_total: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.embedding_space_id:
            raise ValueError("embedding_space_id must be non-empty")

    # ----- recording -------------------------------------------------------------
    def record(
        self,
        *,
        bbp: BBP,
        object_id: str,
        embedding: EmbeddingResult,
        category_id: str | None = None,
    ) -> ObservationEvent:
        """Append one event. Frames must be non-decreasing; events are immutable."""
        ensure_same_space(self.embedding_space_id, embedding.space_id, context="episode")
        if not object_id:
            raise ValueError("object_id must be non-empty")
        if self._last_frame_idx is not None and bbp.frame_idx < self._last_frame_idx:
            raise ValueError(
                f"frame {bbp.frame_idx} precedes last recorded frame {self._last_frame_idx}"
            )
        observation_id = self.id_factory()
        event = ObservationEvent(
            observation_id=observation_id,
            object_id=object_id,
            timestamp_s=float(bbp.timestamp_s),
            frame_idx=int(bbp.frame_idx),
            bbox=bbp.bbox,
            embedding_space_id=embedding.space_id,
            embedding=tuple(float(v) for v in embedding.vector),
            confidence=float(bbp.confidence),
            category_id=category_id,
        )
        self._events.append(event)
        self._by_object.setdefault(object_id, []).append(event)
        self._last_frame_idx = event.frame_idx
        self._recorded_total += 1
        self._enforce_capacity()
        return event

    def _enforce_capacity(self) -> None:
        limit = self.config.max_events
        if limit is None:
            return
        while len(self._events) > limit:
            oldest = self._events.popleft()
            history = self._by_object[oldest.object_id]
            history.pop(0)
            if not history:
                del self._by_object[oldest.object_id]
            self._dropped_total += 1

    # ----- queries ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> list[ObservationEvent]:
        """All retained events in recording order."""
        return list(self._events)

    def last_seen(self, object_id: str) -> ObservationEvent | None:
        history = self._by_object.get(object_id)
        return history[-1] if history else None

    def observations_for(self, object_id: str) -> list[ObservationEvent]:
        """Retained events for one object, oldest first."""
        return list(self._by_object.get(object_id, ()))

    def objects_seen_in_range(self, start_s: float, end_s: float) -> list[str]:
        """Object ids with at least one event in ``[start_s, end_s]``.

        Ordered by each object's first event inside the range, so the answer is
        deterministic for a given log.
        """
        if end_s < start_s:
            raise ValueError("end_s must be >= start_s")
        seen: dict[str, None] = {}
        for event in self._events:
            if start_s <= event.timestamp_s <= end_s and event.object_id not in seen:
                seen[event.object_id] = None
        return list(seen)

    def counts(self) -> dict[str, int]:
        return {
            "retained": len(self._events),
            "recorded_total": self._recorded_total,
            "dropped_total": self._dropped_total,
            "objects_with_history": len(self._by_object),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_space_id": self.embedding_space_id,
            "config": self.config.to_dict(),
            "counts": self.counts(),
            "events": [event.to_dict() for event in self._events],
        }
