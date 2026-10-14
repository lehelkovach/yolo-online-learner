"""Stage 8: K-slot working memory of expected objects.

Working memory is the small set of object files the system currently has in mind:
the expected concepts that top-down prediction and, later, attention draw on. It
holds references and utility bookkeeping, never copies of belief; the object file
stays the source of truth.

- An attention cue (a percept bound to an object file) loads that object into a
  slot, or refreshes it if it is already there.
- When all ``capacity`` slots are taken, the lowest-utility slot is evicted:
  ``utility = recency + evidence_weight * log1p(cues) - error_weight * last_error``,
  where recency halves every ``recency_half_life_frames`` since the last cue.
  ``eviction = "recency"`` keeps only the recency term.
- An object that goes LOST or is forgotten releases its slot; OCCLUDED objects stay
  in mind, which is the point of permanence.

Nothing reads working memory yet: Stage 5 expectations still come from every object
file and attention keeps its proxies until the recorded-clip validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvictionRule = Literal["utility", "recency"]


@dataclass(frozen=True, slots=True)
class WorkingMemoryConfig:
    """Capacity and eviction policy. Nothing here is implicit."""

    capacity: int = 4
    eviction: EvictionRule = "utility"
    recency_half_life_frames: float = 30.0
    evidence_weight: float = 0.1
    error_weight: float = 0.5
    release_on_lost: bool = True

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        if self.eviction not in ("utility", "recency"):
            raise ValueError(f"unknown eviction rule {self.eviction!r}")
        if self.recency_half_life_frames <= 0.0:
            raise ValueError("recency_half_life_frames must be positive")
        if self.evidence_weight < 0.0 or self.error_weight < 0.0:
            raise ValueError("weights must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Slot:
    object_id: str
    loaded_frame: int
    last_cued_frame: int
    cue_count: int
    last_error: float

    def utility(self, frame_idx: int, config: WorkingMemoryConfig) -> float:
        elapsed = max(0, int(frame_idx) - self.last_cued_frame)
        recency = math.exp(-math.log(2.0) * elapsed / config.recency_half_life_frames)
        if config.eviction == "recency":
            return recency
        return (
            recency
            + config.evidence_weight * math.log1p(self.cue_count)
            - config.error_weight * self.last_error
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkingMemoryUpdate:
    object_id: str
    loaded: bool
    reloaded: bool
    refreshed: bool
    evicted_object_id: str | None
    evicted_utility: float | None
    occupancy: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "loaded": self.loaded,
            "reloaded": self.reloaded,
            "refreshed": self.refreshed,
            "evicted_object_id": self.evicted_object_id,
            "evicted_utility": self.evicted_utility,
            "occupancy": self.occupancy,
        }


def empty_working_memory_metrics(status: str) -> dict[str, Any]:
    """Fixed-key schema for a frame without a cue; the runner fills the state keys."""
    return {
        "status": status,
        "object_id": None,
        "loaded": None,
        "reloaded": None,
        "refreshed": None,
        "evicted_object_id": None,
        "evicted_utility": None,
        "occupancy": None,
    }


@dataclass(slots=True)
class WorkingMemory:
    """Bounded register of object-file references with deterministic eviction."""

    config: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    _slots: dict[str, Slot] = field(default_factory=dict, init=False)
    _ever_loaded: set[str] = field(default_factory=set, init=False)
    _loads: int = field(default=0, init=False)
    _reloads: int = field(default=0, init=False)
    _refreshes: int = field(default=0, init=False)
    _evictions: int = field(default=0, init=False)
    _releases: int = field(default=0, init=False)
    _max_occupancy: int = field(default=0, init=False)

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._slots

    def contents(self) -> list[str]:
        """Object ids in load order."""
        return list(self._slots)

    def slot(self, object_id: str) -> Slot:
        return self._slots[object_id]

    def utilities(self, frame_idx: int) -> dict[str, float]:
        return {k: s.utility(frame_idx, self.config) for k, s in self._slots.items()}

    def cue(self, object_id: str, *, frame_idx: int, error: float) -> WorkingMemoryUpdate:
        """Load or refresh ``object_id``; evict the lowest-utility slot if needed."""
        if not object_id:
            raise ValueError("object_id must be non-empty")
        if not math.isfinite(error) or error < 0.0:
            raise ValueError("error must be a finite non-negative number")
        if object_id in self._slots:
            slot = self._slots[object_id]
            slot.last_cued_frame = int(frame_idx)
            slot.cue_count += 1
            slot.last_error = float(error)
            self._refreshes += 1
            return WorkingMemoryUpdate(
                object_id=object_id, loaded=False, reloaded=False, refreshed=True,
                evicted_object_id=None, evicted_utility=None, occupancy=len(self._slots),
            )

        evicted_id: str | None = None
        evicted_utility: float | None = None
        if len(self._slots) >= self.config.capacity:
            evicted_id, evicted_utility = self._evict(frame_idx)
        reloaded = object_id in self._ever_loaded
        self._slots[object_id] = Slot(
            object_id=object_id,
            loaded_frame=int(frame_idx),
            last_cued_frame=int(frame_idx),
            cue_count=1,
            last_error=float(error),
        )
        self._ever_loaded.add(object_id)
        self._loads += 1
        if reloaded:
            self._reloads += 1
        self._max_occupancy = max(self._max_occupancy, len(self._slots))
        return WorkingMemoryUpdate(
            object_id=object_id, loaded=True, reloaded=reloaded, refreshed=False,
            evicted_object_id=evicted_id, evicted_utility=evicted_utility,
            occupancy=len(self._slots),
        )

    def _evict(self, frame_idx: int) -> tuple[str, float]:
        # Lowest utility first; ties fall to the earliest load, then insertion order.
        victim = min(
            enumerate(self._slots.values()),
            key=lambda item: (
                item[1].utility(frame_idx, self.config), item[1].loaded_frame, item[0]
            ),
        )[1]
        del self._slots[victim.object_id]
        self._evictions += 1
        return victim.object_id, victim.utility(frame_idx, self.config)

    def release(self, object_id: str) -> bool:
        """Drop ``object_id`` if present (the object is no longer expected)."""
        if object_id not in self._slots:
            return False
        del self._slots[object_id]
        self._releases += 1
        return True

    def counts(self) -> dict[str, Any]:
        return {
            "capacity": self.config.capacity,
            "occupancy": len(self._slots),
            "max_occupancy": self._max_occupancy,
            "loads": self._loads,
            "reloads": self._reloads,
            "refreshes": self._refreshes,
            "evictions": self._evictions,
            "releases": self._releases,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "counts": self.counts(),
            "slots": [s.to_dict() for s in self._slots.values()],
        }
