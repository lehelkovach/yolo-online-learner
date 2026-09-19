"""Per-frame glue between attention, object memory and prototype memory.

This module owns no learning rule. It sequences the memory layers for one frame and
returns the JSON-ready trace blocks that ``experiments/run.py`` logs and
``experiments/replay.py`` reproduces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.config import ExperimentConfig
from features.encoder import EmbeddingResult, ensure_same_space
from memory.episodes import EpisodicMemory
from memory.prototypes import PrototypeMemory
from objects.ids import seeded_uuid_factory
from objects.memory import ObjectMemory
from perception.bbp import BBP

OBJECT_ID_PREFIX = "obj-"
PROTOTYPE_ID_PREFIX = "proto-"
OBSERVATION_ID_PREFIX = "obs-"

# Each memory draws its ids from its own seeded stream derived from the session seed.
_ID_STREAMS = 3


def object_id_seed(seed: int) -> int:
    return _ID_STREAMS * int(seed)


def prototype_id_seed(seed: int) -> int:
    return _ID_STREAMS * int(seed) + 1


def observation_id_seed(seed: int) -> int:
    return _ID_STREAMS * int(seed) + 2


def empty_object_file_metrics(status: str) -> dict[str, Any]:
    """Fixed-key schema for frames without a bindable attended percept."""
    return {
        "status": status,
        "object_id": None,
        "created": None,
        "reidentified": None,
        "match_score": None,
        "candidate_count": None,
        "previous_visibility": None,
        "appearance_similarity": None,
        "spatial_similarity": None,
        "class_compatibility": None,
        "runner_up_score": None,
        "visibility": None,
        "observation_count": None,
    }


def empty_learning_metrics(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "novelty": None,
        "nearest_prototype_id": None,
        "nearest_similarity": None,
        "band": None,
        "memory_empty": None,
        "prototype_id": None,
        "prototype_created": None,
        "prototype_updated": None,
        "evicted_prototype_id": None,
        "prototype_count": None,
    }


class PerceptualLearner:
    """Object files + permanence + prototype memory, stepped once per frame."""

    def __init__(self, cfg: ExperimentConfig, *, encoder_space_id: str) -> None:
        ensure_same_space(
            encoder_space_id, cfg.prototypes.embedding_space_id, context="prototype memory"
        )
        self.cfg = cfg
        self.encoder_space_id = encoder_space_id
        self.object_memory = ObjectMemory(
            binder_config=cfg.binder,
            permanence=cfg.permanence,
            id_factory=seeded_uuid_factory(object_id_seed(cfg.seed), prefix=OBJECT_ID_PREFIX),
        )
        self.prototype_memory = PrototypeMemory(
            config=cfg.prototypes,
            id_factory=seeded_uuid_factory(
                prototype_id_seed(cfg.seed), prefix=PROTOTYPE_ID_PREFIX
            ),
        )
        self.episodic_memory = EpisodicMemory(
            embedding_space_id=encoder_space_id,
            config=cfg.episodes,
            id_factory=seeded_uuid_factory(
                observation_id_seed(cfg.seed), prefix=OBSERVATION_ID_PREFIX
            ),
        )

    def schema(self) -> dict[str, Any]:
        return {
            "object_id_prefix": OBJECT_ID_PREFIX,
            "prototype_id_prefix": PROTOTYPE_ID_PREFIX,
            "observation_id_prefix": OBSERVATION_ID_PREFIX,
            "id_factory": "seeded_uuid4",
            "object_id_seed": object_id_seed(self.cfg.seed),
            "prototype_id_seed": prototype_id_seed(self.cfg.seed),
            "observation_id_seed": observation_id_seed(self.cfg.seed),
            "embedding_space_id": self.encoder_space_id,
        }

    def step(
        self,
        *,
        frame_idx: int,
        timestamp_s: float,
        bbps: Sequence[BBP],
        selected_index: int | None,
        embedding: EmbeddingResult | None,
    ) -> dict[str, Any]:
        """Observe, record the episode, glimpse the rest, then age every object."""
        observation_id: str | None = None
        if selected_index is None:
            object_file = empty_object_file_metrics("no_selection")
            learning = empty_learning_metrics("no_selection")
        elif embedding is None:
            object_file = empty_object_file_metrics("no_embedding")
            learning = empty_learning_metrics("no_embedding")
        else:
            binding = self.object_memory.observe(bbps[selected_index], embedding)
            obj = self.object_memory.get(binding.object_id)
            object_file = {
                "status": "ok",
                **binding.to_dict(),
                "visibility": obj.visibility.value,
                "observation_count": obj.observation_count,
            }
            episode = self.episodic_memory.record(
                bbp=bbps[selected_index],
                object_id=binding.object_id,
                embedding=embedding,
                category_id=obj.category_id,
            )
            observation_id = episode.observation_id
            update = self.prototype_memory.observe(embedding, timestamp_s=timestamp_s)
            learning = {"status": "ok", **update.to_dict()}

        unattended = [bbp for index, bbp in enumerate(bbps) if index != selected_index]
        glimpsed = self.object_memory.glimpse(unattended, frame_idx=frame_idx)
        transitions = self.object_memory.advance(frame_idx)
        return {
            "observation_id": observation_id,
            "object_file": object_file,
            "object_memory": {
                "counts": self.object_memory.counts(),
                "glimpsed": glimpsed,
                "transitions": [t.to_dict() for t in transitions],
            },
            "learning": learning,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "object_counts": self.object_memory.counts(),
            "prototype_count": len(self.prototype_memory),
            "episodes": self.episodic_memory.counts(),
        }
