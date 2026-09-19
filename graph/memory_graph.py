"""Typed object/observation/prototype layer on top of ``PerceptGraph``.

Node types: ``object``, ``observation``, ``prototype``.
Edge types: ``OBSERVATION_OF`` (observation -> object), ``SIMILAR_TO``
(observation -> nearest prototype, weight = similarity), ``SEEN_WITH``
(object -> object, canonical id order, weight = co-visible frame count).

The graph is short-term relational memory: a workspace that is snapshotted next
to the session log, not a durable store. It never invents identity; every node
comes from a decision already taken by object, prototype or episodic memory.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from graph.percept_graph import NodeRef, PerceptGraph
from memory.episodes import ObservationEvent

OBJECT = "object"
OBSERVATION = "observation"
PROTOTYPE = "prototype"

OBSERVATION_OF = "OBSERVATION_OF"
SIMILAR_TO = "SIMILAR_TO"
SEEN_WITH = "SEEN_WITH"


@dataclass(frozen=True, slots=True)
class MemoryGraphConfig:
    """Retention limit for observation nodes; object and prototype nodes persist."""

    # ``None`` keeps every observation node; an int evicts the oldest beyond it.
    max_observation_nodes: int | None = 10_000

    def __post_init__(self) -> None:
        if self.max_observation_nodes is not None and self.max_observation_nodes < 1:
            raise ValueError("max_observation_nodes must be >= 1 or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryGraph:
    """Records identity, provenance and co-occurrence structure per frame."""

    config: MemoryGraphConfig = field(default_factory=MemoryGraphConfig)
    graph: PerceptGraph = field(default_factory=PerceptGraph)
    _observation_order: deque[str] = field(default_factory=deque, init=False)
    _evicted_observations: int = field(default=0, init=False)

    # ----- helpers ---------------------------------------------------------------
    @property
    def g(self):  # networkx.DiGraph
        return self.graph.g

    def has_node(self, node_id: str) -> bool:
        return node_id in self.g

    def node_type(self, node_id: str) -> str | None:
        return self.g.nodes[node_id].get("node_type") if node_id in self.g else None

    def _ensure_object(
        self, object_id: str, *, timestamp_s: float, visibility: str | None
    ) -> NodeRef:
        ref = NodeRef(object_id, OBJECT)
        if object_id not in self.g:
            self.graph.add_node(
                ref,
                first_seen_s=float(timestamp_s),
                last_seen_s=float(timestamp_s),
                observation_count=0,
                visibility=visibility,
            )
        return ref

    def _ensure_prototype(self, prototype_id: str) -> NodeRef:
        ref = NodeRef(prototype_id, PROTOTYPE)
        if prototype_id not in self.g:
            self.graph.add_node(ref, match_count=0)
        return ref

    # ----- recording -------------------------------------------------------------
    def record_observation(
        self,
        event: ObservationEvent,
        *,
        visibility: str | None = None,
        nearest_prototype_id: str | None = None,
        nearest_similarity: float | None = None,
    ) -> None:
        """Add an observation node with its OBSERVATION_OF and SIMILAR_TO edges.

        Re-inserting an observation id is an error: one episodic event is one node,
        and a silent overwrite could re-attach it to a different object.
        """
        if event.observation_id in self.g:
            raise ValueError(f"observation {event.observation_id!r} already in graph")

        obj_ref = self._ensure_object(
            event.object_id, timestamp_s=event.timestamp_s, visibility=visibility
        )
        obs_ref = NodeRef(event.observation_id, OBSERVATION)
        self.graph.add_node(
            obs_ref,
            frame_idx=int(event.frame_idx),
            timestamp_s=float(event.timestamp_s),
            confidence=float(event.confidence),
        )
        self.graph.add_edge(obs_ref, obj_ref, edge_type=OBSERVATION_OF, weight=1.0)

        node = self.g.nodes[event.object_id]
        node["observation_count"] = int(node.get("observation_count", 0)) + 1
        node["last_seen_s"] = float(event.timestamp_s)
        if visibility is not None:
            node["visibility"] = visibility

        if nearest_prototype_id is not None and nearest_similarity is not None:
            proto_ref = self._ensure_prototype(nearest_prototype_id)
            self.graph.add_edge(
                obs_ref, proto_ref, edge_type=SIMILAR_TO, weight=float(nearest_similarity)
            )
            pnode = self.g.nodes[nearest_prototype_id]
            pnode["match_count"] = int(pnode.get("match_count", 0)) + 1

        self._observation_order.append(event.observation_id)
        self._enforce_capacity()

    def set_visibility(self, object_id: str, visibility: str) -> None:
        if object_id in self.g:
            self.g.nodes[object_id]["visibility"] = visibility

    def record_cooccurrence(self, object_ids: Iterable[str], *, timestamp_s: float) -> int:
        """Increment SEEN_WITH between every pair of objects visible together.

        Pairs use canonical (sorted) id order so each pair has exactly one edge.
        Returns the number of pairs touched.
        """
        ids = sorted(set(object_ids))
        pairs = 0
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                ref_a = self._ensure_object(a, timestamp_s=timestamp_s, visibility=None)
                ref_b = self._ensure_object(b, timestamp_s=timestamp_s, visibility=None)
                if self.g.has_edge(a, b):
                    data = self.g.edges[a, b]
                    data["weight"] = float(data.get("weight", 0.0)) + 1.0
                    data["last_seen_s"] = float(timestamp_s)
                else:
                    self.graph.add_edge(
                        ref_a, ref_b, edge_type=SEEN_WITH, weight=1.0,
                        last_seen_s=float(timestamp_s),
                    )
                pairs += 1
        return pairs

    def _enforce_capacity(self) -> None:
        limit = self.config.max_observation_nodes
        if limit is None:
            return
        while len(self._observation_order) > limit:
            oldest = self._observation_order.popleft()
            if oldest in self.g:
                self.g.remove_node(oldest)
            self._evicted_observations += 1

    # ----- queries ---------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        types = {OBJECT: 0, OBSERVATION: 0, PROTOTYPE: 0}
        for _, data in self.g.nodes(data=True):
            key = data.get("node_type")
            if key in types:
                types[key] += 1
        edges = {OBSERVATION_OF: 0, SIMILAR_TO: 0, SEEN_WITH: 0}
        for _, _, data in self.g.edges(data=True):
            key = data.get("edge_type")
            if key in edges:
                edges[key] += 1
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "object_nodes": types[OBJECT],
            "observation_nodes": types[OBSERVATION],
            "prototype_nodes": types[PROTOTYPE],
            "observation_of_edges": edges[OBSERVATION_OF],
            "similar_to_edges": edges[SIMILAR_TO],
            "seen_with_edges": edges[SEEN_WITH],
            "evicted_observations": self._evicted_observations,
        }

    def observations_of(self, object_id: str) -> list[str]:
        """Observation node ids attached to an object, in frame order."""
        if object_id not in self.g:
            return []
        obs = [
            src for src, _, data in self.g.in_edges(object_id, data=True)
            if data.get("edge_type") == OBSERVATION_OF
        ]
        return sorted(obs, key=lambda o: (self.g.nodes[o]["frame_idx"], o))

    def seen_with(self, object_id: str) -> dict[str, float]:
        """Co-visibility counts with every other object."""
        result: dict[str, float] = {}
        if object_id not in self.g:
            return result
        for a, b, data in self.g.edges(data=True):
            if data.get("edge_type") != SEEN_WITH:
                continue
            if a == object_id:
                result[b] = float(data["weight"])
            elif b == object_id:
                result[a] = float(data["weight"])
        return dict(sorted(result.items()))

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready snapshot with deterministic node and edge ordering."""
        payload = self.graph.as_dict()
        payload["nodes"].sort(key=lambda n: n["id"])
        payload["edges"].sort(key=lambda e: (e["src"], e["dst"], e.get("edge_type", "")))
        return {"config": self.config.to_dict(), "counts": self.counts(), **payload}

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        self.graph.load_dict({"nodes": payload["nodes"], "edges": payload["edges"]})
        self._observation_order = deque(
            n["id"] for n in sorted(
                (n for n in payload["nodes"] if n.get("node_type") == OBSERVATION),
                key=lambda n: (n.get("frame_idx", 0), n["id"]),
            )
        )
        self._evicted_observations = int(
            payload.get("counts", {}).get("evicted_observations", 0)
        )
