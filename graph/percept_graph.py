from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NodeRef:
    """Lightweight node reference for graph debugging/serialization."""

    node_id: str
    node_type: str


class PerceptGraph:
    """
    Dynamic percept graph (DAG-ish) reference implementation.

    Phase-1 scope: provide a place to attach BBPs later, without committing to schema yet.
    """

    def __init__(self) -> None:
        try:
            import networkx as nx  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "networkx is required for the percept graph. Install with: pip install networkx"
            ) from e
        self._nx = nx
        self.g = nx.DiGraph()

    def add_node(self, node: NodeRef, **attrs: Any) -> None:
        self.g.add_node(node.node_id, node_type=node.node_type, **attrs)

    def add_edge(
        self, src: NodeRef, dst: NodeRef, *, edge_type: str, weight: float = 1.0, **attrs: Any
    ) -> None:
        self.g.add_edge(
            src.node_id, dst.node_id, edge_type=edge_type, weight=float(weight), **attrs
        )

    def decay_edges(self, *, rate: float, min_weight: float = 1e-6) -> int:
        """Multiply all edge weights by (1-rate) and prune tiny edges."""
        rate = float(rate)
        if not (0.0 <= rate < 1.0):
            raise ValueError("rate must be in [0, 1).")

        to_remove = []
        for u, v, data in self.g.edges(data=True):
            w = float(data.get("weight", 1.0)) * (1.0 - rate)
            data["weight"] = w
            if w < min_weight:
                to_remove.append((u, v))
        for u, v in to_remove:
            self.g.remove_edge(u, v)
        return len(to_remove)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": n, **self.g.nodes[n]}
                for n in self.g.nodes
            ],
            "edges": [
                {"src": u, "dst": v, **data}
                for u, v, data in self.g.edges(data=True)
            ],
        }

    def load_dict(self, payload: dict[str, Any]) -> None:
        self.g.clear()
        for n in payload.get("nodes", []):
            nid = n["id"]
            attrs = dict(n)
            attrs.pop("id", None)
            self.g.add_node(nid, **attrs)
        for e in payload.get("edges", []):
            src = e["src"]
            dst = e["dst"]
            attrs = dict(e)
            attrs.pop("src", None)
            attrs.pop("dst", None)
            self.g.add_edge(src, dst, **attrs)

