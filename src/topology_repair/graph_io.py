"""Graph parsing utilities. Stored degree/type fields are deliberately ignored."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import math
import networkx as nx

@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float

@dataclass
class RoadGraph:
    nodes: dict[int, Node]
    edges: list[tuple[int, int]]
    graph: nx.Graph
    components: dict[int, set[int]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoadGraph":
        nodes = {int(n["id"]): Node(int(n["id"]), float(n["x"]), float(n["y"])) for n in data.get("nodes", [])}
        seen: set[tuple[int, int]] = set(); edges = []
        for e in data.get("edges", []):
            a, b = int(e.get("source", e.get("u"))), int(e.get("target", e.get("v")))
            if a == b or a not in nodes or b not in nodes: continue
            key = (min(a,b), max(a,b))
            if key not in seen: seen.add(key); edges.append(key)
        g = nx.Graph(); g.add_nodes_from(nodes); g.add_edges_from(edges)
        components = {i: set(c) for i, c in enumerate(nx.connected_components(g))}
        return cls(nodes, edges, g, components)

    def component_id(self, node_id: int) -> int:
        return next(i for i, c in self.components.items() if node_id in c)

    def component_length(self, component: set[int]) -> float:
        return sum(math.hypot(self.nodes[a].x-self.nodes[b].x, self.nodes[a].y-self.nodes[b].y)
                   for a,b in self.edges if a in component and b in component)

    def endpoints(self, component: set[int]) -> list[int]:
        return [n for n in component if self.graph.degree[n] == 1] or ([next(iter(component))] if len(component) == 1 else [])

def load_graph(path: str | Path) -> RoadGraph:
    with open(path, encoding="utf-8") as f: return RoadGraph.from_dict(json.load(f))

def save_repair(path: str | Path, actions: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f: json.dump({"actions": actions}, f, indent=2, ensure_ascii=False)
