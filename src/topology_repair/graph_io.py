"""Read-only Gemini graph parsing with canonical curved polylines."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import networkx as nx

@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float

@dataclass(frozen=True)
class Edge:
    """Undirected edge retaining its original curved geometry."""
    source: int
    target: int
    coordinates: tuple[tuple[float, float], ...]
    edge_id: str = ""
    def __iter__(self): yield self.source; yield self.target
    def __getitem__(self, index): return (self.source, self.target)[index]
    def __eq__(self, other):
        if isinstance(other, tuple): return (self.source, self.target) == other
        if isinstance(other, Edge): return (self.source, self.target, self.coordinates) == (other.source, other.target, other.coordinates)
        return NotImplemented
    def length(self):
        return sum(math.hypot(x2-x1, y2-y1) for (x1,y1),(x2,y2) in zip(self.coordinates, self.coordinates[1:]))

@dataclass
class RoadGraph:
    nodes: dict[int, Node]
    edges: list[Edge]
    graph: nx.Graph
    components: dict[int, set[int]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoadGraph":
        nodes = {int(n["id"]): Node(int(n["id"]), float(n["x"]), float(n["y"])) for n in data.get("nodes", [])}
        seen: set[tuple[int, int]] = set(); edges: list[Edge] = []
        for raw in data.get("edges", []):
            a = int(raw.get("source", raw.get("u"))); b = int(raw.get("target", raw.get("v")))
            if a == b or a not in nodes or b not in nodes: continue
            key = (min(a,b), max(a,b))
            if key in seen: continue
            seen.add(key)
            coords = raw.get("coordinates", raw.get("polyline"))
            if coords is None and isinstance(raw.get("geometry"), dict): coords = raw["geometry"].get("coordinates")
            if coords is None and isinstance(raw.get("geometry"), list): coords = raw["geometry"]
            if not coords: coords = [[nodes[a].x, nodes[a].y], [nodes[b].x, nodes[b].y]]
            points = tuple((float(p[0]), float(p[1])) for p in coords if len(p) >= 2)
            if len(points) < 2: points = ((nodes[a].x,nodes[a].y),(nodes[b].x,nodes[b].y))
            if (a,b) != key: points = tuple(reversed(points))
            edges.append(Edge(key[0], key[1], points, f"edge_{len(edges):06d}"))
        g = nx.Graph(); g.add_nodes_from(nodes)
        for e in edges: g.add_edge(e.source, e.target, edge_id=e.edge_id)
        components = {i: set(c) for i, c in enumerate(nx.connected_components(g))}
        return cls(nodes, edges, g, components)

    def component_id(self, node_id: int) -> int: return next(i for i,c in self.components.items() if node_id in c)
    def edge_by_id(self, edge_id: str) -> Edge: return next(e for e in self.edges if e.edge_id == edge_id)
    def component_edges(self, component: set[int]) -> list[Edge]: return [e for e in self.edges if e.source in component and e.target in component]
    def component_length(self, component: set[int]) -> float: return sum(e.length() for e in self.component_edges(component))
    def endpoints(self, component: set[int]) -> list[int]:
        return sorted([n for n in component if self.graph.degree[n] == 1] or ([next(iter(component))] if len(component) == 1 else []))

def load_graph(path: str | Path) -> RoadGraph:
    with open(path, encoding="utf-8") as f: return RoadGraph.from_dict(json.load(f))

def save_repair(path: str | Path, actions: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f: json.dump({"actions": actions}, f, indent=2, ensure_ascii=False)
