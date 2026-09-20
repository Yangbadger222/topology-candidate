from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SimplificationConfig:
    direction_group_angle_deg: float = 15.0
    opposite_axis_tolerance_deg: float = 15.0
    chain_ratio: float = 1.10
    intermediate_line_distance_px: float = 20.0
    alternative_hops: int = 3
    max_turn_deg: float = 35.0
    corridor_opposite_tolerance_deg: float = 35.0


def canonical_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def circular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def edge_angle(points: dict[int, np.ndarray], source: int, target: int) -> float:
    d = points[target] - points[source]
    return math.degrees(math.atan2(float(d[1]), float(d[0]))) % 360.0


def cluster_angles(angles: list[tuple[int, float]], threshold: float) -> list[dict]:
    if not angles:
        return []
    ordered = sorted(angles, key=lambda item: item[1])
    if len(ordered) == 1:
        return [{"members": [ordered[0][0]], "angles": [ordered[0][1]], "center": ordered[0][1]}]
    gaps = [ordered[i + 1][1] - ordered[i][1] for i in range(len(ordered) - 1)]
    gaps.append(ordered[0][1] + 360.0 - ordered[-1][1])
    start = (int(np.argmax(gaps)) + 1) % len(ordered)
    unwrapped = ordered[start:] + ordered[:start]
    groups: list[list[tuple[int, float]]] = [[]]
    previous = unwrapped[0][1]
    groups[0].append(unwrapped[0])
    for node, angle in unwrapped[1:]:
        value = angle
        while value < previous:
            value += 360.0
        if value - previous > threshold:
            groups.append([])
        groups[-1].append((node, angle))
        previous = value
    result = []
    for group in groups:
        radians = np.radians([a for _, a in group])
        center = math.degrees(math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))) % 360.0
        result.append({"members": [n for n, _ in group], "angles": [a for _, a in group], "center": center})
    return result


def axis_count(group_centers: list[float], tolerance: float) -> int:
    axes: list[float] = []
    for angle in group_centers:
        axis = angle % 180.0
        if not any(min(abs(axis - a), 180.0 - abs(axis - a)) <= tolerance for a in axes):
            axes.append(axis)
    return len(axes)


def classify_nodes(graph: nx.Graph, nodes: list[dict], cfg: SimplificationConfig) -> dict[int, dict]:
    points = {int(n["id"]): np.array([float(n["x"]), float(n["y"])]) for n in nodes}
    components = {}
    for component_id, component in enumerate(nx.connected_components(graph)):
        for node in component:
            components[node] = component_id
    roles = {}
    for node in graph.nodes:
        angles = [(neighbor, edge_angle(points, node, neighbor)) for neighbor in graph.neighbors(node)]
        groups = cluster_angles(angles, cfg.direction_group_angle_deg)
        centers = [g["center"] for g in groups]
        degree = graph.degree(node)
        branch_count = len(groups)
        opposite = branch_count == 2 and abs(circular_difference(centers[0], centers[1]) - 180.0) <= cfg.corridor_opposite_tolerance_deg
        if degree == 1:
            role = "ENDPOINT_LIKE"
        elif branch_count == 2 and opposite:
            role = "CORRIDOR_LIKE"
        elif branch_count >= 3:
            role = "JUNCTION_LIKE"
        else:
            role = "AMBIGUOUS"
        roles[node] = {
            "node_id": node,
            "raw_degree": degree,
            "direction_group_count": branch_count,
            "axis_count": axis_count(centers, cfg.opposite_axis_tolerance_deg),
            "branch_count": branch_count,
            "branch_angles": centers,
            "groups": groups,
            "role": role,
            "component_id": components[node],
        }
    return roles


def make_graph(nodes: list[dict], edges: list[dict]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(int(n["id"]) for n in nodes)
    for edge in edges:
        a, b = int(edge["source"]), int(edge["target"])
        score = float(edge.get("score", 0.0))
        if graph.has_edge(a, b):
            graph[a][b]["score"] = max(score, graph[a][b]["score"])
        else:
            graph.add_edge(a, b, score=score)
    return graph


def graph_edges(graph: nx.Graph) -> list[dict]:
    return [{"source": min(a, b), "target": max(a, b), "score": float(data.get("score", 0.0))}
            for a, b, data in sorted(graph.edges(data=True), key=lambda item: canonical_pair(item[0], item[1]))]


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    if float(np.dot(delta, delta)) <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, delta) / np.dot(delta, delta), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * delta)))


def _path_diagnostics(path: list[int], points: dict[int, np.ndarray], direct_length: float) -> dict:
    segments = [points[b] - points[a] for a, b in zip(path, path[1:])]
    lengths = [float(np.linalg.norm(v)) for v in segments]
    turns = []
    for a, b in zip(segments, segments[1:]):
        cosine = float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))
        turns.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
    line_distances = [_point_segment_distance(points[n], points[path[0]], points[path[-1]]) for n in path[1:-1]]
    return {
        "alternative_length": sum(lengths),
        "alternative_ratio": sum(lengths) / max(1e-12, direct_length),
        "chain_smoothness": max(turns, default=0.0),
        "max_intermediate_line_distance": max(line_distances, default=0.0),
    }


def local_alternatives(graph: nx.Graph, source: int, target: int, points: dict[int, np.ndarray],
                       max_hops: int) -> list[tuple[list[int], dict]]:
    if not graph.has_edge(source, target):
        return []
    direct = float(np.linalg.norm(points[target] - points[source]))
    edge_data = dict(graph[source][target])
    graph.remove_edge(source, target)
    paths = []
    try:
        for path in nx.all_simple_paths(graph, source, target, cutoff=max_hops):
            if 2 <= len(path) - 1 <= max_hops:
                paths.append((path, _path_diagnostics(path, points, direct)))
    finally:
        graph.add_edge(source, target, **edge_data)
    return sorted(paths, key=lambda item: (item[1]["alternative_ratio"], len(item[0])))


def _endpoint_relations(graph: nx.Graph, endpoints: set[int]) -> dict[tuple[int, int], bool]:
    component = {}
    for index, members in enumerate(nx.connected_components(graph)):
        for node in members:
            component[node] = index
    return {canonical_pair(a, b): component[a] == component[b]
            for i, a in enumerate(sorted(endpoints)) for b in sorted(endpoints)[i + 1:]}


def _junction_reachability(graph: nx.Graph, junctions: set[int], endpoints: set[int]) -> dict[int, set[int]]:
    result = {}
    for junction in junctions:
        reachable = nx.node_connected_component(graph, junction)
        result[junction] = endpoints & reachable
    return result


def _group_for_neighbor(role: dict, neighbor: int) -> int | None:
    for index, group in enumerate(role["groups"]):
        if neighbor in group["members"]:
            return index
    return None


def _safe_remove(graph: nx.Graph, edge: tuple[int, int], baseline_components: int,
                 baseline_endpoints: set[int], endpoint_relations: dict, baseline_junctions: dict[int, int],
                 junction_reachability: dict, nodes: list[dict], cfg: SimplificationConfig) -> tuple[bool, str]:
    a, b = edge
    data = dict(graph[a][b])
    graph.remove_edge(a, b)
    reason = "passed"
    if nx.number_connected_components(graph) != baseline_components:
        reason = "component_guard"
    elif _endpoint_relations(graph, baseline_endpoints) != endpoint_relations:
        reason = "endpoint_connectivity_guard"
    elif _junction_reachability(graph, baseline_junctions, baseline_endpoints) != junction_reachability:
        reason = "junction_reachability_guard"
    else:
        current = classify_nodes(graph, nodes, cfg)
        for junction, branch_count in baseline_junctions.items():
            if current[junction]["branch_count"] < branch_count:
                reason = "junction_branch_guard"
                break
    if reason != "passed":
        graph.add_edge(a, b, **data)
        return False, reason
    return True, reason


def simplify(nodes: list[dict], edges: list[dict], cfg: SimplificationConfig) -> dict:
    started = perf_counter()
    points = {int(n["id"]): np.array([float(n["x"]), float(n["y"])]) for n in nodes}
    s0 = make_graph(nodes, edges)
    baseline_components = nx.number_connected_components(s0)
    t0 = perf_counter()
    roles0 = classify_nodes(s0, nodes, cfg)
    role_time = perf_counter() - t0
    endpoints = {n for n, r in roles0.items() if r["raw_degree"] == 1}
    junctions = {n: r["branch_count"] for n, r in roles0.items() if r["role"] == "JUNCTION_LIKE"}
    endpoint_relations = _endpoint_relations(s0, endpoints)
    junction_reach = _junction_reachability(s0, set(junctions), endpoints)
    triangles0 = {canonical_pair(a, b) for cycle in nx.enumerate_all_cliques(s0) if len(cycle) == 3
                  for a, b in ((cycle[0], cycle[1]), (cycle[1], cycle[2]), (cycle[0], cycle[2]))}
    bridge_pairs = {canonical_pair(a, b) for a, b in nx.bridges(s0)}

    decision = {}
    for a, b, data in s0.edges(data=True):
        pair = canonical_pair(a, b)
        decision[pair] = {"u": pair[0], "v": pair[1], "length": float(np.linalg.norm(points[a] - points[b])),
                          "toponet_score": float(data.get("score", 0.0)), "u_node_type": roles0[a]["role"],
                          "v_node_type": roles0[b]["role"], "u_direction_group": _group_for_neighbor(roles0[a], b),
                          "v_direction_group": _group_for_neighbor(roles0[b], a),
                          "u_road_axis_count": roles0[a]["axis_count"], "v_road_axis_count": roles0[b]["axis_count"],
                          "is_triangle_edge": pair in triangles0, "is_cycle_edge": pair not in bridge_pairs,
                          "alternative_path_exists": False, "alternative_hops": "", "alternative_length": "",
                          "alternative_ratio": "", "chain_smoothness": "", "intermediate_nodes": "",
                          "S1_keep": True, "S1_reason": "kept", "S2_keep": True, "S2_reason": "kept", "S3_mapping": ""}

    t0 = perf_counter()
    s1 = s0.copy()
    traces = []
    for a, b in sorted(s0.edges(), key=lambda e: float(np.linalg.norm(points[e[0]] - points[e[1]])), reverse=True):
        if not s1.has_edge(a, b):
            continue
        alternatives = local_alternatives(s1, a, b, points, cfg.alternative_hops)
        pair = canonical_pair(a, b)
        if alternatives:
            path, diag = alternatives[0]
            decision[pair].update({"alternative_path_exists": True, "alternative_hops": len(path) - 1,
                                   "alternative_length": diag["alternative_length"], "alternative_ratio": diag["alternative_ratio"],
                                   "chain_smoothness": diag["chain_smoothness"],
                                   "intermediate_nodes": ";".join(map(str, path[1:-1]))})
        valid = [(path, diag) for path, diag in alternatives
                 if diag["alternative_ratio"] <= cfg.chain_ratio
                 and diag["chain_smoothness"] <= cfg.max_turn_deg
                 and diag["max_intermediate_line_distance"] <= cfg.intermediate_line_distance_px]
        if not valid:
            continue
        path, diag = valid[0]
        removed, guard = _safe_remove(s1, (a, b), baseline_components, endpoints, endpoint_relations,
                                      junctions, junction_reach, nodes, cfg)
        if removed:
            decision[pair]["S1_keep"] = False
            decision[pair]["S1_reason"] = "redundant_chord"
            traces.append({"stage": "S1", "edge": list(pair), "reason": "redundant_chord",
                           "alternative_path": path, "diagnostics": diag})
        elif guard != "passed":
            decision[pair]["S1_reason"] = f"kept_{guard}"
    chord_time = perf_counter() - t0

    t0 = perf_counter()
    s2 = s1.copy()
    roles1 = classify_nodes(s1, nodes, cfg)
    for node in sorted(s2.nodes):
        role = roles1[node]
        if role["role"] != "CORRIDOR_LIKE":
            continue
        for group in role["groups"]:
            active = [n for n in group["members"] if s2.has_edge(node, n)]
            if len(active) <= 1:
                continue
            ranked = sorted(active, key=lambda n: (float(np.linalg.norm(points[n] - points[node])),
                                                    -float(s2[node][n].get("score", 0.0))))
            for neighbor in ranked[1:]:
                if not s2.has_edge(node, neighbor):
                    continue
                alternatives = local_alternatives(s2, node, neighbor, points, cfg.alternative_hops)
                valid = [(path, diag) for path, diag in alternatives
                         if diag["alternative_ratio"] <= cfg.chain_ratio
                         and diag["chain_smoothness"] <= cfg.max_turn_deg]
                if not valid:
                    continue
                pair = canonical_pair(node, neighbor)
                removed, guard = _safe_remove(s2, (node, neighbor), baseline_components, endpoints,
                                              endpoint_relations, junctions, junction_reach, nodes, cfg)
                if removed:
                    decision[pair]["S2_keep"] = False
                    decision[pair]["S2_reason"] = "same_branch_duplicate"
                    path, diag = valid[0]
                    traces.append({"stage": "S2", "edge": list(pair), "reason": "same_branch_duplicate",
                                   "alternative_path": path, "diagnostics": diag})
                else:
                    decision[pair]["S2_reason"] = f"kept_{guard}"
    for pair, row in decision.items():
        if not row["S1_keep"]:
            row["S2_keep"] = False
            row["S2_reason"] = "removed_in_S1"
    corridor_time = perf_counter() - t0

    t0 = perf_counter()
    s3 = contract_degree_two(s2, nodes, cfg, endpoints, set(junctions))
    for nav_edge in s3["edges"]:
        mapping = f"{nav_edge['source']}-{nav_edge['target']}"
        for a, b in zip(nav_edge["path_nodes"], nav_edge["path_nodes"][1:]):
            pair = canonical_pair(a, b)
            if pair in decision:
                decision[pair]["S3_mapping"] = mapping
    contraction_time = perf_counter() - t0
    return {
        "config": asdict(cfg), "S0": s0, "S1": s1, "S2": s2, "S3": s3,
        "roles": {"S0": roles0, "S1": roles1, "S2": classify_nodes(s2, nodes, cfg)},
        "decisions": list(decision.values()), "trace": traces,
        "runtime": {"role_classification_s": role_time, "chord_simplification_s": chord_time,
                    "corridor_simplification_s": corridor_time, "node_contraction_s": contraction_time,
                    "total_simplification_s": perf_counter() - started},
    }


def contract_degree_two(graph: nx.Graph, nodes: list[dict], cfg: SimplificationConfig,
                        protected_endpoints: set[int], protected_junctions: set[int]) -> dict:
    points = {int(n["id"]): [float(n["x"]), float(n["y"])] for n in nodes}
    work = nx.Graph()
    work.add_nodes_from(graph.nodes)
    for a, b, data in graph.edges(data=True):
        work.add_edge(a, b, score=float(data.get("score", 0.0)), path_nodes=[a, b], polyline=[points[a], points[b]])
    cycle_nodes = {n for cycle in nx.cycle_basis(graph) for n in cycle}
    contracted = []
    changed = True
    while changed:
        changed = False
        roles = classify_nodes(work, nodes, cfg)
        for node in sorted(work.nodes):
            if node in protected_endpoints or node in protected_junctions or node in cycle_nodes:
                continue
            if work.degree(node) != 2 or roles[node]["role"] != "CORRIDOR_LIKE":
                continue
            u, v = list(work.neighbors(node))
            if u == v or work.has_edge(u, v):
                continue
            left, right = dict(work[u][node]), dict(work[node][v])
            def orient(data, start, end):
                path = list(data["path_nodes"]); poly = list(data["polyline"])
                return (path, poly) if path[0] == start and path[-1] == end else (path[::-1], poly[::-1])
            lp, lxy = orient(left, u, node); rp, rxy = orient(right, node, v)
            work.add_edge(u, v, score=min(left["score"], right["score"]),
                          path_nodes=lp + rp[1:], polyline=lxy + rxy[1:])
            work.remove_node(node); contracted.append(node); changed = True; break
    nav_edges = []
    for a, b, data in work.edges(data=True):
        path = list(data["path_nodes"]); poly = list(data["polyline"])
        if path[0] != a:
            path.reverse(); poly.reverse()
        nav_edges.append({"source": a, "target": b, "score": data["score"], "path_nodes": path, "polyline": poly})
    return {"nodes": [{**next(n for n in nodes if int(n["id"]) == node)} for node in work.nodes],
            "edges": nav_edges, "contracted_nodes": contracted,
            "components": nx.number_connected_components(work)}


def metrics(graph: nx.Graph, roles: dict[int, dict]) -> dict:
    degrees = [graph.degree(n) for n in graph.nodes]
    corridor = [graph.degree(n) for n, role in roles.items() if role["role"] == "CORRIDOR_LIKE"]
    triangles = sum(nx.triangles(graph).values()) // 3
    components = nx.number_connected_components(graph)
    component_sizes = [len(c) for c in nx.connected_components(graph)]
    return {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "median_degree": float(np.median(degrees)) if degrees else 0.0,
        "p90_degree": float(np.percentile(degrees, 90)) if degrees else 0.0,
        "max_degree": max(degrees, default=0),
        "mean_effective_branch_count": float(np.mean([r["branch_count"] for r in roles.values()])) if roles else 0.0,
        "corridor_node_count": len(corridor),
        "junction_like_node_count": sum(r["role"] == "JUNCTION_LIKE" for r in roles.values()),
        "endpoint_like_node_count": sum(r["role"] == "ENDPOINT_LIKE" for r in roles.values()),
        "ambiguous_node_count": sum(r["role"] == "AMBIGUOUS" for r in roles.values()),
        "triangle_count": triangles, "triangles_per_node": triangles / max(1, graph.number_of_nodes()),
        "local_clustering_coefficient": float(nx.average_clustering(graph)) if graph else 0.0,
        "cycle_rank": graph.number_of_edges() - graph.number_of_nodes() + components,
        "components": components,
        "largest_component_ratio": max(component_sizes, default=0) / max(1, graph.number_of_nodes()),
        "edges_per_component": graph.number_of_edges() / max(1, components),
        "junctions_per_component": sum(r["role"] == "JUNCTION_LIKE" for r in roles.values()) / max(1, components),
        "corridor_nodes_per_component": len(corridor) / max(1, components),
        "corridor_mean_degree": float(np.mean(corridor)) if corridor else 0.0,
        "corridor_median_degree": float(np.median(corridor)) if corridor else 0.0,
        "corridor_p90_degree": float(np.percentile(corridor, 90)) if corridor else 0.0,
    }
