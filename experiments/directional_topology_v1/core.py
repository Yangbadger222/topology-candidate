from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class DirectionalConfig:
    direction_patch_radius: int = 30
    direction_gamma: float = 1.0
    anisotropy_threshold: float = 0.7
    axis_angle_deg: float = 20.0
    branch_separation_deg: float = 35.0
    same_road_direction_deg: float = 15.0
    cluster_distance_px: float = 50.0
    cluster_lateral_px: float = 25.0
    chain_angle_deg: float = 15.0
    chain_detour_ratio: float = 1.18
    sample_step_px: float = 4.0
    min_samples: int = 16
    corridor_radius_px: int = 5
    unsupported_probability: float = 0.1
    max_unsupported_gap_fraction: float = 0.5
    min_unsupported_gap_px: float = 40.0
    topo_threshold: float = 0.373


def _sample(im: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(xs).astype(int), 0, im.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, im.shape[0] - 1)
    return im[yi, xi].astype(float)


def estimate_directions(nodes: list[dict], road: np.ndarray, cfg: DirectionalConfig) -> list[dict]:
    result = []
    radius = cfg.direction_patch_radius
    h, w = road.shape
    for node in nodes:
        x, y = float(node["x"]), float(node["y"])
        x0, x1 = max(0, int(math.floor(x - radius))), min(w, int(math.ceil(x + radius + 1)))
        y0, y1 = max(0, int(math.floor(y - radius))), min(h, int(math.ceil(y + radius + 1)))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        weights = np.power(np.clip(road[y0:y1, x0:x1], 0, 1), cfg.direction_gamma) * inside
        total = float(weights.sum())
        if total <= 1e-8:
            vx, vy, l1, l2 = 1.0, 0.0, 0.0, 0.0
        else:
            mx = float((weights * xx).sum() / total)
            my = float((weights * yy).sum() / total)
            dx, dy = xx - mx, yy - my
            cov = np.array([
                [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
            ]) / total
            values, vectors = np.linalg.eigh(cov)
            order = np.argsort(values)[::-1]
            l1, l2 = float(values[order[0]]), float(values[order[1]])
            vx, vy = map(float, vectors[:, order[0]])
            if vx < 0 or (abs(vx) < 1e-12 and vy < 0):
                vx, vy = -vx, -vy
        anis = (l1 - l2) / (l1 + l2 + 1e-12)
        result.append({"node": int(node["id"]), "x": x, "y": y, "direction_x": vx,
                       "direction_y": vy, "anisotropy": float(anis),
                       "eigenvalue_1": l1, "eigenvalue_2": l2, "road_weight": total})
    return result


def _angle_axis_deg(vector: np.ndarray, axis: np.ndarray) -> float:
    nv = float(np.linalg.norm(vector))
    if nv <= 1e-12:
        return 90.0
    cosine = min(1.0, max(0.0, abs(float(np.dot(vector / nv, axis)))))
    return math.degrees(math.acos(cosine))


def _direction_difference_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = min(1.0, max(0.0, abs(float(np.dot(a, b)))))
    return math.degrees(math.acos(cosine))


def _ray_angle(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0


def _angular_groups(angles: list[float], separation: float) -> int:
    if not angles:
        return 0
    ordered = sorted(angles)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)] + [ordered[0] + 360 - ordered[-1]]
    return max(1, sum(g > separation for g in gaps))


def edge_support(road: np.ndarray, a: np.ndarray, b: np.ndarray, cfg: DirectionalConfig) -> dict:
    delta = b - a
    length = float(np.linalg.norm(delta))
    count = max(cfg.min_samples, int(math.ceil(length / cfg.sample_step_px)) + 1)
    t = np.linspace(0.0, 1.0, count)
    xs, ys = a[0] + t * delta[0], a[1] + t * delta[1]
    line = _sample(road, xs, ys)
    if length > 1e-8:
        normal = np.array([-delta[1], delta[0]]) / length
    else:
        normal = np.zeros(2)
    offsets = np.arange(-cfg.corridor_radius_px, cfg.corridor_radius_px + 1)
    corridor_stack = np.stack([_sample(road, xs + o * normal[0], ys + o * normal[1]) for o in offsets])
    corridor = corridor_stack.max(axis=0)

    def longest_gap(values: np.ndarray) -> tuple[int, float, float]:
        longest = current = 0
        for low in values < cfg.unsupported_probability:
            current = current + 1 if low else 0
            longest = max(longest, current)
        fraction = longest / max(1, len(values))
        return longest, fraction, fraction * length

    line_gap = longest_gap(line)
    corridor_gap = longest_gap(corridor)
    return {
        "distance": length,
        "line_mean": float(line.mean()), "line_median": float(np.median(line)),
        "line_p10": float(np.percentile(line, 10)),
        "line_ratio_gt_0_1": float((line > 0.1).mean()),
        "line_ratio_gt_0_2": float((line > 0.2).mean()),
        "line_ratio_gt_0_5": float((line > 0.5).mean()),
        "corridor_mean": float(corridor.mean()), "corridor_median": float(np.median(corridor)),
        "corridor_p10": float(np.percentile(corridor, 10)),
        "corridor_ratio_gt_0_1": float((corridor > 0.1).mean()),
        "corridor_ratio_gt_0_2": float((corridor > 0.2).mean()),
        "corridor_ratio_gt_0_5": float((corridor > 0.5).mean()),
        "line_unsupported_gap_samples": line_gap[0], "line_unsupported_gap_fraction": line_gap[1],
        "line_unsupported_gap_px": line_gap[2],
        "corridor_unsupported_gap_samples": corridor_gap[0],
        "corridor_unsupported_gap_fraction": corridor_gap[1],
        "corridor_unsupported_gap_px": corridor_gap[2],
    }


def same_road_clusters(nodes: list[dict], directions: list[dict], road: np.ndarray,
                       cfg: DirectionalConfig) -> list[int]:
    n = len(nodes)
    parent = list(range(n))
    points = np.array([[float(x["x"]), float(x["y"])] for x in nodes])

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(n):
        axis_i = np.array([directions[i]["direction_x"], directions[i]["direction_y"]])
        if directions[i]["anisotropy"] < cfg.anisotropy_threshold:
            continue
        for j in range(i + 1, n):
            delta = points[j] - points[i]
            distance = float(np.linalg.norm(delta))
            if distance > cfg.cluster_distance_px or directions[j]["anisotropy"] < cfg.anisotropy_threshold:
                continue
            axis_j = np.array([directions[j]["direction_x"], directions[j]["direction_y"]])
            if _direction_difference_deg(axis_i, axis_j) > cfg.same_road_direction_deg:
                continue
            lateral = abs(float(np.cross(axis_i, delta)))
            if lateral > cfg.cluster_lateral_px:
                continue
            if edge_support(road, points[i], points[j], cfg)["corridor_ratio_gt_0_1"] >= 0.5:
                union(i, j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def cleanup_candidates(nodes: list[dict], candidates: list[dict], road: np.ndarray,
                       cfg: DirectionalConfig) -> dict:
    started = perf_counter()
    points = np.array([[float(n["x"]), float(n["y"])] for n in nodes])
    t0 = perf_counter()
    directions = estimate_directions(nodes, road, cfg)
    direction_time = perf_counter() - t0
    t0 = perf_counter()
    clusters = same_road_clusters(nodes, directions, road, cfg)
    cluster_time = perf_counter() - t0
    selection_started = perf_counter()

    outgoing = defaultdict(list)
    supports = []
    records = []
    for candidate_id, edge in enumerate(candidates):
        s, t = int(edge["source"]), int(edge["target"])
        vector = points[t] - points[s]
        src_axis = np.array([directions[s]["direction_x"], directions[s]["direction_y"]])
        tgt_axis = np.array([directions[t]["direction_x"], directions[t]["direction_y"]])
        support = edge_support(road, points[s], points[t], cfg)
        supports.append(support)
        projection = float(np.dot(vector, src_axis))
        rec = {
            "candidate_id": candidate_id, "source": s, "target": t,
            **support,
            "source_anisotropy": directions[s]["anisotropy"],
            "target_anisotropy": directions[t]["anisotropy"],
            "source_direction": f"{src_axis[0]:.6f},{src_axis[1]:.6f}",
            "target_direction": f"{tgt_axis[0]:.6f},{tgt_axis[1]:.6f}",
            "angle_to_source_axis": _angle_axis_deg(vector, src_axis),
            "angle_to_target_axis": _angle_axis_deg(vector, tgt_axis),
            "direction_score": math.cos(math.radians(_angle_axis_deg(vector, src_axis))),
            "forward_or_backward": "forward" if projection >= 0 else "backward",
            "same_road_cluster_id": clusters[s] if clusters[s] == clusters[t] else -1,
            "toponet_score": float(edge["score"]),
            "toponet_legacy_accept": float(edge["score"]) > cfg.topo_threshold,
        }
        records.append(rec)
        outgoing[s].append(candidate_id)

    # Junction evidence combines low PCA anisotropy with multiple separated rays.
    junction_like = []
    for i in range(len(nodes)):
        ray_angles = [_ray_angle(points[int(candidates[k]["target"])] - points[i]) for k in outgoing[i]]
        groups = _angular_groups(ray_angles, cfg.branch_separation_deg)
        junction_like.append(directions[i]["anisotropy"] < cfg.anisotropy_threshold and groups >= 3)
        directions[i]["candidate_direction_groups"] = groups
        directions[i]["junction_like"] = junction_like[-1]

    c1 = np.ones(len(candidates), dtype=bool)
    c1_reason = ["passed"] * len(candidates)
    for source, ids in outgoing.items():
        if junction_like[source] or directions[source]["anisotropy"] < cfg.anisotropy_threshold:
            continue
        by_side = defaultdict(list)
        for idx in ids:
            by_side[records[idx]["forward_or_backward"]].append(idx)
        for side_ids in by_side.values():
            aligned = [idx for idx in side_ids if records[idx]["angle_to_source_axis"] <= cfg.axis_angle_deg]
            if not aligned:
                continue
            nearest_aligned = min(records[idx]["distance"] for idx in aligned)
            for idx in side_ids:
                if records[idx]["angle_to_source_axis"] > cfg.axis_angle_deg and records[idx]["distance"] >= nearest_aligned:
                    c1[idx] = False
                    c1_reason[idx] = "off_axis_redundant"
            ranked = sorted(aligned, key=lambda idx: (records[idx]["angle_to_source_axis"],
                                                       records[idx]["distance"],
                                                       -records[idx]["corridor_ratio_gt_0_1"],
                                                       -float(nodes[records[idx]["target"]].get("score", 0.0))))
            for idx in ranked[1:]:
                nearer = [other for other in ranked if records[other]["distance"] < .8 * records[idx]["distance"]
                          and abs(records[other]["angle_to_source_axis"] - records[idx]["angle_to_source_axis"]) <= 10
                          and records[other]["corridor_unsupported_gap_fraction"] <= records[idx]["corridor_unsupported_gap_fraction"] + .15]
                if nearer:
                    c1[idx] = False
                    c1_reason[idx] = "same_branch_farther_neighbor"

    # C2 suppresses an undirected long edge only if two shorter kept edges form a local chain.
    c2 = c1.copy()
    c2_reason = c1_reason.copy()
    pair_to_ids = defaultdict(list)
    for idx, edge in enumerate(candidates):
        pair_to_ids[tuple(sorted((int(edge["source"]), int(edge["target"]))))].append(idx)
    active_pairs = {p for p, ids in pair_to_ids.items() if any(c1[i] for i in ids)}
    chord_pairs = set()
    intermediate_for_pair = {}
    for a, c in active_pairs:
        ac = float(np.linalg.norm(points[c] - points[a]))
        if ac <= 1e-8:
            continue
        for b in range(len(nodes)):
            if b in (a, c) or tuple(sorted((a, b))) not in active_pairs or tuple(sorted((b, c))) not in active_pairs:
                continue
            ab = float(np.linalg.norm(points[b] - points[a])); bc = float(np.linalg.norm(points[c] - points[b]))
            if max(ab, bc) >= ac or ab + bc > cfg.chain_detour_ratio * ac:
                continue
            angle_a = _angle_axis_deg(points[b] - points[a], (points[c] - points[a]) / ac)
            angle_c = _angle_axis_deg(points[b] - points[c], (points[a] - points[c]) / ac)
            if max(angle_a, angle_c) <= cfg.chain_angle_deg:
                chord_pairs.add((a, c)); intermediate_for_pair[(a, c)] = b; break
    for pair in chord_pairs:
        for idx in pair_to_ids[pair]:
            if c2[idx]:
                c2[idx] = False; c2_reason[idx] = "redundant_chord"

    c3 = c2.copy()
    c3_reason = c2_reason.copy()
    for pair, ids in pair_to_ids.items():
        active = [idx for idx in ids if c2[idx]]
        if not active:
            continue
        support = supports[active[0]]
        long_gap = (support["line_unsupported_gap_fraction"] > cfg.max_unsupported_gap_fraction and
                    support["corridor_unsupported_gap_fraction"] > cfg.max_unsupported_gap_fraction and
                    support["corridor_unsupported_gap_px"] >= cfg.min_unsupported_gap_px)
        if long_gap:
            for idx in ids:
                if c3[idx]:
                    c3[idx] = False; c3_reason[idx] = "long_unsupported_shortcut"

    for idx, rec in enumerate(records):
        pair = tuple(sorted((rec["source"], rec["target"])))
        rec.update({
            "intermediate_node_exists": pair in intermediate_for_pair,
            "intermediate_node": intermediate_for_pair.get(pair, -1),
            "chain_exists": pair in chord_pairs,
            "is_chord_candidate": pair in chord_pairs,
            "C1_decision": "keep" if c1[idx] else "reject",
            "C2_decision": "keep" if c2[idx] else "reject",
            "C3_decision": "keep" if c3[idx] else "reject",
            "C1_reject_reason": c1_reason[idx], "C2_reject_reason": c2_reason[idx],
            "C3_reject_reason": c3_reason[idx], "reject_reason": c3_reason[idx],
        })
    return {
        "directions": directions, "clusters": clusters, "decisions": records,
        "masks": {"C0": np.ones(len(candidates), bool), "C1": c1, "C2": c2, "C3": c3},
        "runtime": {"direction_estimation_s": direction_time, "clustering_s": cluster_time,
                    "candidate_selection_s": perf_counter() - selection_started,
                    "total_cleanup_s": perf_counter() - started},
    }


def aggregate_variant(candidates: list[dict], mask: np.ndarray, topo_threshold: float) -> tuple[list[dict], list[dict]]:
    kept = [dict(edge) for edge, include in zip(candidates, mask) if include]
    scores = defaultdict(list)
    for edge in kept:
        if float(edge["score"]) > topo_threshold:
            scores[tuple(sorted((int(edge["source"]), int(edge["target"]))))].append(float(edge["score"]))
    final = [{"source": p[0], "target": p[1], "score": float(np.mean(values))}
             for p, values in sorted(scores.items())]
    return kept, final


def graph_metrics(nodes: list[dict], candidates: list[dict], final: list[dict], decisions: list[dict]) -> dict:
    def build(edges: list[dict]) -> nx.Graph:
        graph = nx.Graph(); graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from((int(x["source"]), int(x["target"])) for x in edges)
        return graph
    candidate_graph, final_graph = build(candidates), build(final)
    degrees = np.array([candidate_graph.degree(i) for i in range(len(nodes))], dtype=float)
    components = list(nx.connected_components(final_graph))
    pair_chords = {tuple(sorted((r["source"], r["target"]))) for r in decisions if r["is_chord_candidate"]}
    active_pairs = {tuple(sorted((int(e["source"]), int(e["target"])))) for e in candidates}
    crossings = 0
    points = np.array([[float(n["x"]), float(n["y"])] for n in nodes])
    edges = list(active_pairs)
    def orient(a, b, c): return float(np.cross(b - a, c - a))
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1:]:
            if len({a, b, c, d}) < 4: continue
            if orient(points[a], points[b], points[c]) * orient(points[a], points[b], points[d]) < 0 and orient(points[c], points[d], points[a]) * orient(points[c], points[d], points[b]) < 0:
                crossings += 1
    return {
        "nodes": len(nodes), "directed_candidates": len(candidates), "candidates": len(active_pairs),
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "median_degree": float(np.median(degrees)) if len(degrees) else 0.0,
        "p90_degree": float(np.percentile(degrees, 90)) if len(degrees) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "candidate_triangles": int(sum(nx.triangles(candidate_graph).values()) // 3),
        "candidate_crossings": crossings, "chords": len(pair_chords & active_pairs),
        "chord_fraction": len(pair_chords & active_pairs) / max(1, len(active_pairs)),
        "final_edges": final_graph.number_of_edges(),
        "final_triangles": int(sum(nx.triangles(final_graph).values()) // 3),
        "components": len(components), "largest_component": max(map(len, components), default=0),
    }
