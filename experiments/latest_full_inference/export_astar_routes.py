"""Select diverse long-distance A* routes from a MaGRoad navigation graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import networkx as nx
from PIL import Image, ImageDraw, ImageFont


COLORS = ["#ff2d55", "#00c2ff", "#ffd60a", "#32d74b", "#bf5af2", "#ff9f0a"]


def polyline_length(points: list[list[float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def load_graph(path: Path) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    data = json.loads(path.read_text())
    positions = {int(n["id"]): (float(n["x"]), float(n["y"])) for n in data["nodes"]}
    graph = nx.Graph()
    for node_id, position in positions.items():
        graph.add_node(node_id, pos=position)
    for edge in data["edges"]:
        points = [[float(x), float(y)] for x, y in edge["polyline"]]
        graph.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            weight=polyline_length(points),
            polyline=points,
            score=float(edge.get("score", 0.0)),
        )
    return graph, positions


def oriented_edge_points(graph: nx.Graph, positions: dict, source: int, target: int) -> list[list[float]]:
    points = graph.edges[source, target]["polyline"]
    if math.dist(points[0], positions[source]) > math.dist(points[-1], positions[source]):
        return list(reversed(points))
    return points


def route_polyline(graph: nx.Graph, positions: dict, nodes: list[int]) -> list[list[float]]:
    result: list[list[float]] = []
    for source, target in zip(nodes, nodes[1:]):
        points = oriented_edge_points(graph, positions, source, target)
        result.extend(points if not result else points[1:])
    return result


def select_routes(graph: nx.Graph, positions: dict, count: int, max_overlap: float) -> tuple[list[dict], int]:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component = graph.subgraph(components[0]).copy()
    nodes = list(component.nodes)
    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    anchors = [
        min(nodes, key=lambda n: positions[n][0]),
        max(nodes, key=lambda n: positions[n][0]),
        min(nodes, key=lambda n: positions[n][1]),
        max(nodes, key=lambda n: positions[n][1]),
        min(nodes, key=lambda n: positions[n][0] + positions[n][1]),
        max(nodes, key=lambda n: positions[n][0] + positions[n][1]),
        min(nodes, key=lambda n: positions[n][0] - positions[n][1]),
        max(nodes, key=lambda n: positions[n][0] - positions[n][1]),
    ]
    # Add grid-extreme anchors so the candidates cover different parts of a large component.
    for fx, fy in [(0, 0), (0.5, 0), (1, 0), (0, 0.5), (1, 0.5), (0, 1), (0.5, 1), (1, 1)]:
        tx, ty = min(xs) + fx * (max(xs) - min(xs)), min(ys) + fy * (max(ys) - min(ys))
        anchors.append(min(nodes, key=lambda n: math.dist(positions[n], (tx, ty))))
    anchors = list(dict.fromkeys(anchors))

    candidates: list[dict] = []
    for start in anchors:
        distances, paths = nx.single_source_dijkstra(component, start, weight="weight")
        far_nodes = sorted(distances, key=distances.get, reverse=True)[:12]
        for goal in far_nodes:
            path = paths[goal]
            edge_set = {tuple(sorted(edge)) for edge in zip(path, path[1:])}
            candidates.append(
                {
                    "start": start,
                    "goal": goal,
                    "nodes": path,
                    "length": float(distances[goal]),
                    "direct": math.dist(positions[start], positions[goal]),
                    "edges": edge_set,
                }
            )
    candidates.sort(key=lambda item: (item["direct"], item["length"]), reverse=True)

    selected: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()
    used_endpoints: set[int] = set()
    for candidate in candidates:
        pair = tuple(sorted((candidate["start"], candidate["goal"])))
        if pair in used_pairs or candidate["start"] in used_endpoints or candidate["goal"] in used_endpoints:
            continue
        overlap_ok = True
        for old in selected:
            intersection = len(candidate["edges"] & old["edges"])
            denominator = min(len(candidate["edges"]), len(old["edges"]))
            if denominator and intersection / denominator > max_overlap:
                overlap_ok = False
                break
        if overlap_ok:
            selected.append(candidate)
            used_pairs.add(pair)
            used_endpoints.update(pair)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} sufficiently diverse routes")
    return selected, len(component)


def draw_route(base: Image.Image, points: list[list[float]], color: str, label: str) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    scale = max(image.size) / 2048
    width = max(8, round(5 * scale))
    outline = max(width + 5, round(8 * scale))
    xy = [(round(x), round(y)) for x, y in points]
    draw.line(xy, fill="#101010", width=outline, joint="curve")
    draw.line(xy, fill=color, width=width, joint="curve")
    radius = max(18, round(13 * scale))
    for point, marker in [(xy[0], "S"), (xy[-1], "G")]:
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=max(3, round(2 * scale)))
        font = ImageFont.load_default(size=max(14, round(11 * scale)))
        box = draw.textbbox((0, 0), marker, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), marker, fill="#111111", font=font)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--max-overlap", type=float, default=0.75)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    graph, positions = load_graph(args.graph)
    selected, largest_component_nodes = select_routes(graph, positions, args.count, args.max_overlap)
    base = Image.open(args.image).convert("RGB")
    results = []
    overview = base.copy()
    for index, (candidate, color) in enumerate(zip(selected, COLORS), start=1):
        points = route_polyline(graph, positions, candidate["nodes"])
        route_id = f"route_{index:02d}"
        result = {
            "id": route_id,
            "color": color,
            "start_node": candidate["start"],
            "goal_node": candidate["goal"],
            "start_xy": list(positions[candidate["start"]]),
            "goal_xy": list(positions[candidate["goal"]]),
            "straight_line_distance_px": candidate["direct"],
            "astar_path_length_px": candidate["length"],
            "detour_ratio": candidate["length"] / candidate["direct"],
            "path_node_count": len(candidate["nodes"]),
            "path_nodes": candidate["nodes"],
            "route_polyline": points,
        }
        results.append(result)
        route_image = draw_route(base, points, color, route_id)
        route_image.save(args.output / f"{route_id}.jpg", quality=92, subsampling=1)
        overview = draw_route(overview, points, color, route_id)

    overview.save(args.output / "all_routes_overlay.jpg", quality=92, subsampling=1)
    preview = overview.copy()
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    preview.save(args.output / "all_routes_preview.jpg", quality=94, subsampling=1)
    payload = {
        "graph": str(args.graph.resolve()),
        "graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "image": str(args.image.resolve()),
        "method": "A* with Euclidean heuristic and edge-polyline Euclidean length",
        "selection": f"diverse long-distance pairs in the largest connected component; pairwise route overlap capped at {args.max_overlap:.0%} of the shorter route",
        "largest_component_nodes": largest_component_nodes,
        "route_count": len(results),
        "routes": results,
    }
    (args.output / "astar_routes.json").write_text(json.dumps(payload, indent=2) + "\n")
    pairs = [{"id": r["id"], "start": r["start_xy"], "goal": r["goal_xy"]} for r in results]
    (args.output / "astar_pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "routes": [{k: r[k] for k in ("id", "start_xy", "goal_xy", "straight_line_distance_px", "astar_path_length_px", "detour_ratio", "path_node_count")} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
