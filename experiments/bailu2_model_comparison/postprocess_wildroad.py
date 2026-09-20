#!/usr/bin/env python3
"""Apply the frozen C2/S0 graph stages to WildRoad bailu_2 outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import fields
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from experiments.directional_topology_v1.core import (
    DirectionalConfig,
    aggregate_variant,
    cleanup_candidates,
)
from experiments.final_graph_simplification_v1.core import (
    SimplificationConfig,
    graph_edges,
    metrics,
    simplify,
)
from experiments.final_graph_simplification_v1.visualize import graph_overlay, navigation_overlay


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "095bf4f1688d7172604ff855a22960173c869eeee20061ed61d92a07540ef8a6"


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def dataclass_from_json(cls, path: Path):
    raw = read_json(path)
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in names})


def original_graph(nodes: list[dict], edges: list[dict], scale: list[float]) -> dict:
    sx, sy = map(float, scale)
    return {
        "coordinate_order": "x,y",
        "coordinate_frame": "original_image_pixels",
        "nodes": [{**node, "x": float(node["x"]) * sx, "y": float(node["y"]) * sy} for node in nodes],
        "edges": edges,
    }


def original_navigation(navigation: dict, scale: list[float]) -> dict:
    sx, sy = map(float, scale)
    result = dict(navigation)
    result["coordinate_order"] = "x,y"
    result["coordinate_frame"] = "original_image_pixels"
    result["nodes"] = [
        {**node, "x": float(node["x"]) * sx, "y": float(node["y"]) * sy}
        for node in navigation["nodes"]
    ]
    result["edges"] = [
        {**edge, "polyline": [[float(x) * sx, float(y) * sy] for x, y in edge["polyline"]]}
        for edge in navigation["edges"]
    ]
    return result


def road_overlay(rgb: np.ndarray, road: np.ndarray, threshold: float) -> np.ndarray:
    mask = road >= threshold
    color = np.zeros_like(rgb)
    color[..., 1] = 235
    color[..., 2] = 235
    output = rgb.astype(np.float32) * 0.62
    output[mask] = output[mask] + color[mask].astype(np.float32) * 0.38
    return np.clip(output, 0, 255).astype(np.uint8)


def probability_heatmap(road: np.ndarray) -> Image.Image:
    grayscale = Image.fromarray(np.clip(np.rint(road * 255), 0, 255).astype(np.uint8))
    return ImageOps.colorize(grayscale, black=(20, 10, 45), mid=(20, 170, 180), white=(255, 235, 60))


def panel(items: list[tuple[Path, str]], destination: Path) -> None:
    cell_w, image_h, label_h = 1024, 705, 38
    canvas = Image.new("RGB", (cell_w * 2, (image_h + label_h) * 2), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (path, label) in enumerate(items):
        image = Image.open(path).convert("RGB").resize((cell_w, image_h), Image.Resampling.LANCZOS)
        x = (index % 2) * cell_w
        y = (index // 2) * (image_h + label_h)
        canvas.paste(image, (x, y + label_h))
        draw.text((x + 12, y + 12), label, fill="white", font=font)
    canvas.save(destination, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()

    directional_path = ROOT / "outputs/directional_topology_v1/config/frozen_directional_candidate_config.json"
    simplification_path = ROOT / "outputs/final_graph_simplification_v1/config/frozen_final_graph_simplification_config.json"
    directional_cfg = dataclass_from_json(DirectionalConfig, directional_path)
    simplification_cfg = dataclass_from_json(SimplificationConfig, simplification_path)
    raw_manifest = read_json(args.raw_root / "run_manifest.json")
    if raw_manifest["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("Raw inference checkpoint identity does not match the frozen model")

    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directional_path, args.output_root / "frozen_directional_candidate_config.json")
    shutil.copy2(simplification_path, args.output_root / "frozen_final_graph_simplification_config.json")
    all_rows = []

    for name in args.names:
        stem = Path(name).stem
        source = args.raw_root / stem
        destination = args.output_root / stem
        destination.mkdir(parents=True, exist_ok=True)
        candidate = read_json(source / "candidate_graph.json")
        diagnostics = read_json(source / "diagnostics.json")
        nodes = candidate["nodes"]
        road = np.load(source / "road_prob.npy")
        rgb = np.asarray(Image.open(source / "rgb.png").convert("RGB"))

        cleanup = cleanup_candidates(nodes, candidate["candidate_edges"], road, directional_cfg)
        c2_candidates, c2_edges = aggregate_variant(
            candidate["candidate_edges"], cleanup["masks"]["C2"], directional_cfg.topo_threshold
        )
        final = simplify(nodes, c2_edges, simplification_cfg)
        s0_edges = graph_edges(final["S0"])

        write_json(destination / "C2_candidate_graph.json", {
            "coordinate_order": "x,y",
            "coordinate_frame": "inference_pixels",
            "nodes": nodes,
            "candidate_edges": c2_candidates,
        })
        write_json(destination / "S0_recommended_graph.json", {
            "coordinate_order": "x,y",
            "coordinate_frame": "inference_pixels",
            "nodes": nodes,
            "edges": s0_edges,
        })
        write_json(destination / "S0_recommended_graph_original_coordinates.json", original_graph(
            nodes, s0_edges, diagnostics["original_per_inference_scale_xy"]
        ))
        write_json(destination / "S3_experimental_navigation_graph.json", {
            **final["S3"], "coordinate_order": "x,y", "coordinate_frame": "inference_pixels"
        })
        write_json(destination / "S3_experimental_navigation_graph_original_coordinates.json", original_navigation(
            final["S3"], diagnostics["original_per_inference_scale_xy"]
        ))
        write_json(destination / "simplification_trace.json", final["trace"])

        Image.fromarray(rgb).save(destination / "rgb_inference.png")
        probability_heatmap(road).save(destination / "road_probability_heatmap.png")
        Image.fromarray(road_overlay(rgb, road, float(diagnostics["road_threshold"]))).save(destination / "road_overlay.png")
        Image.fromarray(graph_overlay(rgb, nodes, s0_edges)).save(destination / "S0_recommended_graph_overlay.png")
        Image.fromarray(navigation_overlay(rgb, final["S3"])).save(destination / "S3_experimental_navigation_overlay.png")
        panel([
            (destination / "rgb_inference.png", "RGB (4096 x 2821 inference frame)"),
            (destination / "road_overlay.png", "Road mask at frozen threshold 0.839"),
            (destination / "S0_recommended_graph_overlay.png", "S0 recommended C2 final graph"),
            (destination / "S3_experimental_navigation_overlay.png", "S3 experimental navigation graph"),
        ], destination / "comparison_panel.jpg")

        s0_metrics = metrics(final["S0"], final["roles"]["S0"])
        s3_graph = nx.Graph()
        s3_graph.add_nodes_from(int(node["id"]) for node in final["S3"]["nodes"])
        s3_graph.add_edges_from((int(edge["source"]), int(edge["target"])) for edge in final["S3"]["edges"])
        row = {
            "image": name,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "raw_candidates": len(candidate["candidate_edges"]),
            "C2_candidates": len(c2_candidates),
            "S0_nodes": final["S0"].number_of_nodes(),
            "S0_edges": final["S0"].number_of_edges(),
            "S0_components": s0_metrics["components"],
            "S0_triangles": s0_metrics["triangle_count"],
            "S3_nodes": s3_graph.number_of_nodes(),
            "S3_edges": s3_graph.number_of_edges(),
            "S3_components": nx.number_connected_components(s3_graph),
            "contracted_nodes": len(final["S3"]["contracted_nodes"]),
            "road_positive_ratio": diagnostics["road_positive_ratio"],
            "inference_time_ms": diagnostics["inference_time_ms"],
            "recommendation": "S0",
        }
        write_json(destination / "result_summary.json", row)
        all_rows.append(row)

    write_json(args.output_root / "run_manifest.json", {
        "status": "complete",
        "model": "WildRoad_baseline",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "candidate_variant": "C2",
        "recommended_graph": "S0",
        "experimental_graph": "S3",
        "raw_inference": raw_manifest,
        "results": all_rows,
    })
    print(json.dumps(all_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
