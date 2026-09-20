#!/usr/bin/env python3
"""Diagnose road-probability, threshold, and graph recall on CVAT polylines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, maximum_filter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = Path("/Users/badger/Downloads/annotations.xml")
DEFAULT_RAW = ROOT / "outputs/latest_full_inference_20260917/raw/bailu_2"
DEFAULT_FINAL = ROOT / "outputs/latest_full_inference_20260917/final/bailu_2"
DEFAULT_OUTPUT = ROOT / "outputs/bailu2_gt_small_path_diagnosis"
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.839, 0.9)
TOLERANCES_ORIGINAL = (8.0, 16.0, 32.0)
WEAK_THRESHOLD = 0.3
FROZEN_THRESHOLD = 0.839


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_polylines(path: Path, image_name: str) -> tuple[list[np.ndarray], tuple[int, int]]:
    root = ET.parse(path).getroot()
    image = next((node for node in root.findall("image") if node.get("name") == image_name), None)
    if image is None:
        raise ValueError(f"No image named {image_name!r} in {path}")
    lines = []
    for node in image.findall("polyline"):
        if node.get("label") != "path":
            continue
        points = [[float(value) for value in pair.split(",")] for pair in node.get("points", "").split(";")]
        if len(points) >= 2:
            lines.append(np.asarray(points, dtype=np.float64))
    return lines, (int(image.get("width", "0")), int(image.get("height", "0")))


def sample_polyline(points: np.ndarray, scale_xy: np.ndarray, spacing_inference: float = 1.0) -> np.ndarray:
    points = points / scale_xy
    pieces = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / spacing_inference)))
        values = np.linspace(0.0, 1.0, count + 1, endpoint=True)
        if index:
            values = values[1:]
        pieces.append(start[None, :] + values[:, None] * (end - start)[None, :])
    return np.concatenate(pieces)


def edge_mask(shape: tuple[int, int], nodes_value: list[dict], edges: list[dict]) -> np.ndarray:
    image = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    nodes = {int(node["id"]): (float(node["x"]), float(node["y"])) for node in nodes_value}
    for edge in edges:
        draw.line([nodes[int(edge["source"])], nodes[int(edge["target"])]], fill=1, width=1)
    return np.asarray(image, dtype=bool)


def sampled_edge_distance(shape: tuple[int, int], nodes: list[dict], edges: list[dict], xy: np.ndarray) -> np.ndarray:
    distance = distance_transform_edt(~edge_mask(shape, nodes, edges))
    return distance[xy[:, 1], xy[:, 0]]


def q(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {name: float(value) for name, value in zip(("p10", "p25", "p50", "p75", "p90"), result)}


def ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def save_curve(rows: list[dict], destination: Path) -> None:
    width, height = 1120, 700
    left, right, top, bottom = 90, 35, 45, 85
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = left, top, width - right, height - bottom
    draw.line((x0, y1, x1, y1), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, x0, y1), fill=(40, 40, 40), width=2)
    for value in np.linspace(0, 1, 6):
        y = y1 - value * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
        draw.text((20, y - 7), f"{value:.1f}", fill=(40, 40, 40), font=font)
    colors = {8.0: (220, 60, 60), 16.0: (25, 130, 210), 32.0: (40, 160, 80)}
    thresholds = [row["threshold"] for row in rows if row["tolerance_original_px"] == 8.0]
    for tolerance, color in colors.items():
        subset = [row for row in rows if row["tolerance_original_px"] == tolerance]
        points = []
        for row in subset:
            x = x0 + row["threshold"] * (x1 - x0)
            y = y1 - row["gt_recall"] * (y1 - y0)
            points.append((x, y))
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
    for threshold in thresholds:
        x = x0 + threshold * (x1 - x0)
        draw.text((x - 12, y1 + 12), f"{threshold:g}", fill=(40, 40, 40), font=font)
    frozen_x = x0 + FROZEN_THRESHOLD * (x1 - x0)
    draw.line((frozen_x, y0, frozen_x, y1), fill=(120, 40, 140), width=2)
    draw.text((frozen_x - 30, y0 - 22), "frozen 0.839", fill=(120, 40, 140), font=font)
    draw.text((width // 2 - 80, height - 35), "Road probability threshold", fill=(20, 20, 20), font=font)
    draw.text((20, 15), "GT centerline recall within spatial tolerance", fill=(20, 20, 20), font=font)
    for index, (tolerance, color) in enumerate(colors.items()):
        y = 52 + index * 22
        draw.line((width - 190, y, width - 160, y), fill=color, width=4)
        draw.text((width - 150, y - 7), f"{tolerance:g}px original", fill=(20, 20, 20), font=font)
    canvas.save(destination)


def save_overlay(
    rgb: np.ndarray,
    samples: np.ndarray,
    local_probability: np.ndarray,
    graph_hit: np.ndarray,
    destination: Path,
) -> None:
    image = Image.fromarray(rgb).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Graph recovery takes precedence; remaining misses are attributed to the latest available signal.
    categories = np.full(len(samples), "A", dtype="<U1")
    categories[(local_probability >= WEAK_THRESHOLD) & (local_probability < FROZEN_THRESHOLD)] = "B"
    categories[local_probability >= FROZEN_THRESHOLD] = "C"
    categories[graph_hit] = "D"
    colors = {"A": (230, 40, 40, 220), "B": (255, 165, 0, 220), "C": (30, 190, 230, 220), "D": (30, 210, 90, 220)}
    for category in ("A", "B", "C", "D"):
        for x, y in samples[categories == category]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[category])
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    legend = ImageDraw.Draw(image)
    labels = [
        ("A p<0.3, graph miss", colors["A"]),
        ("B 0.3<=p<0.839, graph miss", colors["B"]),
        ("C p>=0.839, graph miss", colors["C"]),
        ("Graph recalled", colors["D"]),
    ]
    legend.rectangle((12, 12, 245, 112), fill=(0, 0, 0))
    for index, (label, color) in enumerate(labels):
        y = 23 + index * 22
        legend.rectangle((22, y, 34, y + 12), fill=color[:3])
        legend.text((43, y), label, fill="white")
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_XML)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    road = np.load(args.raw_dir / "road_prob.npy")
    rgb = np.asarray(Image.open(args.raw_dir / "rgb.png").convert("RGB"))
    graph = json.loads((args.final_dir / "S0_recommended_graph.json").read_text())
    c2_graph = json.loads((args.final_dir / "C2_candidate_graph.json").read_text())
    raw_candidate = json.loads((args.raw_dir / "candidate_graph.json").read_text())
    raw_final = json.loads((args.raw_dir / "final_graph.json").read_text())
    diagnostics = json.loads((args.raw_dir / "diagnostics.json").read_text())
    lines, original_size = load_polylines(args.annotations, "bailu_2.png")
    scale_xy = np.asarray(diagnostics["original_per_inference_scale_xy"], dtype=np.float64)
    if tuple(road.shape) != (rgb.shape[0], rgb.shape[1]):
        raise ValueError("Probability map and RGB dimensions differ")
    if original_size != tuple(diagnostics["original_size_wh"]):
        raise ValueError("CVAT image dimensions do not match inference metadata")

    sampled_lines = [sample_polyline(points, scale_xy) for points in lines]
    samples = np.concatenate(sampled_lines)
    line_ids = np.concatenate([np.full(len(points), index + 1, dtype=np.int32) for index, points in enumerate(sampled_lines)])
    xy = np.rint(samples).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, road.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, road.shape[0] - 1)

    graph_distance_samples = sampled_edge_distance(road.shape, graph["nodes"], graph["edges"], xy)
    curve_rows = []
    threshold_distances = {}
    for threshold in THRESHOLDS:
        distance = distance_transform_edt(road < threshold)
        threshold_distances[threshold] = distance[xy[:, 1], xy[:, 0]]
        for tolerance_original in TOLERANCES_ORIGINAL:
            tolerance_inference = tolerance_original / float(scale_xy.mean())
            curve_rows.append({
                "threshold": threshold,
                "tolerance_original_px": tolerance_original,
                "gt_recall": ratio(threshold_distances[threshold] <= tolerance_inference),
            })

    main_tolerance_original = 16.0
    main_tolerance = main_tolerance_original / float(scale_xy.mean())
    radius = int(round(main_tolerance))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx * xx + yy * yy <= main_tolerance * main_tolerance
    local_probability_map = maximum_filter(road, footprint=disk, mode="nearest")
    local_probability = local_probability_map[xy[:, 1], xy[:, 0]]
    del local_probability_map
    graph_hit = graph_distance_samples <= main_tolerance
    weak_hit = local_probability >= WEAK_THRESHOLD
    strong_hit = local_probability >= FROZEN_THRESHOLD

    categories = {
        "A_perception_below_0.3_and_graph_missed": (~graph_hit) & (~weak_hit),
        "B_weak_0.3_to_0.839_and_graph_missed": (~graph_hit) & weak_hit & (~strong_hit),
        "C_at_or_above_0.839_but_graph_missed": (~graph_hit) & strong_hit,
        "D_graph_recalled": graph_hit,
    }
    category_counts = {name: int(mask.sum()) for name, mask in categories.items()}
    category_ratios = {name: ratio(mask) for name, mask in categories.items()}
    missed = ~graph_hit
    missed_attribution = {name: ratio(mask[missed]) for name, mask in categories.items() if name != "D_graph_recalled"}

    graph_recall_by_tolerance = {}
    for tolerance_original in TOLERANCES_ORIGINAL:
        tolerance = tolerance_original / float(scale_xy.mean())
        graph_recall_by_tolerance[str(int(tolerance_original))] = ratio(graph_distance_samples <= tolerance)

    graph_stage_recall = {}
    graph_variants = (
        ("raw_candidates_upper_bound", raw_candidate["nodes"], raw_candidate["candidate_edges"]),
        ("C2_candidates_upper_bound", c2_graph["nodes"], c2_graph["candidate_edges"]),
        ("raw_accepted_graph", raw_final["nodes"], raw_final["edges"]),
        ("S0_recommended_graph", graph["nodes"], graph["edges"]),
    )
    for name, nodes_value, edges in graph_variants:
        distances = sampled_edge_distance(road.shape, nodes_value, edges, xy)
        graph_stage_recall[name] = {
            "edges": len(edges),
            "recall_at_16_original_px": ratio(distances <= main_tolerance),
        }

    stage_outcomes_by_tolerance = {}
    for tolerance_original in TOLERANCES_ORIGINAL:
        tolerance = tolerance_original / float(scale_xy.mean())
        graph_at_tolerance = graph_distance_samples <= tolerance
        weak_at_tolerance = threshold_distances[WEAK_THRESHOLD] <= tolerance
        strong_at_tolerance = threshold_distances[FROZEN_THRESHOLD] <= tolerance
        graph_missed = ~graph_at_tolerance
        stage_outcomes_by_tolerance[str(int(tolerance_original))] = {
            "A_perception_below_0.3_and_graph_missed": ratio(graph_missed & ~weak_at_tolerance),
            "B_weak_0.3_to_0.839_and_graph_missed": ratio(graph_missed & weak_at_tolerance & ~strong_at_tolerance),
            "C_at_or_above_0.839_but_graph_missed": ratio(graph_missed & strong_at_tolerance),
            "D_graph_recalled": ratio(graph_at_tolerance),
        }

    per_line = []
    for line_id in range(1, len(lines) + 1):
        mask = line_ids == line_id
        line_graph = graph_hit[mask]
        line_weak = weak_hit[mask]
        line_strong = strong_hit[mask]
        line_missed = ~line_graph
        per_line.append({
            "polyline_id": line_id,
            "vertices": len(lines[line_id - 1]),
            "length_original_px": float(np.linalg.norm(np.diff(lines[line_id - 1], axis=0), axis=1).sum()),
            "samples": int(mask.sum()),
            "local_probability_mean": float(local_probability[mask].mean()),
            "local_probability_median": float(np.median(local_probability[mask])),
            "recall_at_0.3": ratio(line_weak),
            "recall_at_0.839": ratio(line_strong),
            "graph_recall": ratio(line_graph),
            "A_share_of_line": ratio(line_missed & ~line_weak),
            "B_share_of_line": ratio(line_missed & line_weak & ~line_strong),
            "C_share_of_line": ratio(line_missed & line_strong),
        })

    summary = {
        "scope": {
            "image": "bailu_2.png",
            "annotation_label": "path",
            "annotation_polylines": len(lines),
            "annotation_vertices": int(sum(len(line) for line in lines)),
            "sample_spacing_original_px": float(scale_xy.mean()),
            "sample_count": len(samples),
            "annotation_length_original_px": float(sum(row["length_original_px"] for row in per_line)),
            "inference_size_wh": [road.shape[1], road.shape[0]],
            "main_tolerance_original_px": main_tolerance_original,
            "main_tolerance_inference_px": main_tolerance,
            "checkpoint_sha256": diagnostics["checkpoint_sha256"],
            "frozen_road_threshold": diagnostics["road_threshold"],
            "input_sha256": {
                "annotations_xml": sha256(args.annotations),
                "road_prob_npy": sha256(args.raw_dir / "road_prob.npy"),
                "S0_graph_json": sha256(args.final_dir / "S0_recommended_graph.json"),
            },
        },
        "probability": {
            "centerline_exact_quantiles": q(road[xy[:, 1], xy[:, 0]]),
            "local_max_within_16_original_px_quantiles": q(local_probability),
            "recall_at_0.3_within_16_original_px": ratio(weak_hit),
            "recall_at_0.839_within_16_original_px": ratio(strong_hit),
            "threshold_recall_gain_0.839_to_0.3": ratio(weak_hit) - ratio(strong_hit),
        },
        "graph": {
            "S0_nodes": len(graph["nodes"]),
            "S0_edges": len(graph["edges"]),
            "recall_by_original_pixel_tolerance": graph_recall_by_tolerance,
            "strong_supported_but_graph_missed": ratio(strong_hit & ~graph_hit),
            "graph_recalled_without_0.839_support": ratio(graph_hit & ~strong_hit),
            "stage_recall_at_16_original_px": graph_stage_recall,
        },
        "exclusive_final_outcome_at_16_original_px": {
            "counts": category_counts,
            "shares_of_all_gt_samples": category_ratios,
            "shares_of_graph_missed_samples": missed_attribution,
        },
        "threshold_curve": curve_rows,
        "stage_outcomes_by_tolerance": stage_outcomes_by_tolerance,
        "per_polyline": per_line,
        "limitations": [
            "CVAT uses one path label and has no major-road/small-path subtype.",
            "Polyline centerlines are evaluated with spatial tolerance; they are not polygon masks.",
            "S0 edges are evaluated as straight node-to-node segments in inference coordinates.",
            "There is no keypoint GT, so keypoint selection and edge inference cannot be separated rigorously.",
            "Candidate-edge coverage is an upper bound, not a valid final graph precision result.",
        ],
    }
    write_json(args.output_dir / "metrics.json", summary)
    with (args.output_dir / "threshold_recall.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)
    with (args.output_dir / "per_polyline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_line[0].keys())
        writer.writeheader()
        writer.writerows(per_line)
    save_curve(curve_rows, args.output_dir / "threshold_recall_curve.png")
    save_overlay(rgb, samples, local_probability, graph_hit, args.output_dir / "gt_stage_diagnosis_overlay.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
