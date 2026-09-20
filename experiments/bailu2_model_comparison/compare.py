#!/usr/bin/env python3
"""Paired GT comparison of original WildRoad and A_reconstructed on bailu_2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = Path("/Users/badger/Downloads/annotations.xml")
DEFAULT_A_RAW = ROOT / "outputs/latest_full_inference_20260917/raw/bailu_2"
DEFAULT_A_FINAL = ROOT / "outputs/latest_full_inference_20260917/final/bailu_2"
DEFAULT_W_RAW = ROOT / "outputs/bailu2_wildroad_vs_a/raw/bailu_2"
DEFAULT_W_FINAL = ROOT / "outputs/bailu2_wildroad_vs_a/final/bailu_2"
DEFAULT_OUTPUT = ROOT / "outputs/bailu2_wildroad_vs_a/comparison"
TOLERANCE_ORIGINAL = 16.0
SEED = 20260917
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.839, 0.9)


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
        count = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / spacing_inference)))
        values = np.linspace(0.0, 1.0, count + 1, endpoint=True)
        if index:
            values = values[1:]
        pieces.append(start[None, :] + values[:, None] * (end - start)[None, :])
    return np.concatenate(pieces)


def q(values: np.ndarray) -> dict[str, float]:
    result = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {name: float(value) for name, value in zip(("p10", "p25", "p50", "p75", "p90"), result)}


def ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def edge_mask(shape: tuple[int, int], nodes_value: list[dict], edges: list[dict]) -> np.ndarray:
    image = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    nodes = {int(node["id"]): (float(node["x"]), float(node["y"])) for node in nodes_value}
    for edge in edges:
        draw.line([nodes[int(edge["source"])], nodes[int(edge["target"])]], fill=1, width=1)
    return np.asarray(image, dtype=bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def graph_recall(graph_path: Path, shape: tuple[int, int], xy: np.ndarray, tolerance: float) -> float:
    graph = read_json(graph_path)
    mask = edge_mask(shape, graph["nodes"], graph["edges"])
    distances = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return ratio(distances[xy[:, 1], xy[:, 0]] <= tolerance)


def local_max(road: np.ndarray, xy: np.ndarray, tolerance: float) -> np.ndarray:
    radius = int(round(tolerance))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx * xx + yy * yy <= tolerance * tolerance
    values = cv2.dilate(road, disk.astype(np.uint8), borderType=cv2.BORDER_REPLICATE)
    return values[xy[:, 1], xy[:, 0]]


def clustered_bootstrap(values_a: np.ndarray, values_w: np.ndarray, line_ids: np.ndarray, threshold: float) -> dict:
    rng = np.random.default_rng(SEED + int(threshold * 1000))
    unique = np.unique(line_ids)
    deltas = np.empty(10000, dtype=np.float64)
    for index in range(len(deltas)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        masks = [line_ids == line_id for line_id in selected]
        a_hits = sum(int(np.count_nonzero(values_a[mask] >= threshold)) for mask in masks)
        w_hits = sum(int(np.count_nonzero(values_w[mask] >= threshold)) for mask in masks)
        count = sum(int(mask.sum()) for mask in masks)
        deltas[index] = (a_hits - w_hits) / count
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"delta_A_minus_WildRoad": float(np.mean(values_a >= threshold) - np.mean(values_w >= threshold)),
            "cluster_bootstrap_95pct_ci": [float(low), float(high)], "bootstrap_replicates": len(deltas)}


def save_delta_overlay(rgb: np.ndarray, samples: np.ndarray, a: np.ndarray, w: np.ndarray, destination: Path) -> None:
    base = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    delta = a - w
    categories = np.full(len(delta), "stable", dtype="<U8")
    categories[delta <= -0.2] = "loss"
    categories[delta >= 0.2] = "gain"
    colors = {"loss": (235, 45, 45, 220), "stable": (220, 200, 50, 190), "gain": (35, 210, 100, 220)}
    for category in ("loss", "stable", "gain"):
        for x, y in samples[categories == category]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[category])
    output = Image.alpha_composite(base, layer).convert("RGB")
    legend = ImageDraw.Draw(output)
    legend.rectangle((12, 12, 300, 91), fill="black")
    for i, (label, category) in enumerate((("A - WildRoad <= -0.2", "loss"), ("|delta| < 0.2", "stable"), ("A - WildRoad >= +0.2", "gain"))):
        y = 22 + 23 * i
        legend.rectangle((22, y, 34, y + 12), fill=colors[category][:3])
        legend.text((43, y), label, fill="white")
    output.save(destination)


def save_curve(rows: list[dict], destination: Path) -> None:
    width, height = 1100, 700
    x0, y0, x1, y1 = 90, 50, 1060, 610
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for value in np.linspace(0, 1, 6):
        y = y1 - value * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
        draw.text((35, y - 7), f"{value:.1f}", fill=(30, 30, 30), font=font)
    draw.line((x0, y1, x1, y1), fill=(30, 30, 30), width=2)
    draw.line((x0, y0, x0, y1), fill=(30, 30, 30), width=2)
    for model, color in (("WildRoad_baseline", (40, 115, 220)), ("A_reconstructed", (225, 70, 55))):
        subset = [row for row in rows if row["model"] == model]
        points = [(x0 + row["threshold"] * (x1 - x0), y1 - row["recall"] * (y1 - y0)) for row in subset]
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
    for threshold in sorted({row["threshold"] for row in rows}):
        x = x0 + threshold * (x1 - x0)
        draw.text((x - 10, y1 + 12), f"{threshold:g}", fill=(30, 30, 30), font=font)
    draw.text((360, 20), "bailu_2 path-GT recall within 16 original pixels", fill=(20, 20, 20), font=font)
    draw.text((420, 660), "Road probability threshold", fill=(20, 20, 20), font=font)
    draw.line((815, 70, 855, 70), fill=(40, 115, 220), width=4); draw.text((865, 63), "WildRoad", fill=(20, 20, 20), font=font)
    draw.line((815, 93, 855, 93), fill=(225, 70, 55), width=4); draw.text((865, 86), "A_reconstructed", fill=(20, 20, 20), font=font)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--a-raw", type=Path, default=DEFAULT_A_RAW)
    parser.add_argument("--a-final", type=Path, default=DEFAULT_A_FINAL)
    parser.add_argument("--wildroad-raw", type=Path, default=DEFAULT_W_RAW)
    parser.add_argument("--wildroad-final", type=Path, default=DEFAULT_W_FINAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a_diag = read_json(args.a_raw / "diagnostics.json")
    w_diag = read_json(args.wildroad_raw / "diagnostics.json")
    for key in ("input_sha256", "original_size_wh", "inference_size_wh", "original_per_inference_scale_xy",
                "road_threshold", "keypoint_threshold", "topo_threshold"):
        if a_diag[key] != w_diag[key]:
            raise RuntimeError(f"Uncontrolled comparison: {key} differs: {a_diag[key]} vs {w_diag[key]}")

    road_a = np.load(args.a_raw / "road_prob.npy")
    road_w = np.load(args.wildroad_raw / "road_prob.npy")
    if road_a.shape != road_w.shape:
        raise RuntimeError("Probability-map shapes differ")
    rgb = np.asarray(Image.open(args.a_raw / "rgb.png").convert("RGB"))
    lines, original_size = load_polylines(args.annotations, "bailu_2.png")
    if list(original_size) != a_diag["original_size_wh"]:
        raise RuntimeError("GT and inference dimensions differ")
    scale = np.asarray(a_diag["original_per_inference_scale_xy"], dtype=np.float64)
    sampled_lines = [sample_polyline(line, scale) for line in lines]
    samples = np.concatenate(sampled_lines)
    line_ids = np.concatenate([np.full(len(points), i + 1, dtype=np.int32) for i, points in enumerate(sampled_lines)])
    xy = np.rint(samples).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, road_a.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, road_a.shape[0] - 1)
    tolerance = TOLERANCE_ORIGINAL / float(scale.mean())
    local_a = local_max(road_a, xy, tolerance)
    local_w = local_max(road_w, xy, tolerance)

    curve = []
    for threshold in THRESHOLDS:
        curve.extend((
            {"model": "WildRoad_baseline", "threshold": threshold, "recall": ratio(local_w >= threshold)},
            {"model": "A_reconstructed", "threshold": threshold, "recall": ratio(local_a >= threshold)},
        ))

    per_line = []
    for line_id in np.unique(line_ids):
        mask = line_ids == line_id
        per_line.append({
            "polyline_id": int(line_id), "samples": int(mask.sum()),
            "WildRoad_local_probability_mean": float(local_w[mask].mean()),
            "A_local_probability_mean": float(local_a[mask].mean()),
            "mean_probability_delta_A_minus_WildRoad": float(np.mean(local_a[mask] - local_w[mask])),
            "WildRoad_recall_at_0.3": ratio(local_w[mask] >= 0.3),
            "A_recall_at_0.3": ratio(local_a[mask] >= 0.3),
            "recall_delta_at_0.3_A_minus_WildRoad": ratio(local_a[mask] >= 0.3) - ratio(local_w[mask] >= 0.3),
            "WildRoad_recall_at_0.839": ratio(local_w[mask] >= 0.839),
            "A_recall_at_0.839": ratio(local_a[mask] >= 0.839),
            "recall_delta_at_0.839_A_minus_WildRoad": ratio(local_a[mask] >= 0.839) - ratio(local_w[mask] >= 0.839),
        })

    transitions = {}
    for threshold in (0.3, 0.839):
        wh = local_w >= threshold
        ah = local_a >= threshold
        transitions[str(threshold)] = {
            "both_recalled": ratio(wh & ah), "WildRoad_only_lost_after_adaptation": ratio(wh & ~ah),
            "A_only_gained_after_adaptation": ratio(~wh & ah), "neither_recalled": ratio(~wh & ~ah),
            "net_delta_A_minus_WildRoad": ratio(ah) - ratio(wh),
        }

    graph = {
        "WildRoad_raw_accepted_recall": graph_recall(args.wildroad_raw / "final_graph.json", road_w.shape, xy, tolerance),
        "A_raw_accepted_recall": graph_recall(args.a_raw / "final_graph.json", road_a.shape, xy, tolerance),
        "WildRoad_S0_recall": graph_recall(args.wildroad_final / "S0_recommended_graph.json", road_w.shape, xy, tolerance),
        "A_S0_recall": graph_recall(args.a_final / "S0_recommended_graph.json", road_a.shape, xy, tolerance),
    }
    graph["S0_delta_A_minus_WildRoad"] = graph["A_S0_recall"] - graph["WildRoad_S0_recall"]

    metrics = {
        "scope": {"polylines": len(lines), "samples": len(samples), "tolerance_original_px": TOLERANCE_ORIGINAL,
                  "sample_spacing_original_px": float(scale.mean()), "same_input_sha256": a_diag["input_sha256"],
                  "frozen_thresholds": {"road": a_diag["road_threshold"], "keypoint": a_diag["keypoint_threshold"], "topology": a_diag["topo_threshold"]}},
        "identity": {
            "WildRoad_baseline_checkpoint_sha256": w_diag["checkpoint_sha256"],
            "A_reconstructed_checkpoint_sha256": a_diag["checkpoint_sha256"],
            "input_hashes": {"annotations": sha256(args.annotations), "WildRoad_probability": sha256(args.wildroad_raw / "road_prob.npy"),
                             "A_probability": sha256(args.a_raw / "road_prob.npy")}},
        "probability": {
            "WildRoad_local_quantiles": q(local_w), "A_local_quantiles": q(local_a),
            "delta_A_minus_WildRoad_quantiles": q(local_a - local_w),
            "mean_delta_A_minus_WildRoad": float(np.mean(local_a - local_w)),
            "share_delta_le_minus_0.2": ratio(local_a - local_w <= -0.2),
            "share_delta_ge_plus_0.2": ratio(local_a - local_w >= 0.2),
        },
        "threshold_transitions": transitions,
        "cluster_bootstrap": {str(t): clustered_bootstrap(local_a, local_w, line_ids, t) for t in (0.3, 0.839)},
        "graph": graph,
        "per_polyline": per_line,
        "threshold_curve": curve,
        "limitations": [
            "The CVAT annotation has one path label and does not distinguish small paths from major roads.",
            "This is one image; it can establish bailu_2 behavior but not population-wide forgetting.",
            "Centerline recall uses a 16-original-pixel tolerance and does not measure false positives or precision.",
        ],
    }
    write_json(args.output_dir / "metrics.json", metrics)
    with (args.output_dir / "per_polyline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_line[0].keys()); writer.writeheader(); writer.writerows(per_line)
    with (args.output_dir / "threshold_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys()); writer.writeheader(); writer.writerows(curve)
    save_curve(curve, args.output_dir / "threshold_recall_comparison.png")
    save_delta_overlay(rgb, samples, local_a, local_w, args.output_dir / "gt_probability_delta_overlay.png")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
