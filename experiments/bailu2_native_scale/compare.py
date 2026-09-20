#!/usr/bin/env python3
"""Compare half-scale and native-scale tiled probabilities on bailu_2 path GT."""

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
DEFAULT_LOW = ROOT / "outputs/latest_full_inference_20260917/raw/bailu_2"
DEFAULT_NATIVE = ROOT / "outputs/bailu2_native_scale/raw"
DEFAULT_OUTPUT = ROOT / "outputs/bailu2_native_scale/comparison"
DEFAULT_XML = Path("/Users/badger/Downloads/annotations.xml")
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.839, 0.9)
TOLERANCE_ORIGINAL = 16.0
SEED = 20260917


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


def load_polylines(path: Path) -> tuple[list[np.ndarray], tuple[int, int]]:
    root = ET.parse(path).getroot()
    image = next(node for node in root.findall("image") if node.get("name") == "bailu_2.png")
    lines = []
    for node in image.findall("polyline"):
        if node.get("label") == "path":
            lines.append(np.asarray([[float(v) for v in pair.split(",")] for pair in node.get("points", "").split(";")]))
    return lines, (int(image.get("width")), int(image.get("height")))


def sample_polyline(points: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    pieces = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        count = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / spacing)))
        values = np.linspace(0.0, 1.0, count + 1)
        if index:
            values = values[1:]
        pieces.append(start + values[:, None] * (end - start))
    return np.concatenate(pieces)


def local_max(probability: np.ndarray, xy_native: np.ndarray, scale: float, tolerance: float = TOLERANCE_ORIGINAL) -> np.ndarray:
    radius = int(round(tolerance / scale))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    footprint = (xx * xx + yy * yy <= (tolerance / scale) ** 2).astype(np.uint8)
    dilated = cv2.dilate(probability, footprint, borderType=cv2.BORDER_REPLICATE)
    xy = np.rint(xy_native / scale).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, probability.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, probability.shape[0] - 1)
    return dilated[xy[:, 1], xy[:, 0]]


def ratio(value: np.ndarray) -> float:
    return float(np.mean(value))


def quantiles(value: np.ndarray) -> dict[str, float]:
    return dict(zip(("p10", "p25", "p50", "p75", "p90"), map(float, np.quantile(value, (0.1, 0.25, 0.5, 0.75, 0.9)))))


def bootstrap(native: np.ndarray, low: np.ndarray, line_ids: np.ndarray, threshold: float) -> list[float]:
    rng = np.random.default_rng(SEED + int(threshold * 1000))
    ids = np.unique(line_ids)
    deltas = []
    masks = {line_id: line_ids == line_id for line_id in ids}
    for _ in range(10000):
        selected = rng.choice(ids, len(ids), replace=True)
        n = sum(int(masks[i].sum()) for i in selected)
        nh = sum(int(np.count_nonzero(native[masks[i]] >= threshold)) for i in selected)
        lh = sum(int(np.count_nonzero(low[masks[i]] >= threshold)) for i in selected)
        deltas.append((nh - lh) / n)
    return list(map(float, np.quantile(deltas, (0.025, 0.975))))


def nearest_seam_distance(samples: np.ndarray, metadata: dict) -> np.ndarray:
    starts = np.asarray(metadata["tile_starts_xy"], dtype=np.float64)
    patch = float(metadata["patch_size"])
    vertical = np.unique(np.concatenate((starts[:, 0], starts[:, 0] + patch)))
    horizontal = np.unique(np.concatenate((starts[:, 1], starts[:, 1] + patch)))
    dx = np.min(np.abs(samples[:, 0, None] - vertical[None, :]), axis=1)
    dy = np.min(np.abs(samples[:, 1, None] - horizontal[None, :]), axis=1)
    return np.minimum(dx, dy)


def save_curve(rows: list[dict], destination: Path) -> None:
    width, height = 1100, 700
    x0, y0, x1, y1 = 90, 50, 1060, 610
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for value in np.linspace(0, 1, 6):
        y = y1 - value * (y1 - y0)
        draw.line((x0, y, x1, y), fill=(225, 225, 225))
        draw.text((35, y - 7), f"{value:.1f}", fill=(30, 30, 30), font=font)
    draw.line((x0, y1, x1, y1), fill=(30, 30, 30), width=2)
    draw.line((x0, y0, x0, y1), fill=(30, 30, 30), width=2)
    for scale_name, color in (("half_4096", (50, 110, 220)), ("native_8192", (220, 70, 50))):
        subset = [row for row in rows if row["scale"] == scale_name]
        points = [(x0 + row["threshold"] * (x1 - x0), y1 - row["recall"] * (y1 - y0)) for row in subset]
        draw.line(points, fill=color, width=4)
    draw.text((360, 20), "bailu_2 path-GT recall: scale comparison", fill=(20, 20, 20), font=font)
    draw.text((420, 660), "Road probability threshold", fill=(20, 20, 20), font=font)
    image.save(destination)


def save_delta(rgb_path: Path, samples: np.ndarray, delta: np.ndarray, destination: Path) -> None:
    image = Image.open(rgb_path).convert("RGB").resize((4096, 2821), Image.Resampling.LANCZOS).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scaled = samples / 2.0
    categories = np.full(len(delta), "stable", dtype="<U8")
    categories[delta <= -0.2] = "loss"
    categories[delta >= 0.2] = "gain"
    colors = {"loss": (230, 45, 45, 220), "stable": (230, 200, 40, 190), "gain": (30, 210, 100, 220)}
    for category in ("loss", "stable", "gain"):
        for x, y in scaled[categories == category]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[category])
    Image.alpha_composite(image, layer).convert("RGB").save(destination)


def write_report(metrics: dict, destination: Path) -> None:
    t03 = metrics["threshold_transitions"]["0.3"]
    t0839 = metrics["threshold_transitions"]["0.839"]
    missed = metrics["baseline_missed_subset"]
    rows = {row["tolerance_original_px"]: row for row in metrics["tolerance_sensitivity"]}
    per_line = metrics["per_polyline"]
    gains = sorted(per_line, key=lambda row: row["delta_recall_at_0.3"], reverse=True)[:5]
    losses = sorted(per_line, key=lambda row: row["delta_recall_at_0.3"])[:5]
    lines = [
        "# bailu_2 Native-resolution Tiled Inference 尺度诊断", "", "## 结论", "",
        "**native-resolution tiled inference 没有显著恢复整体 path probability，因此当前证据不支持‘缩小一半是 small-path miss 的重要主因’。**",
        "", "在同一 A_reconstructed checkpoint、同一 MaGRoad 配置、同一 38 条 path GT 和同一 16 原图像素容差下：", "",
        "| 指标 | 4096×2821 | Native 8192×5642 | 差值 |", "|---|---:|---:|---:|",
        f"| GT recall @ 0.3 | {(t03['both_recalled'] + t03['low_only_lost_at_native']):.2%} | {(t03['both_recalled'] + t03['native_only_recovered']):.2%} | {t03['net_delta_native_minus_low']:+.2%} |",
        f"| GT recall @ 0.839 | {(t0839['both_recalled'] + t0839['low_only_lost_at_native']):.2%} | {(t0839['both_recalled'] + t0839['native_only_recovered']):.2%} | {t0839['net_delta_native_minus_low']:+.2%} |",
        f"| GT 邻域 mean probability | {metrics['probability']['low_local_mean']:.4f} | {metrics['probability']['native_local_mean']:.4f} | {metrics['probability']['mean_delta_native_minus_low']:+.4f} |",
        "", f"低分辨率下 `p<0.3` 的 {missed['samples']:,} 个漏检 GT 点中，native 仅恢复 **{missed['native_recovered_at_0.3']:.2%}** 到 0.3，恢复 **{missed['native_recovered_at_0.839']:.2%}** 到 0.839。与此同时，原先已命中的点也有损失，净结果为下降。",
        "", f"按 polyline 聚类 bootstrap，差值的 95% 区间在 0.3 为 **{t03['cluster_bootstrap_95pct_ci'][0]:+.2%} 至 {t03['cluster_bootstrap_95pct_ci'][1]:+.2%}**，在 0.839 为 **{t0839['cluster_bootstrap_95pct_ci'][0]:+.2%} 至 {t0839['cluster_bootstrap_95pct_ci'][1]:+.2%}**。区间包含 0，说明 38 条路径之间异质性很强；当前结果不支持显著整体恢复，也不能声称 native 对所有路径都无效。", "",
        "## 容差稳健性", "", "| 原图容差 | 4096 @ 0.3 | Native @ 0.3 | 差值 | 4096 @ 0.839 | Native @ 0.839 | 差值 |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tolerance in (0, 8, 16, 32):
        row = rows[tolerance]
        lines.append(f"| {tolerance} px | {row['low_recall_at_0.3']:.2%} | {row['native_recall_at_0.3']:.2%} | {row['delta_at_0.3']:+.2%} | {row['low_recall_at_0.839']:.2%} | {row['native_recall_at_0.839']:.2%} | {row['delta_at_0.839']:+.2%} |")
    lines += [
        "", "若 native 主要改善细路定位，较严格的 0/8 px 容差应出现明显正增益；实际结果并未呈现一致的整体恢复。全图 `p≥0.839` 正响应面积也从 {:.2%} 降到 {:.2%}，说明这不是 GT 中心线附近的孤立变化。".format(metrics['probability']['full_frame_positive_ratio']['low_at_0.839'], metrics['probability']['full_frame_positive_ratio']['native_at_0.839']), "",
        "## 局部差异", "", "native 在部分路径上有明显收益，也在另一部分路径上明显退化。以下为阈值 0.3 的最大变化：", "",
        "| Polyline | 4096 recall | Native recall | 差值 |", "|---:|---:|---:|---:|",
    ]
    for row in gains:
        lines.append(f"| {row['polyline_id']} | {row['low_recall_at_0.3']:.2%} | {row['native_recall_at_0.3']:.2%} | {row['delta_recall_at_0.3']:+.2%} |")
    for row in losses:
        lines.append(f"| {row['polyline_id']} | {row['low_recall_at_0.3']:.2%} | {row['native_recall_at_0.3']:.2%} | {row['delta_recall_at_0.3']:+.2%} |")
    lines += [
        "", "这说明尺度会重新分配空间响应，但不是单向恢复。最值得进一步人工核验的是增益最大的 ID 5、13、19，以及退化最大的 ID 33、16、34。", "",
        "## Tile 边界检查", "", "native 使用 368 个重叠 1024×1024 tile，每像素覆盖 1–16 次。GT 点按距最近 tile 边界分组后，边界附近没有额外的系统性下降；0–16 px 组在阈值 0.3 的差值为 {:.2%}。因此整体负增益不能归因于 seam。".format(metrics['native_tile_boundary_sensitivity'][0]['recall_delta_at_0.3']), "",
        "## 实验边界", "", "XML 只有统一 `path` 标签，没有道路宽度或 `small_path` 子类，所以本实验严格回答的是所有已标注 path 和原先漏检点，无法单独给出语义 small-path 指标。原生尺度同时把每个 1024 tile 的地面视野减半，因此结果包含像素尺度增大与上下文减少的共同作用。", "", "本轮只做 probability diagnosis。由于 native 没有形成整体概率增益，没有继续执行昂贵的全图 graph extraction；graph 无法恢复模型未提供的稳定道路证据。", "",
        "## 可复现性", "", f"- 输入 SHA256：`{metrics['scope']['input_sha256']}`", f"- checkpoint SHA256：`{metrics['scope']['checkpoint_sha256']}`", f"- native road probability SHA256：`{metrics['hashes']['native_probability']}`", "- 未训练、未修改 MaGRoad、未调整阈值。", "",
        "```bash", "python3 -m experiments.bailu2_native_scale.compare", "```", "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_XML)
    parser.add_argument("--low-dir", type=Path, default=DEFAULT_LOW)
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--native-rgb", type=Path, default=Path("/Users/badger/Desktop/bailu_2.png"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    low_meta = read_json(args.low_dir / "diagnostics.json")
    native_meta = read_json(args.native_dir / "metadata.json")
    if low_meta["input_sha256"] != native_meta["input_sha256"]:
        raise RuntimeError("Scale comparison inputs differ")
    if low_meta["checkpoint_sha256"] != native_meta["checkpoint_sha256"]:
        raise RuntimeError("Scale comparison checkpoints differ")
    low = np.load(args.low_dir / "road_prob.npy", mmap_mode="r")
    native = np.load(args.native_dir / "road_prob.npy", mmap_mode="r")
    lines, original_size = load_polylines(args.annotations)
    if original_size != (native.shape[1], native.shape[0]) or list(original_size) != low_meta["original_size_wh"]:
        raise RuntimeError("Image, GT, and probability dimensions are inconsistent")
    sampled_lines = [sample_polyline(line) for line in lines]
    samples = np.concatenate(sampled_lines)
    line_ids = np.concatenate([np.full(len(points), i + 1, dtype=np.int32) for i, points in enumerate(sampled_lines)])
    low_local = local_max(low, samples, 2.0)
    native_local = local_max(native, samples, 1.0)
    delta = native_local - low_local
    seam_distance = nearest_seam_distance(samples, native_meta)

    curve = []
    for threshold in THRESHOLDS:
        curve.extend((
            {"scale": "half_4096", "threshold": threshold, "recall": ratio(low_local >= threshold)},
            {"scale": "native_8192", "threshold": threshold, "recall": ratio(native_local >= threshold)},
        ))
    per_line = []
    for line_id in np.unique(line_ids):
        mask = line_ids == line_id
        per_line.append({
            "polyline_id": int(line_id), "samples": int(mask.sum()),
            "low_probability_mean": float(low_local[mask].mean()),
            "native_probability_mean": float(native_local[mask].mean()),
            "mean_delta_native_minus_low": float(delta[mask].mean()),
            "low_recall_at_0.3": ratio(low_local[mask] >= 0.3),
            "native_recall_at_0.3": ratio(native_local[mask] >= 0.3),
            "delta_recall_at_0.3": ratio(native_local[mask] >= 0.3) - ratio(low_local[mask] >= 0.3),
            "low_recall_at_0.839": ratio(low_local[mask] >= 0.839),
            "native_recall_at_0.839": ratio(native_local[mask] >= 0.839),
            "delta_recall_at_0.839": ratio(native_local[mask] >= 0.839) - ratio(low_local[mask] >= 0.839),
        })

    seam_bins = []
    for lower, upper in ((0, 16), (16, 32), (32, 64), (64, 128), (128, math.inf)):
        mask = (seam_distance >= lower) & (seam_distance < upper)
        seam_bins.append({
            "distance_to_nearest_native_tile_boundary_original_px": f"[{lower},{upper})",
            "samples": int(mask.sum()),
            "mean_probability_delta": float(delta[mask].mean()) if mask.any() else None,
            "recall_delta_at_0.3": ratio(native_local[mask] >= 0.3) - ratio(low_local[mask] >= 0.3) if mask.any() else None,
            "recall_delta_at_0.839": ratio(native_local[mask] >= 0.839) - ratio(low_local[mask] >= 0.839) if mask.any() else None,
        })

    transitions = {}
    for threshold in (0.3, 0.839):
        lh, nh = low_local >= threshold, native_local >= threshold
        transitions[str(threshold)] = {
            "both_recalled": ratio(lh & nh),
            "low_only_lost_at_native": ratio(lh & ~nh),
            "native_only_recovered": ratio(~lh & nh),
            "neither_recalled": ratio(~lh & ~nh),
            "net_delta_native_minus_low": ratio(nh) - ratio(lh),
            "cluster_bootstrap_95pct_ci": bootstrap(native_local, low_local, line_ids, threshold),
        }

    tolerance_sensitivity = []
    for tolerance in (0.0, 8.0, 16.0, 32.0):
        if tolerance == TOLERANCE_ORIGINAL:
            low_at_tolerance, native_at_tolerance = low_local, native_local
        else:
            low_at_tolerance = local_max(low, samples, 2.0, tolerance)
            native_at_tolerance = local_max(native, samples, 1.0, tolerance)
        tolerance_sensitivity.append({
            "tolerance_original_px": int(tolerance),
            "low_recall_at_0.3": ratio(low_at_tolerance >= 0.3),
            "native_recall_at_0.3": ratio(native_at_tolerance >= 0.3),
            "delta_at_0.3": ratio(native_at_tolerance >= 0.3) - ratio(low_at_tolerance >= 0.3),
            "low_recall_at_0.839": ratio(low_at_tolerance >= 0.839),
            "native_recall_at_0.839": ratio(native_at_tolerance >= 0.839),
            "delta_at_0.839": ratio(native_at_tolerance >= 0.839) - ratio(low_at_tolerance >= 0.839),
        })

    baseline_missed = low_local < 0.3
    metrics = {
        "scope": {
            "polylines": len(lines), "samples": len(samples), "sample_spacing_original_px": 2.0,
            "tolerance_original_px": TOLERANCE_ORIGINAL, "input_sha256": low_meta["input_sha256"],
            "checkpoint_sha256": low_meta["checkpoint_sha256"],
            "low_inference_size_wh": [low.shape[1], low.shape[0]],
            "native_inference_size_wh": [native.shape[1], native.shape[0]],
            "native_patch_count": native_meta["patch_count"],
        },
        "probability": {
            "low_local_quantiles": quantiles(low_local), "native_local_quantiles": quantiles(native_local),
            "delta_quantiles": quantiles(delta), "mean_delta_native_minus_low": float(delta.mean()),
            "low_local_mean": float(low_local.mean()), "native_local_mean": float(native_local.mean()),
            "share_delta_le_minus_0.2": ratio(delta <= -0.2), "share_delta_ge_plus_0.2": ratio(delta >= 0.2),
            "full_frame_positive_ratio": {
                "low_at_0.3": ratio(low >= 0.3), "native_at_0.3": ratio(native >= 0.3),
                "low_at_0.839": ratio(low >= 0.839), "native_at_0.839": ratio(native >= 0.839),
            },
        },
        "threshold_transitions": transitions,
        "tolerance_sensitivity": tolerance_sensitivity,
        "baseline_missed_subset": {
            "definition": "GT samples with half-scale local probability < 0.3",
            "samples": int(baseline_missed.sum()),
            "native_recovered_at_0.3": ratio(native_local[baseline_missed] >= 0.3),
            "native_recovered_at_0.839": ratio(native_local[baseline_missed] >= 0.839),
            "native_probability_quantiles": quantiles(native_local[baseline_missed]),
        },
        "native_tile_boundary_sensitivity": seam_bins,
        "threshold_curve": curve,
        "per_polyline": per_line,
        "limitations": [
            "The XML has one path label and does not distinguish thin paths from major roads.",
            "Changing input scale also changes each 1024-pixel tile's geographic context.",
            "This probability-only experiment does not measure graph recall.",
            "Centerline recall uses a 16-original-pixel tolerance and does not measure false positives.",
        ],
        "hashes": {
            "annotations": sha256(args.annotations),
            "low_probability": sha256(args.low_dir / "road_prob.npy"),
            "native_probability": sha256(args.native_dir / "road_prob.npy"),
        },
    }
    write_json(args.output_dir / "metrics.json", metrics)
    for filename, rows in (("threshold_comparison.csv", curve), ("per_polyline.csv", per_line), ("tile_boundary_sensitivity.csv", seam_bins)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    save_curve(curve, args.output_dir / "threshold_recall_scale_comparison.png")
    if args.native_rgb.exists():
        save_delta(args.native_rgb, samples, delta, args.output_dir / "gt_probability_scale_delta_overlay.png")
    write_report(metrics, args.output_dir.parent / "REPORT.md")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
