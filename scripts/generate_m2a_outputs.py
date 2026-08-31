#!/usr/bin/env python3
"""Summarize frozen M2A probes and generate LoveDA/campus inspection images."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from overhead_ssl.m2a import (
    CLASS_MAPPING, CLASS_NAMES, INPUT_SIZE, LinearDenseProbe, SEEDS, cache_path,
    load_feature, native_mask,
)
from overhead_ssl.models import Dinov2SmallEncoder, Dinov3SatelliteEncoder
from overhead_ssl.preprocessing import prepare_aspect_preserved_input


BACKBONES = ("dinov2_vits14", "dinov3_vitl16_sat493m")
PALETTE = np.array([
    [0, 0, 0], [120, 120, 120], [220, 55, 55], [245, 205, 55],
    [50, 120, 220], [180, 130, 70], [45, 145, 65], [165, 205, 85],
], dtype=np.uint8)


def load_head(path: Path) -> LinearDenseProbe:
    record = torch.load(path, map_location="cpu", weights_only=True)
    head = LinearDenseProbe(record["feature_dim"])
    head.load_state_dict(record["state_dict"])
    return head.cuda().eval()


def predict_cached(head, cache_root, backbone, row):
    feature = load_feature(cache_path(cache_root, backbone, row.split, row.id)).permute(2, 0, 1).unsqueeze(0).cuda().float()
    with torch.inference_mode():
        logits = head(feature, (INPUT_SIZE, INPUT_SIZE))
    return logits


def label_image(native: np.ndarray) -> Image.Image:
    return Image.fromarray(PALETTE[native.astype(np.uint8)])


def deterministic_val_examples(rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for domain, group in rows.groupby("domain", sort=True):
        scores = group.id.map(lambda value: hashlib.sha256(f"M2A-qualitative:{domain}:{value}".encode()).hexdigest())
        parts.append(group.assign(_score=scores).sort_values("_score").iloc[:10].drop(columns="_score"))
    return pd.concat(parts).sort_values(["domain", "id"]).reset_index(drop=True)


def qualitative(rows, data_root, cache_root, heads, output):
    selected = deterministic_val_examples(rows)
    width, height = 192, 192
    canvas = Image.new("RGB", (width * 4, 24 + height * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Original", "Ground truth", "DINOv2-S", "DINOv3 SAT-L")):
        draw.text((column * width + 4, 5), title, fill="black")
    root = Path(data_root)
    for index, row in enumerate(selected.itertuples()):
        with Image.open(root / row.image_rel) as image:
            original = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        truth = label_image(native_mask(root / row.mask_rel, INPUT_SIZE).numpy()).resize((width, height), Image.Resampling.NEAREST)
        predictions = []
        for backbone in BACKBONES:
            logits = predict_cached(heads[backbone], cache_root, backbone, row)
            native = logits.argmax(1)[0].cpu().numpy() + 1
            predictions.append(label_image(native).resize((width, height), Image.Resampling.NEAREST))
        y = 24 + index * height
        for column, image in enumerate((original, truth, *predictions)):
            canvas.paste(image, (column * width, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    selected.to_csv(output.with_name("loveda_val_qualitative_selection.csv"), index=False)


def encoder_for(backbone):
    return Dinov2SmallEncoder(device="cuda") if backbone == "dinov2_vits14" else Dinov3SatelliteEncoder(device="cuda")


def campus_logits(backbone, head, source: Image.Image):
    encoder = encoder_for(backbone)
    prepared = prepare_aspect_preserved_input(source, 1024, encoder.patch_size, encoder.image_mean, encoder.image_std)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = encoder.encode(prepared.tensor.unsqueeze(0).cuda())
    features = output.features.permute(0, 3, 1, 2).float()
    with torch.inference_mode():
        logits = F.interpolate(head(features), size=(prepared.encoder_height, prepared.encoder_width), mode="bilinear", align_corners=False)
    return logits[:, :, :prepared.content_height, :prepared.content_width], prepared


def logits_to_original(logits, prepared, original_size):
    labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8) + 1
    return label_image(labels).resize(original_size, Image.Resampling.NEAREST)


def road_heatmap(logits, original_size):
    probability = logits.softmax(1)[0, 2].cpu().numpy()
    # A single-channel map: brightness equals P(road), with no semantic postprocessing.
    return Image.fromarray(np.clip(probability * 255, 0, 255).astype(np.uint8), "L").resize(original_size, Image.Resampling.BILINEAR).convert("RGB")


def campus(source_path, heads, output_dir):
    with Image.open(source_path) as image:
        source = image.convert("RGB")
    semantic, road = [], []
    for backbone in BACKBONES:
        logits, prepared = campus_logits(backbone, heads[backbone], source)
        semantic.append(logits_to_original(logits, prepared, source.size))
        road.append(road_heatmap(logits, source.size))
    width = 512; height = round(width * source.height / source.width)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, panels, labels in (
        ("campus_semantic_comparison.jpg", (source, *semantic), ("Original", "DINOv2-S semantic", "DINOv3 SAT-L semantic")),
        ("campus_road_probability_comparison.jpg", (source, *road), ("Original", "DINOv2-S P(road)", "DINOv3 SAT-L P(road)")),
    ):
        canvas = Image.new("RGB", (width * 3, height + 26), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (label, panel) in enumerate(zip(labels, panels)):
            canvas.paste(panel.resize((width, height), Image.Resampling.BILINEAR), (index * width, 26))
            draw.text((index * width + 4, 5), label, fill="black")
        canvas.save(output_dir / filename, quality=92)


def metric_rows(metrics_dir):
    rows = []
    for backbone in BACKBONES:
        for seed in SEEDS:
            value = json.loads((metrics_dir / f"{backbone}_seed_{seed}.json").read_text())
            final = value["official_val"]
            row = {"backbone": backbone, "seed": seed, "miou": final["miou"], "mean_f1": final["mean_f1"], "pixel_accuracy": final["pixel_accuracy"], "road_iou": final["per_class"]["road"]["iou"], "road_f1": final["per_class"]["road"]["f1"], "training_seconds": value["training_seconds"], "peak_allocated_mib": value["peak_allocated_mib"], "peak_reserved_mib": value["peak_reserved_mib"]}
            for name in CLASS_NAMES:
                for metric, number in final["per_class"][name].items():
                    row[f"{name}_{metric}"] = number
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--head-root", required=True)
    parser.add_argument("--val-csv", default="data/splits/m2a_loveda_val.csv")
    parser.add_argument("--metrics-dir", default="outputs/M2A/metrics")
    parser.add_argument("--target", default="assets/external_target/campus_target_001.png")
    parser.add_argument("--output-dir", default="outputs/M2A")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M2A output generation requires CUDA")
    output, metrics_dir = Path(args.output_dir), Path(args.metrics_dir)
    frame = metric_rows(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / "all_seeds.csv", index=False)
    summary_rows = []
    for backbone, group in frame.groupby("backbone"):
        summary_rows.append({
            "backbone": backbone,
            "miou_mean": group.miou.mean(), "miou_std": group.miou.std(ddof=1),
            "mean_f1_mean": group.mean_f1.mean(), "mean_f1_std": group.mean_f1.std(ddof=1),
            "pixel_accuracy_mean": group.pixel_accuracy.mean(), "pixel_accuracy_std": group.pixel_accuracy.std(ddof=1),
            "road_iou_mean": group.road_iou.mean(), "road_iou_std": group.road_iou.std(ddof=1),
            "road_f1_mean": group.road_f1.mean(), "road_f1_std": group.road_f1.std(ddof=1),
            "training_seconds_mean": group.training_seconds.mean(), "training_seconds_std": group.training_seconds.std(ddof=1),
            "peak_allocated_mib_max": group.peak_allocated_mib.max(), "peak_reserved_mib_max": group.peak_reserved_mib.max(),
        })
    summary = pd.DataFrame(summary_rows).set_index("backbone")
    summary.to_csv(metrics_dir / "summary.csv")
    per_class = []
    for backbone, group in frame.groupby("backbone"):
        for name in CLASS_NAMES:
            for metric in ("iou", "precision", "recall", "f1"):
                values = group[f"{name}_{metric}"]
                per_class.append({"backbone": backbone, "class": name, "metric": metric, "mean": values.mean(), "std": values.std(ddof=1)})
    pd.DataFrame(per_class).to_csv(metrics_dir / "per_class_summary.csv", index=False)
    heads = {backbone: load_head(Path(args.head_root) / backbone / "seed_20260901.pt") for backbone in BACKBONES}
    val_rows = pd.read_csv(args.val_csv)
    qualitative(val_rows, args.data_root, args.cache_root, heads, output / "qualitative" / "loveda_val_comparison.jpg")
    campus(Path(args.target), heads, output / "campus_target")
    summary_text = summary.to_string()
    key_rows = summary.reset_index()[["backbone", "miou_mean", "miou_std", "road_iou_mean", "road_iou_std", "road_f1_mean", "road_f1_std", "pixel_accuracy_mean", "pixel_accuracy_std"]]
    key_rows = key_rows.to_string(index=False, float_format=lambda value: f"{value:.6f}")
    class_rows = pd.DataFrame(per_class)
    class_table = class_rows[class_rows.metric.isin(["iou", "f1"])].pivot(index=["class", "metric"], columns="backbone", values=["mean", "std"]).to_string(float_format=lambda value: f"{value:.6f}")
    report = ["# M2A report", "", "Task: Frozen Dense Semantic Linear Probe", "Dataset: LoveDA", "Backbone training: NO", "Probe: single 1x1 convolution / linear dense classifier", "Input: 448x448 RGB", "Train: LoveDA Train internal probe_train", "Dev: LoveDA Train internal probe_dev", "Final test: official LoveDA Val", "Campus: qualitative external target only", "", "## Official Val summary (mean +/- std over 3 seeds)", "", "```text", key_rows, "```", "", "## Per-class IoU/F1 summary", "", "```text", class_table, "```", "", "Detailed precision/recall/IoU/F1: `metrics/per_class_summary.csv`.", "Training time and peak VRAM are in `metrics/all_seeds.csv`; feature extraction/cache reports are outside Git under `/home/badger/datasets/overhead_features/M2A/`.", "", "Qualitative outputs use the fixed seed 20260901 best-dev head; no LoveDA Val or campus result selected any protocol choice.", "", "No foundation model training was performed."]
    (output / "report.md").write_text("\n".join(report) + "\n")
    print(output / "report.md")


if __name__ == "__main__":
    main()
