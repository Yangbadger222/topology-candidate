#!/usr/bin/env python3
"""M2B post-SSL downstream comparison. Run only after final SSL export and probes exist."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from overhead_ssl.m2a import CLASS_NAMES, LinearDenseProbe, cache_path, load_feature, native_mask
from overhead_ssl.models import AdaptedDinov2SmallEncoder, Dinov2SmallEncoder, Dinov3SatelliteEncoder
from overhead_ssl.preprocessing import prepare_aspect_preserved_input


DATA = Path("/home/badger/datasets/overhead_rgb")
M2A_CACHE = Path("/home/badger/datasets/overhead_features/M2A")
M2B_CACHE = Path("/home/badger/datasets/overhead_features/M2B")
M2A_HEADS = Path("/home/badger/datasets/overhead_features/M2A/heads_m2a_v1")
M2B_HEADS = Path("/home/badger/datasets/overhead_features/M2B/heads")
M2B_ENCODER = Path("/home/badger/datasets/overhead_checkpoints/M2B/dinov2_vits14_overhead_ssl_v1/dinov2_vits14_overhead_ssl_v1_encoder_epoch20.pt")
M2A_METRICS = Path("outputs/M2A/metrics_v1")
M2B_METRICS = Path("outputs/M2B/metrics")
OUTPUT = Path("outputs/M2B")
SEED = 20260901
PALETTE = np.array([[0, 0, 0], [120, 120, 120], [220, 55, 55], [245, 205, 55], [50, 120, 220], [180, 130, 70], [45, 145, 65], [165, 205, 85]], dtype=np.uint8)


MODELS = {
    "dinov2_vits14": {"cache": M2A_CACHE, "heads": M2A_HEADS, "label": "DINOv2 original"},
    "dinov2_vits14_overhead_ssl_v1": {"cache": M2B_CACHE, "heads": M2B_HEADS, "label": "DINOv2 overhead SSL"},
    "dinov3_vitl16_sat493m": {"cache": M2A_CACHE, "heads": M2A_HEADS, "label": "DINOv3 SAT"},
}


def load_head(path: Path) -> LinearDenseProbe:
    record = torch.load(path, map_location="cpu", weights_only=True)
    head = LinearDenseProbe(record["feature_dim"])
    head.load_state_dict(record["state_dict"])
    return head.cuda().eval()


def label_image(native: np.ndarray) -> Image.Image:
    return Image.fromarray(PALETTE[native.astype(np.uint8)])


def predict_cached(backbone: str, head: LinearDenseProbe, row) -> torch.Tensor:
    path = cache_path(MODELS[backbone]["cache"], backbone, row.split, row.id)
    feature = load_feature(path).permute(2, 0, 1).unsqueeze(0).cuda().float()
    with torch.inference_mode():
        return head(feature, (448, 448))


def metric_summary(backbone: str, metrics_root: Path) -> dict:
    values = []
    for seed in (20260901, 20260902, 20260903):
        payload = json.loads((metrics_root / f"{backbone}_seed_{seed}.json").read_text())
        final = payload["official_val"]
        values.append({
            "miou": final["miou"], "mean_f1": final["mean_f1"], "road_iou": final["per_class"]["road"]["iou"],
            "road_f1": final["per_class"]["road"]["f1"], "pixel_accuracy": final["pixel_accuracy"],
            "training_seconds": payload["training_seconds"], "peak_allocated_mib": payload["peak_allocated_mib"],
            "peak_reserved_mib": payload["peak_reserved_mib"],
        })
    frame = pd.DataFrame(values)
    return {
        "miou_mean": frame.miou.mean(), "miou_std": frame.miou.std(ddof=1),
        "mean_f1_mean": frame.mean_f1.mean(), "mean_f1_std": frame.mean_f1.std(ddof=1),
        "road_iou_mean": frame.road_iou.mean(), "road_iou_std": frame.road_iou.std(ddof=1),
        "road_f1_mean": frame.road_f1.mean(), "road_f1_std": frame.road_f1.std(ddof=1),
        "pixel_accuracy_mean": frame.pixel_accuracy.mean(), "pixel_accuracy_std": frame.pixel_accuracy.std(ddof=1),
        "training_seconds_mean": frame.training_seconds.mean(), "training_seconds_std": frame.training_seconds.std(ddof=1),
        "peak_allocated_mib_max": frame.peak_allocated_mib.max(), "peak_reserved_mib_max": frame.peak_reserved_mib.max(),
    }


def make_qualitative(heads: dict[str, LinearDenseProbe]):
    selected = pd.read_csv("outputs/M2A/qualitative/loveda_val_qualitative_selection.csv")
    width = height = 160
    canvas = Image.new("RGB", (width * 5, 24 + height * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("Original", "Ground truth", "DINOv2 original", "DINOv2 overhead SSL", "DINOv3 SAT")
    for column, label in enumerate(labels): draw.text((column * width + 3, 4), label, fill="black")
    for index, row in enumerate(selected.itertuples()):
        with Image.open(DATA / row.image_rel) as image:
            original = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        truth = label_image(native_mask(DATA / row.mask_rel, 448).numpy()).resize((width, height), Image.Resampling.NEAREST)
        panels = [original, truth]
        for backbone in MODELS:
            native = predict_cached(backbone, heads[backbone], row).argmax(1)[0].cpu().numpy() + 1
            panels.append(label_image(native).resize((width, height), Image.Resampling.NEAREST))
        y = 24 + index * height
        for column, panel in enumerate(panels): canvas.paste(panel, (column * width, y))
    destination = OUTPUT / "qualitative" / "loveda_val_three_way_comparison.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True); canvas.save(destination, quality=92)
    return destination


def encoder_for(backbone):
    if backbone == "dinov2_vits14": return Dinov2SmallEncoder(device="cuda")
    if backbone == "dinov2_vits14_overhead_ssl_v1": return AdaptedDinov2SmallEncoder(M2B_ENCODER, device="cuda")
    return Dinov3SatelliteEncoder(device="cuda")


def target_logits(backbone, head, image):
    encoder = encoder_for(backbone)
    prepared = prepare_aspect_preserved_input(image, 1024, encoder.patch_size, encoder.image_mean, encoder.image_std)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = encoder.encode(prepared.tensor.unsqueeze(0).cuda())
    features = output.features.permute(0, 3, 1, 2).float()
    with torch.inference_mode():
        logits = F.interpolate(head(features), size=(prepared.encoder_height, prepared.encoder_width), mode="bilinear", align_corners=False)
    return logits[:, :, :prepared.content_height, :prepared.content_width], prepared


def target_panel(logits, prepared, original_size, road=False):
    if road:
        values = (logits.softmax(1)[0, 2].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(values, "L").resize(original_size, Image.Resampling.BILINEAR).convert("RGB")
    native = logits.argmax(1)[0].cpu().numpy() + 1
    return label_image(native).resize(original_size, Image.Resampling.NEAREST)


def make_campus(heads: dict[str, LinearDenseProbe]):
    source_path = Path("assets/external_target/campus_target_001.png")
    with Image.open(source_path) as image: source = image.convert("RGB")
    semantic, roads = [], []
    for backbone in MODELS:
        logits, prepared = target_logits(backbone, heads[backbone], source)
        semantic.append(target_panel(logits, prepared, source.size))
        roads.append(target_panel(logits, prepared, source.size, road=True))
    width = 420; height = round(width * source.height / source.width)
    output = OUTPUT / "campus_target"; output.mkdir(parents=True, exist_ok=True)
    for name, panels, titles in (
        ("campus_semantic_three_way.jpg", [source, *semantic], ["Original", "DINOv2 original", "DINOv2 overhead SSL", "DINOv3 SAT"]),
        ("campus_road_probability_three_way.jpg", [source, *roads], ["Original", "DINOv2 original P(road)", "DINOv2 overhead SSL P(road)", "DINOv3 SAT P(road)"]),
    ):
        canvas = Image.new("RGB", (width * 4, height + 26), "white"); draw = ImageDraw.Draw(canvas)
        for index, (panel, title) in enumerate(zip(panels, titles)):
            canvas.paste(panel.resize((width, height), Image.Resampling.BILINEAR), (index * width, 26)); draw.text((index * width + 3, 4), title, fill="black")
        canvas.save(output / name, quality=92)
    return output


def main():
    if not M2B_ENCODER.is_file(): raise SystemExit("M2B final encoder export missing")
    heads = {backbone: load_head(MODELS[backbone]["heads"] / backbone / f"seed_{SEED}.pt") for backbone in MODELS}
    original = pd.read_csv(M2A_METRICS / "summary.csv").set_index("backbone")
    summaries = []
    for backbone, label in (("dinov2_vits14", "DINOv2-S original"), ("dinov2_vits14_overhead_ssl_v1", "DINOv2-S overhead SSL"), ("dinov3_vitl16_sat493m", "DINOv3-SAT-L")):
        stats = metric_summary(backbone, M2B_METRICS) if backbone == "dinov2_vits14_overhead_ssl_v1" else original.loc[backbone].to_dict()
        summaries.append({"model": label, **stats})
    summary = pd.DataFrame(summaries)
    original_road, adapted_road, dino3_road = summary.road_iou_mean.tolist()
    gain = adapted_road - original_road
    comparison = {"road_iou_absolute_gain_vs_original": gain, "road_iou_relative_gain_vs_original_percent": 100 * gain / original_road, "road_iou_remaining_gap_to_dinov3": dino3_road - adapted_road}
    M2B_METRICS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(M2B_METRICS / "three_way_summary.csv", index=False)
    (M2B_METRICS / "adaptation_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    qualitative = make_qualitative(heads)
    campus = make_campus(heads)
    train = json.loads((OUTPUT / "training_summary.json").read_text())
    display_columns = ["model", "miou_mean", "miou_std", "road_iou_mean", "road_iou_std", "road_f1_mean", "road_f1_std", "pixel_accuracy_mean", "pixel_accuracy_std"]
    report = ["# M2B report", "", "Dataset: LoveDA Train RGB 2522 + NAIP Train RGB 4320 = 6842 SSL images.", "Labels used during SSL: NONE.", "Student initialization: official DINOv2 ViT-S/14. Teacher: EMA DINOv2-S.", "SSL objectives: DINO global self-distillation + iBOT masked patch prediction.", "Epochs: 20. GPU: RTX 4070 Laptop 8GB.", "", f"Actual micro batch / accumulation / effective batch: {train['config']['micro_batch']} / {train['config']['accumulation_steps']} / {train['config']['effective_batch']}", f"dtype: {train['config']['dtype']}; peak reserved MiB: {max(row['peak_reserved_mib'] for row in pd.read_csv(OUTPUT / 'training_summary.csv').to_dict('records')):.1f}", f"Training duration seconds: {train['duration_seconds']:.1f}", f"Final encoder: {train['encoder_checkpoint']}", f"Final encoder SHA-256: {train['encoder_sha256']}", f"Collapse diagnostic pass: {train['collapse_pass']}", "", "## Frozen M2A protocol comparison (mean +/- std, 3 seeds)", "", summary[display_columns].to_string(index=False, float_format=lambda value: f'{value:.6f}'), "", f"Road IoU original -> adapted: {gain:+.6f} absolute ({comparison['road_iou_relative_gain_vs_original_percent']:+.2f}%).", f"Remaining Road IoU gap to DINOv3: {comparison['road_iou_remaining_gap_to_dinov3']:+.6f}.", "", f"LoveDA qualitative comparison: {qualitative}", f"Campus qualitative outputs: {campus}", "", "No semantic labels were used during M2B SSL training.", "DINOv3 was not trained.", "Campus data was not used for training."]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    print(OUTPUT / "report.md")


if __name__ == "__main__": main()
