#!/usr/bin/env python3
"""Native-resolution tiled DINOv3 SAT feature and road-probe extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA

import sys

sys.path.insert(0, "/home/badger/overhead-ssl/src")
from overhead_ssl.m2a import LinearDenseProbe  # noqa: E402
from overhead_ssl.models import Dinov3SatelliteEncoder  # noqa: E402


MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
MODEL_REVISION = "f692fa42da72c6797b67cd73494a168d1120d3ee"
SEEDS = (20260901, 20260902, 20260903)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


def load_heads(head_dir: Path, device: str) -> list[LinearDenseProbe]:
    heads = []
    for seed in SEEDS:
        record = torch.load(head_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        head = LinearDenseProbe(record["feature_dim"])
        head.load_state_dict(record["state_dict"])
        heads.append(head.to(device).eval())
    return heads


def tile_tensor(image: Image.Image, box: tuple[int, int, int, int], size: int, mean, std, device: str):
    crop = image.crop(box)
    if crop.size != (size, size):
        canvas = Image.new("RGB", (size, size))
        canvas.paste(crop, (0, 0))
        crop = canvas
    array = np.asarray(crop, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    return tensor.unsqueeze(0).to(device)


def encode(encoder, image, x, y, tile, device):
    tensor = tile_tensor(image, (x, y, min(x + tile, image.width), min(y + tile, image.height)), tile,
                         encoder.image_mean, encoder.image_std, device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = encoder.encode(tensor)
    return output.features[0].float()


def blend_window(tile: int) -> np.ndarray:
    edge = max(16, tile // 12)
    axis = np.ones(tile, dtype=np.float32)
    ramp = np.linspace(0.05, 1.0, edge, dtype=np.float32)
    axis[:edge] = ramp
    axis[-edge:] = ramp[::-1]
    return np.outer(axis, axis)


def save_panel(path: Path, original: Image.Image, pca_img: Image.Image, heatmap: Image.Image, overlay: Image.Image):
    width = 1200
    height = round(width * original.height / original.width)
    header = 42
    canvas = Image.new("RGB", (width * 2, (height + header) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    panels = (
        ("Original", original),
        ("DINOv3 SAT-L dense features (PCA)", pca_img),
        ("LoveDA linear probe: road probability", heatmap),
        ("Road probability overlay", overlay),
    )
    for index, (label, panel) in enumerate(panels):
        row, col = divmod(index, 2)
        x, y = col * width, row * (height + header)
        draw.rectangle((x, y, x + width, y + header), fill=(24, 32, 44))
        draw.text((x + 14, y + 13), label, fill="white")
        canvas.paste(panel.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), (x, y + header))
    canvas.save(path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--calibration-tiles", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.tile % 16:
        raise ValueError("Tile size must be divisible by patch size 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.input).convert("RGB")
    width, height = image.size
    stride = args.tile - args.overlap
    xs, ys = positions(width, args.tile, stride), positions(height, args.tile, stride)
    coords = [(x, y) for y in ys for x in xs]

    started = time.perf_counter()
    encoder = Dinov3SatelliteEncoder(device="cuda")
    heads = load_heads(args.head_dir, "cuda")
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())

    # Fit one stable PCA basis from tiles spread over the complete scene.
    indices = np.linspace(0, len(coords) - 1, min(args.calibration_tiles, len(coords)), dtype=int)
    samples = []
    for index in indices:
        x, y = coords[int(index)]
        features = encode(encoder, image, x, y, args.tile, "cuda").cpu().numpy().reshape(-1, encoder.feature_dim)
        samples.append(features)
    calibration = np.concatenate(samples, axis=0)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=20260917).fit(calibration)
    calibration_projected = pca.transform(calibration)
    pca_low = np.percentile(calibration_projected, 1, axis=0)
    pca_high = np.percentile(calibration_projected, 99, axis=0)

    road_sum = np.memmap(args.output_dir / ".road_sum.dat", mode="w+", dtype="float32", shape=(height, width))
    pca_sum = np.memmap(args.output_dir / ".pca_sum.dat", mode="w+", dtype="float32", shape=(height, width, 3))
    weight_sum = np.memmap(args.output_dir / ".weight_sum.dat", mode="w+", dtype="float32", shape=(height, width))
    road_sum[:] = 0
    pca_sum[:] = 0
    weight_sum[:] = 0
    window = blend_window(args.tile)

    torch.cuda.reset_peak_memory_stats()
    for number, (x, y) in enumerate(coords, start=1):
        features = encode(encoder, image, x, y, args.tile, "cuda")
        feature_map = features.permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            logits = torch.stack([head.classifier(feature_map) for head in heads]).mean(dim=0)
            logits = F.interpolate(logits, size=(args.tile, args.tile), mode="bilinear", align_corners=False)
            road = logits.softmax(dim=1)[0, 2].cpu().numpy()
        feature_np = features.cpu().numpy()
        projected = pca.transform(feature_np.reshape(-1, feature_np.shape[-1])).reshape(features.shape[0], features.shape[1], 3)
        projected = np.clip((projected - pca_low) / np.maximum(pca_high - pca_low, 1e-8), 0, 1)
        projected_img = Image.fromarray((projected * 255).astype(np.uint8)).resize((args.tile, args.tile), Image.Resampling.BILINEAR)
        projected = np.asarray(projected_img, dtype=np.float32) / 255.0

        valid_w, valid_h = min(args.tile, width - x), min(args.tile, height - y)
        local_weight = window[:valid_h, :valid_w]
        road_sum[y:y + valid_h, x:x + valid_w] += road[:valid_h, :valid_w] * local_weight
        pca_sum[y:y + valid_h, x:x + valid_w] += projected[:valid_h, :valid_w] * local_weight[:, :, None]
        weight_sum[y:y + valid_h, x:x + valid_w] += local_weight
        if number == 1 or number % 10 == 0 or number == len(coords):
            print(f"tile {number}/{len(coords)}", flush=True)

    weights = np.maximum(np.asarray(weight_sum), 1e-8)
    road_probability = np.asarray(road_sum) / weights
    pca_rgb = np.asarray(pca_sum) / weights[:, :, None]
    np.save(args.output_dir / "road_probability.npy", road_probability.astype(np.float16))
    Image.fromarray((np.clip(road_probability, 0, 1) * 255).astype(np.uint8)).save(args.output_dir / "road_probability.png")
    pca_img = Image.fromarray((np.clip(pca_rgb, 0, 1) * 255).astype(np.uint8))
    pca_img.save(args.output_dir / "dinov3_feature_pca.jpg", quality=95)

    heatmap_array = (plt.get_cmap("turbo")(road_probability)[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap_array)
    heatmap.save(args.output_dir / "road_probability_heatmap.jpg", quality=95)
    original_np = np.asarray(image, dtype=np.float32)
    alpha = np.clip(road_probability[:, :, None] * 0.72, 0, 0.72)
    overlay_array = original_np * (1 - alpha) + np.array([255, 225, 0], dtype=np.float32) * alpha
    overlay = Image.fromarray(np.clip(overlay_array, 0, 255).astype(np.uint8))
    overlay.save(args.output_dir / "road_probability_overlay.jpg", quality=95)
    save_panel(args.output_dir / "comparison_panel.jpg", image, pca_img, heatmap, overlay)

    elapsed = time.perf_counter() - started
    metadata = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "source_size_wh": [width, height],
        "model_id": MODEL_ID,
        "model_revision": encoder.revision,
        "parameter_count": parameter_count,
        "feature_layer": "last hidden patch tokens",
        "feature_dim": encoder.feature_dim,
        "patch_size": encoder.patch_size,
        "tile_size": args.tile,
        "overlap": args.overlap,
        "stride": stride,
        "tile_grid_xy": [len(xs), len(ys)],
        "tile_count": len(coords),
        "pca_fit_tile_count": len(indices),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "road_probe": "LoveDA seven-class linear probe, three-seed logit ensemble",
        "road_class_index": 2,
        "road_probability_statistics": {
            "min": float(road_probability.min()),
            "mean": float(road_probability.mean()),
            "median": float(np.median(road_probability)),
            "max": float(road_probability.max()),
            "fraction_gt_0_3": float((road_probability > 0.3).mean()),
            "fraction_gt_0_4": float((road_probability > 0.4).mean()),
            "fraction_gt_0_5": float((road_probability > 0.5).mean()),
        },
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "elapsed_seconds": elapsed,
        "training_steps": 0,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for temporary in (".road_sum.dat", ".pca_sum.dat", ".weight_sum.dat"):
        (args.output_dir / temporary).unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
