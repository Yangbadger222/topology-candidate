#!/usr/bin/env python3
"""M1A-T frozen DINOv2 inspection for a user-provided external RGB target."""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import v2

from overhead_ssl.data import load_manifest, load_rgb, pad_to_patch_multiple, resolve_path, select_samples
from overhead_ssl.models import Dinov2SmallEncoder
from overhead_ssl.pca import SharedPCA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_transform(size: int):
    return v2.Compose([
        v2.ToImage(), v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def encode(model, image: Image.Image, transform, device: str):
    tensor = pad_to_patch_multiple(transform(image), model.patch_size).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.encode(tensor)
    return output, tensor.shape[-2:]


def benchmark(model, image, transform, device):
    for _ in range(3):
        encode(model, image, transform, device)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(10):
        torch.cuda.synchronize(); start = time.perf_counter()
        output, shape = encode(model, image, transform, device)
        torch.cuda.synchronize(); timings.append((time.perf_counter() - start) * 1000)
    return output, shape, float(np.mean(timings)), float(np.median(timings)), torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20


def save_comparison(output: Path, original: Image.Image, per_image: np.ndarray, shared: np.ndarray):
    width = 384
    height = round(width * original.height / original.width)
    canvas = Image.new("RGB", (width * 3, height + 25), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate((("Original", original), ("Per-image PCA", Image.fromarray(per_image)), ("Shared PCA (public fit)", Image.fromarray(shared)))):
        canvas.paste(image.convert("RGB").resize((width, height)), (index * width, 25))
        draw.text((index * width + 5, 5), title, fill="black")
    canvas.save(output, quality=92)


def verify_external_not_in_manifests(target: Path, target_hash: str, manifests: list[Path]) -> None:
    """Fail closed if the user target is mentioned as a source path or checksum."""
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Required D3 manifest missing: {manifest}")
        content = manifest.read_text()
        if str(target) in content or target_hash in content or target.name in content:
            raise RuntimeError(f"External target unexpectedly appears in {manifest}")


def public_features(model, samples, data_root, transform, device):
    features = []
    for row in samples.itertuples():
        output, _ = encode(model, load_rgb(resolve_path(data_root, row.path)), transform, device)
        feature = output.features[0].float().cpu().numpy()
        features.append(feature.reshape(-1, feature.shape[-1]))
    return np.concatenate(features, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--reserve-manifest", type=Path)
    parser.add_argument("--output-dir", default="outputs/M1A/external_target")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M1A-T requires CUDA")
    source_hash_before = sha256(args.target)
    train_manifest = Path(args.manifest)
    verify_external_not_in_manifests(
        args.target,
        source_hash_before,
        [
            train_manifest,
            args.holdout_manifest or train_manifest.with_name("ssl_holdout_naip.csv"),
            args.reserve_manifest or train_manifest.with_name("downstream_reserved_loveda_val.csv"),
        ],
    )
    target = load_rgb(args.target)
    original_size = target.size
    samples = select_samples(load_manifest(args.manifest))
    output_root = Path(args.output_dir) / "dinov2_small" / "campus_target_001"
    output_root.mkdir(parents=True, exist_ok=True)
    target.convert("RGB").save(output_root / "original.jpg", quality=95)
    model = Dinov2SmallEncoder(device="cuda")
    results = []
    for size in (256, 512):
        transform = make_transform(size)
        # Fit only public M1A samples. The external image is encoded only after fit().
        pca = SharedPCA(n_components=3, l2_normalize=False).fit(public_features(model, samples, args.data_root, transform, "cuda"))
        output, tensor_shape, mean_ms, median_ms, peak_alloc, peak_reserved = benchmark(model, target, transform, "cuda")
        feature = output.features[0].float().cpu().numpy()
        per_image = SharedPCA(n_components=3, l2_normalize=False).fit(feature.reshape(-1, feature.shape[-1])).transform_grid(feature)
        shared = pca.transform_grid(feature)
        target.convert("RGB").resize((size, size)).save(output_root / f"input_{size}.jpg", quality=95)
        Image.fromarray(per_image).resize((size, size)).save(output_root / f"pca_per_image_{size}.jpg", quality=95)
        Image.fromarray(shared).resize((size, size)).save(output_root / f"pca_shared_{size}.jpg", quality=95)
        save_comparison(output_root / f"campus_dinov2_{size}_comparison.jpg", target, per_image, shared)
        results.append({"content_resize": size, "encoder_input": f"{tensor_shape[0]}x{tensor_shape[1]}", "feature_shape": str(tuple(output.features.shape)), "mean_ms": mean_ms, "median_ms": median_ms, "peak_allocated_mib": peak_alloc, "peak_reserved_mib": peak_reserved, "explained_variance_ratio": ";".join(f"{value:.8f}" for value in pca.pca.explained_variance_ratio_)})
    samples.to_csv(output_root / "public_pca_fit_samples.csv", index=False)
    pd.DataFrame(results).to_csv(output_root / "benchmark.csv", index=False)
    if source_hash_before != sha256(args.target):
        raise RuntimeError("External target source file changed during M1A-T")
    report = [
        "# M1A-T external target report", "", "Source: user-provided external target image", f"Source path: {args.target}",
        f"Source SHA-256: {source_hash_before}", f"Source resolution: {original_size[0]}x{original_size[1]} RGB (decoded from user PNG)",
        "Model: official DINOv2 ViT-S/14 pretrained weights", "Frozen: yes", "Training steps: 0", "",
        "Shared PCA fit: only the 80 public M1A samples listed in public_pca_fit_samples.csv", "PCA feature preprocessing: no L2 normalization", "PCA random state: none; deterministic PCA solver for this input", "Campus participation in PCA fit: NO", "Dataset participation: NO", "Model training participation: NO", "",
        "| content resize | encoder input | feature shape | mean ms | median ms | peak allocated MiB | peak reserved MiB | explained variance ratio |", "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        report.append(f"| {result['content_resize']} | {result['encoder_input']} | {result['feature_shape']} | {result['mean_ms']:.3f} | {result['median_ms']:.3f} | {result['peak_allocated_mib']:.1f} | {result['peak_reserved_mib']:.1f} | {result['explained_variance_ratio']} |")
    report.extend(["", "Visual observations require human inspection; PCA colors carry no semantic labels.", "", "No model training was performed."])
    (output_root / "report.md").write_text("\n".join(report) + "\n")
    print(output_root)


if __name__ == "__main__":
    main()
