#!/usr/bin/env python3
"""Frozen official DINOv3 SAT493M M1A target baseline. Training steps: zero."""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from overhead_ssl.data import load_manifest, load_rgb, resolve_path, select_samples
from overhead_ssl.models import Dinov3SatelliteEncoder
from overhead_ssl.pca import SharedPCA
from overhead_ssl.preprocessing import pca_grid_to_original_aspect, prepare_aspect_preserved_input


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_external_not_in_manifests(target: Path, target_hash: str, manifests: list[Path]) -> None:
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Required D3 manifest missing: {manifest}")
        text = manifest.read_text()
        if str(target) in text or target_hash in text or target.name in text:
            raise RuntimeError(f"External target unexpectedly appears in {manifest}")


def encode(model: Dinov3SatelliteEncoder, image: Image.Image, long_side: int):
    prepared = prepare_aspect_preserved_input(image, long_side, model.patch_size, model.image_mean, model.image_std)
    tensor = prepared.tensor.unsqueeze(0).to("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.encode(tensor)
    return output, prepared


def benchmark(model: Dinov3SatelliteEncoder, image: Image.Image, long_side: int):
    for _ in range(3):
        encode(model, image, long_side)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output, prepared = encode(model, image, long_side)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return output, prepared, float(np.mean(timings)), float(np.median(timings)), torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20


def public_features(model: Dinov3SatelliteEncoder, samples: pd.DataFrame, data_root: str, long_side: int) -> np.ndarray:
    tokens = []
    for row in samples.itertuples():
        output, _ = encode(model, load_rgb(resolve_path(data_root, row.path)), long_side)
        feature = output.features[0].float().cpu().numpy()
        tokens.append(feature.reshape(-1, feature.shape[-1]))
    return np.concatenate(tokens, axis=0)


def save_comparison(path: Path, original: Image.Image, per_image: Image.Image, shared: Image.Image) -> None:
    width = 512
    height = round(width * original.height / original.width)
    canvas = Image.new("RGB", (width * 3, height + 26), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate((("Original", original), ("Per-image PCA", per_image), ("Shared PCA (80 public fit)", shared))):
        canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), (index * width, 26))
        draw.text((index * width + 5, 5), title, fill="black")
    canvas.save(path, quality=92)


def run_scale(model, target, samples, data_root, output_root, long_side):
    """Benchmark target first so a 1024 OOM cannot trigger 80 unnecessary public passes."""
    output, prepared, mean_ms, median_ms, peak_alloc, peak_reserved = benchmark(model, target, long_side)
    pca = SharedPCA(n_components=3, l2_normalize=False).fit(public_features(model, samples, data_root, long_side))
    feature = output.features[0].float().cpu().numpy()
    per_grid = SharedPCA(n_components=3, l2_normalize=False).fit(feature.reshape(-1, feature.shape[-1])).transform_grid(feature)
    shared_grid = pca.transform_grid(feature)
    per_image = pca_grid_to_original_aspect(per_grid, prepared.encoder_height, prepared.encoder_width, prepared.content_height, prepared.content_width, target.size)
    shared_image = pca_grid_to_original_aspect(shared_grid, prepared.encoder_height, prepared.encoder_width, prepared.content_height, prepared.content_width, target.size)
    target.resize((prepared.content_width, prepared.content_height), Image.Resampling.BICUBIC).save(output_root / f"input_aspect_preserved_{long_side}.jpg", quality=95)
    per_image.save(output_root / f"pca_per_image_aspect_preserved_{long_side}.jpg", quality=95)
    shared_image.save(output_root / f"pca_shared_aspect_preserved_{long_side}.jpg", quality=95)
    save_comparison(output_root / f"campus_dinov3_sat_aspect_preserved_{long_side}.jpg", target, per_image, shared_image)
    return {
        "model": "official DINOv3 ViT-L/16 SAT493M",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pretraining_domain": "SAT-493M satellite imagery",
        "input_scale": long_side,
        "content_size_hw": f"{prepared.content_height}x{prepared.content_width}",
        "actual_input_size": f"{prepared.encoder_height}x{prepared.encoder_width}",
        "feature_grid": f"{output.grid_height}x{output.grid_width}",
        "feature_dim": output.feature_dim,
        "feature_shape": str(tuple(output.features.shape)),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "peak_allocated_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
        "explained_variance_ratio": ";".join(f"{value:.8f}" for value in pca.pca.explained_variance_ratio_),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default="outputs/M1A/external_target", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M1A requires CUDA")
    source_hash = sha256(args.target)
    verify_external_not_in_manifests(args.target, source_hash, [args.manifest, args.manifest.with_name("ssl_holdout_naip.csv"), args.manifest.with_name("downstream_reserved_loveda_val.csv")])
    target = load_rgb(args.target)
    samples = select_samples(load_manifest(args.manifest))
    output_root = args.output_dir / "dinov3_sat" / "campus_target_001" / "aspect_preserved"
    output_root.mkdir(parents=True, exist_ok=True)
    target.save(output_root / "original.jpg", quality=95)
    model = Dinov3SatelliteEncoder(device="cuda")
    results = []
    for long_side in (256, 512):
        results.append(run_scale(model, target, samples, args.data_root, output_root, long_side))
    # 512 succeeded. A single 1024 target attempt is allowed; never retry an OOM.
    try:
        results.append(run_scale(model, target, samples, args.data_root, output_root, 1024))
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        results.append({"model": "official DINOv3 ViT-L/16 SAT493M", "parameters": sum(parameter.numel() for parameter in model.parameters()), "pretraining_domain": "SAT-493M satellite imagery", "input_scale": 1024, "status": "OOM"})
    samples.to_csv(output_root / "public_pca_fit_samples.csv", index=False)
    pd.DataFrame(results).to_csv(output_root / "benchmark.csv", index=False)
    if source_hash != sha256(args.target):
        raise RuntimeError("External target source file changed during M1A")
    report = [
        "# DINOv3 SAT493M M1A target report", "", "Model repository: facebook/dinov3-vitl16-pretrain-sat493m", f"Model revision: {model.revision}",
        "Weight format: official model.safetensors via Hugging Face standard cache", f"Cache snapshot: ~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-sat493m/snapshots/{model.revision}",
        "License/provenance: Meta DINOv3 License; official facebook Hugging Face gated repository", f"Parameter count: {sum(parameter.numel() for parameter in model.parameters()):,}",
        f"Source SHA-256: {source_hash}", "Frozen: yes", "Training steps: 0", "",
        "Preprocessing: aspect-preserved whole-image resize to requested long side; official SAT493M AutoImageProcessor mean/std; right/bottom padding only to patch size 16.",
        "Shared PCA: one DINOv3-only basis per scale, fit solely to the same 80 public M1A samples; external target excluded.", "",
        "| input scale | actual input HxW | feature grid HfxWf | feature dim | mean ms | median ms | peak allocated MiB | peak reserved MiB | status |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        report.append(f"| {row['input_scale']} | {row.get('actual_input_size', 'n/a')} | {row.get('feature_grid', 'n/a')} | {row.get('feature_dim', 'n/a')} | {row.get('mean_ms', float('nan')):.3f} | {row.get('median_ms', float('nan')):.3f} | {row.get('peak_allocated_mib', float('nan')):.1f} | {row.get('peak_reserved_mib', float('nan')):.1f} | {row.get('status', 'success')} |")
    report.extend(["", "PCA colors have no semantic labels. Compare only spatial boundary clarity, elongated road-like structures, vegetation/building region coherence, thin structure preservation, and cross-domain visual consistency.", "", "External campus image was NOT used for training.", "External campus image was NOT used to fit PCA.", "No model training was performed."])
    (output_root / "report.md").write_text("\n".join(report) + "\n")
    print(output_root)


if __name__ == "__main__":
    main()
