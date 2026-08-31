#!/usr/bin/env python3
"""Generate M1A DINOv3 SAT public PCA visualizations without saving tensors."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from overhead_ssl.data import load_manifest, load_rgb, resolve_path, select_samples
from overhead_ssl.models import Dinov3SatelliteEncoder
from overhead_ssl.pca import SharedPCA
from overhead_ssl.preprocessing import prepare_aspect_preserved_input


def encode(model, image, long_side):
    prepared = prepare_aspect_preserved_input(image, long_side, model.patch_size, model.image_mean, model.image_std)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.encode(prepared.tensor.unsqueeze(0).to("cuda"))
    return output, prepared


def fit_public_pca(model, samples, data_root, long_side):
    tokens = []
    for row in samples.itertuples():
        output, _ = encode(model, load_rgb(resolve_path(data_root, row.path)), long_side)
        feature = output.features[0].float().cpu().numpy()
        tokens.append(feature.reshape(-1, feature.shape[-1]))
    return SharedPCA(n_components=3, l2_normalize=False).fit(np.concatenate(tokens, axis=0))


def make_sheet(samples, output_root, output_path, scale, filter_frame=None):
    frame = samples if filter_frame is None else filter_frame
    cell = 128
    canvas = Image.new("RGB", (cell * 3, 24 + cell * len(frame)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(("Original", "Per-image PCA", "Shared PCA")):
        draw.text((index * cell + 4, 5), title, fill="black")
    for index, row in enumerate(frame.itertuples()):
        prefix = output_root / f"{row.sample_id}_"
        paths = (prefix.with_name(prefix.name + "original.jpg"), prefix.with_name(prefix.name + "pca_per_image.jpg"), prefix.with_name(prefix.name + "pca_shared.jpg"))
        for column, path in enumerate(paths):
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB").resize((cell, cell)), (column * cell, 24 + index * cell))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/M1A")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M1A requires CUDA")
    samples = select_samples(load_manifest(args.manifest))
    model = Dinov3SatelliteEncoder(device="cuda")
    base = Path(args.output_dir)
    for scale in (256, 512):
        output_root = base / "dinov3_sat" / str(scale)
        output_root.mkdir(parents=True, exist_ok=True)
        pca = fit_public_pca(model, samples, args.data_root, scale)
        for row in samples.itertuples():
            image = load_rgb(resolve_path(args.data_root, row.path))
            output, _ = encode(model, image, scale)
            feature = output.features[0].float().cpu().numpy()
            per = SharedPCA(3, False).fit(feature.reshape(-1, feature.shape[-1])).transform_grid(feature)
            shared = pca.transform_grid(feature)
            image.resize((scale, scale), Image.Resampling.BICUBIC).save(output_root / f"{row.sample_id}_original.jpg", quality=95)
            Image.fromarray(per).resize((scale, scale), Image.Resampling.NEAREST).save(output_root / f"{row.sample_id}_pca_per_image.jpg", quality=95)
            Image.fromarray(shared).resize((scale, scale), Image.Resampling.NEAREST).save(output_root / f"{row.sample_id}_pca_shared.jpg", quality=95)
        make_sheet(samples, output_root, base / "comparisons" / f"dinov3_sat_{scale}.jpg", scale)
        for dataset, scene in (("LoveDA", "urban"), ("LoveDA", "rural"), ("NAIP", "campus_like"), ("NAIP", "wooded_roads")):
            group = samples[(samples.dataset == dataset) & (samples.domain_or_scene == scene)]
            make_sheet(samples, output_root, base / "comparisons" / f"dinov3_sat_{dataset.lower()}_{scene}_{scale}.jpg", scale, group)
    print("DINOv3 public PCA visualizations generated")


if __name__ == "__main__":
    main()
