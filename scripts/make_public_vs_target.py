#!/usr/bin/env python3
"""Compose a shared-PCA-only public-vs-external-target inspection sheet."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="outputs/M1A/sample_selection.csv")
    parser.add_argument("--public-root", default="outputs/M1A/dinov2_small/256")
    parser.add_argument("--target", default="outputs/M1A/external_target/dinov2_small/campus_target_001/pca_shared_256.jpg")
    parser.add_argument("--output", default="outputs/M1A/external_target/dinov2_small/campus_target_001/public_vs_target_shared_pca.jpg")
    args = parser.parse_args()
    samples = pd.read_csv(args.samples)
    choices = [("LoveDA", "urban"), ("LoveDA", "rural"), ("NAIP", "campus_like"), ("NAIP", "wooded_roads")]
    columns = []
    root = Path(args.public_root)
    for dataset, scene in choices:
        row = samples[(samples.dataset == dataset) & (samples.domain_or_scene == scene)].iloc[0]
        columns.append((f"{dataset} {scene}", Image.open(root / f"{row.sample_id}_pca_shared.jpg").convert("RGB")))
    columns.append(("Campus target", Image.open(args.target).convert("RGB")))
    cell = 256
    canvas = Image.new("RGB", (cell * len(columns), cell + 26), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(columns):
        canvas.paste(image.resize((cell, cell)), (index * cell, 26))
        draw.text((index * cell + 4, 5), label, fill="black")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output, quality=92)


if __name__ == "__main__":
    main()
