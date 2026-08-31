#!/usr/bin/env python3
"""Create aspect-preserved shared-PCA comparisons for manual backbone review."""
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("outputs/M1A/external_target")
TARGET = ROOT / "dinov2_small/campus_target_001/aspect_preserved/original.jpg"
DINO2 = ROOT / "dinov2_small/campus_target_001/aspect_preserved"
DINO3 = ROOT / "dinov3_sat/campus_target_001/aspect_preserved"
OUTPUT = ROOT / "comparisons"


def compose(scale: int) -> None:
    paths = [TARGET, DINO2 / f"pca_shared_aspect_preserved_{scale}.jpg", DINO3 / f"pca_shared_aspect_preserved_{scale}.jpg"]
    if not all(path.exists() for path in paths):
        return
    original = Image.open(TARGET).convert("RGB")
    width = 512
    height = round(width * original.height / original.width)
    canvas = Image.new("RGB", (width * 3, height + 26), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("Original", "DINOv2-S shared PCA", "DINOv3 SAT-L shared PCA")
    for index, (label, path) in enumerate(zip(labels, paths)):
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), (index * width, 26))
        draw.text((index * width + 5, 5), label, fill="black")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT / f"campus_dinov2_vs_dinov3_{scale}.jpg", quality=92)


for requested_scale in (512, 1024):
    compose(requested_scale)
