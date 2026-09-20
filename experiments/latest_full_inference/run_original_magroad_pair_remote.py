#!/usr/bin/env python3
"""Run the two untouched official MaGRoad ViT-B checkpoints on one image.

The semantic inference configuration (patch size, thresholds, NMS, candidate
radius and TopoNet settings) comes directly from each official YAML.  Only the
execution batch size is reduced to one for the 8 GiB deployment GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from magroad_graph_regression import run_graph_regression as regression


MODELS = {
    "wildroad_vitb_original": {
        "label": "Original WildRoad ViT-B",
        "checkpoint": Path("/home/badger/datasets/overhead_checkpoints/MaGRoad/wildroad_vitb.ckpt"),
        "checkpoint_sha256": "095bf4f1688d7172604ff855a22960173c869eeee20061ed61d92a07540ef8a6",
        "config": Path("config/toponet_vitb_1024_wild_road.yaml"),
    },
    "globalscale_vitb_original": {
        "label": "Original Global-Scale ViT-B",
        "checkpoint": Path("/home/badger/datasets/overhead_checkpoints/MaGRoad/globalscale_vitb.ckpt"),
        "checkpoint_sha256": "e834dbf0f37cfe7617405df8ca9c098be3d1d262a326d4dde8bf7d22f918e361",
        "config": Path("config/toponet_vitb_512_globalscale.yaml"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in state.items():
        while key.startswith(("module.", "model.", "net.")):
            key = key.split(".", 1)[1]
        if key in output:
            raise RuntimeError(f"Checkpoint key collision: {key}")
        output[key] = value
    return output


def rectangular_infer_function():
    source = inspect.getsource(regression.infer_detailed)
    square_guard = (
        "    if img.shape[:2] != (config.PATCH_SIZE, config.PATCH_SIZE):\n"
        "        raise ValueError(f\"This controlled regression expects native "
        "{config.PATCH_SIZE} square images, got {img.shape}\")\n"
    )
    if square_guard not in source:
        raise RuntimeError("Could not locate the audited square-input guard")
    namespace = dict(regression.__dict__)
    exec(source.replace(square_guard, ""), namespace)
    return namespace["infer_detailed"]


def preview(path: Path, width: int = 2048) -> Image.Image:
    image = Image.open(path).convert("RGB")
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def comparison_panel(output_root: Path, image_stem: str) -> None:
    columns = 2
    cell_width = 1024
    cell_image_height = round(cell_width * 4320 / 8192)
    label_height = 34
    rows = [
        ("rgb.png", "RGB input"),
        ("road_heatmap.png", "Road probability"),
        ("road_binary.png", "Road mask at official threshold"),
        ("keypoints.png", "Selected keypoints"),
        ("final_graph_overlay.png", "Raw official graph"),
    ]
    canvas = Image.new("RGB", (columns * cell_width, len(rows) * (cell_image_height + label_height)), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (model_name, spec) in enumerate(MODELS.items()):
        model_dir = output_root / model_name / image_stem
        diagnostics = json.loads((model_dir / "diagnostics.json").read_text())
        for row, (filename, label) in enumerate(rows):
            image = Image.open(model_dir / filename).convert("RGB").resize(
                (cell_width, cell_image_height), Image.Resampling.LANCZOS
            )
            x = column * cell_width
            y = row * (cell_image_height + label_height)
            canvas.paste(image, (x, y + label_height))
            if row == 0:
                text = f"{spec['label']} | {label}"
            elif filename == "road_binary.png":
                text = f"{spec['label']} | {label} {diagnostics['road_threshold']:.3f}"
            else:
                text = f"{spec['label']} | {label}"
            draw.text((x + 10, y + 10), text, fill="white", font=font)
    canvas.save(output_root / "original_pair_comparison.jpg", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    # NumPy 2.x checkpoint compatibility in the validated NumPy 1.x runtime.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

    inferencer, graph_extraction, MaGRoad, load_config = regression.import_magroad()
    infer_rectangular = rectangular_infer_function()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inferencer.args.device = device
    torch.backends.cudnn.benchmark = True

    with Image.open(args.image) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    height, width = rgb.shape[:2]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "original_magroad_vitb_pair",
        "input_path": str(args.image.resolve()),
        "input_sha256": sha256(args.image),
        "inference_size_wh": [width, height],
        "training": False,
        "autograd_enabled": False,
        "models": [],
    }

    for model_name, spec in MODELS.items():
        actual_sha = sha256(spec["checkpoint"])
        if actual_sha != spec["checkpoint_sha256"]:
            raise RuntimeError(f"{model_name} checkpoint SHA256 mismatch: {actual_sha}")
        config = load_config(str(spec["config"]))
        official_batch_size = int(config.INFER_BATCH_SIZE)
        config.INFER_BATCH_SIZE = 1
        net = MaGRoad(config)
        raw = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        load_result = net.load_state_dict(canonical(raw.get("state_dict", raw)), strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"{model_name}: strict checkpoint load failed")
        net.eval().to(device)
        for parameter in net.parameters():
            parameter.requires_grad_(False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = infer_rectangular(net, rgb, config, device, inferencer, graph_extraction)
        elapsed = time.perf_counter() - started
        result.pop("mask_logits_capture", None)
        result.pop("topo_score_capture", None)
        metadata = {
            "experiment": "original_magroad_vitb_pair",
            "model": model_name,
            "model_label": spec["label"],
            "checkpoint": str(spec["checkpoint"].resolve()),
            "checkpoint_sha256": actual_sha,
            "config": str(spec["config"]),
            "input_path": str(args.image.resolve()),
            "input_sha256": manifest["input_sha256"],
            "original_size_wh": [width, height],
            "inference_size_wh": [width, height],
            "original_per_inference_scale_xy": [1.0, 1.0],
            "patch_size": int(config.PATCH_SIZE),
            "patches_per_edge": int(config.INFER_PATCHES_PER_EDGE),
            "official_infer_batch_size": official_batch_size,
            "execution_infer_batch_size": 1,
            "road_threshold": float(config.ROAD_THRESHOLD),
            "keypoint_threshold": float(config.ITSC_THRESHOLD),
            "topo_threshold": float(config.TOPO_THRESHOLD),
            "neighbor_radius": int(config.NEIGHBOR_RADIUS),
            "training": False,
            "autograd_enabled": False,
            "inference_time_ms": elapsed * 1000.0,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else 0.0,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 2**20 if torch.cuda.is_available() else 0.0,
        }
        result["diagnostics"].update({
            "inference_time_ms": metadata["inference_time_ms"],
            "peak_allocated_vram_mb": metadata["peak_allocated_vram_mb"],
            "peak_reserved_vram_mb": metadata["peak_reserved_vram_mb"],
        })
        destination = args.output_root / model_name / args.image.stem
        regression.save_result(destination, rgb, result, metadata)
        preview(destination / "final_graph_overlay.png").save(destination / "raw_graph_preview.jpg", quality=95)
        preview(destination / "road_heatmap.png").save(destination / "road_heatmap_preview.jpg", quality=95)
        row = {**metadata, **result["diagnostics"]}
        manifest["models"].append(row)
        print(json.dumps({"completed": model_name, **row}, ensure_ascii=False), flush=True)
        del net, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_panel(args.output_root, args.image.stem)
    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
