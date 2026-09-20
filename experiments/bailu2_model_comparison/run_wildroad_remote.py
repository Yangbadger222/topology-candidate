#!/usr/bin/env python3
"""Run the frozen original WildRoad MaGRoad on the bailu_2 inference frame."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from magroad_graph_regression import run_graph_regression as regression


CHECKPOINT_SHA256 = "095bf4f1688d7172604ff855a22960173c869eeee20061ed61d92a07540ef8a6"
INFERENCE_SIZE = (4096, 2821)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()

    checkpoint = Path(regression.MODELS["baseline"]["checkpoint"])
    actual_sha = sha256(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {actual_sha}")

    inferencer, graph_extraction, MaGRoad, load_config = regression.import_magroad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inferencer.args.device = device
    torch.backends.cudnn.benchmark = True
    net, config = regression.load_model("baseline", device, MaGRoad, load_config)
    config.INFER_BATCH_SIZE = 1
    infer_rectangular = rectangular_infer_function()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "WildRoad_baseline",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "epoch": 22,
        "train_mode": "wildroad_baseline",
        "lora_rank": 0,
        "inference_size_wh": list(INFERENCE_SIZE),
        "patch_size": int(config.PATCH_SIZE),
        "patches_per_edge": int(config.INFER_PATCHES_PER_EDGE),
        "road_threshold": float(config.ROAD_THRESHOLD),
        "keypoint_threshold": float(config.ITSC_THRESHOLD),
        "topo_threshold": float(config.TOPO_THRESHOLD),
        "training": False,
        "images": [],
    }

    for name in args.names:
        source_path = args.input_dir / name
        with Image.open(source_path) as opened:
            original = opened.convert("RGB")
            original_size = original.size
            resized = original.resize(INFERENCE_SIZE, Image.Resampling.LANCZOS)
        rgb = np.asarray(resized)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = infer_rectangular(net, rgb, config, device, inferencer, graph_extraction)
        elapsed = time.perf_counter() - started
        result.pop("mask_logits_capture", None)
        result.pop("topo_score_capture", None)
        metadata = {
            "experiment": "bailu2_wildroad_vs_a_reconstructed",
            "model": "WildRoad_baseline",
            "image_name": source_path.stem,
            "input_path": str(source_path.resolve()),
            "input_sha256": sha256(source_path),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": actual_sha,
            "original_size_wh": list(original_size),
            "inference_size_wh": list(INFERENCE_SIZE),
            "original_per_inference_scale_xy": [
                original_size[0] / INFERENCE_SIZE[0],
                original_size[1] / INFERENCE_SIZE[1],
            ],
            "road_threshold": float(config.ROAD_THRESHOLD),
            "keypoint_threshold": float(config.ITSC_THRESHOLD),
            "topo_threshold": float(config.TOPO_THRESHOLD),
            "inference_time_ms": elapsed * 1000.0,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else 0.0,
            "peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device) / 2**20 if torch.cuda.is_available() else 0.0,
            "training": False,
            "autograd_enabled": False,
        }
        result["diagnostics"].update({key: metadata[key] for key in (
            "inference_time_ms", "peak_allocated_vram_mb", "peak_reserved_vram_mb"
        )})
        destination = args.output_root / source_path.stem
        regression.save_result(destination, rgb, result, metadata)
        Image.fromarray(regression.overlay_graph(rgb, result["nodes_rc"], result["accepted_undirected"])).resize(
            (2048, 1410), Image.Resampling.LANCZOS
        ).save(destination / "raw_graph_preview.jpg", quality=94)
        row = {**metadata, **result["diagnostics"]}
        manifest["images"].append(row)
        print(json.dumps({"completed": name, **row}, ensure_ascii=False), flush=True)

    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
