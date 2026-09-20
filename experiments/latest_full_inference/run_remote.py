#!/usr/bin/env python3
"""Run the frozen A_reconstructed MaGRoad graph pipeline on full-size inputs.

This is an inference-only wrapper around the formally checked graph-regression
implementation. Inputs are resized to the established XJTLU inference frame;
the original image dimensions and coordinate scale are recorded for export.
"""

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


CHECKPOINT_SHA256 = "4b5831f40cc292b3230d782280f9dc939a93d3dab85532089219fa3504d341ff"
INFERENCE_SIZE = (4096, 2821)  # width, height


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
    parser.add_argument("--road-threshold", type=float)
    parser.add_argument("--experiment", default="latest_full_inference_20260917")
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()

    checkpoint = Path(regression.MODELS["encoder_lora"]["checkpoint"])
    actual_sha = sha256(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {actual_sha}")

    inferencer, graph_extraction, MaGRoad, load_config = regression.import_magroad()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inferencer.args.device = device
    torch.backends.cudnn.benchmark = True
    net, config = regression.load_model("encoder_lora", device, MaGRoad, load_config)
    if args.road_threshold is not None:
        config.ROAD_THRESHOLD = float(args.road_threshold)
    # CPU-backed patch caches keep the 8 GiB GPU within its memory budget.
    config.INFER_BATCH_SIZE = 1
    infer_rectangular = rectangular_infer_function()

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "model": "A_reconstructed",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "epoch": 8,
        "train_mode": "loveda_lora_encoder",
        "lora_rank": 4,
        "lora_targets": ["Q", "V"],
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
        # Captures exist for the prior equivalence test and are not output artifacts.
        result.pop("mask_logits_capture", None)
        result.pop("topo_score_capture", None)
        metadata = {
            "experiment": args.experiment,
            "model": "A_reconstructed",
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
            "peak_allocated_vram_mb": (
                torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else 0.0
            ),
            "peak_reserved_vram_mb": (
                torch.cuda.max_memory_reserved(device) / 2**20 if torch.cuda.is_available() else 0.0
            ),
            "training": False,
            "autograd_enabled": False,
        }
        result["diagnostics"].update({
            "inference_time_ms": metadata["inference_time_ms"],
            "peak_allocated_vram_mb": metadata["peak_allocated_vram_mb"],
            "peak_reserved_vram_mb": metadata["peak_reserved_vram_mb"],
        })
        destination = args.output_root / source_path.stem
        regression.save_result(destination, rgb, result, metadata)
        Image.fromarray(regression.overlay_graph(rgb, result["nodes_rc"], result["accepted_undirected"])).resize(
            (2048, 1410), Image.Resampling.LANCZOS
        ).save(destination / "raw_graph_preview.jpg", quality=94)
        row = {**metadata, **result["diagnostics"]}
        run_manifest["images"].append(row)
        print(json.dumps({"completed": name, **row}, ensure_ascii=False), flush=True)

    run_manifest["status"] = "complete"
    run_manifest["finished_unix"] = time.time()
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
