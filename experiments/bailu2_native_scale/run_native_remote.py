#!/usr/bin/env python3
"""Run probability-only MaGRoad inference on bailu_2 at native resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


CHECKPOINT_SHA256 = "4b5831f40cc292b3230d782280f9dc939a93d3dab85532089219fa3504d341ff"
INPUT_SHA256 = "e8d34acb62839466e09dc832a1b75801725bfdc75facd06387d3cd9baadcce2a"
NATIVE_SIZE = (8192, 5642)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in state.items():
        while key.startswith(("module.", "model.", "net.")):
            key = key.split(".", 1)[1]
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("bailu_2.png"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("magroad_loveda/outputs/encoder/checkpoints/best_road_iou.ckpt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.input) != INPUT_SHA256:
        raise RuntimeError("Input SHA256 mismatch")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA256 mismatch")

    sys.path.insert(0, str(Path.cwd()))
    sys.argv = ["inferencer.py", "--device", "0"]
    import inferencer  # type: ignore
    from model import MaGRoad  # type: ignore
    from utils import load_config  # type: ignore

    config = load_config("magroad_loveda/config/loveda/remote_encoder.yaml")
    official = load_config("config/toponet_vitb_1024_wild_road.yaml")
    for name in (
        "PATCH_SIZE", "SAMPLE_MARGIN", "INFER_PATCHES_PER_EDGE",
        "ITSC_THRESHOLD", "ROAD_THRESHOLD", "TOPO_THRESHOLD",
    ):
        config[name] = official[name]
    config.MAGROAD_INIT_ONLY = True
    config.INFER_BATCH_SIZE = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inferencer.args.device = device
    net = MaGRoad(config)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = net.load_state_dict(canonical(raw.get("state_dict", raw)), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Checkpoint did not load strictly")
    net.eval().to(device)
    for parameter in net.parameters():
        parameter.requires_grad_(False)

    with Image.open(args.input) as opened:
        image = np.asarray(opened.convert("RGB"))
    if (image.shape[1], image.shape[0]) != NATIVE_SIZE:
        raise RuntimeError(f"Expected native size {NATIVE_SIZE}, got {(image.shape[1], image.shape[0])}")

    patches = inferencer.get_patch_info_rectangular(
        0, image.shape[0], image.shape[1], config.SAMPLE_MARGIN,
        config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE,
    )
    road_sum = torch.zeros(image.shape[:2], dtype=torch.float32, device=device)
    keypoint_sum = torch.zeros(image.shape[:2], dtype=torch.float32, device=device)
    counter = torch.zeros(image.shape[:2], dtype=torch.float32, device=device)
    starts = []
    started = time.perf_counter()
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for index, patch_info in enumerate(patches):
            _, (x0, y0), (x1, y1) = patch_info
            batch = inferencer.get_batch_img_patches(image, [patch_info]).to(device)
            _, _, scores = net.infer_masks_and_img_features(batch)
            keypoint_sum[y0:y1, x0:x1] += scores[0, :, :, 0]
            road_sum[y0:y1, x0:x1] += scores[0, :, :, 1]
            counter[y0:y1, x0:x1] += 1
            starts.append([int(x0), int(y0)])
            if (index + 1) % 25 == 0 or index + 1 == len(patches):
                print(json.dumps({"patches_complete": index + 1, "patches_total": len(patches)}), flush=True)

    if int(counter.min().item()) < 1:
        raise RuntimeError("Native tile grid left uncovered pixels")
    road = (road_sum / counter).cpu().numpy().astype(np.float32)
    keypoint = (keypoint_sum / counter).cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - started
    coverage = counter.cpu().numpy()
    del road_sum, keypoint_sum, counter

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "road_prob.npy", road)
    np.save(args.output_dir / "keypoint_prob.npy", keypoint)
    Image.fromarray(np.rint(road * 65535).astype(np.uint16)).save(args.output_dir / "road_prob.png")
    metadata = {
        "experiment": "bailu2_native_scale_probability",
        "model": "A_reconstructed",
        "input": str(args.input.resolve()),
        "input_sha256": INPUT_SHA256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "native_size_wh": list(NATIVE_SIZE),
        "inference_size_wh": list(NATIVE_SIZE),
        "patch_size": int(config.PATCH_SIZE),
        "patches_per_edge": int(config.INFER_PATCHES_PER_EDGE),
        "sample_margin": int(config.SAMPLE_MARGIN),
        "patch_count": len(patches),
        "tile_starts_xy": starts,
        "coverage_count_min": int(coverage.min()),
        "coverage_count_max": int(coverage.max()),
        "coverage_count_mean": float(coverage.mean()),
        "road_threshold": float(config.ROAD_THRESHOLD),
        "keypoint_threshold": float(config.ITSC_THRESHOLD),
        "topology_threshold": float(config.TOPO_THRESHOLD),
        "probability_only": True,
        "inference_time_seconds": elapsed,
        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else 0.0,
        "road_probability_sha256": sha256(args.output_dir / "road_prob.npy"),
        "keypoint_probability_sha256": sha256(args.output_dir / "keypoint_prob.npy"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
