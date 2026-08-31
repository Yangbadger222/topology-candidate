#!/usr/bin/env python3
"""Precompute validated frozen LoveDA dense features for M2A."""
import argparse
from pathlib import Path
import time

import pandas as pd
import torch

from overhead_ssl.m2a import (
    INPUT_SIZE, cache_metadata, cache_path, cache_valid, ensure_rows_readable,
    rgb_tensor, sha256_file, write_feature_cache,
)
from overhead_ssl.models import Dinov2SmallEncoder, Dinov3SatelliteEncoder


def model_for(name):
    if name == "dinov2_vits14":
        return Dinov2SmallEncoder(device="cuda")
    if name == "dinov3_sat_vitl16":
        return Dinov3SatelliteEncoder(device="cuda")
    raise ValueError(f"Unknown backbone {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["dinov2_vits14", "dinov3_sat_vitl16"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M2A feature caching requires CUDA")
    model = model_for(args.backbone)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("M2A foundation backbone must be frozen and in eval mode")
    root, cache_root = Path(args.data_root), Path(args.cache_root)
    all_rows = [(name, pd.read_csv(path)) for name, path in args.split]
    for split_name, rows in all_rows:
        if set(rows["split"]) != {split_name}:
            raise ValueError(f"CSV split field must equal cache split {split_name}")
        ensure_rows_readable(rows, root)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    generated = reused = 0
    for split_name, rows in all_rows:
        for row in rows.itertuples():
            image_path = root / row.image_rel
            actual_sha = sha256_file(image_path)
            if actual_sha != row.source_sha256:
                raise RuntimeError(f"Source SHA mismatch for {row.id}: expected {row.source_sha256}, got {actual_sha}")
            path = cache_path(cache_root, model.model_name, split_name, row.id)
            expected = {"schema": 1, "backbone_id": model.model_name, "model_revision": model.revision, "source_sha256": actual_sha, "input_size": INPUT_SIZE}
            if cache_valid(path, expected):
                reused += 1
                continue
            input_tensor = rgb_tensor(image_path, INPUT_SIZE, model.image_mean, model.image_std).unsqueeze(0).to("cuda")
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.encode(input_tensor)
            feature = output.features[0].detach().to(device="cpu", dtype=torch.float16)
            write_feature_cache(path, feature, cache_metadata(model, actual_sha, feature))
            generated += 1
    seconds = time.perf_counter() - start
    files = list((cache_root / model.model_name).rglob("*.pt"))
    bytes_used = sum(path.stat().st_size for path in files)
    report = cache_root / model.model_name / "cache_report.json"
    report.write_text(__import__("json").dumps({
        "backbone": model.model_name, "revision": model.revision, "generated": generated, "reused": reused,
        "files": len(files), "bytes": bytes_used, "seconds": seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
