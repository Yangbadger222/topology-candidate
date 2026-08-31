#!/usr/bin/env python3
"""Combine aspect-preserved external-target DINOv2/DINOv3 benchmark records."""
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/M1A/external_target")
DINO2 = ROOT / "dinov2_small/campus_target_001/aspect_preserved/benchmark.csv"
DINO3 = ROOT / "dinov3_sat/campus_target_001/aspect_preserved/benchmark.csv"
OUTPUT = ROOT / "benchmark.csv"


def main():
    dino2 = pd.read_csv(DINO2).rename(columns={
        "target_long_side": "input_scale", "encoder_input_hw": "actual_input_size",
        "dense_grid_hw": "feature_grid",
    })
    dino2.insert(0, "model", "official DINOv2 ViT-S/14")
    dino2.insert(1, "parameters", 22_056_576)
    dino2.insert(2, "pretraining_domain", "LVD-142M web imagery")
    dino3 = pd.read_csv(DINO3)
    columns = [
        "model", "parameters", "pretraining_domain", "input_scale", "actual_input_size", "feature_grid", "feature_dim",
        "mean_ms", "median_ms", "peak_allocated_mib", "peak_reserved_mib", "feature_shape", "content_size_hw", "explained_variance_ratio",
    ]
    combined = pd.concat([dino2.reindex(columns=columns), dino3.reindex(columns=columns)], ignore_index=True)
    combined.to_csv(OUTPUT, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
