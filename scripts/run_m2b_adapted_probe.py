#!/usr/bin/env python3
"""Cache and evaluate M2B's final exported DINOv2 encoder under the frozen M2A protocol."""
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA = Path("/home/badger/datasets/overhead_rgb")
CACHE = Path("/home/badger/datasets/overhead_features/M2B")
HEADS = Path("/home/badger/datasets/overhead_features/M2B/heads")
CHECKPOINT = Path("/home/badger/datasets/overhead_checkpoints/M2B/dinov2_vits14_overhead_ssl_v1/dinov2_vits14_overhead_ssl_v1_encoder_epoch20.pt")
PYTHON = Path("/home/badger/overhead-ssl/.venv/bin/python")
BACKBONE = "dinov2_vits14_overhead_ssl_v1"


def run(*arguments):
    subprocess.run([str(PYTHON), *arguments], cwd=REPO, check=True)


if not CHECKPOINT.is_file():
    raise SystemExit(f"M2B final exported encoder missing: {CHECKPOINT}")

run("scripts/cache_m2a_features.py", "--backbone", BACKBONE, "--checkpoint", str(CHECKPOINT), "--data-root", str(DATA), "--cache-root", str(CACHE),
    "--split", "probe_train", "data/splits/m2a_loveda_probe_train.csv", "--split", "probe_dev", "data/splits/m2a_loveda_probe_dev.csv", "--split", "val", "data/splits/m2a_loveda_val.csv")
for seed in (20260901, 20260902, 20260903):
    run("scripts/train_m2a_probe.py", "--backbone-id", BACKBONE, "--seed", str(seed), "--data-root", str(DATA), "--cache-root", str(CACHE),
        "--train-csv", "data/splits/m2a_loveda_probe_train.csv", "--dev-csv", "data/splits/m2a_loveda_probe_dev.csv", "--val-csv", "data/splits/m2a_loveda_val.csv",
        "--weights-json", "outputs/M2A/class_weights.json", "--head-root", str(HEADS), "--output-dir", "outputs/M2B/metrics")
