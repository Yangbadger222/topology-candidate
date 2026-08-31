#!/usr/bin/env python3
"""Audit official LoveDA labels and make the deterministic M2A probe split."""
import argparse
from pathlib import Path

import pandas as pd
import torch

from overhead_ssl.m2a import (
    CLASS_MAPPING, CLASS_NAMES, INPUT_SIZE, NATIVE_IGNORE_LABEL, SPLIT_SEED,
    class_pixel_counts, deterministic_probe_split, ensure_rows_readable,
    inverse_sqrt_class_weights, loveda_rows, native_mask,
)


def native_counts(rows, data_root):
    counts = torch.zeros(8, dtype=torch.long)
    root = Path(data_root)
    for row in rows.itertuples():
        counts += torch.bincount(native_mask(root / row.mask_rel).flatten().long(), minlength=8)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--m2a-data-root", default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output-dir", default="outputs/M2A")
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser()
    m2a_root = Path(args.m2a_data_root).expanduser() if args.m2a_data_root else data_root / "loveda_m2a"
    train_rows = loveda_rows(data_root, m2a_root, args.train_manifest, "train")
    val_rows = loveda_rows(data_root, m2a_root, args.val_manifest, "val")
    ensure_rows_readable(train_rows, data_root)
    ensure_rows_readable(val_rows, data_root)
    probe_train, probe_dev = deterministic_probe_split(train_rows, seed=SPLIT_SEED)
    if set(val_rows.id) & (set(probe_train.id) | set(probe_dev.id)):
        raise AssertionError("Official LoveDA Val must not enter probe split")
    split_dir = Path(args.split_dir); split_dir.mkdir(parents=True, exist_ok=True)
    probe_train = probe_train.assign(split="probe_train")
    probe_dev = probe_dev.assign(split="probe_dev")
    val_rows = val_rows.assign(split="val")
    probe_train.to_csv(split_dir / "m2a_loveda_probe_train.csv", index=False)
    probe_dev.to_csv(split_dir / "m2a_loveda_probe_dev.csv", index=False)
    val_rows.to_csv(split_dir / "m2a_loveda_val.csv", index=False)
    native_train, native_val = native_counts(train_rows, data_root), native_counts(val_rows, data_root)
    resized_probe_counts = class_pixel_counts(probe_train, data_root, size=INPUT_SIZE)
    weights = inverse_sqrt_class_weights(resized_probe_counts)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "class_weights.json").write_text("{\n" + ",\n".join(
        f'  "{name}": {float(weights[index]):.10f}' for index, name in enumerate(CLASS_NAMES)
    ) + "\n}\n")
    lines = [
        "# LoveDA label audit", "", "Source mapping: official LoveDA README, Category labels section (Junjue-Wang/LoveDA).",
        "Native label `0` is official no-data and is mapped to `ignore_index=255`; it is never treated as background.", "",
        "| native id | class | Train pixels | Val pixels |", "|---:|---|---:|---:|",
        f"| 0 | no-data (ignored) | {int(native_train[0]):,} | {int(native_val[0]):,} |",
    ]
    for value, name in CLASS_MAPPING.items():
        lines.append(f"| {value} | {name} | {int(native_train[value]):,} | {int(native_val[value]):,} |")
    lines.extend(["", f"Train RGB/mask pairs audited: {len(train_rows):,}", f"Official Val RGB/mask pairs audited: {len(val_rows):,}",
                  f"Probe train/dev split seed: {SPLIT_SEED}", f"Probe train: {len(probe_train):,}; probe dev: {len(probe_dev):,}",
                  f"Class-weight pixel basis: 448x448 resized probe-train masks only ({int(resized_probe_counts.sum()):,} non-ignore pixels)."])
    (output / "label_audit.md").write_text("\n".join(lines) + "\n")
    print(f"probe_train={len(probe_train)} probe_dev={len(probe_dev)} val={len(val_rows)}")


if __name__ == "__main__":
    main()
