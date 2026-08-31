#!/usr/bin/env python3
import argparse
from pathlib import Path
from overhead_ssl.data import load_manifest, select_samples

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--output", default="outputs/M1A/sample_selection.csv")
parser.add_argument("--seed", type=int, default=20260831)
args = parser.parse_args()
selected = select_samples(load_manifest(args.manifest), args.seed)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(args.output, index=False)
print(f"selected {len(selected)} samples -> {args.output}")

