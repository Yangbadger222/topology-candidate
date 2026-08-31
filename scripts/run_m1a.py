#!/usr/bin/env python3
"""Run M1A setup checks and benchmark when official weights are available."""
import argparse
from pathlib import Path
import pandas as pd
from overhead_ssl.data import load_manifest, select_samples, resolve_path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root", required=True); p.add_argument("--manifest", required=True); p.add_argument("--output-dir", default="outputs/M1A"); args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    selected=select_samples(load_manifest(args.manifest)); selected.to_csv(out/"sample_selection.csv", index=False)
    missing=[]
    for path in selected.path:
        resolved=resolve_path(args.data_root, path)
        if not resolved.exists(): missing.append(str(resolved))
    print(f"selected={len(selected)} missing={len(missing)}")
    if missing: print("first_missing=", missing[0])
    target=Path("assets/external_target")
    targets=list(target.glob("*.png"))+list(target.glob("*.jpg"))+list(target.glob("*.jpeg"))
    print("external_target=provided" if targets else "external target image not provided")
    print("No model training has been performed in M1A.")

if __name__ == "__main__": main()

