#!/usr/bin/env python3
"""Train one M2A linear dense probe; foundation features remain frozen caches."""
import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from overhead_ssl.m2a import (
    CachedFeatureDataset, CLASS_NAMES, IGNORE_INDEX, INPUT_SIZE, LinearDenseProbe,
    SegmentationMetrics, dump_json, load_feature, set_seed,
)


def load_weights(path: Path) -> torch.Tensor:
    values = json.loads(path.read_text())
    return torch.tensor([values[name] for name in CLASS_NAMES], dtype=torch.float32)


def run_epoch(head, loader, device, criterion=None, optimizer=None):
    training = optimizer is not None
    head.train(training)
    metrics = SegmentationMetrics()
    loss_total = 0.0
    valid_total = 0
    for features, labels, _ in loader:
        features, labels = features.to(device, dtype=torch.float32, non_blocking=True), labels.to(device, non_blocking=True)
        logits = head(features, output_size=labels.shape[-2:])
        valid = labels != IGNORE_INDEX
        if criterion is not None:
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = int(valid.sum())
            loss_total += float(loss.detach()) * count
            valid_total += count
        metrics.update(logits, labels)
    output = metrics.as_dict()
    output["loss"] = loss_total / max(valid_total, 1) if criterion is not None else None
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--dev-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--weights-json", required=True)
    parser.add_argument("--head-root", required=True)
    parser.add_argument("--output-dir", default="outputs/M2A/metrics")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M2A probe training requires CUDA")
    set_seed(args.seed)
    train_rows, dev_rows, val_rows = (pd.read_csv(path) for path in (args.train_csv, args.dev_csv, args.val_csv))
    if set(val_rows.id) & (set(train_rows.id) | set(dev_rows.id)):
        raise AssertionError("Official Val cannot be used for training or dev")
    train_data = CachedFeatureDataset(train_rows, args.data_root, args.cache_root, args.backbone_id)
    dev_data = CachedFeatureDataset(dev_rows, args.data_root, args.cache_root, args.backbone_id)
    val_data = CachedFeatureDataset(val_rows, args.data_root, args.cache_root, args.backbone_id)
    first_path = Path(args.cache_root) / args.backbone_id / "probe_train" / f"{train_rows.iloc[0].id}.pt"
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    feature_dim = first["features"].shape[-1]
    revision = first["metadata"]["model_revision"]
    head = LinearDenseProbe(feature_dim).cuda()
    if not all(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("Only linear probe parameters must be trainable")
    generator = torch.Generator().manual_seed(args.seed)
    # Feature and mask data are preloaded once per seed; no repeated cache I/O per epoch.
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    eval_loader = lambda dataset: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    weights = load_weights(Path(args.weights_json)).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint = Path(args.head_root) / args.backbone_id / f"seed_{args.seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    best_miou, best_epoch, wait, history = -1.0, 0, 0, []
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = run_epoch(head, train_loader, "cuda", criterion, optimizer)
        with torch.inference_mode():
            dev_metrics = run_epoch(head, eval_loader(dev_data), "cuda", criterion)
        history.append({"epoch": epoch, "train": train_metrics, "dev": dev_metrics})
        if dev_metrics["miou"] > best_miou:
            best_miou, best_epoch, wait = dev_metrics["miou"], epoch, 0
            torch.save({"state_dict": head.state_dict(), "backbone_id": args.backbone_id, "model_revision": revision, "feature_dim": feature_dim, "seed": args.seed, "best_epoch": epoch}, checkpoint)
        else:
            wait += 1
            if wait >= args.patience:
                break
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    head.load_state_dict(state["state_dict"]); head.eval()
    with torch.inference_mode():
        final_val = run_epoch(head, eval_loader(val_data), "cuda", criterion)
        best_dev = run_epoch(head, eval_loader(dev_data), "cuda", criterion)
    result = {
        "task": "Frozen Dense Semantic Linear Probe", "backbone_id": args.backbone_id, "model_revision": revision,
        "seed": args.seed, "feature_dim": feature_dim, "linear_head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "best_epoch": best_epoch, "dev": best_dev, "official_val": final_val, "history": history,
        "training_seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "checkpoint": str(checkpoint), "foundation_training_steps": 0,
    }
    output = Path(args.output_dir) / f"{args.backbone_id}_seed_{args.seed}.json"
    dump_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
