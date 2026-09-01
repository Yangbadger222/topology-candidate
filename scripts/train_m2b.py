#!/usr/bin/env python3
"""Run M2B-v1 smoke, runtime estimate, or fixed 20-epoch DINOv2 overhead SSL."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from overhead_ssl.ssl.datasets import load_ssl_train_rows
from overhead_ssl.ssl.teacher_student import SSLConfig, TeacherStudentDINOv2
from overhead_ssl.ssl.trainer import (
    config_hash, export_encoder, make_loader, optimizer_for, run_optimizer_steps,
    save_checkpoint, load_checkpoint, select_autocast_dtype, write_csv,
)


ROOT = Path("outputs/M2B")
CHECKPOINT_ROOT = Path("/home/badger/datasets/overhead_checkpoints/M2B/dinov2_vits14_overhead_ssl_v1")


def fixed_config(micro_batch: int) -> dict:
    accumulation = math.ceil(64 / micro_batch)
    return {
        "experiment": "M2B-v1", "seed": 20260901, "epochs": 20, "samples_per_epoch": 6842,
        "micro_batch": micro_batch, "accumulation_steps": accumulation, "effective_batch": micro_batch * accumulation,
        "global_size": 224, "local_size": 98, "global_views": 2, "local_views": 2,
        "mask_ratio": 0.4, "patch_grid": 16, "prototypes": 65536, "head_hidden_dim": 2048,
        "head_bottleneck_dim": 256, "head_layers": 3, "student_temperature": 0.1, "center_momentum": 0.9,
        "warmup_epochs": 2, "teacher_temperature_start": 0.04, "teacher_temperature_end": 0.07,
        "teacher_momentum_start": 0.996, "teacher_momentum_end": 1.0,
        "backbone_lr": 1e-5, "head_lr": 1e-4, "min_lr": 1e-6,
        "weight_decay_start": 0.04, "weight_decay_end": 0.4, "gradient_clip_norm": 3.0,
        "dino_weight": 1.0, "ibot_weight": 1.0, "labels_used_during_ssl": "NONE",
    }


def ssl_config(config: dict) -> SSLConfig:
    return SSLConfig(
        prototypes=config["prototypes"], head_hidden_dim=config["head_hidden_dim"], head_bottleneck_dim=config["head_bottleneck_dim"],
        head_layers=config["head_layers"], student_temperature=config["student_temperature"], center_momentum=config["center_momentum"],
        mask_ratio=config["mask_ratio"], patch_grid=config["patch_grid"], dino_weight=config["dino_weight"], ibot_weight=config["ibot_weight"],
    )


def total_steps(rows: int, config: dict) -> int:
    micro_steps = math.ceil(rows / config["micro_batch"])
    return math.ceil(micro_steps / config["accumulation_steps"]) * config["epochs"]


def build(rows, data_root, config, samples_per_epoch=None):
    model = TeacherStudentDINOv2(ssl_config(config)).cuda()
    optimizer = optimizer_for(model, config["backbone_lr"], config["head_lr"], config["weight_decay_start"])
    loader = make_loader(rows, data_root, config["micro_batch"], config["seed"], samples_per_epoch=samples_per_epoch or config["samples_per_epoch"])
    dtype = select_autocast_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    return model, optimizer, loader, dtype, scaler


def run_smoke(rows, data_root):
    results = []
    for candidate in (8, 4, 2, 1):
        config = fixed_config(candidate)
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            # 128 RGB samples means 16 micro-batches at candidate=8, enough for two EMA optimizer updates.
            model, optimizer, loader, dtype, scaler = build(rows, data_root, config, samples_per_epoch=128)
            steps, micro_steps, history = run_optimizer_steps(model, loader, optimizer, torch.device("cuda"), dtype, config["accumulation_steps"], 0, 2, 0, config["warmup_epochs"], {**config, "warmup_steps": 1}, max_optimizer_steps=1, scaler=scaler)
            checkpoint = CHECKPOINT_ROOT / "smoke_step1.pt"
            save_checkpoint(checkpoint, model, optimizer, 0, steps, config, scaler)
            # Resume must load the saved teacher, centers, optimizer, and scaler before another finite update.
            resumed, resumed_opt, resume_loader, resume_dtype, resume_scaler = build(rows, data_root, config, samples_per_epoch=128)
            epoch, step = load_checkpoint(checkpoint, resumed, resumed_opt, config, resume_scaler)
            final_step, _, resumed_history = run_optimizer_steps(resumed, resume_loader, resumed_opt, torch.device("cuda"), resume_dtype, config["accumulation_steps"], step, 2, epoch, config["warmup_epochs"], {**config, "warmup_steps": 1}, max_optimizer_steps=1, scaler=resume_scaler)
            save_checkpoint(CHECKPOINT_ROOT / "smoke_resume_step2.pt", resumed, resumed_opt, 0, final_step, config, resume_scaler)
            record = {"status": "pass", "config": config, "dtype": str(dtype).removeprefix("torch."), "history": history + resumed_history, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "checkpoint_resume": True}
            ROOT.mkdir(parents=True, exist_ok=True); (ROOT / "smoke_test.json").write_text(json.dumps(record, indent=2) + "\n")
            print(ROOT / "smoke_test.json")
            return record
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"micro_batch": candidate, "status": "OOM"})
    raise RuntimeError(f"M2B smoke test OOM at all candidates: {results}")


def run_estimate(rows, data_root, config):
    # 100 stable micro-batches, expressed as optimizer updates under the selected accumulation rule.
    estimate_updates = math.ceil(100 / config["accumulation_steps"])
    model, optimizer, loader, dtype, scaler = build(rows, data_root, config, samples_per_epoch=100 * config["micro_batch"])
    actual_total = estimate_updates
    start = time.perf_counter()
    _, micro_steps, history = run_optimizer_steps(model, loader, optimizer, torch.device("cuda"), dtype, config["accumulation_steps"], 0, actual_total, 0, config["warmup_epochs"], {**config, "warmup_steps": max(1, math.ceil(2 * actual_total / 20))}, max_optimizer_steps=estimate_updates, scaler=scaler)
    seconds_per_micro = (time.perf_counter() - start) / micro_steps
    epoch_micro_steps = math.ceil(config["samples_per_epoch"] / config["micro_batch"])
    epoch_seconds = seconds_per_micro * epoch_micro_steps
    total_seconds = epoch_seconds * config["epochs"]
    text = ["# M2B runtime estimate", "", f"dtype: {str(dtype).removeprefix('torch.')}", f"micro batch: {config['micro_batch']}", f"gradient accumulation: {config['accumulation_steps']}", f"effective batch: {config['effective_batch']}", f"stable micro-steps measured: {micro_steps}", f"optimizer steps measured: {len(history)}", f"seconds per micro-step: {seconds_per_micro:.4f}", f"estimated seconds per optimizer step: {seconds_per_micro * config['accumulation_steps']:.4f}", f"estimated epoch seconds: {epoch_seconds:.1f}", f"estimated 20 epoch seconds: {total_seconds:.1f}", f"estimated 20 epoch hours: {total_seconds / 3600:.2f}", f"peak allocated MiB: {torch.cuda.max_memory_allocated() / 2**20:.1f}", f"peak reserved MiB: {torch.cuda.max_memory_reserved() / 2**20:.1f}"]
    ROOT.mkdir(parents=True, exist_ok=True); (ROOT / "runtime_estimate.md").write_text("\n".join(text) + "\n")
    return total_seconds


def run_full(rows, data_root, config, resume: Path | None = None):
    model, optimizer, _, dtype, scaler = build(rows, data_root, config)
    updates_per_epoch = math.ceil(math.ceil(len(rows) / config["micro_batch"]) / config["accumulation_steps"])
    total = updates_per_epoch * config["epochs"]
    config = {**config, "warmup_steps": updates_per_epoch * config["warmup_epochs"], "total_optimizer_steps": total, "dtype": str(dtype).removeprefix("torch.")}
    start_epoch, global_step = 0, 0
    if resume:
        start_epoch, global_step = load_checkpoint(resume, model, optimizer, config, scaler)
    else:
        save_checkpoint(CHECKPOINT_ROOT / "epoch_000.pt", model, optimizer, 0, 0, config, scaler)
    history = []
    run_start = time.perf_counter()
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        loader = make_loader(rows, data_root, config["micro_batch"], config["seed"], epoch=epoch, samples_per_epoch=config["samples_per_epoch"])
        torch.cuda.reset_peak_memory_stats()
        global_step, _, epoch_history = run_optimizer_steps(model, loader, optimizer, torch.device("cuda"), dtype, config["accumulation_steps"], global_step, total, epoch, config["warmup_epochs"], config, scaler=scaler)
        history.extend(epoch_history)
        if epoch in (5, 10, 15, 20):
            save_checkpoint(CHECKPOINT_ROOT / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, global_step, config, scaler)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "training_summary.csv", history)
    encoder_path = CHECKPOINT_ROOT / "dinov2_vits14_overhead_ssl_v1_encoder_epoch20.pt"
    encoder_sha = export_encoder(encoder_path, model, config, config["epochs"])
    summary = {"config": config, "config_hash": config_hash(config), "duration_seconds": time.perf_counter() - run_start, "final_checkpoint": str(CHECKPOINT_ROOT / "epoch_020.pt"), "encoder_checkpoint": str(encoder_path), "encoder_sha256": encoder_sha, "final": history[-1], "collapse_pass": all(row["student_feature_variance"] > 1e-6 and row["teacher_feature_variance"] > 1e-6 and row["student_effective_rank"] > 1.0 and row["teacher_effective_rank"] > 1.0 for row in history)}
    (ROOT / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(ROOT / "training_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "estimate", "full"), required=True)
    parser.add_argument("--data-root", default="/home/badger/datasets/overhead_rgb")
    parser.add_argument("--manifest", default="/home/badger/datasets/overhead_rgb/splits/ssl_train.csv")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M2B requires CUDA")
    rows = load_ssl_train_rows(args.manifest)
    if args.mode == "smoke":
        run_smoke(rows, args.data_root); return
    smoke = json.loads((ROOT / "smoke_test.json").read_text())
    micro = args.micro_batch or smoke["config"]["micro_batch"]
    config = fixed_config(micro)
    if args.mode == "estimate":
        estimated = run_estimate(rows, args.data_root, config)
        if estimated > 12 * 3600:
            raise SystemExit("M2B estimated runtime exceeds 12 hours; full training not started")
    else:
        estimate = (ROOT / "runtime_estimate.md").read_text()
        hours = float(next(line.split(": ", 1)[1] for line in estimate.splitlines() if line.startswith("estimated 20 epoch hours")))
        if hours > 12:
            raise SystemExit("M2B estimate exceeds 12 hours; full training not started")
        run_full(rows, args.data_root, config, args.resume)


if __name__ == "__main__":
    main()
