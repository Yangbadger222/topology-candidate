"""Fixed-recipe single-GPU M2B trainer with EMA, schedules, and resumable checkpoints."""
from __future__ import annotations

from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from .datasets import OverheadMultiCrop, OverheadMultiCropDataset, SourceBalancedSampler, load_ssl_train_rows, multicrop_collate
from .teacher_student import SSLConfig, TeacherStudentDINOv2


M2B_BACKBONE_ID = "dinov2_vits14_overhead_ssl_v1"


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cosine(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def select_autocast_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_loader(rows, data_root, micro_batch, seed, epoch=0, samples_per_epoch=6842):
    transform = OverheadMultiCrop()
    dataset = OverheadMultiCropDataset(rows, data_root, transform)
    sampler = SourceBalancedSampler(rows, seed=seed, samples_per_epoch=samples_per_epoch)
    sampler.set_epoch(epoch)
    return DataLoader(dataset, batch_size=micro_batch, sampler=sampler, num_workers=2, pin_memory=True, drop_last=False, collate_fn=multicrop_collate)


def optimizer_for(model: TeacherStudentDINOv2, backbone_lr: float, head_lr: float, weight_decay: float):
    teacher_ids = {id(parameter) for parameter in model.teacher_parameters()}
    backbone = [parameter for parameter in model.student_backbone.parameters() if parameter.requires_grad]
    head = [parameter for parameter in model.student_head.parameters() if parameter.requires_grad]
    if teacher_ids & {id(parameter) for group in (backbone, head) for parameter in group}:
        raise AssertionError("Teacher parameters must never enter the optimizer")
    return torch.optim.AdamW([
        {"params": backbone, "lr": backbone_lr, "base_lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head, "lr": head_lr, "base_lr": head_lr, "weight_decay": weight_decay},
    ], betas=(0.9, 0.999))


def apply_schedule(optimizer, step: int, total_steps: int, warmup_steps: int, min_lr: float, wd_start: float, wd_end: float):
    if step < warmup_steps:
        factor = (step + 1) / max(warmup_steps, 1)
    else:
        factor = cosine(1.0, min_lr, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
    for group in optimizer.param_groups:
        # Head and backbone use their fixed base learning rates. min_lr is a shared absolute floor.
        group["lr"] = max(group["base_lr"] * factor, min_lr)
        group["weight_decay"] = cosine(wd_start, wd_end, step / max(total_steps - 1, 1))
    return optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], optimizer.param_groups[0]["weight_decay"]


def teacher_temperature(epoch: int, warmup_epochs: int, start: float, end: float) -> float:
    return end if epoch >= warmup_epochs else start + (end - start) * epoch / max(warmup_epochs, 1)


def teacher_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    return cosine(start, end, step / max(total_steps - 1, 1))


def save_checkpoint(path: Path, model: TeacherStudentDINOv2, optimizer, epoch: int, global_step: int, config: dict[str, Any], scaler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1, "backbone_id": M2B_BACKBONE_ID, "epoch": epoch, "global_step": global_step,
        "config_hash": config_hash(config), "config": config,
        "student_backbone": model.exported_encoder_state(), "teacher_backbone": model.teacher_backbone.state_dict(),
        "student_head": model.student_head.state_dict(), "teacher_head": model.teacher_head.state_dict(),
        "dino_loss": model.dino_loss.state_dict(), "ibot_loss": model.ibot_loss.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model: TeacherStudentDINOv2, optimizer, config: dict[str, Any], scaler=None) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config_hash"] != config_hash(config):
        raise ValueError("M2B checkpoint config hash does not match requested resume config")
    model.student_backbone.load_state_dict(state["student_backbone"])
    model.teacher_backbone.load_state_dict(state["teacher_backbone"])
    model.student_head.load_state_dict(state["student_head"])
    model.teacher_head.load_state_dict(state["teacher_head"])
    model.dino_loss.load_state_dict(state["dino_loss"])
    model.ibot_loss.load_state_dict(state["ibot_loss"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return int(state["epoch"]), int(state["global_step"])


def export_encoder(path: Path, model: TeacherStudentDINOv2, config: dict[str, Any], epoch: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone_id": M2B_BACKBONE_ID, "epoch": epoch, "config_hash": config_hash(config), "student_backbone": model.exported_encoder_state()}, path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_optimizer_steps(model, loader, optimizer, device, autocast_dtype, accumulation_steps, start_step, total_steps, epoch, warmup_epochs, config, max_optimizer_steps=None, scaler=None):
    model.train()
    model.to(device)
    global_step = start_step
    micro_steps = 0
    rows = []
    iterator = iter(loader)
    while True:
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        accum_records = []
        for _ in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            micro_steps += 1
            temp = teacher_temperature(epoch, config["warmup_epochs"], config["teacher_temperature_start"], config["teacher_temperature_end"])
            context = torch.autocast(device_type="cuda", dtype=autocast_dtype)
            with context:
                loss, record = model.forward_losses(batch, temp, device)
                loss = loss / accumulation_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_records.append(record)
        if not accum_records:
            break
        backbone_lr, head_lr, weight_decay = apply_schedule(optimizer, global_step, total_steps, config["warmup_steps"], config["min_lr"], config["weight_decay_start"], config["weight_decay_end"])
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.student_parameters(), config["gradient_clip_norm"])
        if scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        momentum = teacher_momentum(global_step, total_steps, config["teacher_momentum_start"], config["teacher_momentum_end"])
        model.update_teacher(momentum)
        torch.cuda.synchronize()
        averaged = {key: sum(item[key] for item in accum_records) / len(accum_records) for key in accum_records[0]}
        rows.append({"epoch": epoch, "global_step": global_step + 1, "micro_steps": len(accum_records), "step_seconds": time.perf_counter() - step_start, "backbone_lr": backbone_lr, "head_lr": head_lr, "weight_decay": weight_decay, "teacher_momentum": momentum, "teacher_temperature": temp, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, **averaged})
        global_step += 1
        if max_optimizer_steps and len(rows) >= max_optimizer_steps:
            break
    return global_step, micro_steps, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
