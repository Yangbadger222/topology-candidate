"""Minimal single-GPU M2B teacher/student implementation using official DINOv2 components."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn

from .official import official_components


@dataclass(frozen=True)
class SSLConfig:
    prototypes: int = 65536
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_layers: int = 3
    student_temperature: float = 0.1
    center_momentum: float = 0.9
    mask_ratio: float = 0.4
    patch_grid: int = 16
    dino_weight: float = 1.0
    ibot_weight: float = 1.0


def load_official_dinov2_student() -> nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def initial_weights_identical(student_parameters, teacher_parameters) -> bool:
    return all(torch.equal(student.detach().cpu(), teacher.detach().cpu()) for student, teacher in zip(student_parameters, teacher_parameters))


@torch.no_grad()
def ema_update_pairs(student_parameters, teacher_parameters, momentum: float) -> None:
    for student, teacher in zip(student_parameters, teacher_parameters):
        teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)


class TeacherStudentDINOv2(nn.Module):
    def __init__(self, cfg: SSLConfig):
        super().__init__()
        DINOHead, DINOLoss, iBOTPatchLoss, MaskingGenerator = official_components()
        self.cfg = cfg
        self.student_backbone = load_official_dinov2_student()
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.feature_dim = int(self.student_backbone.embed_dim)
        self.student_head = DINOHead(self.feature_dim, cfg.prototypes, hidden_dim=cfg.head_hidden_dim, bottleneck_dim=cfg.head_bottleneck_dim, nlayers=cfg.head_layers)
        # torch 2.6 cannot deepcopy weight_norm modules; instantiate the same official head and copy its state.
        self.teacher_head = DINOHead(self.feature_dim, cfg.prototypes, hidden_dim=cfg.head_hidden_dim, bottleneck_dim=cfg.head_bottleneck_dim, nlayers=cfg.head_layers)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.dino_loss = DINOLoss(cfg.prototypes, student_temp=cfg.student_temperature, center_momentum=cfg.center_momentum)
        self.ibot_loss = iBOTPatchLoss(cfg.prototypes, student_temp=cfg.student_temperature, center_momentum=cfg.center_momentum)
        self.mask_generator = MaskingGenerator((cfg.patch_grid, cfg.patch_grid), num_masking_patches=round(cfg.mask_ratio * cfg.patch_grid**2))
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        for parameter in self.teacher_parameters():
            parameter.requires_grad_(False)
        self.assert_initially_identical()

    def student_parameters(self):
        return list(self.student_backbone.parameters()) + list(self.student_head.parameters())

    def teacher_parameters(self):
        return list(self.teacher_backbone.parameters()) + list(self.teacher_head.parameters())

    def assert_initially_identical(self) -> None:
        if not initial_weights_identical(self.student_parameters(), self.teacher_parameters()):
            raise AssertionError("Teacher must start as an exact student copy")

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        return self

    def make_masks(self, batch_size: int, device: torch.device) -> torch.Tensor:
        target = round(self.cfg.mask_ratio * self.cfg.patch_grid**2)
        masks = [torch.from_numpy(self.mask_generator(target)).bool() for _ in range(batch_size * 2)]
        result = torch.stack(masks).flatten(1).to(device)
        ratio = result.float().mean().item()
        expected = round(self.cfg.mask_ratio * self.cfg.patch_grid**2) / self.cfg.patch_grid**2
        if abs(ratio - expected) > 0.03:
            raise AssertionError(f"iBOT mask ratio drifted: {ratio} vs {expected}")
        return result

    def forward_losses(self, batch: dict[str, list[torch.Tensor]], teacher_temperature: float, device: torch.device):
        global_student = torch.cat(batch["global_student"], dim=0).to(device, non_blocking=True)
        global_teacher = torch.cat(batch["global_teacher"], dim=0).to(device, non_blocking=True)
        local_student = torch.cat(batch["local_student"], dim=0).to(device, non_blocking=True)
        batch_size = batch["global_student"][0].shape[0]
        masks = self.make_masks(batch_size, device)
        with torch.no_grad():
            teacher = self.teacher_backbone.forward_features(global_teacher)
            teacher_cls_logits = self.teacher_head(teacher["x_norm_clstoken"])
            teacher_patch_logits = self.teacher_head(teacher["x_norm_patchtokens"])
            teacher_cls_probs = self.dino_loss.softmax_center_teacher(teacher_cls_logits, teacher_temperature)
            teacher_patch_probs = self.ibot_loss.softmax_center_teacher(teacher_patch_logits, teacher_temperature)
            self.dino_loss.update_center(teacher_cls_logits)
            self.ibot_loss.update_center(teacher_patch_logits)
        student_global = self.student_backbone.forward_features(global_student, masks=masks)
        student_local = self.student_backbone.forward_features(local_student)
        student_global_logits = self.student_head(student_global["x_norm_clstoken"])
        student_local_logits = self.student_head(student_local["x_norm_clstoken"])
        student_patch_logits = self.student_head(student_global["x_norm_patchtokens"])
        s_global0, s_global1 = student_global_logits.chunk(2)
        t_global0, t_global1 = teacher_cls_probs.chunk(2)
        cross_global = 0.5 * (self.dino_loss([s_global0], [t_global1]) + self.dino_loss([s_global1], [t_global0]))
        s_local = student_local_logits.chunk(2)
        local = self.dino_loss(list(s_local), [t_global0, t_global1]) / 4.0
        dino = (2.0 * cross_global + 4.0 * local) / 6.0
        ibot = self.ibot_loss(student_patch_logits, teacher_patch_probs, masks)
        total = self.cfg.dino_weight * dino + self.cfg.ibot_weight * ibot
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite M2B SSL loss")
        diagnostics = {
            "student_feature_variance": feature_variance(student_global["x_norm_clstoken"]),
            "teacher_feature_variance": feature_variance(teacher["x_norm_clstoken"]),
            "student_effective_rank": effective_rank(student_global["x_norm_clstoken"]),
            "teacher_effective_rank": effective_rank(teacher["x_norm_clstoken"]),
        }
        return total, {"total_loss": float(total.detach()), "dino_global_loss": float(dino.detach()), "ibot_patch_loss": float(ibot.detach()), "dino_cross_global_loss": float(cross_global.detach()), "dino_local_loss": float(local.detach()), **diagnostics}

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        ema_update_pairs(self.student_parameters(), self.teacher_parameters(), momentum)

    def exported_encoder_state(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.student_backbone.state_dict().items()}


def feature_variance(features: torch.Tensor) -> float:
    return float(features.detach().float().var(dim=0, unbiased=False).mean().cpu())


def effective_rank(features: torch.Tensor) -> float:
    centered = features.detach().float() - features.detach().float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).cpu())
