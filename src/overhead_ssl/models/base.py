from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class DenseFeatureOutput:
    features: torch.Tensor  # [B, Hf, Wf, C]
    model_name: str
    patch_size: int
    feature_dim: int
    input_height: int
    input_width: int
    grid_height: int
    grid_width: int


class BaseDenseEncoder(nn.Module):
    model_name: str
    patch_size: int
    feature_dim: int

    def encode(self, x: torch.Tensor) -> DenseFeatureOutput:
        raise NotImplementedError


def freeze_eval(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tokens_to_grid(tokens: torch.Tensor, height: int, width: int, patch: int) -> tuple[torch.Tensor, int, int]:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,C], got {tuple(tokens.shape)}")
    gh, gw = height // patch, width // patch
    expected = gh * gw
    if tokens.shape[1] == expected + 1:
        tokens = tokens[:, 1:]
    elif tokens.shape[1] != expected:
        raise ValueError(f"Cannot infer dense grid: N={tokens.shape[1]}, expected {expected} or {expected + 1}")
    return tokens.reshape(tokens.shape[0], gh, gw, tokens.shape[-1]), gh, gw


def extract_patch_tokens(raw: Any) -> torch.Tensor:
    """Handle official DINOv2/DINOv3 forward_features return conventions."""
    if isinstance(raw, dict):
        for key in ("x_norm_patchtokens", "x_prenorm", "patch_tokens", "tokens"):
            value = raw.get(key)
            if torch.is_tensor(value):
                return value
        for value in raw.values():
            if torch.is_tensor(value) and value.ndim == 3:
                return value
    if torch.is_tensor(raw):
        return raw
    raise TypeError(f"Unable to find patch tokens in {type(raw)!r}")

