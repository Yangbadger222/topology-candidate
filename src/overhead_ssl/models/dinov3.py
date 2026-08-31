import os

import torch

from .base import BaseDenseEncoder, DenseFeatureOutput, extract_patch_tokens, freeze_eval, _tokens_to_grid


class Dinov3SatelliteEncoder(BaseDenseEncoder):
    model_name = "dinov3_vitl16_sat493m"
    patch_size = 16

    def __init__(self, device: str = "cuda", repo_dir: str | None = None, weights: str | None = None):
        super().__init__()
        repo_dir = repo_dir or os.environ.get("DINOV3_REPO")
        weights = weights or os.environ.get("DINOV3_SAT493M_WEIGHTS")
        if not repo_dir or not weights:
            raise RuntimeError("DINOv3 SAT493M requires DINOV3_REPO and DINOV3_SAT493M_WEIGHTS (official access-controlled checkpoint).")
        self.backbone = torch.hub.load(repo_dir, "dinov3_vitl16", source="local", weights=weights)
        freeze_eval(self.backbone)
        self.to(device)
        self.feature_dim = int(getattr(self.backbone, "embed_dim", 1024))

    def encode(self, x: torch.Tensor) -> DenseFeatureOutput:
        with torch.inference_mode():
            raw = self.backbone.forward_features(x)
        tokens = extract_patch_tokens(raw)
        features, gh, gw = _tokens_to_grid(tokens, x.shape[-2], x.shape[-1], self.patch_size)
        return DenseFeatureOutput(features, self.model_name, self.patch_size, features.shape[-1], x.shape[-2], x.shape[-1], gh, gw)

