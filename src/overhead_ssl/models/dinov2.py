import hashlib
from pathlib import Path

import torch

from .base import BaseDenseEncoder, DenseFeatureOutput, extract_patch_tokens, freeze_eval, _tokens_to_grid


class Dinov2SmallEncoder(BaseDenseEncoder):
    model_name = "dinov2_vits14"
    patch_size = 14
    feature_dim = 384

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.to(device)
        freeze_eval(self)
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth"
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.revision = f"official-checkpoint-sha256:{digest.hexdigest()}"
        self.image_mean = (0.485, 0.456, 0.406)
        self.image_std = (0.229, 0.224, 0.225)

    def encode(self, x: torch.Tensor) -> DenseFeatureOutput:
        with torch.inference_mode():
            raw = self.backbone.forward_features(x)
        tokens = extract_patch_tokens(raw)
        features, gh, gw = _tokens_to_grid(tokens, x.shape[-2], x.shape[-1], self.patch_size)
        return DenseFeatureOutput(features, self.model_name, self.patch_size, features.shape[-1], x.shape[-2], x.shape[-1], gh, gw)


class AdaptedDinov2SmallEncoder(Dinov2SmallEncoder):
    """Frozen M2B-exported DINOv2-S encoder; projection heads are intentionally absent."""
    model_name = "dinov2_vits14_overhead_ssl_v1"

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        checkpoint = Path(checkpoint)
        super().__init__(device=device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("backbone_id") != self.model_name or "student_backbone" not in payload:
            raise ValueError("Expected exported M2B student backbone checkpoint")
        self.backbone.load_state_dict(payload["student_backbone"], strict=True)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.revision = f"m2b-encoder-sha256:{digest.hexdigest()}"
        freeze_eval(self)
