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
