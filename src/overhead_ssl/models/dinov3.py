import torch
from transformers import AutoImageProcessor, AutoModel

from .base import BaseDenseEncoder, DenseFeatureOutput, extract_patch_tokens, freeze_eval, _tokens_to_grid


class Dinov3SatelliteEncoder(BaseDenseEncoder):
    model_name = "dinov3_vitl16_sat493m"
    model_id = "facebook/dinov3-vitl16-pretrain-sat493m"
    revision_id = "f692fa42da72c6797b67cd73494a168d1120d3ee"

    def __init__(self, device: str = "cuda", revision: str | None = None):
        super().__init__()
        revision = revision or self.revision_id
        if revision != self.revision_id:
            raise ValueError(f"Only the verified official SAT493M revision is allowed: {self.revision_id}")
        # Access and the official snapshot are verified before this frozen runner starts.
        # Avoid network retries for optional processor files and use that exact cache entry.
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, revision=revision, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(self.model_id, revision=revision, local_files_only=True)
        self.to(device)
        freeze_eval(self)
        self.patch_size = int(self.backbone.config.patch_size)
        self.feature_dim = int(self.backbone.config.hidden_size)
        self.revision = getattr(self.backbone.config, "_commit_hash", revision)
        self.image_mean = tuple(float(value) for value in self.processor.image_mean)
        self.image_std = tuple(float(value) for value in self.processor.image_std)

    def encode(self, x: torch.Tensor) -> DenseFeatureOutput:
        with torch.inference_mode():
            raw = self.backbone(pixel_values=x)
        tokens = extract_patch_tokens(raw)
        features, gh, gw = _tokens_to_grid(tokens, x.shape[-2], x.shape[-1], self.patch_size)
        return DenseFeatureOutput(features, self.model_name, self.patch_size, features.shape[-1], x.shape[-2], x.shape[-1], gh, gw)
