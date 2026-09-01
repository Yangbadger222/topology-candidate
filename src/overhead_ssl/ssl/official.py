"""Explicit access to the locally cached official DINOv2 implementation."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
import torch


def official_dinov2_root() -> Path:
    root = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not (root / "dinov2/loss/dino_clstoken_loss.py").is_file():
        raise FileNotFoundError("Official facebookresearch/dinov2 cache is required for M2B")
    return root


@lru_cache(maxsize=1)
def official_components():
    root = official_dinov2_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dinov2.data.masking import MaskingGenerator
    from dinov2.layers.dino_head import DINOHead
    from dinov2.loss.dino_clstoken_loss import DINOLoss
    from dinov2.loss.ibot_patch_loss import iBOTPatchLoss
    return DINOHead, DINOLoss, iBOTPatchLoss, MaskingGenerator
