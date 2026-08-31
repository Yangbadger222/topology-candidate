"""Geometry-preserving RGB preprocessing for frozen overhead encoder checks."""
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import v2

from .data import pad_to_patch_multiple


@dataclass(frozen=True)
class AspectPreservedInput:
    tensor: torch.Tensor
    content_height: int
    content_width: int
    encoder_height: int
    encoder_width: int


def _nearest_patch_multiple(value: int, patch_size: int) -> int:
    lower = max(patch_size, (value // patch_size) * patch_size)
    upper = lower + patch_size
    return lower if value - lower <= upper - value else upper


def resize_long_side_preserve_aspect(image: Image.Image, long_side: int, patch_size: int) -> Image.Image:
    """Resize only by a single scale factor; no crop or anisotropic stretching."""
    width, height = image.size
    scaled_long = _nearest_patch_multiple(long_side, patch_size)
    scale = scaled_long / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, Image.Resampling.BICUBIC)


def prepare_aspect_preserved_input(image: Image.Image, long_side: int, patch_size: int, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> AspectPreservedInput:
    resized = resize_long_side_preserve_aspect(image, long_side, patch_size)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    content = transform(resized)
    tensor = pad_to_patch_multiple(content, patch_size)
    return AspectPreservedInput(tensor, content.shape[-2], content.shape[-1], tensor.shape[-2], tensor.shape[-1])


def pca_grid_to_original_aspect(pca_rgb, encoder_height: int, encoder_width: int, content_height: int, content_width: int, original_size: tuple[int, int]) -> Image.Image:
    """Remove right/bottom patch padding before returning to the source aspect ratio."""
    rendered = Image.fromarray(pca_rgb).resize((encoder_width, encoder_height), Image.Resampling.BILINEAR)
    content = rendered.crop((0, 0, content_width, content_height))
    return content.resize(original_size, Image.Resampling.BILINEAR)
