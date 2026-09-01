"""RGB-only M2B data access and overhead-valid multi-crop augmentation."""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
import random

import pandas as pd
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import ColorJitter, InterpolationMode, RandomGrayscale
from torchvision.transforms import functional as TF

from ..data import load_manifest, resolve_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_ssl_train_rows(manifest: str | Path) -> pd.DataFrame:
    """Return exactly D3's approved 6842 RGB rows; no labels are read or retained."""
    frame = load_manifest(manifest)
    required = {"id", "dataset", "path", "source_split", "ssl_allowed"}
    if missing := required - set(frame.columns):
        raise ValueError(f"D3 SSL manifest missing {sorted(missing)}")
    frame = frame[frame.ssl_allowed.astype(str).str.lower() == "true"].copy()
    if len(frame) != 6842:
        raise ValueError(f"M2B requires exactly 6842 D3 SSL images, found {len(frame)}")
    counts = frame.dataset.value_counts().to_dict()
    if counts != {"NAIP": 4320, "LoveDA": 2522}:
        raise ValueError(f"Unexpected D3 dataset counts: {counts}")
    forbidden_split = frame.source_split.astype(str).str.contains("val|holdout|test", case=False, regex=True, na=False)
    forbidden = forbidden_split | frame.path.str.contains("Val", case=False, na=False) | frame.path.str.contains("holdout", case=False, na=False)
    if forbidden.any() or frame.id.str.contains("campus_target", case=False, na=False).any():
        raise ValueError("M2B SSL manifest contains a forbidden sample")
    return frame[["id", "dataset", "path", "sha256"]].sort_values("id").reset_index(drop=True)


class SourceBalancedSampler(Sampler[int]):
    """Deterministically draw an equal count from each D3 source with replacement."""
    def __init__(self, rows: pd.DataFrame, seed: int, samples_per_epoch: int | None = None):
        self.rows = rows.reset_index(drop=True)
        self.seed = seed
        self.epoch = 0
        self.samples_per_epoch = samples_per_epoch or len(rows)
        if self.samples_per_epoch % 2:
            raise ValueError("M2B samples_per_epoch must be even for exact source balancing")
        self.indices = {source: self.rows.index[self.rows.dataset == source].tolist() for source in ("LoveDA", "NAIP")}
        if not all(self.indices.values()):
            raise ValueError("Both LoveDA and NAIP are required for source-balanced M2B sampling")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        half = self.samples_per_epoch // 2
        drawn = []
        for source in ("LoveDA", "NAIP"):
            choices = torch.randint(len(self.indices[source]), (half,), generator=generator).tolist()
            drawn.extend(self.indices[source][choice] for choice in choices)
        order = torch.randperm(len(drawn), generator=generator).tolist()
        return iter([drawn[index] for index in order])

    def __len__(self) -> int:
        return self.samples_per_epoch


def _rotate(image: Image.Image, degrees: int) -> Image.Image:
    return image if degrees == 0 else image.rotate(degrees, expand=False)


class OverheadMultiCrop:
    """Two paired-geometry global views plus two local RGB-only DINO views."""
    def __init__(self, global_size=224, local_size=98, global_scale=(0.4, 1.0), local_scale=(0.05, 0.4)):
        self.global_size, self.local_size = global_size, local_size
        self.global_scale, self.local_scale = global_scale, local_scale
        if global_size % 14 or local_size % 14:
            raise ValueError("All M2B crop sizes must be divisible by DINOv2 patch size 14")
        self.jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        self.grayscale = RandomGrayscale(p=0.2)

    def _geometry(self, image: Image.Image, size: int, scale: tuple[float, float]) -> Image.Image:
        top, left, height, width = __import__("torchvision").transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=(0.75, 1.3333333333))
        crop = TF.resized_crop(image, top, left, height, width, (size, size), InterpolationMode.BICUBIC, antialias=True)
        if random.random() < 0.5:
            crop = TF.hflip(crop)
        if random.random() < 0.5:
            crop = TF.vflip(crop)
        return _rotate(crop, random.choice((0, 90, 180, 270)))

    def _appearance(self, image: Image.Image, blur_probability: float, solarize_probability: float) -> torch.Tensor:
        if random.random() < 0.8:
            image = self.jitter(image)
        image = self.grayscale(image)
        if random.random() < blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        if random.random() < solarize_probability:
            image = ImageOps.solarize(image)
        return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, image: Image.Image) -> dict[str, list[torch.Tensor]]:
        global_geometry = [self._geometry(image, self.global_size, self.global_scale) for _ in range(2)]
        # Student/teacher share geometry for every global crop; only appearance differs.
        global_student = [self._appearance(global_geometry[0], 1.0, 0.0), self._appearance(global_geometry[1], 0.1, 0.2)]
        global_teacher = [self._appearance(global_geometry[0], 0.0, 0.0), self._appearance(global_geometry[1], 0.0, 0.0)]
        local_student = [self._appearance(self._geometry(image, self.local_size, self.local_scale), 0.5, 0.0) for _ in range(2)]
        return {"global_student": global_student, "global_teacher": global_teacher, "local_student": local_student}


class OverheadMultiCropDataset(Dataset):
    """D3 SSL RGB only. This class deliberately has no mask or semantic-label field."""
    def __init__(self, rows: pd.DataFrame, data_root: str | Path, transform: OverheadMultiCrop):
        self.rows, self.root, self.transform = rows.reset_index(drop=True), Path(data_root), transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = resolve_path(self.root, row.path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), row.id, row.dataset


def multicrop_collate(batch):
    views = {key: [item[0][key] for item in batch] for key in ("global_student", "global_teacher", "local_student")}
    return {key: [torch.stack([sample[view] for sample in value]) for view in range(2)] for key, value in views.items()}
