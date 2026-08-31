"""Reproducible data, cache, probe, and metric primitives for M2A."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .data import load_manifest, resolve_path


INPUT_SIZE = 448
NATIVE_IGNORE_LABEL = 0
IGNORE_INDEX = 255
CLASS_MAPPING = {
    1: "background", 2: "building", 3: "road", 4: "water",
    5: "barren", 6: "forest", 7: "agriculture",
}
CLASS_NAMES = tuple(CLASS_MAPPING.values())
SEEDS = (20260901, 20260902, 20260903)
SPLIT_SEED = 20260901


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_domain(domain: str) -> str:
    if domain.lower() not in {"urban", "rural"}:
        raise ValueError(f"Unexpected LoveDA domain: {domain}")
    return domain.capitalize()


def loveda_rows(data_root: str | Path, m2a_root: str | Path, manifest: str | Path, split: str) -> pd.DataFrame:
    """Build external-only RGB/mask paths from immutable D3 source manifests."""
    data_root, m2a_root = Path(data_root), Path(m2a_root)
    source = pd.read_csv(manifest)
    source = source[source["dataset"] == "LoveDA"].copy()
    expected_split = "train" if split == "train" else "val"
    if not (source["source_split"] == expected_split).all():
        raise ValueError(f"Manifest contains rows outside expected LoveDA {expected_split} split")
    rows: list[dict[str, str]] = []
    for row in source.itertuples():
        domain = canonical_domain(row.domain_or_scene)
        name = Path(row.path).name
        if split == "train":
            image = resolve_path(data_root, row.path)
            image_rel = str(image.relative_to(data_root))
            semantic_split = "Train"
        else:
            image = m2a_root / "Val" / domain / "images_png" / name
            image_rel = str(image.relative_to(data_root))
            semantic_split = "Val"
        mask = m2a_root / semantic_split / domain / "masks_png" / name
        rows.append({
            "id": row.id,
            "split": split,
            "domain": row.domain_or_scene,
            "image_rel": image_rel,
            "mask_rel": str(mask.relative_to(data_root)),
            "source_sha256": row.sha256,
        })
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if frame.id.duplicated().any():
        raise ValueError("LoveDA IDs must be unique")
    return frame


def ensure_rows_readable(rows: pd.DataFrame, data_root: str | Path) -> None:
    root = Path(data_root)
    for row in rows.itertuples():
        image, mask = root / row.image_rel, root / row.mask_rel
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing M2A pair for {row.id}: {image}, {mask}")
        with Image.open(image) as rgb, Image.open(mask) as label:
            if rgb.size != label.size:
                raise ValueError(f"RGB/mask dimensions mismatch for {row.id}: {rgb.size}, {label.size}")
            if rgb.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"Expected RGB source for {row.id}, got {rgb.mode}")


def deterministic_probe_split(rows: pd.DataFrame, dev_fraction: float = 0.2, seed: int = SPLIT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be within (0, 1)")
    dev_parts = []
    for domain, group in rows.groupby("domain", sort=True):
        scores = group.id.map(lambda image_id: hashlib.sha256(f"{seed}:{domain}:{image_id}".encode()).hexdigest())
        ranked = group.assign(_score=scores).sort_values(["_score", "id"])
        count = round(len(ranked) * dev_fraction)
        dev_parts.append(ranked.iloc[:count].drop(columns="_score"))
    dev = pd.concat(dev_parts).sort_values("id").reset_index(drop=True)
    train = rows[~rows.id.isin(dev.id)].sort_values("id").reset_index(drop=True)
    if set(train.id) & set(dev.id) or len(train) + len(dev) != len(rows):
        raise AssertionError("Probe train/dev split must be a partition")
    return train, dev


def native_mask(path: str | Path, size: int | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        mask = image.convert("L")
        if size is not None:
            mask = mask.resize((size, size), Image.Resampling.NEAREST)
        values = np.asarray(mask, dtype=np.uint8).copy()
    allowed = set(CLASS_MAPPING) | {NATIVE_IGNORE_LABEL}
    unknown = set(np.unique(values)) - allowed
    if unknown:
        raise ValueError(f"Unexpected LoveDA mask values {sorted(unknown)} in {path}")
    return torch.from_numpy(values)


def train_ids(mask: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(mask, IGNORE_INDEX, dtype=torch.long)
    valid = mask != NATIVE_IGNORE_LABEL
    result[valid] = mask[valid].long() - 1
    return result


def class_pixel_counts(rows: pd.DataFrame, data_root: str | Path, size: int | None = None) -> torch.Tensor:
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.long)
    root = Path(data_root)
    for row in rows.itertuples():
        labels = train_ids(native_mask(root / row.mask_rel, size=size))
        valid = labels != IGNORE_INDEX
        counts += torch.bincount(labels[valid], minlength=len(CLASS_NAMES))
    return counts


def inverse_sqrt_class_weights(counts: torch.Tensor) -> torch.Tensor:
    if torch.any(counts <= 0):
        raise ValueError(f"All classes need probe-train pixels, got {counts.tolist()}")
    weights = counts.float().rsqrt()
    return weights / weights.mean()


def rgb_tensor(path: str | Path, size: int, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])(image)


def cache_path(cache_root: str | Path, backbone_id: str, split: str, image_id: str) -> Path:
    return Path(cache_root) / backbone_id / split / f"{image_id}.pt"


def cache_metadata(model, source_sha256: str, feature: torch.Tensor) -> dict[str, Any]:
    return {
        "schema": 1,
        "backbone_id": model.model_name,
        "model_revision": model.revision,
        "source_sha256": source_sha256,
        "input_size": INPUT_SIZE,
        "feature_shape": list(feature.shape),
        "dtype": str(feature.dtype).removeprefix("torch."),
    }


def cache_valid(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        record = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, OSError, EOFError):
        return False
    metadata = record.get("metadata", {})
    return all(metadata.get(key) == value for key, value in expected.items()) and torch.is_tensor(record.get("features"))


def write_feature_cache(path: Path, feature: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"features": feature.cpu().contiguous(), "metadata": metadata}, temporary)
    os.replace(temporary, path)


def load_feature(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)["features"]


class CachedFeatureDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_root: str | Path, cache_root: str | Path, backbone_id: str, preload: bool = True):
        self.rows, self.data_root, self.cache_root, self.backbone_id = rows.reset_index(drop=True), Path(data_root), Path(cache_root), backbone_id
        self._features: list[torch.Tensor] | None = None
        self._labels: list[torch.Tensor] | None = None
        if preload:
            self._features = []
            self._labels = []
            for row in self.rows.itertuples():
                feature = load_feature(cache_path(self.cache_root, self.backbone_id, row.split, row.id)).permute(2, 0, 1)
                label = train_ids(native_mask(self.data_root / row.mask_rel, size=INPUT_SIZE)).to(torch.uint8)
                self._features.append(feature)
                self._labels.append(label)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        if self._features is None or self._labels is None:
            feature = load_feature(cache_path(self.cache_root, self.backbone_id, row["split"], row["id"])).permute(2, 0, 1)
            labels = train_ids(native_mask(self.data_root / row["mask_rel"], size=INPUT_SIZE))
        else:
            feature, labels = self._features[index], self._labels[index].long()
        return feature, labels, row["id"]


class LinearDenseProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)) -> torch.Tensor:
        return F.interpolate(self.classifier(features), size=output_size, mode="bilinear", align_corners=False)


def only_linear_head_trainable(head: nn.Module, backbone: nn.Module) -> bool:
    return all(parameter.requires_grad for parameter in head.parameters()) and all(not parameter.requires_grad for parameter in backbone.parameters())


class SegmentationMetrics:
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        predicted = logits.argmax(dim=1).detach().cpu()
        target = labels.detach().cpu()
        valid = target != IGNORE_INDEX
        if not valid.any():
            return
        encoded = target[valid] * self.num_classes + predicted[valid]
        self.matrix += torch.bincount(encoded, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def as_dict(self) -> dict[str, Any]:
        matrix = self.matrix.float()
        tp = matrix.diag()
        fp = matrix.sum(dim=0) - tp
        fn = matrix.sum(dim=1) - tp
        total = matrix.sum()
        iou = tp / torch.clamp(tp + fp + fn, min=1)
        precision = tp / torch.clamp(tp + fp, min=1)
        recall = tp / torch.clamp(tp + fn, min=1)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
        per_class = {
            name: {"iou": float(iou[index]), "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])}
            for index, name in enumerate(CLASS_NAMES)
        }
        return {
            "miou": float(iou.mean()), "mean_f1": float(f1.mean()), "pixel_accuracy": float(tp.sum() / torch.clamp(total, min=1)),
            "per_class": per_class,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
