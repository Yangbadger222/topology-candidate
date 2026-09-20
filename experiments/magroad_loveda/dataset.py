"""Native LoveDA labels: road=3, valid=1..7, no-data=0.
Source: https://github.com/Junjue-Wang/LoveDA#dataset-and-contest
Original files are never modified; RGB stays float32 0..255 BHWC for MaGRoad.
"""
from pathlib import Path
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def discover(root, split='train', domain='all', train_rgb_root=None):
    root = Path(root).expanduser()
    if domain not in ('all', 'urban', 'rural'):
        raise ValueError(domain)
    rows = []
    folders = [Path(base) / name for base, dirs, files in os.walk(root, followlinks=True) for name in dirs]
    for folder in sorted(folders):
        if not folder.is_dir() or folder.name.lower() not in ('images_png', 'images'):
            continue
        parts = [p.lower() for p in folder.parts]
        if split.lower() not in parts or not any(d in parts for d in ('urban', 'rural')):
            continue
        d = 'urban' if 'urban' in parts else 'rural'
        if domain != 'all' and d != domain:
            continue
        labels = next((folder.parent / n for n in ('masks_png', 'labels', 'masks') if (folder.parent / n).is_dir()), None)
        if labels is None:
            raise ValueError(f'Missing semantic labels: {folder}')
        for image in sorted(folder.glob('*.png')):
            mask = labels / image.name
            if not mask.exists():
                raise ValueError(f'Missing label: {mask}')
            rows.append({'id': f'{split.lower()}_{d}_{image.stem}', 'image': str(image), 'label': str(mask), 'domain': d})
    if split.lower() == 'train' and train_rgb_root:
        if rows:
            raise ValueError('Both official Train RGB and separate TRAIN_RGB_ROOT provided')
        for d in ('urban', 'rural'):
            if domain != 'all' and d != domain: continue
            rgbdir = Path(train_rgb_root) / d
            labeldirs = [f for f in folders if f.name.lower() == 'masks_png' and 'train' in [p.lower() for p in f.parts] and d in [p.lower() for p in f.parts]]
            if len(labeldirs) != 1: raise ValueError(f'Expected one Train/{d}/masks_png')
            for image in sorted(rgbdir.glob('*.png')):
                mask = labeldirs[0] / image.name
                if not mask.exists(): raise ValueError(f'Missing label: {mask}')
                rows.append({'id': f'train_{d}_{image.stem}', 'image': str(image), 'label': str(mask), 'domain': d})
    if not rows:
        raise ValueError(f'No LoveDA {split}/{domain} pairs under {root}')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate LoveDA IDs (multiple copies of data under root)')
    return rows


def road_target(label):
    label = np.asarray(label)
    if label.ndim != 2:
        raise ValueError('Semantic label must be single-channel IDs')
    return torch.from_numpy((label == 3).astype(np.float32)), torch.from_numpy(((label >= 1) & (label <= 7)).copy())


LOVEDA_CLASSES = {
    0: 'ignore', 1: 'background', 2: 'building', 3: 'road',
    4: 'water', 5: 'barren', 6: 'forest', 7: 'agriculture',
}
IGNORE_CLASS_ID = 0
BUILDING_CLASS_ID = 2
ROAD_CLASS_ID = 3


def inner_boundary(mask, width):
    """Return a width-pixel band strictly inside a binary mask."""
    if width < 1:
        raise ValueError('Boundary width must be >= 1')
    mask = torch.as_tensor(mask, dtype=torch.bool)
    inverse = (~mask).float()[None, None]
    inverse = F.pad(inverse, (width, width, width, width), value=1.)
    eroded = ~F.max_pool2d(inverse, 2 * width + 1, stride=1).bool()[0, 0]
    return mask & ~eroded


def hard_negative_targets(label, boundary_width=5, normal_weight=1., building_weight=2., boundary_weight=4., positive_weight=1.):
    label = torch.as_tensor(np.asarray(label).copy(), dtype=torch.uint8)
    valid = (label >= 1) & (label <= 7)
    road = label == ROAD_CLASS_ID
    building = label == BUILDING_CLASS_ID
    boundary = inner_boundary(building, boundary_width) & ~road
    weights = torch.full(label.shape, float(normal_weight), dtype=torch.float32)
    weights[building] = float(building_weight)
    weights[boundary] = float(boundary_weight)
    weights[road] = float(positive_weight)
    weights[~valid] = 0.
    if torch.any(boundary & road):
        raise AssertionError('Road positive was assigned to building boundary')
    return {
        'semantic': label, 'road_mask': road.float(), 'valid': valid,
        'building_mask': building, 'building_boundary': boundary,
        'loss_weight': weights,
    }


class LoveDARoad(Dataset):
    def __init__(self, root, split='train', domain='all', augment=False, limit=None, train_rgb_root=None, hard_negative=None):
        self.rows = discover(root, split, domain, train_rgb_root)
        if limit:
            self.rows = self.rows[:limit]
        self.augment = augment and split.lower() == 'train'
        self.hard_negative = hard_negative

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(row['image']).convert('RGB')
        label = np.array(Image.open(row['label']))
        if image.size != (1024, 1024) or label.shape != (1024, 1024):
            raise ValueError(f'Native 1024x1024 required: {row["id"]}')
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.95, 1.05))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(.95, 1.05))
        rgb = np.array(image)
        if self.augment:
            k = random.randrange(4)
            rgb, label = np.rot90(rgb, k), np.rot90(label, k)
            for axis in (0, 1):
                if random.random() < .5:
                    rgb, label = np.flip(rgb, axis), np.flip(label, axis)
        if self.hard_negative:
            targets = hard_negative_targets(label, **self.hard_negative)
        else:
            target, valid = road_target(label)
            targets = {'road_mask': target, 'valid': valid}
        return {'rgb': torch.from_numpy(rgb.copy()).float(), **targets, 'id': row['id']}
