from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F


def resolve_path(data_root: str | Path, manifest_path: str) -> Path:
    """Resolve D3 relative paths without mutating the source manifest."""
    root = Path(data_root).expanduser().resolve()
    rel = Path(manifest_path)
    return (root / rel.relative_to("data")).resolve() if rel.parts[:1] == ("data",) else (root / rel).resolve()


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "dataset", "domain_or_scene", "path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if "ssl_allowed" in frame:
        allowed = frame["ssl_allowed"].map(lambda value: value is True or str(value).strip().lower() == "true")
        return frame[allowed].copy()
    return frame


def select_samples(frame: pd.DataFrame, seed: int = 20260831, counts: dict[str, int] | None = None) -> pd.DataFrame:
    counts = counts or {"LoveDA/urban": 10, "LoveDA/rural": 10, "NAIP/campus_like": 10, "NAIP/urban_residential": 10, "NAIP/suburban_roads": 10, "NAIP/wooded_roads": 10, "NAIP/parks_open": 10, "NAIP/rural": 10}
    rows = []
    for key, count in counts.items():
        dataset, scene = key.split("/", 1)
        group = frame[(frame.dataset == dataset) & (frame.domain_or_scene == scene)].sort_values("id").copy()
        if len(group) < count:
            raise ValueError(f"Not enough rows for {key}: need {count}, found {len(group)}")
        scored = [(hashlib.sha256(f"{seed}:{key}:{row.id}".encode()).hexdigest(), i) for i, row in group.iterrows()]
        indices = [i for _, i in sorted(scored)[:count]]
        rows.append(group.loc[indices])
    selected = pd.concat(rows, ignore_index=True)
    selected.insert(0, "sample_id", [f"m1a_{i:03d}" for i in range(len(selected))])
    return selected[["sample_id", "id", "dataset", "domain_or_scene", "path"]]


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def pad_to_patch_multiple(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Right/bottom pad a normalized CHW or BCHW tensor without cropping RGB content."""
    h, w = image.shape[-2:]
    pad_h = (-h) % patch_size
    pad_w = (-w) % patch_size
    return F.pad(image, (0, pad_w, 0, pad_h)) if pad_h or pad_w else image
