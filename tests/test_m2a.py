from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from overhead_ssl.m2a import (
    CLASS_MAPPING, IGNORE_INDEX, LinearDenseProbe, SEEDS, cache_valid, deterministic_probe_split,
    inverse_sqrt_class_weights, native_mask, only_linear_head_trainable, train_ids,
)


def sample_rows():
    rows = []
    for domain in ("urban", "rural"):
        for index in range(10):
            rows.append({"id": f"loveda_train_{domain}_{index}", "domain": domain, "split": "train", "image_rel": "unused.png", "mask_rel": "unused.png", "source_sha256": "x"})
    return pd.DataFrame(rows)


def test_probe_split_is_deterministic_and_domain_balanced():
    first = deterministic_probe_split(sample_rows(), seed=20260901)
    second = deterministic_probe_split(sample_rows(), seed=20260901)
    assert first[0].equals(second[0]) and first[1].equals(second[1])
    assert first[1].groupby("domain").size().to_dict() == {"rural": 2, "urban": 2}


def test_official_val_cannot_enter_probe_split():
    train, dev = deterministic_probe_split(sample_rows())
    val_ids = {"loveda_val_urban_3000"}
    assert not (val_ids & set(train.id))
    assert not (val_ids & set(dev.id))


def test_native_ignore_is_not_background(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(__import__("numpy").array([[0, 1, 3, 7]], dtype="uint8")).save(path)
    labels = train_ids(native_mask(path))
    assert labels.tolist() == [[IGNORE_INDEX, 0, 2, 6]]


def test_mask_geometry_uses_nearest_neighbor(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(__import__("numpy").array([[0, 7], [3, 1]], dtype="uint8")).save(path)
    assert set(native_mask(path, size=8).unique().tolist()) == {0, 1, 3, 7}


def test_reading_raw_mask_does_not_change_it(tmp_path):
    path = tmp_path / "official_mask.png"
    Image.fromarray(__import__("numpy").array([[0, 1]], dtype="uint8")).save(path)
    before = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    native_mask(path)
    assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == before


def test_rgb_mask_spatial_alignment_is_enforced(tmp_path):
    from overhead_ssl.m2a import ensure_rows_readable
    root = tmp_path / "data"; root.mkdir()
    Image.new("RGB", (1024, 1024)).save(root / "rgb.png")
    Image.new("L", (1024, 1024)).save(root / "mask.png")
    rows = pd.DataFrame([{"id":"a", "image_rel":"rgb.png", "mask_rel":"mask.png"}])
    ensure_rows_readable(rows, root)
    Image.new("L", (512, 512)).save(root / "mask.png")
    with pytest.raises(ValueError, match="dimensions mismatch"):
        ensure_rows_readable(rows, root)


def test_only_linear_head_is_trainable():
    torch = pytest.importorskip("torch")
    backbone = torch.nn.Linear(2, 2)
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    assert only_linear_head_trainable(LinearDenseProbe(2), backbone)


def test_same_weights_are_model_independent():
    counts = __import__("torch").tensor([10, 20, 30, 40, 50, 60, 70])
    assert __import__("torch").equal(inverse_sqrt_class_weights(counts), inverse_sqrt_class_weights(counts.clone()))


def test_same_probe_split_can_be_reused_by_both_backbones():
    train, dev = deterministic_probe_split(sample_rows())
    dino2_ids = (set(train.id), set(dev.id))
    dino3_ids = (set(train.id), set(dev.id))
    assert dino2_ids == dino3_ids


def test_m2a_uses_the_fixed_three_seeds():
    assert SEEDS == (20260901, 20260902, 20260903)


def test_cache_validation_includes_model_revision(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "feature.pt"
    torch.save({"features": torch.zeros(2, 2, 4), "metadata": {"schema": 1, "backbone_id": "encoder", "model_revision": "revision-a", "source_sha256": "source", "input_size": 448}}, path)
    base = {"schema": 1, "backbone_id": "encoder", "model_revision": "revision-a", "source_sha256": "source", "input_size": 448}
    assert cache_valid(path, base)
    assert not cache_valid(path, {**base, "model_revision": "revision-b"})


def test_preloaded_cached_dataset_does_not_reopen_files_per_item(tmp_path):
    torch = pytest.importorskip("torch")
    from overhead_ssl.m2a import CachedFeatureDataset, cache_path
    root, cache = tmp_path / "data", tmp_path / "cache"
    (root / "masks").mkdir(parents=True)
    Image.fromarray(__import__("numpy").array([[1]], dtype="uint8")).save(root / "masks" / "a.png")
    path = cache_path(cache, "encoder", "probe_train", "a")
    path.parent.mkdir(parents=True)
    torch.save({"features": torch.zeros(2, 2, 4), "metadata": {}}, path)
    rows = pd.DataFrame([{"id":"a","split":"probe_train","mask_rel":"masks/a.png"}])
    dataset = CachedFeatureDataset(rows, root, cache, "encoder", preload=True)
    path.unlink()
    features, labels, _ = dataset[0]
    assert features.shape == (4, 2, 2) and labels.shape == (448, 448)


def test_campus_target_is_not_loveda_or_naip_label_source():
    campus = "campus_target_001.png"
    assert campus not in {f"loveda_{name}" for name in CLASS_MAPPING.values()}
    assert "NAIP" not in "LoveDA official semantic labels"


def test_campus_identifier_cannot_enter_metrics_partitions():
    target_id = "campus_target_001"
    train, dev = deterministic_probe_split(sample_rows())
    val = {"loveda_val_3000"}
    assert target_id not in set(train.id) | set(dev.id) | val


def test_expected_m2a_class_count():
    assert list(CLASS_MAPPING) == [1, 2, 3, 4, 5, 6, 7]
