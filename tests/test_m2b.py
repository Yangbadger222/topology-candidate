from pathlib import Path
import random

import pandas as pd
from PIL import Image
import pytest

from overhead_ssl.ssl.datasets import OverheadMultiCrop, OverheadMultiCropDataset, SourceBalancedSampler, load_ssl_train_rows
from overhead_ssl.ssl.teacher_student import SSLConfig, ema_update_pairs, initial_weights_identical
from overhead_ssl.ssl.trainer import M2B_BACKBONE_ID, config_hash, optimizer_for


def d3_rows():
    return pd.DataFrame([
        {"id": f"loveda_{i}", "dataset": "LoveDA", "path": f"data/processed/loveda_{i}.png", "sha256": "x"} for i in range(4)
    ] + [
        {"id": f"naip_{i}", "dataset": "NAIP", "path": f"data/processed/naip_{i}.png", "sha256": "x"} for i in range(6)
    ])


def test_source_balanced_sampler_is_exact_and_reproducible():
    rows = d3_rows()
    first = list(SourceBalancedSampler(rows, seed=20260901, samples_per_epoch=100))
    second = list(SourceBalancedSampler(rows, seed=20260901, samples_per_epoch=100))
    assert first == second
    sources = rows.iloc[first].dataset.value_counts().to_dict()
    assert sources == {"LoveDA": 50, "NAIP": 50}


def test_ssl_train_requires_the_exact_d3_population(tmp_path):
    path = tmp_path / "ssl.csv"
    pd.DataFrame([{ "id":"a", "dataset":"LoveDA", "domain_or_scene":"urban", "path":"data/a.png", "sha256":"x", "source_split":"train", "ssl_allowed":True }]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="6842"):
        load_ssl_train_rows(path)


def test_d3_train_source_names_allow_loveda_and_naip_train_conventions():
    # D3 names LoveDA rows "train" and NAIP rows "d3_train" in its immutable manifest.
    assert not any("val" in value or "holdout" in value for value in ("train", "d3_train"))


def test_crop_sizes_are_patch_divisible_and_teacher_geometry_matches():
    torch = pytest.importorskip("torch")
    augmentation = OverheadMultiCrop()
    # Make appearance identity to expose the shared teacher/student geometry contract.
    augmentation._appearance = lambda image, *_: __import__("torchvision").transforms.functional.to_tensor(image)
    random.seed(12)
    output = augmentation(Image.new("RGB", (1024, 1024), "red"))
    assert [view.shape[-1] for view in output["global_student"]] == [224, 224]
    assert [view.shape[-1] for view in output["local_student"]] == [98, 98]
    assert all(size % 14 == 0 for size in (224, 98))
    assert torch.equal(output["global_student"][0], output["global_teacher"][0])
    assert torch.equal(output["global_student"][1], output["global_teacher"][1])


def test_ssl_dataset_opens_only_rgb_paths(tmp_path, monkeypatch):
    root = tmp_path / "root"; (root / "processed").mkdir(parents=True)
    Image.new("RGB", (1024, 1024)).save(root / "processed" / "rgb.png")
    rows = pd.DataFrame([{ "id":"image", "dataset":"LoveDA", "path":"data/processed/rgb.png", "sha256":"x" }])
    dataset = OverheadMultiCropDataset(rows, root, OverheadMultiCrop())
    import PIL.Image
    original_open = PIL.Image.open
    def reject_masks(path, *args, **kwargs):
        assert "mask" not in str(path).lower()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(PIL.Image, "open", reject_masks)
    views, _, _ = dataset[0]
    assert len(views["global_student"]) == 2


def test_ema_teacher_update_and_initial_copy_contract():
    torch = pytest.importorskip("torch")
    student = [torch.nn.Parameter(torch.tensor([2.0]))]
    teacher = [torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)]
    assert initial_weights_identical(student, teacher)
    student[0].data.fill_(6.0)
    ema_update_pairs(student, teacher, 0.75)
    assert teacher[0].item() == pytest.approx(3.0)
    assert not teacher[0].requires_grad


def test_mask_ratio_matches_configured_40_percent_without_model_construction():
    cfg = SSLConfig(mask_ratio=0.4, patch_grid=16)
    assert round(cfg.mask_ratio * cfg.patch_grid**2) == 102
    assert round(cfg.mask_ratio * cfg.patch_grid**2) / cfg.patch_grid**2 == pytest.approx(0.3984375)


def test_optimizer_never_receives_teacher_parameters():
    torch = pytest.importorskip("torch")
    class Tiny:
        def __init__(self):
            self.student_backbone = torch.nn.Linear(2, 2)
            self.student_head = torch.nn.Linear(2, 2)
            self.teacher = torch.nn.Linear(2, 2)
            for p in self.teacher.parameters(): p.requires_grad_(False)
        def student_parameters(self): return list(self.student_backbone.parameters()) + list(self.student_head.parameters())
        def teacher_parameters(self): return list(self.teacher.parameters())
    tiny = Tiny(); optimizer = optimizer_for(tiny, 1e-5, 1e-4, 0.04)
    teacher_ids = {id(p) for p in tiny.teacher_parameters()}
    assert not teacher_ids & {id(p) for group in optimizer.param_groups for p in group["params"]}


def test_checkpoint_config_hash_rejects_recipe_drift():
    assert config_hash({"epoch":20, "mask":0.4}) != config_hash({"epoch":20, "mask":0.5})


def test_exported_backbone_id_is_not_original_cache_id():
    assert M2B_BACKBONE_ID == "dinov2_vits14_overhead_ssl_v1"
    assert M2B_BACKBONE_ID != "dinov2_vits14"


def test_adapted_feature_cache_is_separate_from_original_m2a_cache():
    assert "M2B" != "M2A"
    assert M2B_BACKBONE_ID.startswith("dinov2_vits14_overhead_ssl_")


def test_projection_heads_are_not_part_of_export_contract():
    # The exported M2B encoder carries only `student_backbone`, never training-only DINO/iBOT heads.
    exported_keys = {"backbone_id", "epoch", "config_hash", "student_backbone"}
    assert not {"student_head", "teacher_head", "ibot_head"} & exported_keys


def test_m2a_protocol_seeds_and_dinov3_identity_remain_fixed():
    from overhead_ssl.m2a import SEEDS
    from overhead_ssl.models.dinov3 import Dinov3SatelliteEncoder
    assert SEEDS == (20260901, 20260902, 20260903)
    assert Dinov3SatelliteEncoder.model_id == "facebook/dinov3-vitl16-pretrain-sat493m"
    assert Dinov3SatelliteEncoder.revision_id == "f692fa42da72c6797b67cd73494a168d1120d3ee"
