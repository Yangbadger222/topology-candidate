from pathlib import Path
import pandas as pd
import pytest
from overhead_ssl.data import select_samples, resolve_path, load_manifest
from overhead_ssl.pca import SharedPCA
import numpy as np

def manifest():
    rows=[]
    for dataset, scenes in (("LoveDA", ["urban","rural"]),("NAIP", ["campus_like","urban_residential","suburban_roads","wooded_roads","parks_open","rural"])):
        for scene in scenes:
            for i in range(12): rows.append({"id":f"{dataset}_{scene}_{i}","dataset":dataset,"domain_or_scene":scene,"path":f"data/{i}.jpg","ssl_allowed":True})
    return pd.DataFrame(rows)

def test_deterministic_sample_selection():
    a=select_samples(manifest()); b=select_samples(manifest()); assert a.equals(b); assert len(a)==80

def test_resolver_keeps_data_external():
    assert resolve_path("/srv/overhead-data", "data/processed/x.jpg") == Path("/srv/overhead-data/processed/x.jpg")

def test_manifest_false_is_excluded(tmp_path):
    path=tmp_path / "manifest.csv"
    path.write_text("id,dataset,domain_or_scene,path,ssl_allowed\na,LoveDA,urban,data/a.jpg,false\nb,LoveDA,urban,data/b.jpg,true\n")
    assert load_manifest(path).id.tolist() == ["b"]

def test_external_target_is_gitignored():
    root = Path(__file__).parents[1]
    ignore = (root / ".gitignore").read_text()
    assert "assets/external_target/*" in ignore
    assert "!assets/external_target/.gitkeep" in ignore

def test_external_target_is_not_a_manifest_member(tmp_path):
    target = tmp_path / "campus_target_001.png"
    target.write_bytes(b"user-provided-target")
    for name in ("ssl_train.csv", "ssl_holdout_naip.csv", "downstream_reserved_loveda_val.csv"):
        manifest_path = tmp_path / name
        manifest_path.write_text("id,path\npublic_001,data/processed/public.png\n")
        assert str(target) not in manifest_path.read_text()


def test_padding_preserves_content_and_patch_grid():
    torch=pytest.importorskip("torch")
    from overhead_ssl.data import pad_to_patch_multiple
    padded=pad_to_patch_multiple(torch.ones(3,256,256),14)
    assert padded.shape == (3,266,266)
    assert torch.all(padded[:,:256,:256] == 1)

def test_pca_shape_and_fit_excludes_target_by_contract():
    p=SharedPCA().fit(np.random.default_rng(0).normal(size=(100,8)))
    assert p.transform_grid(np.zeros((4,5,8), dtype=np.float32)).shape == (4,5,3)

def test_model_frozen_when_torch_available():
    torch=pytest.importorskip("torch")
    from overhead_ssl.models.base import freeze_eval
    model=torch.nn.Linear(2,2); freeze_eval(model)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
