import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.mixed_dataset import MixedPetsDataset


def _setup_fake_dataset(tmp_path: Path, n_synth_per_class: int = 4) -> tuple[Path, Path, Path]:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    for name in ("cat.jpg", "dog.jpg"):
        Image.new("RGB", (50, 50), color="red").save(real_dir / name)

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({
        "meta": {
            "classes": ["cat", "dog"],
            "class_to_idx": {"cat": 0, "dog": 1},
        },
        "train": [
            {"image": "cat.jpg", "class": "cat", "label": 0},
            {"image": "dog.jpg", "class": "dog", "label": 1},
        ],
        "val": [],
        "test": [],
    }))

    synth_root = tmp_path / "synth"
    for breed in ("cat", "dog"):
        breed_dir = synth_root / breed
        breed_dir.mkdir(parents=True)
        for i in range(n_synth_per_class):
            Image.new("RGB", (50, 50), color="blue").save(breed_dir / f"{breed}_{i:04d}.png")
    return splits_file, real_dir, synth_root


def test_basic_len_real_plus_synth(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path, n_synth_per_class=4)
    ds = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=3, seed=0)
    assert len(ds) == 2 + 2 * 3
    assert ds.num_classes == 2


def test_capped_when_pool_smaller_than_requested(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path, n_synth_per_class=2)
    ds = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=10, seed=0)
    # only 2 synth per class available -> 4 total synth
    assert len(ds.synth_samples) == 4


def test_seed_reproducibility(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path, n_synth_per_class=8)
    ds1 = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=3, seed=42)
    ds2 = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=3, seed=42)
    ds3 = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=3, seed=7)
    assert [s["image"] for s in ds1.synth_samples] == [s["image"] for s in ds2.synth_samples]
    assert [s["image"] for s in ds1.synth_samples] != [s["image"] for s in ds3.synth_samples]


def test_getitem_returns_image_and_label(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path)
    from src.data.transforms import make_eval_transform
    ds = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=2, transform=make_eval_transform(32))
    img, label = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape == (3, 32, 32)
    assert label in {0, 1}
    img_s, label_s = ds[len(ds) - 1]
    assert isinstance(img_s, torch.Tensor)
    assert label_s in {0, 1}


def test_labels_match_class_to_idx(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path)
    ds = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=2, seed=0)
    for s in ds.synth_samples:
        assert ds.class_to_idx[s["class"]] == s["label"]


def test_manifest_filters_pool(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path, n_synth_per_class=5)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "breed,path\n"
        f"cat,{synth / 'cat' / 'cat_0000.png'}\n"
        f"cat,{synth / 'cat' / 'cat_0001.png'}\n"
        f"dog,{synth / 'dog' / 'dog_0003.png'}\n"
    )
    ds = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=10, manifest_path=manifest, seed=0)
    cat_synth = [s for s in ds.synth_samples if s["class"] == "cat"]
    dog_synth = [s for s in ds.synth_samples if s["class"] == "dog"]
    assert len(cat_synth) == 2
    assert len(dog_synth) == 1


def test_missing_breed_synth_raises(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path, n_synth_per_class=4)
    # Remove all synth for "dog" -> should raise
    for f in (synth / "dog").iterdir():
        f.unlink()
    with pytest.raises(FileNotFoundError, match="dog"):
        MixedPetsDataset(splits, real, synth, n_synthetic_per_class=2, seed=0)


def test_real_first_then_synth_ordering(tmp_path):
    splits, real, synth = _setup_fake_dataset(tmp_path, n_synth_per_class=3)
    ds = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=2, seed=0)
    # indices [0, 1] -> real, indices [2..] -> synth
    assert len(ds.real_samples) == 2
    assert len(ds.synth_samples) == 4


def test_manifest_with_missing_data_prefix(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    real = data_root / "real"
    real.mkdir(parents=True)
    for name in ("cat.jpg", "dog.jpg"):
        Image.new("RGB", (50, 50), color="red").save(real / name)
    splits = data_root / "splits.json"
    splits.write_text(json.dumps({
        "meta": {"classes": ["cat", "dog"], "class_to_idx": {"cat": 0, "dog": 1}},
        "train": [
            {"image": "cat.jpg", "class": "cat", "label": 0},
            {"image": "dog.jpg", "class": "dog", "label": 1},
        ],
        "val": [], "test": [],
    }))
    synth = data_root / "synthetic" / "variant_D"
    for breed in ("cat", "dog"):
        (synth / breed).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (50, 50), color="blue").save(synth / breed / f"{breed}_{i:04d}.png")

    manifest = data_root / "manifest.csv"
    # Paths missing the data/ prefix (legacy bug format)
    manifest.write_text(
        "breed,path\n"
        "cat,synthetic/variant_D/cat/cat_0000.png\n"
        "dog,synthetic/variant_D/dog/dog_0001.png\n"
    )
    monkeypatch.chdir(tmp_path)
    ds = MixedPetsDataset(splits, real, synth, n_synthetic_per_class=10, manifest_path=manifest, seed=0)
    for s in ds.synth_samples:
        assert Path(s["image"]).exists(), f"unresolved: {s['image']}"
