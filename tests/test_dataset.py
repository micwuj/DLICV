import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.pets_dataset import PetsDataset

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits" / "pets_10cls_30perclass_seed0.json"
IMAGES = ROOT / "data" / "raw" / "oxford-iiit-pet" / "images"

needs_data = pytest.mark.skipif(
    not (SPLITS.exists() and IMAGES.exists()),
    reason="Brak pobranych danych Pets — uruchom scripts/download_pets.py + scripts/make_all_splits.py",
)


@needs_data
def test_train_split_sizes():
    ds = PetsDataset(SPLITS, "train", IMAGES)
    assert len(ds) == 300
    assert ds.num_classes == 10
    assert ds.classes[0] == "Maine_Coon"


@needs_data
def test_val_and_test_splits():
    val = PetsDataset(SPLITS, "val", IMAGES)
    test = PetsDataset(SPLITS, "test", IMAGES)
    assert len(val) == 300
    assert len(test) == 988


@needs_data
def test_getitem_returns_pil_when_no_transform():
    ds = PetsDataset(SPLITS, "train", IMAGES)
    img, label = ds[0]
    assert isinstance(img, Image.Image)
    assert isinstance(label, int)
    assert 0 <= label < 10


@needs_data
def test_getitem_with_transform_returns_tensor():
    from src.data.transforms import make_eval_transform
    ds = PetsDataset(SPLITS, "train", IMAGES, transform=make_eval_transform(224))
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32


def test_invalid_split_raises(tmp_path):
    j = tmp_path / "fake.json"
    j.write_text(json.dumps({"meta": {"classes": ["a"]}, "train": []}))
    with pytest.raises(ValueError, match="nie istnieje"):
        PetsDataset(j, "val", tmp_path)


def test_synthetic_dataset_with_mock_images(tmp_path):
    """Pelny test bez pobranych Pets — robimy minimalny JSON i obrazy."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (50, 50), color="red").save(img_dir / name)

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({
        "meta": {"classes": ["x", "y"]},
        "train": [
            {"image": "a.jpg", "class": "x", "label": 0},
            {"image": "b.jpg", "class": "y", "label": 1},
        ],
        "val": [], "test": [],
    }))

    from src.data.transforms import make_eval_transform
    ds = PetsDataset(splits_file, "train", img_dir, transform=make_eval_transform(32))
    assert len(ds) == 2
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert label == 0
