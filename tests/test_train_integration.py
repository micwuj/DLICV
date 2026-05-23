import json
from pathlib import Path

from PIL import Image

from src.train import run


def _make_mini_pets(tmp_path: Path) -> dict:
    """Generuje mini-zbior: 2 klasy x 6 obrazow trainval + 4 test = 16 total."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    splits_file = tmp_path / "splits.json"

    train, val, test = [], [], []
    for cls_idx, cls in enumerate(["x", "y"]):
        for i in range(4):
            name = f"{cls}_train_{i}.jpg"
            Image.new("RGB", (64, 64), color=(255 if cls_idx == 0 else 0, 100, 100)).save(img_dir / name)
            train.append({"image": name, "class": cls, "label": cls_idx})
        for i in range(2):
            name = f"{cls}_val_{i}.jpg"
            Image.new("RGB", (64, 64), color=(255 if cls_idx == 0 else 0, 100, 100)).save(img_dir / name)
            val.append({"image": name, "class": cls, "label": cls_idx})
        for i in range(2):
            name = f"{cls}_test_{i}.jpg"
            Image.new("RGB", (64, 64), color=(255 if cls_idx == 0 else 0, 100, 100)).save(img_dir / name)
            test.append({"image": name, "class": cls, "label": cls_idx})

    splits_file.write_text(json.dumps({
        "meta": {"classes": ["x", "y"], "class_to_idx": {"x": 0, "y": 1}},
        "train": train, "val": val, "test": test,
    }))
    return {"splits_path": str(splits_file), "images_dir": str(img_dir)}


def test_train_run_end_to_end(tmp_path):
    paths = _make_mini_pets(tmp_path)
    cfg = {
        "data": {
            "num_classes": 2,
            "image_size": 64,
            "images_dir": paths["images_dir"],
            "splits_path": paths["splits_path"],
        },
        "model": {"name": "resnet18", "pretrained": False},
        "augmentation": {"mode": "none"},
        "training": {
            "batch_size": 4,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "seed": 0,
        "device": "cpu",
        "num_workers": 0,
        "output_dir": str(tmp_path / "out"),
        "run_name": "smoke",
        "variant": "A",
    }
    final = run(cfg)

    # Wynik treningu
    assert "test" in final
    assert 0 <= final["test"]["accuracy"] <= 1
    assert final["best_epoch"] in (0, 1)

    out_dir = Path(cfg["output_dir"]) / "smoke"
    # CSV per epoka
    metrics_csv = out_dir / "metrics_metrics.csv"
    assert metrics_csv.exists()
    lines = metrics_csv.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 epoki
    assert "epoch,train_loss" in lines[0]

    # Best checkpoint
    assert (out_dir / "best.ckpt").exists()
    # Final results JSON
    fr = json.loads((out_dir / "final_results.json").read_text())
    assert fr["variant"] == "A"
    assert fr["model"] == "resnet18"
    assert "test" in fr
    assert "balanced_accuracy" in fr["test"]
    assert "confusion_matrix" in fr["test"]


def test_train_run_with_classical_aug(tmp_path):
    paths = _make_mini_pets(tmp_path)
    cfg = {
        "data": {
            "num_classes": 2,
            "image_size": 64,
            "images_dir": paths["images_dir"],
            "splits_path": paths["splits_path"],
        },
        "model": {"name": "resnet18", "pretrained": False},
        "augmentation": {"mode": "classical"},
        "training": {"batch_size": 4, "num_epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0},
        "seed": 0, "device": "cpu", "num_workers": 0,
        "output_dir": str(tmp_path / "out"), "run_name": "smoke_aug", "variant": "B",
    }
    final = run(cfg)
    assert "test" in final
