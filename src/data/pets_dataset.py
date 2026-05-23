import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class PetsDataset(Dataset):
    def __init__(
        self,
        splits_path: str | Path,
        split: str,
        images_dir: str | Path,
        transform: Callable | None = None,
    ):
        with open(splits_path) as f:
            data = json.load(f)
        if split not in data:
            raise ValueError(f"split={split!r} nie istnieje w {splits_path}")
        self.meta = data["meta"]
        self.samples = data[split]
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.classes = self.meta["classes"]
        self.num_classes = len(self.classes)
        self.split_name = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        path = self.images_dir / entry["image"]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, entry["label"]
