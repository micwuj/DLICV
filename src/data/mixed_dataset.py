import csv
import json
import random
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class MixedPetsDataset(Dataset):
    def __init__(
        self,
        splits_path: str | Path,
        images_dir: str | Path,
        synthetic_root: str | Path,
        n_synthetic_per_class: int,
        transform: Callable | None = None,
        manifest_path: str | Path | None = None,
        seed: int = 0,
    ):
        with open(splits_path) as f:
            data = json.load(f)
        if "train" not in data:
            raise ValueError(f"split 'train' nie istnieje w {splits_path}")
        self.meta = data["meta"]
        self.real_samples = data["train"]
        self.images_dir = Path(images_dir)
        self.synthetic_root = Path(synthetic_root)
        self.transform = transform
        self.classes = self.meta["classes"]
        self.num_classes = len(self.classes)
        self.class_to_idx = self.meta.get("class_to_idx") or {c: i for i, c in enumerate(self.classes)}

        eligible = self._load_eligible_synth(manifest_path)
        self.synth_samples = self._sample_synth(eligible, n_synthetic_per_class, seed)

    def _load_eligible_synth(self, manifest_path: str | Path | None) -> dict[str, list[Path]]:
        if manifest_path and Path(manifest_path).exists():
            by_breed: dict[str, list[Path]] = {c: [] for c in self.classes}
            with open(manifest_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    breed = row["breed"]
                    if breed in by_breed:
                        by_breed[breed].append(Path(row["path"]))
            return by_breed
        return {
            c: sorted((self.synthetic_root / c).glob(f"{c}_*.png"))
            for c in self.classes
        }

    def _sample_synth(self, eligible: dict[str, list[Path]], n: int, seed: int) -> list[dict]:
        rng = random.Random(seed)
        out: list[dict] = []
        for breed in self.classes:
            pool = list(eligible.get(breed, []))
            if not pool:
                raise FileNotFoundError(
                    f"brak syntetycznych obrazow dla klasy {breed!r} "
                    f"(synthetic_root={self.synthetic_root}, manifest sprawdzony)"
                )
            k = min(n, len(pool))
            picked = rng.sample(pool, k=k)
            label = self.class_to_idx[breed]
            for p in picked:
                out.append({"image": str(p), "class": breed, "label": label, "source": "synth"})
        return out

    def __len__(self) -> int:
        return len(self.real_samples) + len(self.synth_samples)

    def __getitem__(self, idx: int):
        if idx < len(self.real_samples):
            entry = self.real_samples[idx]
            path = self.images_dir / entry["image"]
        else:
            entry = self.synth_samples[idx - len(self.real_samples)]
            path = Path(entry["image"])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, entry["label"]
