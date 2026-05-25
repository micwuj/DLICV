"""Copy real images per breed for LoRA fine-tuning.

For each --breed, takes first N images from the train split (seed 0) and
copies them to <output-root>/<breed>/. Idempotent: skips breeds that already
have >= --n-per-breed images.

Defaults to the 5 hardest breeds from phase 2 (Ragdoll, Birman, Maine_Coon,
staffordshire_bull_terrier, american_bulldog).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BREEDS = (
    "Ragdoll",
    "Birman",
    "Maine_Coon",
    "staffordshire_bull_terrier",
    "american_bulldog",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breeds", nargs="*", default=list(DEFAULT_BREEDS))
    ap.add_argument("--n-per-breed", type=int, default=10)
    ap.add_argument("--splits-path", default="data/splits/pets_10cls_30perclass_seed0.json")
    ap.add_argument("--images-dir", default="data/raw/oxford-iiit-pet/images")
    ap.add_argument("--output-root", default="data/lora_train")
    args = ap.parse_args()

    splits_path = ROOT / args.splits_path
    images_dir = ROOT / args.images_dir
    output_root = ROOT / args.output_root

    if not splits_path.exists():
        print(f"missing splits: {splits_path}")
        sys.exit(2)
    if not images_dir.exists():
        print(f"missing images: {images_dir} (run scripts/download_pets.py)")
        sys.exit(2)

    with open(splits_path) as f:
        split = json.load(f)
    class_to_idx = split["meta"].get("class_to_idx") or {
        c: i for i, c in enumerate(split["meta"]["classes"])
    }

    for breed in args.breeds:
        if breed not in class_to_idx:
            print(f"unknown breed: {breed!r}; skipping")
            continue
        label = class_to_idx[breed]
        out_dir = output_root / breed
        existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
        if len(existing) >= args.n_per_breed:
            print(f"{breed}: have {len(existing)} >= {args.n_per_breed}, skip")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        picked = [e for e in split["train"] if e["label"] == label][: args.n_per_breed]
        for i, entry in enumerate(picked):
            src = images_dir / entry["image"]
            dst = out_dir / f"{breed}_{i:02d}{src.suffix.lower()}"
            shutil.copy(src, dst)
        print(f"{breed}: copied {len(picked)} -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
