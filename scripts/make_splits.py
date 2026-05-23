import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "oxford-iiit-pet"
ANNOT_DIR = DATA_RAW / "annotations"
SPLITS_DIR = ROOT / "data" / "splits"


def parse_annot_file(path: Path) -> list[str]:
    stems: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stems.append(line.split()[0])
    return stems


def breed_from_stem(stem: str) -> str:
    # 'Maine_Coon_100' -> 'Maine_Coon', 'boxer_5' -> 'boxer'
    return stem.rsplit("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--images-per-class", type=int, default=30)
    ap.add_argument("--val-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    classes: list[str] = cfg["data"]["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if not ANNOT_DIR.exists():
        raise FileNotFoundError(
            f"Brak katalogu z anotacjami: {ANNOT_DIR}\n"
            "Uruchom najpierw: python scripts/download_pets.py"
        )

    trainval_stems = parse_annot_file(ANNOT_DIR / "trainval.txt")
    test_stems = parse_annot_file(ANNOT_DIR / "test.txt")

    trainval_by_breed: dict[str, list[str]] = defaultdict(list)
    for stem in trainval_stems:
        breed = breed_from_stem(stem)
        if breed in class_to_idx:
            trainval_by_breed[breed].append(stem)

    test_by_breed: dict[str, list[str]] = defaultdict(list)
    for stem in test_stems:
        breed = breed_from_stem(stem)
        if breed in class_to_idx:
            test_by_breed[breed].append(stem)

    missing = [c for c in classes if c not in trainval_by_breed]
    if missing:
        raise ValueError(
            f"Klasy nieobecne w trainval.txt: {missing}. "
            "Sprawdź pisownię ras w configs/base.yaml."
        )

    rng = random.Random(args.seed)
    train: list[dict] = []
    val: list[dict] = []
    n_needed = args.images_per_class + args.val_per_class
    for breed in classes:
        stems = sorted(trainval_by_breed[breed])
        rng.shuffle(stems)
        n_available = len(stems)
        if n_available < n_needed:
            raise ValueError(
                f"Klasa {breed} ma tylko {n_available} obrazów w trainval, "
                f"a poprosiliśmy o {args.images_per_class} (train) + "
                f"{args.val_per_class} (val) = {n_needed}."
            )
        train_stems = stems[: args.images_per_class]
        val_stems = stems[args.images_per_class : args.images_per_class + args.val_per_class]
        for stem in train_stems:
            train.append(
                {"image": f"{stem}.jpg", "class": breed, "label": class_to_idx[breed]}
            )
        for stem in val_stems:
            val.append(
                {"image": f"{stem}.jpg", "class": breed, "label": class_to_idx[breed]}
            )

    test: list[dict] = []
    for breed in classes:
        for stem in sorted(test_by_breed[breed]):
            test.append(
                {"image": f"{stem}.jpg", "class": breed, "label": class_to_idx[breed]}
            )

    if args.out:
        out_path = Path(args.out)
    else:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = (
            SPLITS_DIR
            / f"pets_10cls_{args.images_per_class}perclass_seed{args.seed}.json"
        )

    splits = {
        "meta": {
            "dataset": "oxford_pets",
            "classes": classes,
            "class_to_idx": class_to_idx,
            "images_per_class_train": args.images_per_class,
            "images_per_class_val": args.val_per_class,
            "seed": args.seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_train": len(train),
            "num_val": len(val),
            "num_test": len(test),
        },
        "train": train,
        "val": val,
        "test": test,
    }
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote {out_path.relative_to(ROOT)}")
    print(f"  train: {len(train)} ({args.images_per_class}/class x {len(classes)} classes)")
    print(f"  val:   {len(val)} ({args.val_per_class}/class)")
    print(f"  test:  {len(test)} ({len(test) // len(classes)} avg/class)")


if __name__ == "__main__":
    main()
