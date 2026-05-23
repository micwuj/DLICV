import json

import pytest

from scripts.make_splits import breed_from_stem, parse_annot_file


def test_breed_from_stem_cat():
    assert breed_from_stem("Maine_Coon_100") == "Maine_Coon"
    assert breed_from_stem("Persian_5") == "Persian"


def test_breed_from_stem_dog():
    assert breed_from_stem("boxer_22") == "boxer"
    assert breed_from_stem("staffordshire_bull_terrier_3") == "staffordshire_bull_terrier"
    assert breed_from_stem("english_cocker_spaniel_17") == "english_cocker_spaniel"


def test_parse_annot_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "annot.txt"
    f.write_text(
        "# header\n"
        "# another comment\n"
        "\n"
        "Maine_Coon_1 23 1 12\n"
        "boxer_5 9 2 5\n"
    )
    stems = parse_annot_file(f)
    assert stems == ["Maine_Coon_1", "boxer_5"]


def test_make_splits_end_to_end(tmp_path, monkeypatch):
    """Symuluje minimalne dane Pets i uruchamia make_splits.main()."""
    annot_dir = tmp_path / "data" / "raw" / "oxford-iiit-pet" / "annotations"
    annot_dir.mkdir(parents=True)

    classes = [
        "Maine_Coon", "Ragdoll", "Birman", "Siamese", "Persian",
        "boxer", "american_bulldog", "staffordshire_bull_terrier",
        "english_cocker_spaniel", "english_setter",
    ]
    # 50 obrazów / klasę w trainval, 10 / klasę w test (wystarczy do sprawdzenia)
    trainval_lines = []
    test_lines = []
    for breed in classes:
        for i in range(1, 51):
            trainval_lines.append(f"{breed}_{i} 1 1 1")
        for i in range(51, 61):
            test_lines.append(f"{breed}_{i} 1 1 1")
    (annot_dir / "trainval.txt").write_text("\n".join(trainval_lines))
    (annot_dir / "test.txt").write_text("\n".join(test_lines))

    splits_dir = tmp_path / "data" / "splits"

    import scripts.make_splits as ms
    monkeypatch.setattr(ms, "ROOT", tmp_path)
    monkeypatch.setattr(ms, "DATA_RAW", tmp_path / "data" / "raw" / "oxford-iiit-pet")
    monkeypatch.setattr(ms, "ANNOT_DIR", annot_dir)
    monkeypatch.setattr(ms, "SPLITS_DIR", splits_dir)

    # Config z naszymi klasami — musi być pod tmp_path/configs/base.yaml,
    # bo skrypt resolwuje --config względem tmp_path/ROOT.
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "base.yaml").write_text(
        "data:\n  classes:\n" + "".join(f"    - {c}\n" for c in classes)
    )

    monkeypatch.setattr(
        "sys.argv",
        ["make_splits.py", "--images-per-class", "30", "--val-per-class", "10", "--seed", "0"],
    )
    ms.main()

    out = splits_dir / "pets_10cls_30perclass_seed0.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["meta"]["num_train"] == 30 * 10
    assert data["meta"]["num_val"] == 10 * 10
    assert data["meta"]["num_test"] == 10 * 10
    assert len(data["train"]) == 300
    assert len(data["val"]) == 100
    assert data["meta"]["classes"] == classes
    # Wszystkie labele w zakresie [0, 9]
    assert {x["label"] for x in data["train"]} == set(range(10))
    assert {x["label"] for x in data["val"]} == set(range(10))
    # Train i val muszą być rozłączne
    train_imgs = {x["image"] for x in data["train"]}
    val_imgs = {x["image"] for x in data["val"]}
    assert train_imgs.isdisjoint(val_imgs)


def test_make_splits_seeds_differ(tmp_path, monkeypatch):
    """Różne seedy => różne podzbiory train (przy ograniczonej puli)."""
    annot_dir = tmp_path / "data" / "raw" / "oxford-iiit-pet" / "annotations"
    annot_dir.mkdir(parents=True)
    classes = ["Maine_Coon", "Ragdoll", "Birman", "Siamese", "Persian",
               "boxer", "american_bulldog", "staffordshire_bull_terrier",
               "english_cocker_spaniel", "english_setter"]
    trainval_lines = [f"{b}_{i} 1 1 1" for b in classes for i in range(1, 51)]
    test_lines = [f"{b}_{i} 1 1 1" for b in classes for i in range(51, 61)]
    (annot_dir / "trainval.txt").write_text("\n".join(trainval_lines))
    (annot_dir / "test.txt").write_text("\n".join(test_lines))

    splits_dir = tmp_path / "data" / "splits"
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "base.yaml").write_text(
        "data:\n  classes:\n" + "".join(f"    - {c}\n" for c in classes)
    )

    import scripts.make_splits as ms
    monkeypatch.setattr(ms, "ROOT", tmp_path)
    monkeypatch.setattr(ms, "ANNOT_DIR", annot_dir)
    monkeypatch.setattr(ms, "SPLITS_DIR", splits_dir)

    results = []
    for seed in (0, 1):
        monkeypatch.setattr(
            "sys.argv",
            ["make_splits.py", "--images-per-class", "30", "--val-per-class", "10", "--seed", str(seed)],
        )
        ms.main()
        out = splits_dir / f"pets_10cls_30perclass_seed{seed}.json"
        data = json.loads(out.read_text())
        results.append({x["image"] for x in data["train"]})

    assert results[0] != results[1], "Różne seedy powinny dać różny train set"


def test_make_splits_not_enough_images_fails(tmp_path, monkeypatch):
    annot_dir = tmp_path / "data" / "raw" / "oxford-iiit-pet" / "annotations"
    annot_dir.mkdir(parents=True)
    classes = ["Maine_Coon"] + ["Ragdoll", "Birman", "Siamese", "Persian",
                                "boxer", "american_bulldog", "staffordshire_bull_terrier",
                                "english_cocker_spaniel", "english_setter"]
    # Tylko 5 obrazów dla Maine_Coon — za mało jeśli prosimy o 30
    lines = []
    for breed in classes:
        n = 5 if breed == "Maine_Coon" else 50
        for i in range(1, n + 1):
            lines.append(f"{breed}_{i} 1 1 1")
    (annot_dir / "trainval.txt").write_text("\n".join(lines))
    (annot_dir / "test.txt").write_text(f"{classes[0]}_99 1 1 1")

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "base.yaml").write_text(
        "data:\n  classes:\n" + "".join(f"    - {c}\n" for c in classes)
    )

    import scripts.make_splits as ms
    monkeypatch.setattr(ms, "ROOT", tmp_path)
    monkeypatch.setattr(ms, "ANNOT_DIR", annot_dir)
    monkeypatch.setattr(ms, "SPLITS_DIR", tmp_path / "data" / "splits")
    monkeypatch.setattr(
        "sys.argv",
        ["make_splits.py", "--images-per-class", "30", "--val-per-class", "0", "--seed", "0"],
    )
    with pytest.raises(ValueError, match="Maine_Coon"):
        ms.main()
