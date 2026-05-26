"""Generate Grad-CAM comparison grids across variants A/B/D/E.

For each (model, class), pick test images interesting w.r.t. A↔E disagreement
(extremum), then overlay CAM from all selected variants side-by-side.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.transforms import make_eval_transform
from src.interpret.gradcam import GradCAM, get_target_layer
from src.models.registry import build_model
from src.utils.device import get_device

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "_aggregate" / "gradcam"
SPLIT = ROOT / "data/splits/pets_10cls_30perclass_seed0.json"
PETS = ROOT / "data/raw/oxford-iiit-pet/images"


def load_model(model_name: str, ckpt_path: Path, num_classes: int, device) -> torch.nn.Module:
    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_all(model, paths, tf, device) -> tuple[np.ndarray, np.ndarray]:
    preds, confs = [], []
    with torch.no_grad():
        for p in paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            preds.append(int(probs.argmax()))
            confs.append(float(probs.max()))
    return np.array(preds), np.array(confs)


def pick_examples(true_label: int, preds_a: np.ndarray, preds_e: np.ndarray, n_per_case: int = 2) -> list[tuple[int, str]]:
    """Return (idx, case) tuples covering 4 case types where possible."""
    cases = {
        "A_ok_E_wrong": np.where((preds_a == true_label) & (preds_e != true_label))[0],
        "A_wrong_E_ok": np.where((preds_a != true_label) & (preds_e == true_label))[0],
        "both_ok": np.where((preds_a == true_label) & (preds_e == true_label))[0],
        "both_wrong": np.where((preds_a != true_label) & (preds_e != true_label))[0],
    }
    picks: list[tuple[int, str]] = []
    for case_name, idxs in cases.items():
        for j in idxs[:n_per_case]:
            picks.append((int(j), case_name))
    return picks


def make_grid(model_name: str, breed: str, picks, paths, classes,
              models_by_variant: dict, targets_by_variant: dict,
              tf, device, out_path: Path) -> None:
    n = len(picks)
    if n == 0:
        print(f"  {model_name}/{breed}: no examples")
        return
    variants = list(models_by_variant.keys())
    ncols = 1 + len(variants)
    fig, axes = plt.subplots(n, ncols, figsize=(3 * ncols, 3 * n), squeeze=False)
    cams = {v: GradCAM(models_by_variant[v], targets_by_variant[v]) for v in variants}
    true_label = classes.index(breed)
    for row, (idx, case) in enumerate(picks):
        p = paths[idx]
        pil = Image.open(p).convert("RGB").resize((224, 224))
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)

        axes[row, 0].imshow(pil)
        axes[row, 0].set_title(f"{p.name}\ntrue={breed} | {case}", fontsize=8)
        axes[row, 0].axis("off")
        for col, v in enumerate(variants, start=1):
            cam, pred = cams[v](x, class_idx=true_label)
            axes[row, col].imshow(pil)
            axes[row, col].imshow(cam.cpu().numpy(), alpha=0.5, cmap="jet")
            mark = "OK" if pred == true_label else f"WRONG -> {classes[pred]}"
            axes[row, col].set_title(f"{v} pred={classes[pred]}\n({mark})", fontsize=8)
            axes[row, col].axis("off")
    for c in cams.values():
        c.remove()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "convnext_tiny"])
    ap.add_argument("--variants", nargs="+", default=["A", "B", "D", "E"],
                    help="Variants to compare side-by-side. Picks driven by first vs last.")
    ap.add_argument("--breeds", nargs="+", default=["Ragdoll", "staffordshire_bull_terrier"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-case", type=int, default=1)
    args = ap.parse_args()

    device = get_device("auto")
    print(f"device: {device}")
    split = json.loads(SPLIT.read_text())
    classes = split["meta"]["classes"]
    class_to_idx = split["meta"]["class_to_idx"]
    tf = make_eval_transform(224)

    for model_name in args.models:
        ckpts = {}
        for v in args.variants:
            p = ROOT / f"outputs/{v}_{model_name}_seed{args.seed}/best.ckpt"
            if p.exists():
                ckpts[v] = p
            else:
                print(f"skip {v}/{model_name}: missing ckpt at {p}")
        if len(ckpts) < 2:
            print(f"skip {model_name}: need at least 2 variants, have {list(ckpts)}")
            continue
        models = {v: load_model(model_name, c, len(classes), device) for v, c in ckpts.items()}
        targets = {v: get_target_layer(models[v], model_name) for v in models}

        first_v, last_v = args.variants[0], args.variants[-1]
        for breed in args.breeds:
            label = class_to_idx[breed]
            test_paths = [PETS / e["image"] for e in split["test"] if e["label"] == label]
            preds_first, _ = predict_all(models[first_v], test_paths, tf, device)
            preds_last, _ = predict_all(models[last_v], test_paths, tf, device)
            picks = pick_examples(label, preds_first, preds_last, args.n_per_case)
            tag = "_vs_".join(args.variants)
            out = OUT_DIR / f"{model_name}_{breed}_{tag}.png"
            make_grid(model_name, breed, picks, test_paths, classes, models, targets,
                      tf, device, out)


if __name__ == "__main__":
    main()
