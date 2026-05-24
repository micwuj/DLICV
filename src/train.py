import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.mixed_dataset import MixedPetsDataset
from src.data.pets_dataset import PetsDataset
from src.data.transforms import make_eval_transform, make_train_transform
from src.models.registry import build_model, num_trainable_params, trainable_parameters
from src.utils.config import load_with_overrides
from src.utils.device import device_info, get_device
from src.utils.logger import CSVLogger
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_lbls: list[int] = []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        preds = out.argmax(dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_lbls.extend(lbls.numpy().tolist())
    return compute_metrics(all_lbls, all_preds)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> float:
    model.train()
    losses: list[float] = []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                out = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def run(cfg: dict) -> dict:
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])
    print(f"Device: {device_info(device)}")

    run_name = cfg.get("run_name", "default")
    output_dir = Path(cfg.get("output_dir", "outputs")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    image_size = cfg["data"]["image_size"]
    train_tf = make_train_transform(image_size, cfg["augmentation"]["mode"])
    eval_tf = make_eval_transform(image_size)

    splits_path = cfg["data"]["splits_path"]
    images_dir = cfg["data"]["images_dir"]
    synth_root = cfg["data"].get("synthetic_root")
    if synth_root:
        train_ds = MixedPetsDataset(
            splits_path=splits_path,
            images_dir=images_dir,
            synthetic_root=synth_root,
            n_synthetic_per_class=cfg["data"]["n_synthetic_per_class"],
            transform=train_tf,
            manifest_path=cfg["data"].get("synthetic_manifest"),
            seed=cfg["seed"],
        )
        print(f"Train: real={len(train_ds.real_samples)} + synth={len(train_ds.synth_samples)}")
    else:
        train_ds = PetsDataset(splits_path, "train", images_dir, transform=train_tf)
    val_ds = PetsDataset(splits_path, "val", images_dir, transform=eval_tf)
    test_ds = PetsDataset(splits_path, "test", images_dir, transform=eval_tf)

    nw = cfg["num_workers"]
    bs = cfg["training"]["batch_size"]
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin
    )

    model = build_model(
        cfg["model"]["name"],
        num_classes=train_ds.num_classes,
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)
    print(f"Model: {cfg['model']['name']} | trainable params: {num_trainable_params(model):,}")

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["num_epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger = CSVLogger(output_dir, run_name="metrics")
    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt = output_dir / "best.ckpt"

    for epoch in range(cfg["training"]["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        scheduler.step()
        val = evaluate(model, val_loader, device)

        logger.log(
            epoch=epoch,
            train_loss=round(train_loss, 5),
            val_acc=round(val["accuracy"], 5),
            val_bal_acc=round(val["balanced_accuracy"], 5),
            val_macro_f1=round(val["macro_f1"], 5),
            lr=round(optimizer.param_groups[0]["lr"], 7),
        )

        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            best_epoch = epoch
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc},
                best_ckpt,
            )

        print(
            f"Epoch {epoch:02d}/{cfg['training']['num_epochs']}: "
            f"train_loss={train_loss:.4f} val_acc={val['accuracy']:.4f} "
            f"(best={best_val_acc:.4f} @ ep{best_epoch})"
        )

    # Final test eval — TYLKO RAZ, na koncu, z best-val checkpointu
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_loader, device)
    print(
        f"\nTest acc={test['accuracy']:.4f} | bal_acc={test['balanced_accuracy']:.4f} "
        f"| macro_F1={test['macro_f1']:.4f} (best ckpt @ ep{best_epoch})"
    )

    final = {
        "run_name": run_name,
        "variant": cfg.get("variant", "?"),
        "model": cfg["model"]["name"],
        "seed": cfg["seed"],
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test": test,
    }
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final, f, indent=2)

    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        nargs="+",
        help="Config file(s) mergowane w kolejnosci (base + override(s))",
    )
    args = ap.parse_args()
    cfg = load_with_overrides(*args.config)
    run(cfg)


if __name__ == "__main__":
    main()
