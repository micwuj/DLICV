import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
AGG_DIR = OUTPUTS_DIR / "_aggregate"


def load_all_results() -> tuple[pd.DataFrame, dict, dict]:
    rows: list[dict] = []
    confusion: dict[str, list[list[int]]] = {}
    curves: dict[str, pd.DataFrame] = {}
    for fr in sorted(OUTPUTS_DIR.glob("*/final_results.json")):
        if fr.parent.name.startswith("_"):
            continue
        d = json.loads(fr.read_text())
        run = d["run_name"]
        rows.append({
            "run_name": run,
            "variant": d["variant"],
            "model": d["model"],
            "seed": d["seed"],
            "best_epoch": d["best_epoch"],
            "best_val_acc": d["best_val_acc"],
            "test_acc": d["test"]["accuracy"],
            "test_bal_acc": d["test"]["balanced_accuracy"],
            "test_macro_f1": d["test"]["macro_f1"],
        })
        confusion[run] = d["test"]["confusion_matrix"]
        csv_path = fr.parent / "metrics_metrics.csv"
        if csv_path.exists():
            curves[run] = pd.read_csv(csv_path)
    if not rows:
        raise FileNotFoundError(f"Brak wynikow w {OUTPUTS_DIR}/")
    return pd.DataFrame(rows), confusion, curves


def write_master_table(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values(["variant", "model", "seed"]).reset_index(drop=True)
    out = AGG_DIR / "master_table.csv"
    df_sorted.to_csv(out, index=False, float_format="%.4f")
    print(f"Wrote {out.relative_to(ROOT)}  ({len(df_sorted)} runs)")


def write_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["variant", "model"]).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        test_bal_acc_mean=("test_bal_acc", "mean"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        best_val_acc_mean=("best_val_acc", "mean"),
        n_seeds=("seed", "count"),
    ).round(4)
    out = AGG_DIR / "summary_by_model_variant.csv"
    g.to_csv(out, float_format="%.4f")
    print(f"Wrote {out.relative_to(ROOT)}")
    print(g)
    return g


def per_class_accuracy(confusion: dict, df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    per_run_accs: dict[str, np.ndarray] = {}
    for run, cm in confusion.items():
        cm = np.array(cm)
        row_sums = cm.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # unika dzielenia przez 0
        per_run_accs[run] = cm.diagonal() / row_sums

    # Mean ± std per (variant, model, class)
    rows: list[dict] = []
    grouped = df.groupby(["variant", "model"])
    for (variant, model), group in grouped:
        run_names = group["run_name"].tolist()
        accs = np.array([per_run_accs[r] for r in run_names])  # (n_seeds, n_classes)
        mean = accs.mean(axis=0)
        std = accs.std(axis=0)
        for i, cls in enumerate(classes):
            rows.append({
                "variant": variant,
                "model": model,
                "class": cls,
                "acc_mean": round(mean[i], 4),
                "acc_std": round(std[i], 4),
            })
    res = pd.DataFrame(rows)
    out = AGG_DIR / "per_class_accuracy.csv"
    res.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(ROOT)}")
    return res


def plot_test_accuracy_bars(summary: pd.DataFrame) -> None:
    summary_r = summary.reset_index()
    models = sorted(summary_r["model"].unique())
    variants = sorted(summary_r["variant"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.8 / len(variants)

    for i, variant in enumerate(variants):
        sub = summary_r[summary_r["variant"] == variant].set_index("model").loc[models]
        offset = (i - (len(variants) - 1) / 2) * width
        ax.bar(
            x + offset,
            sub["test_acc_mean"],
            width=width,
            yerr=sub["test_acc_std"],
            capsize=4,
            label=f"Variant {variant}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Test accuracy by (variant, model). Errorbars = std po seedach")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = AGG_DIR / "test_accuracy_bars.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


def plot_training_curves(df: pd.DataFrame, curves: dict) -> None:
    variants = sorted(df["variant"].unique())
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(2, len(variants), figsize=(6 * len(variants), 8), sharex=True)
    if len(variants) == 1:
        axes = axes.reshape(2, 1)

    for col, variant in enumerate(variants):
        for model in models:
            runs = df[(df["variant"] == variant) & (df["model"] == model)]["run_name"].tolist()
            if not runs:
                continue
            # Align curves by epoch
            losses = np.stack([curves[r]["train_loss"].values for r in runs])
            val_accs = np.stack([curves[r]["val_acc"].values for r in runs])
            epochs = curves[runs[0]]["epoch"].values

            axes[0, col].plot(epochs, losses.mean(axis=0), label=model)
            axes[0, col].fill_between(
                epochs,
                losses.mean(axis=0) - losses.std(axis=0),
                losses.mean(axis=0) + losses.std(axis=0),
                alpha=0.2,
            )

            axes[1, col].plot(epochs, val_accs.mean(axis=0), label=model)
            axes[1, col].fill_between(
                epochs,
                val_accs.mean(axis=0) - val_accs.std(axis=0),
                val_accs.mean(axis=0) + val_accs.std(axis=0),
                alpha=0.2,
            )

        axes[0, col].set_title(f"Variant {variant}: train loss")
        axes[0, col].set_ylabel("loss")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(alpha=0.3)

        axes[1, col].set_title(f"Variant {variant}: val accuracy")
        axes[1, col].set_xlabel("epoch")
        axes[1, col].set_ylabel("val acc")
        axes[1, col].set_ylim(0, 1)
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(alpha=0.3)

    plt.tight_layout()
    out = AGG_DIR / "training_curves.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


def plot_confusion_examples(df: pd.DataFrame, confusion: dict, classes: list[str]) -> None:
    pairs = (
        df[df["seed"] == 0]
        .sort_values(["variant", "model"])[["run_name", "variant", "model"]]
        .values
    )

    n = len(pairs)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for i, (run, variant, model) in enumerate(pairs):
        cm = np.array(confusion[run])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax = axes[i]
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{variant} | {model} | seed0", fontsize=10)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=6)
        if i % ncols == 0:
            ax.set_ylabel("true")
        if i >= n - ncols:
            ax.set_xlabel("pred")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = AGG_DIR / "confusion_matrices_seed0.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Wrote {out.relative_to(ROOT)}")


def top_confused_pairs(df: pd.DataFrame, confusion: dict, classes: list[str], k: int = 5) -> None:
    rows: list[dict] = []
    grouped = df.groupby(["variant", "model"])
    for (variant, model), group in grouped:
        # Sum confusion across seeds
        cms = np.array([confusion[r] for r in group["run_name"]])
        cm_sum = cms.sum(axis=0)
        n_classes = len(classes)
        # Bierzemy off-diagonalne, sortujemy
        pairs = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    continue
                pairs.append((cm_sum[i, j], i, j))
        pairs.sort(reverse=True)
        for cnt, i, j in pairs[:k]:
            rows.append({
                "variant": variant,
                "model": model,
                "true_class": classes[i],
                "pred_class": classes[j],
                "count_summed_over_seeds": int(cnt),
            })
    res = pd.DataFrame(rows)
    out = AGG_DIR / "top_confused_pairs.csv"
    res.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Lista klas (jesli pominieta, wezmiemy z dowolnego config.json)")
    args = ap.parse_args()

    AGG_DIR.mkdir(parents=True, exist_ok=True)

    df, confusion, curves = load_all_results()
    print(f"Wczytano {len(df)} runow.")

    # Lista klas — z dowolnego config.json
    if args.classes:
        classes = args.classes
    else:
        any_cfg = next(OUTPUTS_DIR.glob("*/config.json"))
        cfg = json.loads(any_cfg.read_text())
        classes = cfg["data"]["classes"]
    print(f"Klasy ({len(classes)}): {classes}")

    write_master_table(df)
    summary = write_summary(df)
    per_class_accuracy(confusion, df, classes)
    top_confused_pairs(df, confusion, classes, k=5)
    plot_test_accuracy_bars(summary)
    plot_training_curves(df, curves)
    plot_confusion_examples(df, confusion, classes)

    print(f"\nAll artifacts in {AGG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
