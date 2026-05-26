"""Plot acc(N_synth) sweep for D variant across all 4 models.

Reads outputs/D_{model}_*_seed*/final_results.json, parses n_synth from
run_name (or falls back to config.json), plots mean +/- std over seeds.
Adds horizontal A baseline per model.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "_aggregate" / "n_synth_sweep"
PATTERN = re.compile(r"^D_(resnet18|convnext_tiny|deit_tiny|dinov2_small)(?:_n(\d+))?_seed(\d+)$")

COLORS = {
    "resnet18": "tab:blue",
    "convnext_tiny": "tab:orange",
    "deit_tiny": "tab:green",
    "dinov2_small": "tab:red",
}


def collect() -> pd.DataFrame:
    rows = []
    for d in (ROOT / "outputs").glob("D_*"):
        m = PATTERN.match(d.name)
        if not m:
            continue
        model, n_str, seed_str = m.groups()
        # Bare D_<model>_seed{n} runs use default n=120 from generate_configs.py
        n_synth = 120 if n_str is None else int(n_str)
        final = d / "final_results.json"
        if not final.exists():
            continue
        f = json.loads(final.read_text())
        rows.append({
            "run_name": d.name,
            "model": model,
            "n_synth": n_synth,
            "seed": int(seed_str),
            "test_acc": f["test"]["accuracy"],
            "test_bal_acc": f["test"]["balanced_accuracy"],
            "test_macro_f1": f["test"]["macro_f1"],
        })
    return pd.DataFrame(rows).sort_values(["model", "n_synth", "seed"]).reset_index(drop=True)


def baseline_a_per_model() -> dict[str, tuple[float, float]]:
    out = {}
    for d in (ROOT / "outputs").glob("A_*_seed*"):
        parts = d.name.split("_seed")
        model = parts[0].replace("A_", "")
        if model not in COLORS:
            continue
        final = d / "final_results.json"
        if not final.exists():
            continue
        acc = json.loads(final.read_text())["test"]["accuracy"]
        out.setdefault(model, []).append(acc)
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in out.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    print(df.to_string(index=False))
    df.to_csv(OUT / "sweep_raw.csv", index=False)

    agg = df.groupby(["model", "n_synth"]).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    print("\nAggregated:")
    print(agg.to_string(index=False))
    agg.to_csv(OUT / "sweep_summary.csv", index=False)

    baselines = baseline_a_per_model()

    # Combined plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in sorted(agg["model"].unique()):
        sub = agg[agg["model"] == model].sort_values("n_synth")
        color = COLORS.get(model, "gray")
        ax.errorbar(sub["n_synth"], sub["test_acc_mean"], yerr=sub["test_acc_std"],
                    marker="o", capsize=4, lw=2, label=f"{model} (D)", color=color)
        if model in baselines:
            a_mean, _ = baselines[model]
            ax.axhline(a_mean, color=color, linestyle=":", lw=1, alpha=0.6,
                       label=f"{model} A baseline = {a_mean:.3f}")
    ax.set_xlabel("N synthetic per class")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Influence of N_synthetic on test accuracy (variant D, 3 seeds per point)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "n_synth_vs_acc.png", dpi=140)
    plt.close()
    print(f"\nwrote {(OUT / 'n_synth_vs_acc.png').relative_to(ROOT)}")

    # Per-model 2x2 grid (clearer per-model trend)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, model in zip(axes.flat, sorted(agg["model"].unique())):
        sub = agg[agg["model"] == model].sort_values("n_synth")
        color = COLORS.get(model, "gray")
        ax.errorbar(sub["n_synth"], sub["test_acc_mean"], yerr=sub["test_acc_std"],
                    marker="o", capsize=4, lw=2, color=color)
        if model in baselines:
            a_mean, a_std = baselines[model]
            ax.axhline(a_mean, color="gray", linestyle="--", lw=1,
                       label=f"A baseline = {a_mean:.3f}")
            ax.axhspan(a_mean - a_std, a_mean + a_std, color="gray", alpha=0.15)
        ax.set_title(model)
        ax.set_xlabel("N synthetic per class")
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "n_synth_vs_acc_per_model.png", dpi=140)
    plt.close()
    print(f"wrote {(OUT / 'n_synth_vs_acc_per_model.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
