"""Plot acc(N_synth) sweep for dinov2_small + D variant.

Reads outputs/D_dinov2_small_*_seed*/final_results.json, parses n_synth from
run_name (or falls back to config.json), and plots mean +/- std over seeds.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "_aggregate" / "n_synth_sweep"
PATTERN = re.compile(r"^D_dinov2_small(?:_n(\d+))?_seed(\d+)$")


def collect() -> pd.DataFrame:
    rows = []
    for d in (ROOT / "outputs").glob("D_dinov2_small_*"):
        m = PATTERN.match(d.name)
        if not m:
            continue
        n_str, seed_str = m.groups()
        # Standard D_dinov2_small_seed{0,1,2} uses n=120 (from config defaults)
        if n_str is None:
            cfg = json.loads((d / "config.json").read_text())
            n_synth = cfg["data"].get("n_synthetic_per_class", 120)
        else:
            n_synth = int(n_str)
        final = d / "final_results.json"
        if not final.exists():
            continue
        f = json.loads(final.read_text())
        rows.append({
            "run_name": d.name,
            "n_synth": n_synth,
            "seed": int(seed_str),
            "test_acc": f["test"]["accuracy"],
            "test_bal_acc": f["test"]["balanced_accuracy"],
            "test_macro_f1": f["test"]["macro_f1"],
        })
    return pd.DataFrame(rows).sort_values(["n_synth", "seed"]).reset_index(drop=True)


def baseline_a() -> tuple[float, float]:
    """Mean ± std of A_dinov2_small (no synth at all) — gives a left anchor."""
    accs = []
    for d in (ROOT / "outputs").glob("A_dinov2_small_seed*"):
        final = d / "final_results.json"
        if final.exists():
            accs.append(json.loads(final.read_text())["test"]["accuracy"])
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    print(df.to_string(index=False))
    df.to_csv(OUT / "sweep_raw.csv", index=False)

    agg = df.groupby("n_synth").agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    print("\nAggregated:")
    print(agg.to_string(index=False))
    agg.to_csv(OUT / "sweep_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(agg["n_synth"], agg["test_acc_mean"], yerr=agg["test_acc_std"],
                marker="o", capsize=4, lw=2, label="D_dinov2_small")

    a_mean, a_std = baseline_a()
    ax.axhline(a_mean, color="gray", linestyle="--", lw=1, label=f"A_dinov2_small (no synth) = {a_mean:.3f}")
    ax.axhspan(a_mean - a_std, a_mean + a_std, color="gray", alpha=0.15)

    ax.set_xlabel("N synthetic per class")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Influence of N_synthetic on test accuracy (D / dinov2_small, 3 seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = OUT / "n_synth_vs_acc.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
