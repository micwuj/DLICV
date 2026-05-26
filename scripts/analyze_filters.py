"""Analyze filtering methods on existing filter_scores.csv (no retraining).

Compares: CLIP-only, DINOv2-only, joint AND, joint OR, percentile-based.
Outputs scatter, histograms, correlation, and per-method retention table.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "_aggregate" / "filters"

FLOOR_CLIP = 0.65
FLOOR_DINO = 0.55
DROP_PCT = 0.25


def methods(df: pd.DataFrame) -> pd.DataFrame:
    c = df["clip_sim"].to_numpy()
    d = df["dinov2_sim"].to_numpy()
    combined = 0.5 * c + 0.5 * d
    pct_thr = np.quantile(combined, DROP_PCT)
    mask_pct = combined >= pct_thr

    out = df.copy()
    out["kept_clip_only"] = c >= FLOOR_CLIP
    out["kept_dino_only"] = d >= FLOOR_DINO
    out["kept_joint_and"] = (c >= FLOOR_CLIP) & (d >= FLOOR_DINO) & mask_pct
    out["kept_joint_or"] = (c >= FLOOR_CLIP) | (d >= FLOOR_DINO)
    out["kept_pct_only"] = mask_pct
    return out


def retention_table(df: pd.DataFrame, source: str) -> pd.DataFrame:
    methods_cols = [
        "kept_clip_only", "kept_dino_only", "kept_joint_and",
        "kept_joint_or", "kept_pct_only", "kept",
    ]
    rows = []
    n_total = len(df)
    for m in methods_cols:
        n_kept = int(df[m].sum())
        rows.append({
            "source": source,
            "method": m.replace("kept_", "").replace("kept", "current_pipeline"),
            "kept": n_kept,
            "dropped": n_total - n_kept,
            "kept_pct": 100 * n_kept / n_total,
        })
    return pd.DataFrame(rows)


def per_class_retention(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for breed, group in df.groupby("breed"):
        n_total = len(group)
        rows.append({
            "source": source,
            "breed": breed,
            "n_total": n_total,
            "clip_only": int(group["kept_clip_only"].sum()),
            "dino_only": int(group["kept_dino_only"].sum()),
            "joint_and_pipeline": int(group["kept"].sum()),
            "joint_or": int(group["kept_joint_or"].sum()),
            "clip_mean": group["clip_sim"].mean(),
            "dino_mean": group["dinov2_sim"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["source", "breed"])


def plot_scatter(df: pd.DataFrame, source: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    kept = df[df["kept"]]
    dropped = df[~df["kept"]]
    ax.scatter(dropped["clip_sim"], dropped["dinov2_sim"], s=6, c="lightgray",
               alpha=0.6, label=f"dropped (n={len(dropped)})")
    ax.scatter(kept["clip_sim"], kept["dinov2_sim"], s=6, c="tab:blue",
               alpha=0.6, label=f"kept (n={len(kept)})")
    ax.axvline(FLOOR_CLIP, color="r", linestyle="--", lw=0.8, label=f"CLIP floor={FLOOR_CLIP}")
    ax.axhline(FLOOR_DINO, color="g", linestyle="--", lw=0.8, label=f"DINOv2 floor={FLOOR_DINO}")
    rho, p = spearmanr(df["clip_sim"], df["dinov2_sim"])
    ax.set_title(f"{source}: CLIP-sim vs DINOv2-sim (Spearman rho={rho:.3f}, p={p:.2g})")
    ax.set_xlabel("CLIP cosine similarity (top-5 mean to real)")
    ax.set_ylabel("DINOv2 cosine similarity (top-5 mean to real)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  wrote {out_path.relative_to(ROOT)}")


def plot_histograms(df: pd.DataFrame, source: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, floor, label in [
        (axes[0], "clip_sim", FLOOR_CLIP, "CLIP"),
        (axes[1], "dinov2_sim", FLOOR_DINO, "DINOv2"),
    ]:
        ax.hist(df[df["kept"]][col], bins=40, alpha=0.7, label="kept", color="tab:blue")
        ax.hist(df[~df["kept"]][col], bins=40, alpha=0.7, label="dropped", color="lightgray")
        ax.axvline(floor, color="r", linestyle="--", lw=0.8, label=f"floor={floor}")
        ax.set_title(f"{source}: {label} cosine similarity")
        ax.set_xlabel("similarity")
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  wrote {out_path.relative_to(ROOT)}")


def plot_retention_bars(table: pd.DataFrame, out_path: Path) -> None:
    pivot = table.pivot(index="method", columns="source", values="kept_pct")
    method_order = ["clip_only", "dino_only", "joint_or", "pct_only", "joint_and", "current_pipeline"]
    pivot = pivot.reindex(method_order)
    ax = pivot.plot(kind="bar", figsize=(10, 5), edgecolor="black")
    ax.set_ylabel("% kept")
    ax.set_title("Retention rate per filter method")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="source")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=[
        "data/synthetic/variant_D/filter_scores.csv",
        "data/synthetic/variant_E/filter_scores_lora.csv",
    ])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tables: list[pd.DataFrame] = []
    all_per_class: list[pd.DataFrame] = []
    for src in args.sources:
        path = ROOT / src
        if not path.exists():
            print(f"skip {src}: missing")
            continue
        source_label = path.parent.name  # variant_D / variant_E
        print(f"\n=== {source_label} ===")
        df = pd.read_csv(path)
        df = methods(df)

        table = retention_table(df, source_label)
        per_class = per_class_retention(df, source_label)
        all_tables.append(table)
        all_per_class.append(per_class)

        plot_scatter(df, source_label, OUT_DIR / f"scatter_{source_label}.png")
        plot_histograms(df, source_label, OUT_DIR / f"histograms_{source_label}.png")

    combined_table = pd.concat(all_tables, ignore_index=True)
    combined_table.to_csv(OUT_DIR / "retention_summary.csv", index=False)
    print(f"\nwrote {(OUT_DIR / 'retention_summary.csv').relative_to(ROOT)}")
    print(combined_table.to_string(index=False))

    combined_per_class = pd.concat(all_per_class, ignore_index=True)
    combined_per_class.to_csv(OUT_DIR / "retention_per_class.csv", index=False)
    print(f"\nwrote {(OUT_DIR / 'retention_per_class.csv').relative_to(ROOT)}")

    plot_retention_bars(combined_table, OUT_DIR / "retention_bars.png")


if __name__ == "__main__":
    main()
