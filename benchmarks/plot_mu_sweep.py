#!/usr/bin/env python3
"""Plot results from the mu-grid N_mu sweep.

Usage:
    python benchmarks/plot_mu_sweep.py [--input PATH] [--output-dir DIR]

Produces four figures saved to output-dir:
    1. mean_benefit_vs_nmu.png   — mean AF improvement over uniform vs N_mu
    2. benefit_distribution.png  — per-patient benefit distribution (box plot) per N_mu
    3. n_improved_vs_nmu.png     — number of improved patients vs N_mu
    4. runtime_vs_nmu.png        — mean per-patient runtime vs N_mu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure numeric columns
    df["n_mu_actual"] = df["n_mu_actual"].astype(int)
    df["n_mu_target"] = df["n_mu_target"].astype(int)
    return df


def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per N_mu with summary statistics."""
    records = []
    for n_mu, sub in df.groupby("n_mu_target", sort=True):
        records.append({
            "n_mu_target": n_mu,
            "n_mu_actual": int(sub["n_mu_actual"].iloc[0]),
            "mean_benefit": sub["benefit"].mean(),
            "median_benefit": sub["benefit"].median(),
            "std_benefit": sub["benefit"].std(),
            "n_improved": int((sub["benefit"] > 0).sum()),
            "n_patients": len(sub),
            "mean_runtime": sub["runtime_sec"].mean(),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_mean_benefit(stats: pd.DataFrame, ax: plt.Axes) -> None:
    x = stats["n_mu_actual"].values
    y = stats["mean_benefit"].values

    ax.plot(x, y, "o-", color=BLUE, linewidth=2, markersize=7, zorder=3)

    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.4f}", xy=(xi, yi), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Number of μ grid points (N_μ)")
    ax.set_ylabel("Mean AF improvement over uniform (ccGy)")
    ax.set_title("AF benefit vs μ grid resolution")
    ax.set_ylim(1.36, 1.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))



def plot_n_improved(stats: pd.DataFrame, ax: plt.Axes) -> None:
    x = stats["n_mu_actual"].values
    y = stats["n_improved"].values
    n_total = int(stats["n_patients"].iloc[0])

    ax.bar(x, y, width=np.diff(x, prepend=x[0] - 30).clip(min=15) * 0.5, color=GREEN, alpha=0.7, align="center")
    ax.set_xlabel("Number of μ grid points (N_μ)")
    ax.set_ylabel(f"Patients improved (out of {n_total})")
    ax.set_title("Patients with positive AF improvement")
    ax.set_ylim(48, 54)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def plot_runtime(stats: pd.DataFrame, ax: plt.Axes) -> None:
    x = stats["n_mu_actual"].values
    y = stats["mean_runtime"].values

    ax.plot(x, y, "s-", color=ORANGE, linewidth=2, markersize=7)
    ax.set_xlabel("Number of μ grid points (N_μ)")
    ax.set_ylabel("Mean per-patient runtime (s)")
    ax.set_title("Compute cost vs μ grid resolution")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="benchmarks/reports/mu_grid_sweep.csv")
    parser.add_argument("--output-dir", default="benchmarks/reports/mu_sweep_plots")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / args.input
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sweep(input_path)
    stats = group_stats(df)

    print(f"Loaded {len(df)} rows — {df['n_mu_target'].nunique()} configs, {df['patient_number'].nunique()} patients")
    print(stats[["n_mu_target", "n_mu_actual", "mean_benefit", "n_improved", "mean_runtime"]].to_string(index=False))

    # Figure 1: mean benefit vs N_mu
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_mean_benefit(stats, ax)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_benefit_vs_nmu.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'mean_benefit_vs_nmu.png'}")

    # Figure 2: n_improved
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_n_improved(stats, ax)
    fig.tight_layout()
    fig.savefig(output_dir / "n_improved_vs_nmu.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'n_improved_vs_nmu.png'}")

    # Figure 3: runtime
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_runtime(stats, ax)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_nmu.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'runtime_vs_nmu.png'}")

    # Figure 4: benefit + runtime combined (dual y-axis)
    fig, ax_benefit = plt.subplots(figsize=(7, 4))
    x = stats["n_mu_actual"].values

    # Left axis: mean benefit (blue)
    ax_benefit.plot(x, stats["mean_benefit"].values, "o-", color=BLUE, linewidth=2, markersize=7, label="Mean benefit")
    for xi, yi in zip(x, stats["mean_benefit"].values):
        ax_benefit.annotate(f"{yi:.4f}", xy=(xi, yi), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8, color=BLUE)
    ax_benefit.set_xlabel("Number of μ grid points (N_μ)")
    ax_benefit.set_ylabel("Mean AF improvement over uniform (ccGy)", color=BLUE)
    ax_benefit.tick_params(axis="y", labelcolor=BLUE)
    ax_benefit.set_ylim(1.36, 1.4)
    ax_benefit.xaxis.set_major_locator(ticker.MultipleLocator(50))

    # Right axis: runtime (red)
    ax_runtime = ax_benefit.twinx()
    ax_runtime.plot(x, stats["mean_runtime"].values, "s-", color=RED, linewidth=2, markersize=7, label="Runtime")
    ax_runtime.set_ylabel("Mean per-patient runtime (s)", color=RED)
    ax_runtime.tick_params(axis="y", labelcolor=RED)
    ax_runtime.spines["right"].set_visible(True)

    ax_benefit.set_title("AF benefit and compute cost vs μ grid resolution")
    fig.tight_layout()
    fig.savefig(output_dir / "benefit_and_runtime_vs_nmu.png")
    plt.close(fig)
    print(f"Saved: {output_dir / 'benefit_and_runtime_vs_nmu.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
