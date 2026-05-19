#!/usr/bin/env python
"""Plot sequence recovery vs training step for the five series.

Reads recovery_by_step.csv (produced by collect_recovery.py) and renders a
single-axis line plot with the four contexts plus macro-averaged total.
A light band over the last ~20% of the step range visually flags the
convergence zone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "evaluation/sweeps/dfm_sweep/recovery_by_step.csv"
DEFAULT_PNG = REPO_ROOT / "evaluation/sweeps/dfm_sweep/recovery_by_step.png"

SERIES = [
    ("rna_only", "RNA only", "tab:blue"),
    ("rna_with_protein", "RNA + protein", "tab:cyan"),
    ("dna_only", "DNA only", "tab:orange"),
    ("dna_with_protein", "DNA + protein", "tab:red"),
    ("total", "Total (macro mean)", "black"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_PNG)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("step").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    s_min, s_max = df["step"].min(), df["step"].max()
    convergence_start = s_min + 0.8 * (s_max - s_min)
    ax.axvspan(convergence_start, s_max, color="gray", alpha=0.08, label="_nolegend_")

    for col, label, color in SERIES:
        lw = 2.5 if col == "total" else 1.5
        ls = "-" if col == "total" else "--"
        ax.plot(df["step"], df[col], label=label, color=color, linewidth=lw, linestyle=ls, marker="o", markersize=3)

    best = df.loc[df["total"].idxmax()] if df["total"].notna().any() else None
    if best is not None:
        ax.axvline(best["step"], color="black", alpha=0.25, linewidth=0.8)
        ax.annotate(
            f"best: s_{int(best['step'])}\n  total={best['total']:.3f}",
            xy=(best["step"], best["total"]),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("training step")
    ax.set_ylabel("mean sequence recovery")
    ax.set_title("DFM checkpoint sweep — sequence recovery on design_valid")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote plot -> {args.out.resolve().relative_to(REPO_ROOT)}")
    if best is not None:
        print(f"Best checkpoint by total recovery: s_{int(best['step'])}.pt  (total={best['total']:.4f})")


if __name__ == "__main__":
    main()
