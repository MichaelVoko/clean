#!/usr/bin/env python
"""Collect recovery results from a dt sweep and plot them.

Reads: recovery_samples.csv (dt,context,structure,sample,recovery) written during sweep,
       merged with existing recovery_by_dt.csv for dt values run before this approach.
Writes: recovery_by_dt.csv and recovery_by_dt.png in SWEEP_DIR.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path("/home/mh2167/rds/hpc-work/NA-MPNN")
DEFAULT_OUTPUTS = REPO_ROOT / "evaluation/sweeps/dt_sweep"
DEFAULT_CSV     = REPO_ROOT / "evaluation/sweeps/dt_sweep/recovery_by_dt.csv"
DEFAULT_PNG     = REPO_ROOT / "evaluation/sweeps/dt_sweep/recovery_by_dt.png"

CONTEXTS = ("rna_only", "rna_with_protein", "dna_only", "dna_with_protein")
SERIES = [
    ("rna_only",         "RNA only",        "tab:blue"),
    ("rna_with_protein", "RNA + protein",   "tab:cyan"),
    ("dna_only",         "DNA only",        "tab:orange"),
    ("dna_with_protein", "DNA + protein",   "tab:red"),
    ("total",            "Total (macro)",   "black"),
]


def collect(sweep_dir: Path, existing_csv: Path) -> pd.DataFrame:
    samples_csv = sweep_dir / "recovery_samples.csv"
    if not samples_csv.exists():
        raise SystemExit(f"No recovery_samples.csv at {samples_csv}")

    df = pd.read_csv(samples_csv)

    # load existing aggregated rows (dt values run before recovery_samples.csv approach)
    done_dts: set = set()
    existing_rows: list = []
    if existing_csv.exists():
        existing = pd.read_csv(existing_csv)
        done_dts = set(existing["dt"])
        existing_rows = existing.to_dict("records")
        print(f"  {len(done_dts)} dt values already aggregated, skipping", file=sys.stderr)

    new_dts = sorted(dt for dt in df["dt"].unique() if dt not in done_dts)
    rows = list(existing_rows)
    for dt in new_dts:
        df_dt = df[df["dt"] == dt]
        row: dict = {"dt": dt}
        ctx_means: list[float] = []
        for ctx in CONTEXTS:
            vals = df_dt[df_dt["context"] == ctx]["recovery"].dropna().tolist()
            row[ctx] = mean(vals) if vals else float("nan")
            row[f"n_{ctx}"] = len(vals)
            if vals:
                ctx_means.append(row[ctx])
            else:
                print(f"  warn: dt={dt} missing context {ctx}", file=sys.stderr)
        row["total"] = mean(ctx_means) if len(ctx_means) == len(CONTEXTS) else float("nan")
        rows.append(row)

    cols = ["dt"] + list(CONTEXTS) + ["total"] + [f"n_{c}" for c in CONTEXTS]
    return pd.DataFrame(rows)[cols].sort_values("dt")


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label, color in SERIES:
        lw = 2.5 if col == "total" else 1.5
        ls = "-" if col == "total" else "--"
        ax.plot(df["dt"], df[col], label=label, color=color, linewidth=lw,
                linestyle=ls, marker="o", markersize=5)
    ax.invert_xaxis()  # smaller dt (more steps) on the right
    ax.set_xlabel("dfm_dt (Euler step size)")
    ax.set_ylabel("mean sequence recovery")
    ax.set_title("DFM dt sweep — sequence recovery on design_valid")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote plot -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--out-csv",   type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-png",   type=Path, default=DEFAULT_PNG)
    ap.add_argument("--no-plot",   action="store_true")
    args = ap.parse_args()

    df = collect(args.sweep_dir, args.out_csv)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {args.out_csv}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df[["dt", *CONTEXTS, "total"]].to_string(index=False))

    if not args.no_plot:
        plot(df, args.out_png)


if __name__ == "__main__":
    main()
