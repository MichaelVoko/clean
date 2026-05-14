#!/usr/bin/env python
"""Build an Entropic Discrete Schedule (EDS) from a precomputed neural-entropy curve.

Step 2 of Algorithm 1 in Foresti et al.: invert the warping function
Φ(t) = C(t) / C(T) so that uniform spacing in [0,1] corresponds to constant
information gain per step.

Input:  evaluation/entropy/h_na_curve.csv (columns: t, H_neural_mean, C_t_neural)
Output: evaluation/entropy/eds_schedule_K{K}.json with structure
        {
          "K": K,
          "t_grid": [t_0=0, t_1, ..., t_K=1],
          "h_per_step": [t_{k+1}-t_k for k in 0..K-1],
          "source_csv": "...",
          "C_total": float
        }

Usage:
    python evaluation/entropy/build_eds_schedule.py \
        --curve evaluation/entropy/h_na_curve.csv \
        --K 16 32 64 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_schedule(curve_csv: Path, K: int) -> dict:
    df = pd.read_csv(curve_csv)
    t = df["t"].to_numpy()
    C = df["C_t_neural"].to_numpy()

    # Force monotonicity (cummax) and bracket with t=0 and t=1.
    C = np.maximum.accumulate(C)
    if t[0] > 0:
        t = np.concatenate([[0.0], t])
        C = np.concatenate([[C[0]], C])
    if t[-1] < 1.0:
        t = np.concatenate([t, [1.0]])
        C = np.concatenate([C, [C[-1]]])

    C_total = float(C[-1])
    if C_total <= 0:
        raise ValueError(f"Cumulative entropy is non-positive (C_total={C_total}); cannot build schedule.")
    Phi = C / C_total  # in [0, 1]

    # Targets: K+1 uniformly spaced progress values in [0,1].
    targets = np.linspace(0.0, 1.0, K + 1)
    # Numpy interp does linear interpolation; Phi is monotone non-decreasing so
    # this is exactly the inverse of the warping function.
    t_grid = np.interp(targets, Phi, t)
    # Numerical safety: enforce strict monotonicity.
    for i in range(1, len(t_grid)):
        if t_grid[i] <= t_grid[i - 1]:
            t_grid[i] = t_grid[i - 1] + 1e-6
    t_grid[0] = 0.0
    t_grid[-1] = 1.0

    h_per_step = np.diff(t_grid).tolist()
    return {
        "K": int(K),
        "t_grid": t_grid.tolist(),
        "h_per_step": h_per_step,
        "C_total": C_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curve", default="evaluation/entropy/h_na_curve.csv")
    p.add_argument("--out-dir", default="evaluation/entropy")
    p.add_argument("--K", type=int, nargs="+", required=True,
                   help="Number of generation steps (multiple values allowed; one schedule per value).")
    args = p.parse_args()

    curve = Path(args.curve)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for K in args.K:
        sched = build_schedule(curve, K)
        sched["source_csv"] = str(curve.resolve())
        out_path = out_dir / f"eds_schedule_K{K}.json"
        out_path.write_text(json.dumps(sched, indent=2))
        spans = sched["h_per_step"]
        print(f"K={K}  t_grid[:5]={sched['t_grid'][:5]}  h_min={min(spans):.4f} h_max={max(spans):.4f}  -> {out_path}")


if __name__ == "__main__":
    main()
