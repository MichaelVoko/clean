#!/usr/bin/env python
"""Aggregate recovery_samples.csv → recovery_by_step.csv.

recovery_samples.csv columns: step,context,structure,sample,recovery
Written by run_checkpoint_sweep.sh during the sweep (one row per design sample).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP = REPO_ROOT / "evaluation/sweeps/dfm_sweep"

CONTEXTS = ("rna_only", "rna_with_protein", "dna_only", "dna_with_protein")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    args = ap.parse_args()

    samples_csv = args.sweep_dir / "recovery_samples.csv"
    by_step_csv = args.sweep_dir / "recovery_by_step.csv"

    if not samples_csv.exists():
        raise SystemExit(f"No recovery_samples.csv at {samples_csv}")

    df = pd.read_csv(samples_csv, header=None,
                     names=["step", "context", "structure", "sample", "recovery"])

    done_steps: set = set()
    existing_rows: list = []
    if by_step_csv.exists():
        existing = pd.read_csv(by_step_csv)
        done_steps = set(existing["step"])
        existing_rows = existing.to_dict("records")
        print(f"  {len(done_steps)} steps already aggregated, skipping", file=sys.stderr)

    new_steps = sorted(s for s in df["step"].unique() if s not in done_steps)
    if not new_steps:
        print("Nothing new to aggregate.")
        return

    cols = ["step"] + list(CONTEXTS) + ["total"] + [f"n_{c}" for c in CONTEXTS] + ["n_designs_total"]
    rows = list(existing_rows)
    for step in new_steps:
        df_step = df[df["step"] == step]
        row: dict = {"step": step}
        ctx_means: list[float] = []
        total_n = 0
        for ctx in CONTEXTS:
            vals = df_step[df_step["context"] == ctx]["recovery"].dropna().tolist()
            row[ctx] = float(mean(vals)) if vals else float("nan")
            row[f"n_{ctx}"] = len(vals)
            total_n += len(vals)
            if vals:
                ctx_means.append(row[ctx])
            else:
                print(f"  warn: step {step} missing context {ctx}", file=sys.stderr)
        row["total"] = float(mean(ctx_means)) if len(ctx_means) == len(CONTEXTS) else float("nan")
        row["n_designs_total"] = total_n
        rows.append(row)
        print(f"  step {step}: total={row['total']:.4f} n={total_n}", file=sys.stderr)

    out = pd.DataFrame(rows)[cols].sort_values("step")
    out.to_csv(by_step_csv, index=False)
    print(f"Wrote {len(out)} rows -> {by_step_csv}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(out[["step", *CONTEXTS, "total", "n_designs_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
