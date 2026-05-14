#!/usr/bin/env python
"""Aggregate per-design sequence recovery over the sweep tree.

Walks:  outputs/<step>/<context>/<structure>/design_json/*.json
Reads:  tool_reported_sequence_recovery from each design JSON.
Writes: recovery_by_step.csv with columns
        step, rna_only, rna_with_protein, dna_only, dna_with_protein,
        total, n_rna_only, n_rna_with_protein, n_dna_only,
        n_dna_with_protein, n_designs_total

Per-(step, context) value = mean of tool_reported_sequence_recovery across all
design JSONs for that (step, context), pooling structures and samples.
total = macro-average of the four context means (each context weighted equally).
Steps with any missing context emit NaN in `total` and a warning to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import pandas as pd


REPO_ROOT = Path("/home/mh2167/rds/hpc-work/NA-MPNN")
DEFAULT_OUTPUTS = REPO_ROOT / "evaluation/sweeps/dfm_sweep/outputs"
DEFAULT_CSV = REPO_ROOT / "evaluation/sweeps/dfm_sweep/recovery_by_step.csv"

CONTEXTS = ("rna_only", "rna_with_protein", "dna_only", "dna_with_protein")


def collect_context(ctx_dir: Path) -> list[float]:
    if not ctx_dir.is_dir():
        return []
    recoveries = []
    json_fallback_dirs = []
    for struct_dir in ctx_dir.iterdir():
        if not struct_dir.is_dir():
            continue
        txt = struct_dir / "recovery.txt"
        if txt.exists():
            try:
                recoveries.append(float(txt.read_text().strip()))
            except (ValueError, OSError):
                json_fallback_dirs.append(struct_dir)
        else:
            json_fallback_dirs.append(struct_dir)
    # Fall back to grep for structures without recovery.txt (pre-index runs)
    if json_fallback_dirs:
        import subprocess, re
        result = subprocess.run(
            ["grep", "-rh", "--include=*.json", "tool_reported_sequence_recovery"]
            + [str(d) for d in json_fallback_dirs],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            m = re.search(r"tool_reported_sequence_recovery[\"']?\s*:\s*([0-9.]+)", line)
            if m:
                recoveries.append(float(m.group(1)))
    return recoveries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    if not args.outputs_dir.is_dir():
        raise SystemExit(f"No outputs directory at {args.outputs_dir}")

    step_dirs = sorted(
        (p for p in args.outputs_dir.iterdir() if p.is_dir() and p.name.startswith("step_") and p.name[5:].isdigit()),
        key=lambda p: int(p.name[5:]),
    )
    if not step_dirs:
        raise SystemExit(f"No step subdirectories under {args.outputs_dir}")

    cols = ["step"] + list(CONTEXTS) + ["total"] + [f"n_{c}" for c in CONTEXTS] + ["n_designs_total"]
    if args.out_csv.exists():
        existing = pd.read_csv(args.out_csv)
        done_steps = set(existing["step"].tolist())
        rows = existing.to_dict("records")
        print(f"  resuming: {len(done_steps)} steps already collected", file=sys.stderr, flush=True)
    else:
        done_steps, rows = set(), []

    for step_dir in step_dirs:
        step = int(step_dir.name[5:])
        if step in done_steps:
            continue
        print(f"  collecting step {step}...", file=sys.stderr, flush=True)
        row = {"step": step}
        ctx_means: list[float] = []
        total_n = 0
        for ctx in CONTEXTS:
            values = collect_context(step_dir / ctx)
            row[f"n_{ctx}"] = len(values)
            total_n += len(values)
            if values:
                row[ctx] = mean(values)
                ctx_means.append(row[ctx])
            else:
                row[ctx] = float("nan")
                print(f"  warn: step {step} missing context {ctx}", file=sys.stderr)
        row["total"] = mean(ctx_means) if len(ctx_means) == len(CONTEXTS) else float("nan")
        row["n_designs_total"] = total_n
        rows.append(row)
        pd.DataFrame(rows)[cols].sort_values("step").to_csv(args.out_csv, index=False)

    df = pd.DataFrame(rows)[cols].sort_values("step")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Wrote {len(df)} rows -> {args.out_csv}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df[["step", *CONTEXTS, "total", "n_designs_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
