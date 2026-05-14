#!/usr/bin/env python
"""Aggregate per-design recovery for the EDS sweep (one CSV row per K).

Walks: <outputs-dir>/K_<K>/<context>/<structure>/recovery.json
Writes: recovery_by_K.csv with the same column layout as recovery_by_step.csv,
        but keyed on K (number of EDS steps).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import pandas as pd

REPO_ROOT = Path("/home/mh2167/rds/hpc-work/NA-MPNN")
DEFAULT_OUTPUTS = REPO_ROOT / "evaluation/sweeps/eds_sweep/outputs"
DEFAULT_CSV = REPO_ROOT / "evaluation/sweeps/eds_sweep/recovery_by_K.csv"

CONTEXTS = ("rna_only", "rna_with_protein", "dna_only", "dna_with_protein")


def collect_context(ctx_dir: Path) -> list[float]:
    if not ctx_dir.is_dir():
        return []
    recoveries: list[float] = []
    fallback: list[Path] = []
    for struct_dir in ctx_dir.iterdir():
        if not struct_dir.is_dir():
            continue
        rjson = struct_dir / "recovery.json"
        if rjson.exists():
            try:
                recoveries.extend(json.loads(rjson.read_text()).values())
            except (ValueError, OSError):
                fallback.append(struct_dir)
        else:
            fallback.append(struct_dir)
    if fallback:
        import re, subprocess
        result = subprocess.run(
            ["grep", "-rh", "--include=*.json", "tool_reported_sequence_recovery"]
            + [str(d) for d in fallback],
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

    k_dirs = sorted(
        (p for p in args.outputs_dir.iterdir()
         if p.is_dir() and p.name.startswith("K_") and p.name[2:].isdigit()),
        key=lambda p: int(p.name[2:]),
    )
    if not k_dirs:
        raise SystemExit(f"No K_<n> subdirectories under {args.outputs_dir}")

    cols = ["K"] + list(CONTEXTS) + ["total"] + [f"n_{c}" for c in CONTEXTS] + ["n_designs_total"]
    rows: list[dict] = []
    for k_dir in k_dirs:
        K = int(k_dir.name[2:])
        row = {"K": K}
        ctx_means: list[float] = []
        total_n = 0
        for ctx in CONTEXTS:
            values = collect_context(k_dir / ctx)
            row[f"n_{ctx}"] = len(values)
            total_n += len(values)
            if values:
                row[ctx] = mean(values)
                ctx_means.append(row[ctx])
            else:
                row[ctx] = float("nan")
                print(f"  warn: K={K} missing context {ctx}", file=sys.stderr)
        row["total"] = mean(ctx_means) if len(ctx_means) == len(CONTEXTS) else float("nan")
        row["n_designs_total"] = total_n
        rows.append(row)

    df = pd.DataFrame(rows)[cols].sort_values("K")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {args.out_csv}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
