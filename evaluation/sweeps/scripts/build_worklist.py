#!/usr/bin/env python
"""Print a tab-separated (key, context, structure_path) work list to stdout.

Usage: build_worklist.py <csv_dir> <key1> [key2 ...]

key is a checkpoint step number (for run_checkpoint_sweep.sh) or a dt value
(for run_dt_sweep.sh) — the caller interprets it either way.
"""
import sys
import pandas as pd
from pathlib import Path

csv_dir = Path(sys.argv[1])
keys = sys.argv[2:]
contexts = ["rna_only", "rna_with_protein", "dna_only", "dna_with_protein"]

for key in keys:
    for ctx in contexts:
        csv_path = csv_dir / f"design_valid_{ctx}.csv"
        if not csv_path.exists():
            continue
        for sp in pd.read_csv(csv_path)["structure_path"]:
            print(f"{key}\t{ctx}\t{sp}")
