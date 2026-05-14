#!/usr/bin/env python
"""Print a tab-separated (K, context, structure_path) work list for the EDS sweep.

Usage: build_eds_worklist.py <csv_dir> <K1> [K2 ...]

K is the number of EDS steps (so the file eds_schedule_K{K}.json is consumed
downstream).
"""
import sys
import pandas as pd
from pathlib import Path

csv_dir = Path(sys.argv[1])
ks = sys.argv[2:]
contexts = ["rna_only", "rna_with_protein", "dna_only", "dna_with_protein"]

for K in ks:
    for ctx in contexts:
        csv_path = csv_dir / f"design_valid_{ctx}.csv"
        if not csv_path.exists():
            continue
        for sp in pd.read_csv(csv_path)["structure_path"]:
            print(f"{K}\t{ctx}\t{sp}")
