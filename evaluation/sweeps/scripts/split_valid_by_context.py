#!/usr/bin/env python
"""Split design_evaluation_valid.csv into four per-context CSVs.

Contexts are derived from the chain-type columns:
  - rna_only:           polyribonucleotide      + no protein
  - rna_with_protein:   polyribonucleotide      + polypeptide(L)
  - dna_only:           polydeoxyribonucleotide + no protein
  - dna_with_protein:   polydeoxyribonucleotide + polypeptide(L)

Rows with mixed RNA+DNA (none observed in the current valid set) are skipped
with a warning so downstream means remain well-defined.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "data/datasets/design_from_splits/design_evaluation_valid.csv"
DEFAULT_OUT = REPO_ROOT / "evaluation/sweeps/scripts/valid_datasets"

CONTEXTS = ("rna_only", "rna_with_protein", "dna_only", "dna_with_protein")


def _parse_list(cell) -> list:
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return []
    return ast.literal_eval(cell)


def classify(row) -> str | None:
    na = _parse_list(row["nucleic_acid_chain_cluster_ids_chain_types"])
    pr = _parse_list(row["protein_chain_cluster_ids_chain_types"])
    has_rna = any(x == "polyribonucleotide" for x in na)
    has_dna = any(x == "polydeoxyribonucleotide" for x in na)
    has_prot = any("polypeptide" in x for x in pr)
    if has_rna and has_dna:
        return None
    if has_rna:
        return "rna_with_protein" if has_prot else "rna_only"
    if has_dna:
        return "dna_with_protein" if has_prot else "dna_only"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["_context"] = df.apply(classify, axis=1)

    dropped = df["_context"].isna().sum()
    if dropped:
        print(f"Warning: dropped {dropped} rows with mixed/unknown chain types")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'context':<20} {'n_rows':>6}")
    for ctx in CONTEXTS:
        sub = df[df["_context"] == ctx].drop(columns="_context")
        out = args.out_dir / f"design_valid_{ctx}.csv"
        sub.to_csv(out, index=False)
        print(f"{ctx:<20} {len(sub):>6}  ->  {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
