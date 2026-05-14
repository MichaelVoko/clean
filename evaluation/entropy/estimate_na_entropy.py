#!/usr/bin/env python
"""Estimate non-adiabatic entropy production rate H_na(t) for a DFM checkpoint.

Implements step 1 of Algorithm 1 in Foresti et al. (2026), reparameterized for
x_1-prediction DFM with the masking interpolant.

Reparameterized estimator (derivation in plan / chat history):

    H_na(t) = E_{x_t ~ p_t} [ (1/(1-t)) * sum_{d: x_t^d == MAS} (
                              H(p_theta(x_1^d | x_t)) - ln(t/(1-t)) ) ]

where H is Shannon entropy of the per-position predictive distribution over the
canonical token set (special tokens and legacy RNA tokens are zeroed and the
distribution renormalized — same masking as the inference sampler).

We additionally track a "neural entropy rate"

    H_neural(t) = E_{x_t ~ p_t} [ (1/(1-t)) * sum_{d: x_t^d == MAS} H_d ]

which is the model-only contribution (drops the constant -ln(t/(1-t)) offset
that arises from the omitted stationary-distribution terms in the absorbing
case and otherwise breaks monotonicity of the cumulative). H_neural is
non-negative by construction and is what we use to build the EDS schedule.

Outputs:
    h_na_curve.csv  — t, H_na_mean, H_na_std, H_neural_mean, H_neural_std, C_t_neural, n_examples
    h_na_curve.png  — both curves and the cumulative used for EDS

Usage:
    python evaluation/entropy/estimate_na_entropy.py \
        --checkpoint dfm_base/s_54019.pt \
        --csv-dir evaluation/sweeps/scripts/valid_datasets \
        --n-structures 64 --n-time 1024 \
        --out-dir evaluation/entropy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_ROOT / "inference"
sys.path.insert(0, str(INFERENCE_DIR))

from data_utils import parse_PDB, featurize  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402

# Vocab/restypes — mirrors inference/run.py
RESTYPES = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","UNK",
    "DA","DC","DG","DT","DX",
    "A","C","G","U","RX",
    "MAS","PAD",
]
POLYTYPES = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
ATOM_TYPES = [
    "N","CA","C","O",
    "OP1","OP2","P","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'",
]
SPECIAL_TOKENS = ["UNK", "DX", "RX", "MAS", "PAD"]


def build_token_maps(na_shared_tokens: bool = True):
    restype_to_int = dict(zip(RESTYPES, range(len(RESTYPES))))
    polytype_to_int = dict(zip(POLYTYPES, range(len(POLYTYPES))))
    atom_dict = dict(zip(ATOM_TYPES, range(len(ATOM_TYPES))))
    if na_shared_tokens:
        restype_to_int["A"] = restype_to_int["DA"]
        restype_to_int["C"] = restype_to_int["DC"]
        restype_to_int["G"] = restype_to_int["DG"]
        restype_to_int["U"] = restype_to_int["DT"]
        restype_to_int["RX"] = restype_to_int["DX"]
    return restype_to_int, polytype_to_int, atom_dict


def load_model(checkpoint_path: str, device: torch.device):
    restype_to_int, polytype_to_int, atom_dict = build_token_maps()
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=32,
        model_type="na_mpnn",
        vocab=33,
        num_letters=33,
        atom_dict=atom_dict,
        restype_to_int=restype_to_int,
        polytype_to_int=polytype_to_int,
        mode="dfm",
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, restype_to_int


def select_structures(csv_dir: Path, n_structures: int, contexts, seed: int):
    """Pick n_structures evenly-ish from the listed contexts."""
    rng = np.random.RandomState(seed)
    pool = []
    for ctx in contexts:
        csv_path = csv_dir / f"design_valid_{ctx}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for sp in df["structure_path"]:
            pool.append((ctx, sp))
    if not pool:
        raise RuntimeError(f"No structures found in {csv_dir}")
    idx = rng.permutation(len(pool))[:n_structures]
    return [pool[i] for i in idx]


def featurize_one(pdb_path: str, device: torch.device):
    """Parse + featurize a single PDB. Returns (feature_dict, native_S, chain_mask) or None on failure."""
    abs_path = pdb_path
    if not os.path.isabs(abs_path):
        abs_path = str(REPO_ROOT / pdb_path)
    if not os.path.exists(abs_path):
        return None
    try:
        macromolecule_dict, _backbone, _other_atoms, _icodes, _water_atoms = parse_PDB(
            abs_path,
            device=device,
            chains="",
            model_type="na_mpnn",
            parse_na_only=0,
            na_shared_tokens=1,
            load_residues_with_missing_atoms=0,
        )
    except Exception as exc:
        print(f"[skip] parse_PDB failed for {abs_path}: {exc}", file=sys.stderr)
        return None

    L = macromolecule_dict["S"].shape[0]
    # Designable mask = nucleic-acid positions (mirrors design_na_only behaviour):
    # we estimate the entropy contribution where the model is actually predicting.
    na_mask = (macromolecule_dict["dna_mask"] | macromolecule_dict["rna_mask"]).long()
    # Combine with the parser's mask (valid residues with backbones).
    chain_mask = (macromolecule_dict["mask"] * na_mask.float()).long()
    if int(chain_mask.sum()) == 0:
        return None
    macromolecule_dict["chain_mask"] = chain_mask.float()

    feat = featurize(macromolecule_dict)
    feat["batch_size"] = 1
    return feat, macromolecule_dict["S"], chain_mask


def compute_h_na_for_structure(
    model: ProteinMPNN,
    feat: dict,
    native_S: torch.Tensor,
    chain_mask: torch.Tensor,
    t_values: np.ndarray,
    restype_to_int: dict,
    rng: torch.Generator,
    device: torch.device,
) -> np.ndarray:
    """Compute H_na(t_i) for one structure across all t_i. Returns array of shape [T]."""
    mask_id = restype_to_int["MAS"]
    L = native_S.shape[0]
    designable = chain_mask.bool()  # [L]
    n_design = int(designable.sum())
    if n_design == 0:
        return np.full(t_values.shape, np.nan)

    # Special-token suppression set (same as samplers._forward_probs).
    bad_token_ids = sorted({restype_to_int[tok] for tok in SPECIAL_TOKENS})
    bad_token_ids = torch.tensor(bad_token_ids, device=device, dtype=torch.long)

    # Move feat tensors to device once (parse_PDB already placed them on device, but be safe).
    base_feat = {}
    for k, v in feat.items():
        if torch.is_tensor(v):
            base_feat[k] = v.to(device)
        else:
            base_feat[k] = v

    h_na = np.zeros(t_values.shape, dtype=np.float64)
    h_neural = np.zeros(t_values.shape, dtype=np.float64)
    with torch.no_grad():
        for i, t_val in enumerate(t_values):
            t_val = float(t_val)
            unmask_keep = (torch.rand(L, generator=rng, device=device) < t_val)
            S_t = native_S.clone()
            mask_set = designable & (~unmask_keep)
            S_t[mask_set] = mask_id

            feat_t = dict(base_feat)
            feat_t["S"] = S_t.unsqueeze(0)

            t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
            log_probs, _ = model.forward_dfm(feat_t, t_tensor)
            probs = torch.softmax(log_probs, dim=-1)
            probs[..., bad_token_ids] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            eps = 1e-12
            H_d = -(probs * (probs.clamp_min(eps).log())).sum(dim=-1).squeeze(0)  # [L]

            agg_mask = mask_set
            n_masked = int(agg_mask.sum())
            if n_masked == 0:
                h_na[i] = 0.0
                h_neural[i] = 0.0
                continue

            denom = max(1.0 - t_val, 1e-6)
            sum_H = H_d[agg_mask].sum().item()
            log_ratio = float(np.log(max(t_val, 1e-12) / denom))
            # Foresti-style H_na (with constant offset; can go negative at high t).
            h_na[i] = (sum_H - n_masked * log_ratio) / denom
            # Neural entropy rate (model-only, non-negative; used for EDS).
            h_neural[i] = sum_H / denom

    return h_na, h_neural


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to DFM checkpoint, e.g. dfm_base/s_54019.pt")
    p.add_argument("--csv-dir", default=str(REPO_ROOT / "evaluation/sweeps/scripts/valid_datasets"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "evaluation/entropy"))
    p.add_argument("--n-structures", type=int, default=64)
    p.add_argument("--n-time", type=int, default=1024)
    p.add_argument("--t-min", type=float, default=1e-3)
    p.add_argument("--t-max", type=float, default=1.0 - 1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contexts", nargs="+", default=["rna_only", "rna_with_protein", "dna_only", "dna_with_protein"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}", file=sys.stderr)
    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    model, restype_to_int = load_model(args.checkpoint, device)

    print(f"Selecting {args.n_structures} structures from {args.csv_dir}", file=sys.stderr)
    structures = select_structures(Path(args.csv_dir), args.n_structures, args.contexts, args.seed)

    t_values = np.linspace(args.t_min, args.t_max, args.n_time)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    h_na_list = []
    h_neural_list = []
    contexts_used = []
    n_designable = []
    t_start = time.time()
    for k, (ctx, sp) in enumerate(structures):
        try:
            res = featurize_one(sp, device)
        except Exception as exc:
            print(f"[skip] featurize failed {sp}: {exc}", file=sys.stderr)
            continue
        if res is None:
            print(f"[skip] no usable nucleic-acid positions in {sp}", file=sys.stderr)
            continue
        feat, native_S, chain_mask = res
        try:
            h_na_arr, h_neural_arr = compute_h_na_for_structure(
                model, feat, native_S, chain_mask, t_values, restype_to_int, rng, device
            )
        except Exception as exc:
            print(f"[skip] forward failed for {sp}: {exc}", file=sys.stderr)
            continue
        h_na_list.append(h_na_arr)
        h_neural_list.append(h_neural_arr)
        contexts_used.append(ctx)
        n_designable.append(int(chain_mask.sum()))
        elapsed = time.time() - t_start
        print(
            f"[{k+1}/{len(structures)}] ctx={ctx} n_design={n_designable[-1]} "
            f"mean_H_neural={h_neural_arr.mean():.3f} mean_H_na={h_na_arr.mean():.3f} "
            f"elapsed={elapsed:.1f}s",
            file=sys.stderr,
        )

    if not h_neural_list:
        raise RuntimeError("No structures produced an entropy curve.")

    h_na = np.stack(h_na_list, axis=0)         # [N, T]
    h_neural = np.stack(h_neural_list, axis=0)
    h_na_mean = h_na.mean(axis=0)
    h_na_std = h_na.std(axis=0)
    h_neural_mean = h_neural.mean(axis=0)
    h_neural_std = h_neural.std(axis=0)

    # Cumulative for the EDS schedule (uses H_neural, which is non-negative).
    C_t_neural = np.zeros_like(h_neural_mean)
    for i in range(1, len(t_values)):
        dt = t_values[i] - t_values[i - 1]
        C_t_neural[i] = C_t_neural[i - 1] + 0.5 * (h_neural_mean[i] + h_neural_mean[i - 1]) * dt

    df = pd.DataFrame({
        "t": t_values,
        "H_na_mean": h_na_mean,
        "H_na_std": h_na_std,
        "H_neural_mean": h_neural_mean,
        "H_neural_std": h_neural_std,
        "C_t_neural": C_t_neural,
        "n_examples": [h_neural.shape[0]] * len(t_values),
    })
    csv_path = out_dir / "h_na_curve.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", file=sys.stderr)

    monotone = bool(np.all(np.diff(C_t_neural) >= -1e-9))
    meta = {
        "checkpoint": args.checkpoint,
        "n_structures_used": int(h_neural.shape[0]),
        "n_structures_requested": int(args.n_structures),
        "n_time": int(args.n_time),
        "t_min": args.t_min,
        "t_max": args.t_max,
        "seed": args.seed,
        "contexts_used": contexts_used,
        "n_designable_per_struct": n_designable,
        "C_neural_total": float(C_t_neural[-1]),
        "C_neural_monotone": monotone,
        "note": (
            "H_na uses Foresti's formula and may be negative at large t (the "
            "constant -ln(t/(1-t)) offset breaks non-negativity in the absorbing "
            "case). H_neural drops that offset, is non-negative, and is what we "
            "integrate to build the EDS schedule."
        ),
    }
    (out_dir / "h_na_curve_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"C_neural total = {C_t_neural[-1]:.3f}, monotone={monotone}", file=sys.stderr)

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(t_values, h_na_mean, label="H_na (Foresti)")
        axes[0].fill_between(t_values, h_na_mean - h_na_std, h_na_mean + h_na_std, alpha=0.2)
        axes[0].axhline(0, color="grey", lw=0.5)
        axes[0].set_xlabel("t"); axes[0].set_ylabel("H_na(t)")
        axes[0].set_title("Non-adiabatic entropy rate"); axes[0].legend()
        axes[1].plot(t_values, h_neural_mean, color="C1", label="H_neural")
        axes[1].fill_between(t_values, h_neural_mean - h_neural_std, h_neural_mean + h_neural_std, alpha=0.2, color="C1")
        axes[1].set_xlabel("t"); axes[1].set_ylabel("H_neural(t)")
        axes[1].set_title("Neural entropy rate (used for EDS)"); axes[1].legend()
        axes[2].plot(t_values, C_t_neural, color="C2")
        axes[2].set_xlabel("t"); axes[2].set_ylabel("C_neural(t)")
        axes[2].set_title(f"Cumulative neural entropy (total={C_t_neural[-1]:.2f})")
        fig.suptitle(f"DFM neural entropy — {Path(args.checkpoint).name}  (n={h_neural.shape[0]})")
        fig.tight_layout()
        fig.savefig(out_dir / "h_na_curve.png", dpi=140)
        print(f"Wrote {out_dir / 'h_na_curve.png'}", file=sys.stderr)
    except Exception as exc:
        print(f"Plotting skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
