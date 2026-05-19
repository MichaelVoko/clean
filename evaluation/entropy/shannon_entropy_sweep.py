#!/usr/bin/env python
"""Shannon entropy vs diffusion time t, split by polymer type (protein vs RNA/DNA).

For each masked position at time t, computes:

    H_shannon(t) = -sum_k  p_k log p_k

where probs are renormalized over the canonical token set for that polymer type:
  - protein: 20 amino acids (ALA..VAL, indices 0-19)
  - NA:       4 nucleotides (DA,DC,DG,DT = indices 21-24; RNA maps here via na_shared_tokens)

No rate-weighting, no correction terms — just raw per-position predictive entropy.
Structures from rna_with_protein / dna_with_protein contain both types in one complex,
so both curves come from the same forward pass at the same masking state.

Usage:
    python evaluation/entropy/shannon_entropy_sweep.py \\
        --checkpoints dfm_model/s_4085.pt dfm_model/s_26576.pt dfm_model/s_59513.pt \\
        --n-structures 6 --n-time 100 \\
        --out-dir evaluation/entropy/shannon_sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference"))

from data_utils import parse_PDB, featurize  # noqa: E402
from model_utils import ProteinMPNN          # noqa: E402

RESTYPES = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","UNK",
    "DA","DC","DG","DT","DX",
    "A","C","G","U","RX",
    "MAS","PAD",
]
POLYTYPES  = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
ATOM_TYPES = [
    "N","CA","C","O",
    "OP1","OP2","P","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'",
]

# Canonical vocab indices per polymer type (special/degenerate tokens excluded).
# With na_shared_tokens=1, RNA nucleotides are encoded at the DNA positions.
PROTEIN_IDS = list(range(20))  # ALA(0) .. VAL(19)
NA_IDS      = [21, 22, 23, 24] # DA, DC, DG, DT  (RNA A/C/G/U share these indices)


def _build_token_maps():
    r2i = dict(zip(RESTYPES, range(len(RESTYPES))))
    p2i = dict(zip(POLYTYPES, range(len(POLYTYPES))))
    a2i = dict(zip(ATOM_TYPES, range(len(ATOM_TYPES))))
    r2i["A"] = r2i["DA"]; r2i["C"] = r2i["DC"]
    r2i["G"] = r2i["DG"]; r2i["U"] = r2i["DT"]; r2i["RX"] = r2i["DX"]
    return r2i, p2i, a2i


def load_model(checkpoint_path: str, device: torch.device):
    r2i, p2i, a2i = _build_token_maps()
    model = ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32,
        model_type="na_mpnn", vocab=33, num_letters=33,
        atom_dict=a2i, restype_to_int=r2i, polytype_to_int=p2i, mode="dfm",
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, r2i


def select_structures(csv_dir: Path, n: int, contexts: list[str], seed: int):
    rng = np.random.RandomState(seed)
    pool = []
    for ctx in contexts:
        csv_path = csv_dir / f"design_valid_{ctx}.csv"
        if not csv_path.exists():
            print(f"[warn] {csv_path} not found, skipping", file=sys.stderr)
            continue
        for sp in pd.read_csv(csv_path)["structure_path"]:
            pool.append((ctx, sp))
    if not pool:
        raise RuntimeError(f"No structures found for contexts {contexts}")
    idx = rng.permutation(len(pool))[:n]
    return [pool[i] for i in idx]


def featurize_one(pdb_path: str, device: torch.device):
    """Parse a structure, keeping both protein and NA chains. Returns None if either is absent."""
    abs_path = pdb_path if Path(pdb_path).is_absolute() else str(REPO_ROOT / pdb_path)
    if not Path(abs_path).exists():
        print(f"[skip] not found: {abs_path}", file=sys.stderr)
        return None
    try:
        md, _, _, _, _ = parse_PDB(
            abs_path, device=device, chains="", model_type="na_mpnn",
            parse_na_only=0, na_shared_tokens=1, load_residues_with_missing_atoms=0,
        )
    except Exception as exc:
        print(f"[skip] parse_PDB: {exc}", file=sys.stderr)
        return None

    valid    = md["mask"].float()
    prot_m   = md["protein_mask"].float()
    na_m     = (md["dna_mask"] + md["rna_mask"]).clamp_max(1.0).float()

    prot_design = (valid * prot_m).bool()
    na_design   = (valid * na_m).bool()

    if not (prot_design.any() and na_design.any()):
        return None  # structure must have both types

    md["chain_mask"] = (valid * (prot_m + na_m).clamp_max(1.0)).float()
    feat = featurize(md)
    feat["batch_size"] = 1
    return feat, md["S"], prot_design, na_design


def _shannon_per_position(probs: torch.Tensor, token_ids: list[int], eps: float = 1e-12) -> torch.Tensor:
    """Renorm probs over token_ids and return per-position H. probs: [L, V] -> [L]."""
    keep = torch.zeros(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    keep[token_ids] = 1.0
    p = probs * keep
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(p * p.clamp_min(eps).log()).sum(dim=-1)


@torch.no_grad()
def compute_shannon_curves(
    model, feat, native_S, prot_design, na_design,
    t_values, restype_to_int, rng, device,
):
    """Return (H_prot, H_na) each of shape [T] (float64, NaN where no masked positions)."""
    mask_id   = restype_to_int["MAS"]
    L         = native_S.shape[0]
    all_design = (prot_design | na_design).to(device)
    prot_d    = prot_design.to(device)
    na_d      = na_design.to(device)
    native_S  = native_S.to(device)

    base_feat = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feat.items()}

    H_prot = np.full(len(t_values), float("nan"), dtype=np.float64)
    H_na   = np.full(len(t_values), float("nan"), dtype=np.float64)

    for i, t in enumerate(t_values):
        t = float(t)
        keep = torch.rand(L, generator=rng, device=device) < t
        S_t  = native_S.clone()
        to_mask = all_design & ~keep
        S_t[to_mask] = mask_id

        feat_t = {**base_feat, "S": S_t.unsqueeze(0)}
        t_ten  = torch.tensor([[t]], device=device, dtype=torch.float32)

        log_probs, _ = model.forward_dfm(feat_t, t_ten)
        probs = torch.softmax(log_probs, dim=-1).squeeze(0)  # [L, V]

        masked_prot = to_mask & prot_d
        masked_na   = to_mask & na_d

        if masked_prot.any():
            H_prot[i] = _shannon_per_position(probs, PROTEIN_IDS)[masked_prot].mean().item()
        if masked_na.any():
            H_na[i]   = _shannon_per_position(probs, NA_IDS)[masked_na].mean().item()

    return H_prot, H_na


def _plot_summary(all_results: dict, t_values: np.ndarray, n_structs: int, contexts: list, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ckpts = len(all_results)
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, n_ckpts))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    panel_cfg = [
        (axes[0], "H_prot", "Protein (20-AA vocab)", np.log(20), "ln(20)"),
        (axes[1], "H_na",   "NA (4-base vocab)",     np.log(4),  "ln(4)"),
    ]
    for ax, key, title, h_max, h_label in panel_cfg:
        for c, (name, res) in zip(colors, all_results.items()):
            arr  = res[key]
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr, axis=0)
            ax.plot(t_values, mean, color=c, label=name)
            ax.fill_between(t_values, mean - std, mean + std, alpha=0.15, color=c)
        ax.axhline(h_max, color="grey", lw=0.8, ls=":", label=f"{h_label}={h_max:.2f}")
        ax.set_xlabel("t  (0=fully masked, 1=fully revealed)")
        ax.set_ylabel("H_Shannon")
        ax.set_title(f"Shannon entropy — {title}")
        ax.legend(fontsize=7)

    ax = axes[2]
    for c, (name, res) in zip(colors, all_results.items()):
        gap = np.nanmean(res["H_prot"], axis=0) - np.nanmean(res["H_na"], axis=0)
        ax.plot(t_values, gap, color=c, label=name)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel("H_prot − H_NA")
    ax.set_title("Entropy gap  (protein vs NA)")
    ax.legend(fontsize=7)

    fig.suptitle(f"Shannon entropy sweep  (n={n_structs} structs, contexts={contexts})")
    fig.tight_layout()
    path = out_dir / "shannon_sweep_summary.png"
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}", file=sys.stderr)


def _plot_per_checkpoint(name: str, res: dict, t_values: np.ndarray, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [(axes[0], "H_prot", "Protein"), (axes[1], "H_na", "NA")]:
        arr = res[key]
        for j, row in enumerate(arr):
            ax.plot(t_values, row, alpha=0.65, lw=0.9, label=f"s{j}")
        mean = np.nanmean(arr, axis=0)
        ax.plot(t_values, mean, color="black", lw=2, label="mean")
        ax.set_xlabel("t"); ax.set_ylabel("H_Shannon")
        ax.set_title(f"{title} — {name}"); ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / f"{name}_per_structure.png"
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more checkpoint paths (s_NNNNN.pt)")
    ap.add_argument("--csv-dir",    default=str(REPO_ROOT / "evaluation/sweeps/scripts/valid_datasets"))
    ap.add_argument("--contexts",   nargs="+", default=["rna_with_protein", "dna_with_protein"],
                    help="Dataset contexts; must contain structures with both protein and NA chains")
    ap.add_argument("--out-dir",    default=str(REPO_ROOT / "evaluation/entropy/shannon_sweep"))
    ap.add_argument("--n-structures", type=int, default=6)
    ap.add_argument("--n-time",     type=int, default=100)
    ap.add_argument("--t-min",      type=float, default=1e-3)
    ap.add_argument("--t-max",      type=float, default=1.0 - 1e-3)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device(args.device)
    t_values = np.linspace(args.t_min, args.t_max, args.n_time)

    print(f"Device: {device}", file=sys.stderr)
    structures = select_structures(Path(args.csv_dir), args.n_structures, args.contexts, args.seed)
    print(f"Selected {len(structures)} structures", file=sys.stderr)

    all_results: dict[str, dict] = {}

    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).stem
        print(f"\n=== {name} ===", file=sys.stderr)
        model, r2i = load_model(ckpt_path, device)
        rng = torch.Generator(device=device).manual_seed(args.seed)

        H_prot_list, H_na_list = [], []
        for k, (ctx, sp) in enumerate(structures):
            res = featurize_one(sp, device)
            if res is None:
                print(f"  [skip] {Path(sp).name}", file=sys.stderr)
                continue
            feat, S, prot_d, na_d = res
            try:
                hp, hn = compute_shannon_curves(model, feat, S, prot_d, na_d, t_values, r2i, rng, device)
            except Exception as exc:
                print(f"  [skip] forward failed for {Path(sp).name}: {exc}", file=sys.stderr)
                continue
            H_prot_list.append(hp)
            H_na_list.append(hn)
            print(
                f"  [{k+1}/{len(structures)}] {ctx}  {Path(sp).name}"
                f"  H_prot={np.nanmean(hp):.3f}  H_na={np.nanmean(hn):.3f}",
                file=sys.stderr,
            )

        if not H_prot_list:
            print(f"[warn] no usable structures for {name}", file=sys.stderr)
            continue

        H_prot_arr = np.stack(H_prot_list)
        H_na_arr   = np.stack(H_na_list)

        df = pd.DataFrame({
            "t":           t_values,
            "H_prot_mean": np.nanmean(H_prot_arr, axis=0),
            "H_prot_std":  np.nanstd(H_prot_arr, axis=0),
            "H_na_mean":   np.nanmean(H_na_arr, axis=0),
            "H_na_std":    np.nanstd(H_na_arr, axis=0),
            "n":           [(~np.isnan(H_prot_arr[:, i])).sum() for i in range(len(t_values))],
        })
        df.to_csv(out_dir / f"{name}_shannon.csv", index=False)
        print(f"  Wrote {name}_shannon.csv", file=sys.stderr)

        all_results[name] = {"H_prot": H_prot_arr, "H_na": H_na_arr}

    if not all_results:
        print("No results.", file=sys.stderr)
        return

    try:
        _plot_summary(all_results, t_values, args.n_structures, args.contexts, out_dir)
        for name, res in all_results.items():
            _plot_per_checkpoint(name, res, t_values, out_dir)
    except Exception as exc:
        print(f"Plotting failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
