"""Streaming trajectory writer for DFM sampling.

Writes one JSONL line per (sample_id, step_idx). Each line records:
  - structure_name
  - sample_id (which design from the batch this is)
  - step_idx, t
  - sequence (token-id list, length L)
  - mask_bool (True at positions still equal to MAS)
  - n_masked
  - mean_pred_entropy_masked (Shannon entropy of probs averaged over masked positions; null at step 0)

The writer is opened in append mode so multiple structures can share a file
when desired. Caller is responsible for setting `structure_name` and
`sample_id_offset` (to disambiguate samples across batches).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class TrajectoryWriter:
    def __init__(self, path: str, structure_name: str, sample_id_offset: int = 0,
                 chain_mask: torch.Tensor | None = None):
        """
        Args:
            path: JSONL file path; opened append.
            structure_name: identifier copied into every record.
            sample_id_offset: integer added to the row index to label samples
                (so a second batch can continue numbering).
            chain_mask: optional [B,L] tensor — when provided the recorded
                ``mask_bool`` is restricted to positions inside this mask
                (non-designable positions are not flagged as masked even if
                their token happens to equal MAS).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")
        self.structure_name = structure_name
        self.sample_id_offset = int(sample_id_offset)
        self.chain_mask = chain_mask  # [B, L] or None

    def write_step(self, step_idx: int, t: float, S: torch.Tensor,
                   mask_token_id: int, probs: torch.Tensor | None):
        """Persist one snapshot for every sample in batch S [B, L]."""
        S_cpu = S.detach().cpu().numpy()
        is_mask = (S == mask_token_id)
        if self.chain_mask is not None:
            cm_bool = self.chain_mask.bool()
            if cm_bool.shape == is_mask.shape:
                is_mask = is_mask & cm_bool
        is_mask_cpu = is_mask.detach().cpu().numpy()

        mean_H = None
        if probs is not None:
            eps = 1e-12
            H_d = -(probs * (probs.clamp_min(eps).log())).sum(dim=-1)  # [B, L]
            B = H_d.shape[0]
            mean_H = []
            for b in range(B):
                m = is_mask[b]
                if bool(m.any()):
                    mean_H.append(float(H_d[b][m].mean().item()))
                else:
                    mean_H.append(None)

        B, L = S_cpu.shape
        for b in range(B):
            rec = {
                "structure_name": self.structure_name,
                "sample_id": self.sample_id_offset + b,
                "step_idx": int(step_idx),
                "t": float(t),
                "n_masked": int(is_mask_cpu[b].sum()),
                "sequence": S_cpu[b].astype(int).tolist(),
                "mask_bool": is_mask_cpu[b].astype(bool).tolist(),
                "mean_pred_entropy_masked": (mean_H[b] if mean_H is not None else None),
            }
            self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
