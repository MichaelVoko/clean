from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# mask tokens = pdb_dataset.restype_to_int["MAS"]

#Call featurize on the input protein structure to get the feature_dict, then pass it to this function along with the model to sample sequences.
def dfm_sample(model, feature_dict, mask_token_id, dt=0.1, eps=1e-3, return_trajectory=False):
    """
    Sample discrete sequences with the DFM Euler-style categorical update.

    Args:
        model: ProteinMPNN model with forward(feature_dict, mode="dfm", t=...).
        feature_dict: model input dictionary 
        eps: numerical stability constant.
        return_trajectory: return (t, x_t) snapshots.

    Returns:
        x_t: sampled final tokens [B, L].
        trajectory: optional list of (t_float, x_t_tensor) snapshots.
    """
    if "S" not in feature_dict:
        raise ValueError("feature_dict must contain key 'S'.")

    features = feature_dict.copy()  # Avoid modifying the input dict.
    S = features["S"]
    sample_mask = features.get("mask_for_loss", features["mask"]).bool() # fixed mask for sampling, if not provided, sample all positions
    if S.dim() != 2:
        raise ValueError(f"Expected feature_dict['S'] with shape [B, L], got {tuple(S.shape)}.")

    B, L = S.shape
    device = S.device
    dtype_long = S.dtype

    x_t = torch.full((B, L), mask_token_id, dtype=dtype_long, device=device) # initialise at t=0 with all MAS tokens

    t_value = float(0.0)
    trajectory = []
    if return_trajectory:
        trajectory.append((t_value, x_t.clone()))
    model.eval() # set model to eval mode for sampling
    with torch.no_grad():
        while t_value < 1.0 - eps:
            t_tensor = torch.full((B, 1), float(t_value), device=device, dtype=torch.float32)
            h = min(dt, 1.0 - t_value) # prevents overshooting t=1.0 in the last step

            log_probs, _ = model(features, mode="dfm", t=t_tensor)

            p1 = torch.softmax(log_probs, dim=-1) # [B, L, vocab_size]

            vocab_size = p1.size(-1)
            one_hot_x_t = F.one_hot(x_t, num_classes=vocab_size).float()

            denom = max(1.0 - t_value, eps) # numerical stability as t approaches 1.0
            u = (p1 - one_hot_x_t) / denom # calculate the velocity u based on the DFM update rule
            p_next = one_hot_x_t + h * u

            p_next = torch.clamp(p_next, min=0.0) # no negative probabilities
            p_next = p_next / p_next.sum(dim=-1, keepdim=True).clamp_min(eps) # renormalize to get valid probabilities (P=1)

            x_next = torch.distributions.Categorical(probs=p_next).sample()
            x_t = torch.where(sample_mask, x_next, x_t) # Token update, keeping fixed tokens unchanged
            features["S"] = x_t

            t_value += h
            if return_trajectory:
                trajectory.append((float(t_value), x_t.clone())) #t,xt pairs

    return x_t, trajectory if return_trajectory else None


