import torch
import torch.nn.functional as F

_SPECIAL_TOKENS = ["UNK", "DX", "RX", "MAS", "PAD"]


def _forward_probs(model, feat, S, t_value, bias, eps):
    """Forward pass → bias-adjusted probability distribution at T=1."""
    B = S.shape[0]
    device = S.device
    t_tensor = torch.full((B, 1), t_value, device=device, dtype=torch.float32)
    log_probs, _ = model.forward_dfm({**feat, "S": S}, t_tensor)
    probs = F.softmax(log_probs + bias, dim=-1)
    for tok in _SPECIAL_TOKENS:
        probs[..., model.restype_to_int[tok]] = 0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return log_probs, probs


def _velocity(probs, one_hot_S, t_value, eps):
    """Discrete flow matching velocity: u = (p1 - x_t) / (1 - t)."""
    return (probs - one_hot_S) / max(1.0 - t_value, eps)


def _apply_update(one_hot_S, u, h, eps):
    """Euler update: p_next = clamp(x_t + h*u), renormalised."""
    p = (one_hot_S + h * u).clamp(min=0.0)
    return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)


# ---------------------------------------------------------------------------

class EulerSampler:
    def __init__(self, dt):
        self.dt = dt

    def step(self, model, feat, S, t_value, bias, temperature, eps=1e-3):
        h = min(self.dt, 1.0 - t_value)
        log_probs, probs = _forward_probs(model, feat, S, t_value, bias, eps)
        one_hot_S = F.one_hot(S, num_classes=probs.size(-1)).float()
        u = _velocity(probs, one_hot_S, t_value, eps)
        p_next = _apply_update(one_hot_S, u, h, eps)
        return log_probs, probs, p_next, h


class EDSSampler:
    """Entropic Discrete Schedule sampler.

    Pre-computed monotone time grid t_grid = [0=t_0 < t_1 < ... < t_K=1],
    constructed offline so steps are uniform in cumulative neural entropy.
    Each call to ``step`` consumes one grid interval [t_k, t_{k+1}].
    """

    def __init__(self, t_grid):
        self.t_grid = [float(x) for x in t_grid]
        if len(self.t_grid) < 2:
            raise ValueError("EDSSampler requires at least 2 grid points.")
        if self.t_grid[0] != 0.0 or self.t_grid[-1] < 1.0 - 1e-6:
            raise ValueError(f"t_grid must span [0, 1]; got {self.t_grid[0]}..{self.t_grid[-1]}")
        self._k = 0  # next interval index

    def reset(self):
        self._k = 0

    @property
    def num_steps(self):
        return len(self.t_grid) - 1

    @property
    def done(self):
        return self._k >= self.num_steps

    def step(self, model, feat, S, t_value, bias, temperature, eps=1e-3):
        # Drive t from the grid; the caller's t_value should match t_grid[self._k]
        # (we don't enforce strictly so callers can clamp at the boundary).
        if self.done:
            raise RuntimeError("EDSSampler exhausted; reset() before reuse.")
        t_lo = self.t_grid[self._k]
        t_hi = self.t_grid[self._k + 1]
        h = max(t_hi - t_lo, eps)
        log_probs, probs = _forward_probs(model, feat, S, t_lo, bias, eps)
        one_hot_S = F.one_hot(S, num_classes=probs.size(-1)).float()
        u = _velocity(probs, one_hot_S, t_lo, eps)
        p_next = _apply_update(one_hot_S, u, h, eps)
        self._k += 1
        return log_probs, probs, p_next, h
