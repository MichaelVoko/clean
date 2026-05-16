#!/usr/bin/env python
"""Select ~N evenly-spaced checkpoints from dfm_base/s_*.pt.

Emits one step number per line to selected_steps.txt (and stdout). The smallest
and largest available steps are always included; the middle is filled in by
linear interpolation in step-space then snapped to the nearest available
checkpoint. Re-runs produce the same set as long as dfm_base is unchanged.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "dfm_base"
DEFAULT_OUT = REPO_ROOT / "evaluation/sweeps/dfm_sweep/selected_steps.txt"

STEP_RE = re.compile(r"^s_(\d+)\.pt$")


def available_steps(ckpt_dir: Path) -> list[int]:
    steps = []
    for p in ckpt_dir.iterdir():
        m = STEP_RE.match(p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def pick_evenly_spaced(steps: list[int], n: int) -> list[int]:
    if n <= 0:
        return []
    if n >= len(steps):
        return list(steps)
    s_min, s_max = steps[0], steps[-1]
    targets = [s_min + (s_max - s_min) * i / (n - 1) for i in range(n)]
    picked: list[int] = []
    for t in targets:
        nearest = min(steps, key=lambda s: abs(s - t))
        if not picked or nearest != picked[-1]:
            picked.append(nearest)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    steps = available_steps(args.ckpt_dir)
    if not steps:
        raise SystemExit(f"No s_<step>.pt files found in {args.ckpt_dir}")

    picked = pick_evenly_spaced(steps, args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(s) for s in picked) + "\n")

    print(f"Available: {len(steps)} checkpoints, step range [{steps[0]}, {steps[-1]}]")
    print(f"Selected:  {len(picked)} checkpoints -> {args.out}")
    for s in picked:
        print(f"  s_{s}.pt")


if __name__ == "__main__":
    main()
