"""Convert MapAnything per-frame camera poses -> `T_anchor_front` (11,4,4).

Uses the training-data pose convention:

    T_anchor_front[i] = inv(P_c2w[frame_0]) @ P_c2w[frame_{4i}]     (frame_0 := identity)

where P_c2w[k] is the front-camera pose (cam-to-world) at source frame k. For a
41-frame stride-1 clip the 11 latent-frame anchors are frames [0,4,8,...,40].

Convention required by the model: OpenCV camera (+X right, +Y down, +Z FORWARD),
so forward driving yields a positive Z translation. If MapAnything emits OpenGL
cameras (+Z backward) set is_opengl=True; if it emits world-to-cam set is_c2w=False.
"""
from __future__ import annotations

import torch

# 11 latent-frame source indices for a 41-frame stride-1 clip (VAE temporal stride 4).
ANCHOR_SRC_IDX = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]


def _opengl_to_opencv(P_c2w: torch.Tensor) -> torch.Tensor:
    """Flip camera Y,Z axes (right-multiply c2w): OpenGL(+Y up,+Z back) -> OpenCV(+Y down,+Z fwd)."""
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=P_c2w.dtype, device=P_c2w.device))
    return P_c2w @ flip


def build_T_anchor_front(
    poses: torch.Tensor,
    *,
    is_c2w: bool = True,
    is_opengl: bool = False,
    anchor_idx: list[int] | None = None,
) -> torch.Tensor:
    """poses: (N>=41, 4, 4) MapAnything camera poses -> T_anchor_front (11,4,4) float32."""
    anchor_idx = anchor_idx or ANCHOR_SRC_IDX
    P = poses.to(torch.float64)
    if P.ndim != 3 or P.shape[-2:] != (4, 4):
        raise ValueError(f"expected poses (N,4,4), got {tuple(poses.shape)}")
    if P.shape[0] <= max(anchor_idx):
        raise ValueError(f"need >{max(anchor_idx)} poses, got {P.shape[0]}")
    if not is_c2w:
        P = torch.linalg.inv(P)                 # world-to-cam -> cam-to-world
    if is_opengl:
        P = _opengl_to_opencv(P)
    P = P[anchor_idx]                           # (11,4,4)
    T = torch.linalg.inv(P[0]).unsqueeze(0) @ P  # relative to frame 0
    T[0] = torch.eye(4, dtype=torch.float64)
    return T.to(torch.float32)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--mapanything-pt", type=Path, required=True, help="{'poses_c2w': (N,4,4), ...}")
    ap.add_argument("--out", type=Path, required=True, help="writes {'T_anchor_front': (11,4,4)}")
    ap.add_argument("--w2c", action="store_true", help="poses are world-to-cam")
    ap.add_argument("--opengl", action="store_true", help="poses are OpenGL convention")
    ap.add_argument("--key", type=str, default="poses_c2w")
    a = ap.parse_args()
    d = torch.load(a.mapanything_pt, map_location="cpu", weights_only=False)
    T = build_T_anchor_front(d[a.key].float(), is_c2w=not a.w2c, is_opengl=a.opengl)
    torch.save({"T_anchor_front": T}, a.out)
    fwd = float(T[-1, 2, 3])
    print(f"wrote {a.out}  T[-1] translation={T[-1,:3,3].tolist()}  forward(Z)={fwd:.2f}m")
