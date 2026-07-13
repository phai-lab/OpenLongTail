from __future__ import annotations
import torch

def mat3_inverse(M: torch.Tensor) -> torch.Tensor:
    if M.shape[-2:] != (3, 3):
        raise ValueError(f'mat3_inverse expects shape (..., 3, 3), got {tuple(M.shape)}')
    in_dtype = M.dtype
    M32 = M.float()
    (a, b, c) = (M32[..., 0, 0], M32[..., 0, 1], M32[..., 0, 2])
    (d, e, f) = (M32[..., 1, 0], M32[..., 1, 1], M32[..., 1, 2])
    (g, h, i) = (M32[..., 2, 0], M32[..., 2, 1], M32[..., 2, 2])
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I = a * e - b * d
    det = a * A + b * B + c * C
    inv_det = 1.0 / det
    out = torch.stack([torch.stack([A * inv_det, D * inv_det, G * inv_det], dim=-1), torch.stack([B * inv_det, E * inv_det, H * inv_det], dim=-1), torch.stack([C * inv_det, F * inv_det, I * inv_det], dim=-1)], dim=-2)
    return out.to(in_dtype)

def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2:] != (4, 4):
        raise ValueError(f'se3_inverse expects shape (..., 4, 4), got {tuple(T.shape)}')
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_T = R.transpose(-1, -2)
    t_inv = -(R_T @ t)
    top = torch.cat([R_T, t_inv], dim=-1)
    bottom = T[..., 3:4, :]
    return torch.cat([top, bottom], dim=-2)
