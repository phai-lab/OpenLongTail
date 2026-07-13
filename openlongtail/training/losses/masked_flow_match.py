from __future__ import annotations
import torch

def sample_sigma_for_stage(B: int, stage: str, device: torch.device | str) -> torch.Tensor:
    if B <= 0:
        raise ValueError(f'expected B > 0, got {B}')
    if stage in ('A.0', 'A.1'):
        t = torch.rand(B, device=device) * 0.643
    elif stage == 'B':
        t = torch.rand(B, device=device) * (1.0 - 0.643) + 0.643
    else:
        raise ValueError(f'expected stage one of A.0, A.1, B, got {stage!r}')
    return 5.0 * t / (1.0 + 4.0 * t)

def add_masked_flow_matching_noise(z_all: torch.Tensor, sigma: torch.Tensor, front_idx: int=0) -> tuple[torch.Tensor, torch.Tensor]:
    if z_all.ndim != 6:
        raise ValueError(f'expected z_all shape (B, 6, C, T, H, W), got {tuple(z_all.shape)}')
    if sigma.shape != (z_all.shape[0],):
        raise ValueError(f'expected sigma shape ({z_all.shape[0]},), got {tuple(sigma.shape)}')
    if not 0 <= front_idx < z_all.shape[1]:
        raise ValueError(f'expected front_idx in [0, {z_all.shape[1]}), got {front_idx}')
    sigma_b = sigma.to(device=z_all.device, dtype=z_all.dtype).view(-1, 1, 1, 1, 1, 1)
    eps = torch.randn_like(z_all)
    z_noisy = (1.0 - sigma_b) * z_all + sigma_b * eps
    v_target = eps - z_all
    z_noisy[:, front_idx] = z_all[:, front_idx]
    v_target[:, front_idx] = 0
    return (z_noisy, v_target)

def compute_masked_flow_matching_loss(v_pred: torch.Tensor, v_target: torch.Tensor, front_idx: int=0) -> torch.Tensor:
    if v_pred.shape != v_target.shape:
        raise ValueError(f'expected v_pred and v_target same shape, got {tuple(v_pred.shape)} and {tuple(v_target.shape)}')
    if v_pred.ndim != 6:
        raise ValueError(f'expected v_pred shape (B, 6, C, T, H, W), got {tuple(v_pred.shape)}')
    if not 0 <= front_idx < v_pred.shape[1]:
        raise ValueError(f'expected front_idx in [0, {v_pred.shape[1]}), got {front_idx}')
    view_mask = torch.ones(v_pred.shape[1], dtype=torch.bool, device=v_pred.device)
    view_mask[front_idx] = False
    return ((v_pred[:, view_mask] - v_target[:, view_mask]) ** 2).mean()
