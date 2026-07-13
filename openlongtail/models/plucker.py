from __future__ import annotations
import torch
import torch.nn.functional as F
from openlongtail.models.inverse_math import mat3_inverse
TOKEN_DOWNSAMPLE = 16.0
LATENT_TIME = 21
NUM_VIEWS = 6
PLUCKER_MOMENT_SCALE = 25.0

def compute_plucker_rays(K: torch.Tensor, E: torch.Tensor, h_tok: int=30, w_tok: int=52, t_lat: int | None=None) -> torch.Tensor:
    if K.ndim != 4 or K.shape[1:] != (NUM_VIEWS, 3, 3):
        raise ValueError(f'expected K shape (B, 6, 3, 3), got {tuple(K.shape)}')
    if E.ndim not in (4, 5):
        raise ValueError(f'expected E ndim 4 or 5, got shape {tuple(E.shape)}')
    if E.ndim == 4 and E.shape[1:] != (NUM_VIEWS, 4, 4):
        raise ValueError(f'expected E shape (B, 6, 4, 4), got {tuple(E.shape)}')
    if E.ndim == 5 and E.shape[2:] != (NUM_VIEWS, 4, 4):
        raise ValueError(f'expected E shape (B, T, 6, 4, 4), got {tuple(E.shape)}')
    if K.shape[0] != E.shape[0]:
        raise ValueError(f'expected K and E batch dims to match, got {K.shape[0]} and {E.shape[0]}')
    if h_tok <= 0 or w_tok <= 0:
        raise ValueError(f'expected positive token grid, got h_tok={h_tok}, w_tok={w_tok}')
    dtype = K.dtype
    device = K.device
    E = E.to(device=device, dtype=dtype)
    K_tok = K.clone()
    K_tok[..., 0, :] /= TOKEN_DOWNSAMPLE
    K_tok[..., 1, :] /= TOKEN_DOWNSAMPLE
    inv_k = mat3_inverse(K_tok)
    (grid_y, grid_x) = torch.meshgrid(torch.arange(h_tok, device=device, dtype=dtype), torch.arange(w_tok, device=device, dtype=dtype), indexing='ij')
    pixels = torch.stack([grid_x + 0.5, grid_y + 0.5, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    dirs_cam = torch.einsum('bvij,nj->bvni', inv_k, pixels)
    batch = K.shape[0]
    if E.ndim == 4:
        if t_lat is not None and int(t_lat) > LATENT_TIME:
            raise ValueError(f'static Plucker path supports at most {LATENT_TIME} latent frames, got {t_lat}')
        dirs_vehicle = torch.einsum('bvij,bvnj->bvni', E[..., :3, :3], dirs_cam)
        dirs_vehicle = F.normalize(dirs_vehicle, dim=-1, eps=1e-08)
        origin = E[..., :3, 3].unsqueeze(-2)
        moment = torch.cross(origin.expand_as(dirs_vehicle), dirs_vehicle, dim=-1)
        plucker = torch.cat([dirs_vehicle, moment], dim=-1)
        plucker = plucker.reshape(batch, NUM_VIEWS, h_tok, w_tok, 6).permute(0, 1, 4, 2, 3).contiguous()
        return plucker.unsqueeze(2).expand(-1, -1, LATENT_TIME, -1, -1, -1).contiguous()
    dirs_anchor = torch.einsum('btvij,bvnj->btvni', E[..., :3, :3], dirs_cam)
    dirs_anchor = F.normalize(dirs_anchor, dim=-1, eps=1e-08)
    origin = E[..., :3, 3].unsqueeze(-2)
    moment = torch.cross(origin.expand_as(dirs_anchor), dirs_anchor, dim=-1) / PLUCKER_MOMENT_SCALE
    if t_lat is not None and int(t_lat) != E.shape[1]:
        raise ValueError(f'dynamic Plucker temporal dim {E.shape[1]} does not match t_lat={t_lat}')
    t_lat = E.shape[1]
    plucker = torch.cat([dirs_anchor, moment], dim=-1)
    plucker = plucker.reshape(batch, t_lat, NUM_VIEWS, h_tok, w_tok, 6).permute(0, 2, 1, 5, 3, 4).contiguous()
    plucker_abs_max = float(plucker.detach().abs().max().cpu())
    if plucker_abs_max >= 1000.0:
        translation_max = float(E[..., :3, 3].detach().norm(dim=-1).max().cpu())
        raise ValueError(f'dynamic Plucker values exceeded safety bound: abs_max={plucker_abs_max:.3f}, T_anchor_front translation max={translation_max:.3f}')
    return plucker

def compute_plucker_world(K: torch.Tensor, E: torch.Tensor, T_anchor: torch.Tensor, H: int=60, W: int=104) -> torch.Tensor:
    if K.ndim != 3 or K.shape[-2:] != (3, 3):
        raise ValueError(f'expected K shape (B, 3, 3), got {tuple(K.shape)}')
    if E.ndim != 3 or E.shape[-2:] != (4, 4):
        raise ValueError(f'expected E shape (B, 4, 4), got {tuple(E.shape)}')
    if T_anchor.ndim != 4 or T_anchor.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor shape (B, T, 4, 4), got {tuple(T_anchor.shape)}')
    if K.shape[0] != E.shape[0] or K.shape[0] != T_anchor.shape[0]:
        raise ValueError(f'expected matching batch dims, got K={K.shape[0]}, E={E.shape[0]}, T_anchor={T_anchor.shape[0]}')
    if H <= 0 or W <= 0:
        raise ValueError(f'expected positive pixel grid, got H={H}, W={W}')
    dtype = K.dtype
    device = K.device
    E = E.to(device=device, dtype=dtype)
    T_anchor = T_anchor.to(device=device, dtype=dtype)
    B = K.shape[0]
    T = T_anchor.shape[1]
    (grid_y, grid_x) = torch.meshgrid(torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype), indexing='ij')
    pixels = torch.stack([grid_x + 0.5, grid_y + 0.5, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    inv_k = mat3_inverse(K)
    dirs_cam = torch.einsum('bij,nj->bni', inv_k, pixels)
    dirs_cam = F.normalize(dirs_cam, dim=-1, eps=1e-08)
    R_e = E[..., :3, :3]
    t_e = E[..., :3, 3]
    dirs_world = torch.einsum('bij,bnj->bni', R_e, dirs_cam)
    origins_world = t_e.unsqueeze(1).expand_as(dirs_world)
    R_a = T_anchor[..., :3, :3]
    t_a = T_anchor[..., :3, 3]
    dirs_anchor = torch.einsum('btij,bnj->btni', R_a, dirs_world)
    dirs_anchor = F.normalize(dirs_anchor, dim=-1, eps=1e-08)
    origins_anchor = torch.einsum('btij,bnj->btni', R_a, origins_world) + t_a.unsqueeze(2)
    moment = torch.cross(origins_anchor, dirs_anchor, dim=-1)
    plucker = torch.cat([dirs_anchor, moment], dim=-1)
    plucker = plucker.reshape(B, T, H, W, 6).contiguous()
    return plucker
