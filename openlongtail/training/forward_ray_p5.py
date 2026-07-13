from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import torch
import torch.distributed as dist
from openlongtail.data.transforms import se3_inverse, se3_log_rotation_angle
from openlongtail.models.plucker import compute_plucker_rays
from openlongtail.models.dit_p5_vace import DiTP5VACE
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p3 import _prepare_z_all_for_p3, _resolve_batch_device
from openlongtail.training.forward_wan21 import sample_sigma_unified
from openlongtail.training.schedulers import FlowMatchScheduler
FRONT_IDX = 0
P5_TARGET_VIEWS = (1, 2, 3, 4, 5)
P5_ROLE_CONDITION = 0
P5_ROLE_TARGET = 1

@dataclass
class P5StreamBatch:
    z_streams: torch.Tensor
    v_target: torch.Tensor
    target_clean: torch.Tensor
    stream_view_ids: torch.Tensor
    stream_role_ids: torch.Tensor
    target_local_indices: torch.Tensor
    target_noise: torch.Tensor

@dataclass
class RayTrainingStepP5Output:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | str | list[int] | float]
    sigma: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor

def canonical_p5_target_views(target_views: list[int] | tuple[int, ...] | None=None, num_targets: int | None=None) -> list[int]:
    views = [int(item) for item in (P5_TARGET_VIEWS if target_views is None else target_views)]
    expected = len(views) if num_targets is None else int(num_targets)
    if expected <= 0 or expected > len(P5_TARGET_VIEWS):
        raise ValueError(f'P5 num_targets must be in [1, 5], got {expected}')
    if len(views) != expected:
        raise ValueError(f'P5 expects exactly {expected} target views, got {views}')
    if len(set(views)) != expected:
        raise ValueError(f'P5 target views must be unique, got {views}')
    if FRONT_IDX in views:
        raise ValueError(f'P5 targets exclude the front condition view {FRONT_IDX}, got {views}')
    if any((view_id < 0 or view_id >= 6 for view_id in views)):
        raise ValueError(f'P5 target views must be in [0, 5], got {views}')
    return views

def sample_p5_target_views(num_targets: int=5, generator: torch.Generator | None=None) -> list[int]:
    if num_targets <= 0 or num_targets > len(P5_TARGET_VIEWS):
        raise ValueError(f'P5 num_targets must be in [1, 5], got {num_targets}')
    selected = torch.randperm(len(P5_TARGET_VIEWS), generator=generator)[:num_targets]
    return sorted((P5_TARGET_VIEWS[int(idx)] for idx in selected.tolist()))

def broadcast_p5_target_views(target_views: list[int] | None, device: torch.device, num_targets: int=5) -> list[int]:
    if not (dist.is_available() and dist.is_initialized()):
        if target_views is None:
            raise ValueError('target_views must be provided in single-process P5 target resolution')
        return canonical_p5_target_views(target_views, num_targets=num_targets)
    buffer = torch.zeros(num_targets, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        if target_views is None:
            raise ValueError('rank 0 target_views must be provided for P5 broadcast')
        buffer[:] = torch.tensor(canonical_p5_target_views(target_views, num_targets=num_targets), dtype=torch.long, device=device)
    dist.broadcast(buffer, src=0)
    return canonical_p5_target_views([int(item) for item in buffer.tolist()], num_targets=num_targets)

def resolve_p5_target_views(device: torch.device, num_targets: int=5) -> list[int]:
    target_views_local = sample_p5_target_views(num_targets) if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0 else None
    return broadcast_p5_target_views(target_views_local, device, num_targets=num_targets)

def p5_stream_ids(target_views: list[int] | tuple[int, ...] | None=None, device: torch.device | str | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    views = canonical_p5_target_views(target_views)
    stream_view_ids = torch.tensor([FRONT_IDX, *views], device=device, dtype=torch.long)
    stream_role_ids = torch.tensor([P5_ROLE_CONDITION, *[P5_ROLE_TARGET] * len(views)], device=device, dtype=torch.long)
    return (stream_view_ids, stream_role_ids)

def _normalize_shared_noise(shared_noise: torch.Tensor, target_clean: torch.Tensor) -> torch.Tensor:
    expected = (target_clean.shape[0], 1, *target_clean.shape[2:])
    if tuple(shared_noise.shape) == expected:
        return shared_noise
    expanded = target_clean[:, :1].shape
    if tuple(shared_noise.shape) != expanded:
        raise ValueError(f'expected shared_noise shape {expected}, got {tuple(shared_noise.shape)}')
    return shared_noise

def build_shared_private_noise(target_clean: torch.Tensor, alpha: float=0.5, private_noise: torch.Tensor | None=None, shared_noise: torch.Tensor | None=None) -> torch.Tensor:
    if target_clean.ndim != 6:
        raise ValueError(f'expected target_clean shape (B, V, C, T, H, W), got {tuple(target_clean.shape)}')
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f'expected alpha in [0, 1], got {alpha}')
    if private_noise is None:
        private_noise = torch.randn_like(target_clean)
    if tuple(private_noise.shape) != tuple(target_clean.shape):
        raise ValueError(f'expected private_noise shape {tuple(target_clean.shape)}, got {tuple(private_noise.shape)}')
    if shared_noise is None:
        shared_noise = torch.randn_like(target_clean[:, :1])
    else:
        shared_noise = _normalize_shared_noise(shared_noise, target_clean).to(device=target_clean.device, dtype=target_clean.dtype)
    private_noise = private_noise.to(device=target_clean.device, dtype=target_clean.dtype)
    alpha_f = float(alpha)
    private_scale = math.sqrt(max(0.0, 1.0 - alpha_f * alpha_f))
    return alpha_f * shared_noise + private_scale * private_noise

def build_p5_streams_from_latents(z_all: torch.Tensor, sigma: torch.Tensor, target_views: list[int] | tuple[int, ...] | None=None, shared_noise_alpha: float=0.5, private_noise: torch.Tensor | None=None, shared_noise: torch.Tensor | None=None) -> P5StreamBatch:
    if z_all.ndim != 6 or z_all.shape[1] != 6 or z_all.shape[2] != 16:
        raise ValueError(f'expected z_all shape (B, 6, 16, T, H, W), got {tuple(z_all.shape)}')
    views = canonical_p5_target_views(target_views)
    batch = z_all.shape[0]
    if sigma.shape != (batch,):
        raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
    target_clean = z_all[:, views]
    target_noise = build_shared_private_noise(target_clean, alpha=shared_noise_alpha, private_noise=private_noise, shared_noise=shared_noise)
    sigma_b = sigma.view(batch, 1, 1, 1, 1, 1).to(dtype=z_all.dtype)
    target_noisy = (1.0 - sigma_b) * target_clean + sigma_b * target_noise
    v_target = target_noise - target_clean
    z_streams = torch.cat([z_all[:, FRONT_IDX:FRONT_IDX + 1], target_noisy], dim=1)
    (stream_view_ids, stream_role_ids) = p5_stream_ids(views, device=z_all.device)
    target_local_indices = torch.arange(1, len(views) + 1, device=z_all.device, dtype=torch.long)
    return P5StreamBatch(z_streams=z_streams, v_target=v_target, target_clean=target_clean, stream_view_ids=stream_view_ids, stream_role_ids=stream_role_ids, target_local_indices=target_local_indices, target_noise=target_noise)

def build_p5_dynamic_plucker_streams(K: torch.Tensor, E_rig: torch.Tensor, T_anchor_front: torch.Tensor, stream_view_ids: torch.Tensor, latent_frames: int, h_tok: int, w_tok: int, dtype: torch.dtype | None=None) -> torch.Tensor:
    if K.ndim != 4 or K.shape[1:] != (6, 3, 3):
        raise ValueError(f'expected K shape (B, 6, 3, 3), got {tuple(K.shape)}')
    if E_rig.ndim != 4 or E_rig.shape[1:] != (6, 4, 4):
        raise ValueError(f'expected E shape (B, 6, 4, 4), got {tuple(E_rig.shape)}')
    if T_anchor_front.ndim != 4 or T_anchor_front.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor_front shape (B, T, 4, 4), got {tuple(T_anchor_front.shape)}')
    if T_anchor_front.shape[0] != K.shape[0]:
        raise ValueError(f'expected T_anchor_front batch {K.shape[0]}, got {T_anchor_front.shape[0]}')
    if T_anchor_front.shape[1] < latent_frames:
        raise ValueError(f'expected T_anchor_front temporal dim >= {latent_frames}, got {T_anchor_front.shape[1]}')
    if stream_view_ids.ndim != 1:
        raise ValueError(f'expected stream_view_ids shape (S,), got {tuple(stream_view_ids.shape)}')
    device = K.device
    E_front = E_rig[:, FRONT_IDX:FRONT_IDX + 1].to(device=device, dtype=torch.float32)
    E_rig_f = E_rig.to(device=device, dtype=torch.float32)
    T = T_anchor_front.to(device=device, dtype=torch.float32)
    E_front_cam = se3_inverse(E_front) @ E_rig_f
    E_anchor_cam = T[:, :, None] @ E_front_cam[:, None]
    plucker_all = compute_plucker_rays(K.to(dtype=torch.float32), E_anchor_cam, h_tok=h_tok, w_tok=w_tok)
    plucker = plucker_all[:, stream_view_ids.to(device)]
    if plucker.shape[2] < latent_frames:
        raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker.shape[2]}')
    plucker = plucker[:, :, :latent_frames]
    return plucker if dtype is None else plucker.to(dtype=dtype)

def training_step_ray_p5(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, device: torch.device | str | None=None, target_views: list[int] | tuple[int, ...] | None=None, num_targets: int=5, shared_noise_alpha: float=0.5) -> RayTrainingStepP5Output:
    resolved_device = _resolve_batch_device(batch, device)
    views = canonical_p5_target_views(target_views, num_targets=num_targets) if target_views is not None else resolve_p5_target_views(resolved_device, num_targets=num_targets)
    z_all = _prepare_z_all_for_p3(batch, components, resolved_device, views)
    sigma = sample_sigma_unified(z_all.shape[0], scheduler, resolved_device).to(dtype=z_all.dtype)
    streams = build_p5_streams_from_latents(z_all, sigma, target_views=views, shared_noise_alpha=shared_noise_alpha)
    if 'T_anchor_front' not in batch:
        raise KeyError("P5 training requires batch['T_anchor_front']")
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E_rig = batch['E'].to(device=resolved_device, dtype=torch.float32)
    T_anchor_front = batch['T_anchor_front'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    latent_frames = z_all.shape[3]
    with torch.no_grad():
        plucker = build_p5_dynamic_plucker_streams(K, E_rig, T_anchor_front, streams.stream_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_all.dtype)
    dit = components.low_dit
    if dit is None:
        raise ValueError('P5 requires components.low_dit')
    v_pred = dit(streams.z_streams, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), plucker, streams.stream_view_ids, streams.stream_role_ids, T_anchor_front)
    pred_targets = v_pred[:, streams.target_local_indices]
    loss = ((pred_targets - streams.v_target) ** 2).mean()
    front_translation_norm = T_anchor_front[:, :latent_frames, :3, 3].norm(dim=-1)
    metrics: dict[str, torch.Tensor | str | list[int] | float] = {'loss': loss.detach(), 'stage': 'unified', 'training_method': 'p5_vace_sync_front_multiview_completion', 'target_views': views, 'stream_view_ids': [int(item) for item in streams.stream_view_ids.detach().cpu().tolist()], 'stream_role_ids': [int(item) for item in streams.stream_role_ids.detach().cpu().tolist()], 'sigma_mean': sigma.detach().float().mean(), 'sigma_min': sigma.detach().float().min(), 'sigma_max': sigma.detach().float().max(), 'shared_noise_alpha': float(shared_noise_alpha), 'front_translation_norm_mean': front_translation_norm.mean().detach(), 'front_translation_norm_max': front_translation_norm.max().detach(), 'front_rotation_angle_max': se3_log_rotation_angle(T_anchor_front[:, :latent_frames]).abs().max().detach()}
    if isinstance(dit, DiTP5VACE):
        metrics['sync_temporal_window'] = int(dit.sync_temporal_window)
    else:
        sync_window = getattr(dit, 'sync_temporal_window', None)
        if sync_window is not None:
            metrics['sync_temporal_window'] = int(sync_window)
    for (local_idx, global_view_idx) in enumerate(views):
        metrics[f'flow_view_{global_view_idx}'] = ((pred_targets[:, local_idx] - streams.v_target[:, local_idx]) ** 2).mean().detach()
    return RayTrainingStepP5Output(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=streams.v_target)
