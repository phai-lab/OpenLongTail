from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.distributed as dist
from openlongtail.models.wan_vae import normalize_rgb_for_wan_vae
from openlongtail.models.plucker import compute_plucker_rays
from openlongtail.training.losses.masked_flow_match import add_masked_flow_matching_noise, compute_masked_flow_matching_loss, sample_sigma_for_stage
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.view_parallel import ViewParallelContext, broadcast_sigma_within_node, gather_view_tensor
VIEW_METRIC_NAMES: tuple[tuple[int, str], ...] = ((1, 'cross_left'), (2, 'cross_right'), (3, 'rear_left'), (4, 'rear_right'), (5, 'rear_tele'))

def sample_active_views(num_active_targets: int, generator: torch.Generator | None=None) -> list[int]:
    if num_active_targets == 5:
        return [0, 1, 2, 3, 4, 5]
    if num_active_targets == 3:
        indices = torch.randperm(5, generator=generator)[:3].tolist()
        non_front = sorted((int(idx) + 1 for idx in indices))
        return [0] + non_front
    raise ValueError(f'num_active_targets must be 3 or 5, got {num_active_targets}')

def broadcast_active_views(active_views: list[int] | None, num_active_targets: int, device: torch.device) -> list[int]:
    if num_active_targets == 5:
        return [0, 1, 2, 3, 4, 5]
    length = num_active_targets + 1
    if not (dist.is_available() and dist.is_initialized()):
        assert active_views is not None
        return active_views
    buffer = torch.zeros(length, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        assert active_views is not None and len(active_views) == length
        buffer[:] = torch.tensor(active_views, device=device, dtype=torch.long)
    dist.broadcast(buffer, src=0)
    return [int(item) for item in buffer.tolist()]

@dataclass
class RayTrainingComponents:
    vae: Any
    low_dit: torch.nn.Module
    high_dit: torch.nn.Module | None = None
    shared_modules: torch.nn.Module | None = None

@dataclass
class RayTrainingStepOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | str | list[int]]
    sigma: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor

def _move_vae_to_device(vae: Any, device: torch.device) -> None:
    if hasattr(vae, 'model'):
        vae.model.to(device)
    if hasattr(vae, 'mean'):
        vae.mean = vae.mean.to(device)
    if hasattr(vae, 'std'):
        vae.std = vae.std.to(device)
    if hasattr(vae, 'mean') and hasattr(vae, 'std'):
        vae.scale = [vae.mean, 1.0 / vae.std]
    if hasattr(vae, 'device'):
        vae.device = str(device)

@torch.no_grad()
def encode_batch_with_vae(vae: Any, rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 6 or rgb.shape[3] != 3:
        raise ValueError(f'expected rgb shape (B, V, T, 3, H, W), got {tuple(rgb.shape)}')
    _move_vae_to_device(vae, rgb.device)
    (batch, views) = rgb.shape[:2]
    z_views: list[torch.Tensor] = []
    for view_idx in range(views):
        videos = [normalize_rgb_for_wan_vae(rgb[item_idx, view_idx].permute(1, 0, 2, 3).contiguous(), source_range='raw_0_255') for item_idx in range(batch)]
        encoded = vae.encode(videos)
        z_views.append(torch.stack([item.to(device=rgb.device) for item in encoded], dim=0))
    return torch.stack(z_views, dim=1)

@torch.no_grad()
def encode_batch_view_with_vae(vae: Any, rgb: torch.Tensor, view_id: int) -> torch.Tensor:
    if rgb.ndim != 6 or rgb.shape[1] != 6 or rgb.shape[3] != 3:
        raise ValueError(f'expected rgb shape (B, 6, T, 3, H, W), got {tuple(rgb.shape)}')
    if view_id < 0 or view_id >= rgb.shape[1]:
        raise ValueError(f'expected view_id in [0, {rgb.shape[1] - 1}], got {view_id}')
    _move_vae_to_device(vae, rgb.device)
    videos = [normalize_rgb_for_wan_vae(rgb[item_idx, view_id].permute(1, 0, 2, 3).contiguous(), source_range='raw_0_255') for item_idx in range(rgb.shape[0])]
    encoded = vae.encode(videos)
    return torch.stack([item.to(device=rgb.device) for item in encoded], dim=0)

def _select_dit(components: RayTrainingComponents, stage: str) -> torch.nn.Module:
    if stage in ('A.0', 'A.1'):
        return components.low_dit
    if stage == 'B':
        if components.high_dit is None:
            raise ValueError('stage B requires components.high_dit')
        return components.high_dit
    raise ValueError(f'expected stage one of A.0, A.1, B, got {stage!r}')

def _resolve_batch_device(batch: dict[str, Any], device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if 'z_all' in batch:
        return batch['z_all'].device
    return batch['rgb'].device

def _resolve_active_views(num_active_targets: int, device: torch.device) -> list[int]:
    if num_active_targets == 5:
        return [0, 1, 2, 3, 4, 5]
    active_views_local = sample_active_views(num_active_targets) if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0 else None
    return broadcast_active_views(active_views_local, num_active_targets, device)

def _prepare_z_all(batch: dict[str, Any], components: RayTrainingComponents, resolved_device: torch.device, active_views: list[int]) -> torch.Tensor:
    if 'z_all' in batch:
        z_all = batch['z_all'].to(resolved_device)
        if z_all.ndim != 6 or z_all.shape[1] != 6:
            raise ValueError(f"expected cached z_all shape (B, 6, 16, T', H', W'), got {tuple(z_all.shape)}")
        return z_all if active_views == [0, 1, 2, 3, 4, 5] else z_all[:, active_views]
    rgb = batch['rgb'].to(resolved_device)
    rgb_active = rgb if active_views == [0, 1, 2, 3, 4, 5] else rgb[:, active_views]
    z_all = encode_batch_with_vae(components.vae, rgb_active)
    if resolved_device.type == 'cuda':
        _move_vae_to_device(components.vae, torch.device('cpu'))
        torch.cuda.empty_cache()
    return z_all

def training_step_ray(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, stage: str, device: torch.device | str | None=None, num_active_targets: int=5) -> RayTrainingStepOutput:
    del scheduler
    resolved_device = _resolve_batch_device(batch, device)
    active_views = _resolve_active_views(num_active_targets, resolved_device)
    z_all = _prepare_z_all(batch, components, resolved_device, active_views)
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E = batch['E'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    with torch.no_grad():
        plucker = compute_plucker_rays(K, E, h_tok=h_tok, w_tok=w_tok).to(dtype=z_all.dtype)
        if active_views != [0, 1, 2, 3, 4, 5]:
            plucker = plucker[:, active_views]
        latent_frames = z_all.shape[3]
        if plucker.shape[2] < latent_frames:
            raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker.shape[2]}')
        plucker = plucker[:, :, :latent_frames]
    sigma = sample_sigma_for_stage(z_all.shape[0], stage, resolved_device).to(dtype=z_all.dtype)
    (z_noisy, v_target) = add_masked_flow_matching_noise(z_all, sigma, front_idx=0)
    dit = _select_dit(components, stage)
    view_ids = batch['view_ids'][0].to(resolved_device) if batch['view_ids'].ndim == 2 else batch['view_ids'].to(resolved_device)
    if active_views != [0, 1, 2, 3, 4, 5]:
        view_ids = torch.tensor(active_views, device=resolved_device, dtype=torch.long)
    forward_kwargs: dict[str, object] = {}
    if active_views != [0, 1, 2, 3, 4, 5]:
        forward_kwargs['active_views'] = active_views
    v_pred = dit(z_noisy, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), plucker, view_ids, **forward_kwargs)
    loss = compute_masked_flow_matching_loss(v_pred, v_target, front_idx=0)
    metrics: dict[str, torch.Tensor | str | list[int]] = {'loss': loss.detach(), 'stage': stage, 'active_views': active_views}
    name_by_view = dict(VIEW_METRIC_NAMES)
    for (local_idx, global_view_idx) in enumerate(active_views):
        if global_view_idx == 0:
            continue
        metrics[f'flow_{name_by_view[global_view_idx]}'] = ((v_pred[:, local_idx] - v_target[:, local_idx]) ** 2).mean().detach()
    return RayTrainingStepOutput(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=v_target)

def _sample_view_parallel_sigma(batch: int, stage: str, device: torch.device, dtype: torch.dtype, view_context: ViewParallelContext) -> torch.Tensor:
    if view_context.group is not None and view_context.view_id != 0:
        sigma = torch.zeros(batch, device=device, dtype=dtype)
    else:
        sigma = sample_sigma_for_stage(batch, stage, device).to(dtype=dtype)
    return broadcast_sigma_within_node(sigma, view_context)

def training_step_ray_view_parallel(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, stage: str, view_context: ViewParallelContext, device: torch.device | str | None=None, num_active_targets: int=5) -> RayTrainingStepOutput:
    assert num_active_targets == 5, 'view_parallel training does not support view dropout in V1.1'
    if 'z_all' in batch:
        raise ValueError('latent cache training is not supported with view_parallel mode')
    del scheduler
    rgb = batch['rgb']
    resolved_device = torch.device(device) if device is not None else rgb.device
    rgb = rgb.to(resolved_device)
    view_id = view_context.view_id
    z_view = encode_batch_view_with_vae(components.vae, rgb, view_id)
    if resolved_device.type == 'cuda':
        _move_vae_to_device(components.vae, torch.device('cpu'))
        torch.cuda.empty_cache()
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E = batch['E'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_view.shape[-2] // 2
    w_tok = z_view.shape[-1] // 2
    with torch.no_grad():
        plucker = compute_plucker_rays(K, E, h_tok=h_tok, w_tok=w_tok).to(dtype=z_view.dtype)
        latent_frames = z_view.shape[2]
        if plucker.shape[2] < latent_frames:
            raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker.shape[2]}')
        plucker = plucker[:, :, :latent_frames]
    sigma = _sample_view_parallel_sigma(z_view.shape[0], stage, resolved_device, z_view.dtype, view_context)
    if view_id == 0:
        z_noisy = z_view
        v_target = torch.zeros_like(z_view)
    else:
        sigma_b = sigma.view(-1, 1, 1, 1, 1)
        eps = torch.randn_like(z_view)
        z_noisy = (1.0 - sigma_b) * z_view + sigma_b * eps
        v_target = eps - z_view
    dit = _select_dit(components, stage)
    hidden_gather = lambda hidden: gather_view_tensor(hidden, view_id=view_id, group=view_context.group, views_per_node=view_context.views_per_node)
    v_pred = dit(z_noisy, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), plucker, view_id, hidden_gather)
    loss = v_pred.sum() * 0.0 if view_id == 0 else ((v_pred - v_target) ** 2).mean()
    metrics: dict[str, torch.Tensor | str | list[int]] = {'loss': loss.detach(), 'stage': stage, 'view_name': view_context.view_name, 'active_views': [0, 1, 2, 3, 4, 5]}
    if view_id != 0:
        metrics[f'flow_{view_context.view_name}'] = ((v_pred - v_target) ** 2).mean().detach()
    return RayTrainingStepOutput(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=v_target)

def training_step_ray_svt_recompute(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, stage: str, device: torch.device | str | None=None, num_active_targets: int=5) -> RayTrainingStepOutput:
    del scheduler
    resolved_device = _resolve_batch_device(batch, device)
    active_views = _resolve_active_views(num_active_targets, resolved_device)
    z_all = _prepare_z_all(batch, components, resolved_device, active_views)
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E = batch['E'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    with torch.no_grad():
        plucker = compute_plucker_rays(K, E, h_tok=h_tok, w_tok=w_tok).to(dtype=z_all.dtype)
        if active_views != [0, 1, 2, 3, 4, 5]:
            plucker = plucker[:, active_views]
        latent_frames = z_all.shape[3]
        if plucker.shape[2] < latent_frames:
            raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker.shape[2]}')
        plucker = plucker[:, :, :latent_frames]
    sigma = sample_sigma_for_stage(z_all.shape[0], stage, resolved_device).to(dtype=z_all.dtype)
    (z_noisy, v_target) = add_masked_flow_matching_noise(z_all, sigma, front_idx=0)
    dit = _select_dit(components, stage)
    view_ids = batch['view_ids'][0].to(resolved_device) if batch['view_ids'].ndim == 2 else batch['view_ids'].to(resolved_device)
    if active_views != [0, 1, 2, 3, 4, 5]:
        view_ids = torch.tensor(active_views, device=resolved_device, dtype=torch.long)
    forward_kwargs = {'training_mode': 'sequential_view_recompute'}
    if active_views != [0, 1, 2, 3, 4, 5]:
        forward_kwargs['active_views'] = active_views
    v_pred = dit(z_noisy, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), plucker, view_ids, **forward_kwargs)
    loss = compute_masked_flow_matching_loss(v_pred, v_target, front_idx=0)
    metrics: dict[str, torch.Tensor | str | list[int]] = {'loss': loss.detach(), 'stage': stage, 'training_method': 'sequential_view_recompute', 'active_views': active_views}
    name_by_view = dict(VIEW_METRIC_NAMES)
    for (local_idx, global_view_idx) in enumerate(active_views):
        if global_view_idx == 0:
            continue
        metrics[f'flow_{name_by_view[global_view_idx]}'] = ((v_pred[:, local_idx] - v_target[:, local_idx]) ** 2).mean().detach()
    return RayTrainingStepOutput(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=v_target)
