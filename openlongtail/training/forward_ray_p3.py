from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.distributed as dist
from openlongtail.models.plucker import compute_plucker_rays
from openlongtail.training.forward_ray import RayTrainingComponents, encode_batch_with_vae
from openlongtail.training.losses.masked_flow_match import sample_sigma_for_stage
from openlongtail.training.schedulers import FlowMatchScheduler

@dataclass
class P3StreamBatch:
    z_streams: torch.Tensor
    v_target: torch.Tensor
    target_clean: torch.Tensor
    stream_view_ids: torch.Tensor
    stream_role_ids: torch.Tensor
    target_local_indices: torch.Tensor

@dataclass
class RayTrainingStepP3Output:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | str | list[int]]
    sigma: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor

def sample_p3_target_views(num_targets: int=3, generator: torch.Generator | None=None) -> list[int]:
    if num_targets != 3:
        raise ValueError(f'num_targets must be 3 for P3, got {num_targets}')
    return sorted((int(item) for item in torch.randperm(6, generator=generator)[:num_targets].tolist()))

def broadcast_p3_target_views(target_views: list[int] | None, device: torch.device, num_targets: int=3) -> list[int]:
    if num_targets != 3:
        raise ValueError(f'num_targets must be 3 for P3, got {num_targets}')
    if not (dist.is_available() and dist.is_initialized()):
        assert target_views is not None
        return target_views
    buffer = torch.zeros(num_targets, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        assert target_views is not None and len(target_views) == num_targets
        buffer[:] = torch.tensor(target_views, dtype=torch.long, device=device)
    dist.broadcast(buffer, src=0)
    return [int(item) for item in buffer.tolist()]

def resolve_p3_target_views(device: torch.device, num_targets: int=3) -> list[int]:
    target_views_local = sample_p3_target_views(num_targets) if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0 else None
    return broadcast_p3_target_views(target_views_local, device, num_targets=num_targets)

def build_p3_streams_from_latents(z_all: torch.Tensor, target_views: list[int], sigma: torch.Tensor, noise: torch.Tensor | None=None) -> P3StreamBatch:
    if z_all.ndim != 6 or z_all.shape[1] != 6 or z_all.shape[2] != 16:
        raise ValueError(f'expected z_all shape (B, 6, 16, T, H, W), got {tuple(z_all.shape)}')
    if len(target_views) != 3 or len(set(target_views)) != 3:
        raise ValueError(f'expected 3 unique target views, got {target_views}')
    if any((view_id < 0 or view_id >= 6 for view_id in target_views)):
        raise ValueError(f'expected target views in [0, 5], got {target_views}')
    batch = z_all.shape[0]
    if sigma.shape != (batch,):
        raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
    target_clean = z_all[:, target_views]
    if noise is None:
        noise = torch.randn_like(target_clean)
    if noise.shape != target_clean.shape:
        raise ValueError(f'expected noise shape {tuple(target_clean.shape)}, got {tuple(noise.shape)}')
    sigma_b = sigma.view(batch, 1, 1, 1, 1, 1).to(dtype=z_all.dtype)
    target_noisy = (1.0 - sigma_b) * target_clean + sigma_b * noise
    v_target = noise - target_clean
    z_streams = torch.cat([z_all[:, 0:1], target_noisy], dim=1)
    stream_view_ids = torch.tensor([0, *target_views], device=z_all.device, dtype=torch.long)
    stream_role_ids = torch.tensor([0, 1, 1, 1], device=z_all.device, dtype=torch.long)
    target_local_indices = torch.tensor([1, 2, 3], device=z_all.device, dtype=torch.long)
    return P3StreamBatch(z_streams=z_streams, v_target=v_target, target_clean=target_clean, stream_view_ids=stream_view_ids, stream_role_ids=stream_role_ids, target_local_indices=target_local_indices)

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

def _prepare_z_all_for_p3(batch: dict[str, Any], components: RayTrainingComponents, device: torch.device, target_views: list[int]) -> torch.Tensor:
    if 'z_all' in batch:
        z_all = batch['z_all'].to(device)
        if z_all.ndim != 6 or z_all.shape[1] != 6:
            raise ValueError(f'expected cached z_all shape (B, 6, 16, T, H, W), got {tuple(z_all.shape)}')
        return z_all
    rgb = batch['rgb'].to(device)
    unique_views = sorted(set([0, *target_views]))
    z_unique = encode_batch_with_vae(components.vae, rgb[:, unique_views])
    z_all = torch.zeros(z_unique.shape[0], 6, *z_unique.shape[2:], device=device, dtype=z_unique.dtype)
    for (local_idx, global_view_id) in enumerate(unique_views):
        z_all[:, global_view_id] = z_unique[:, local_idx]
    if device.type == 'cuda':
        if hasattr(components.vae, 'model'):
            components.vae.model.to('cpu')
        torch.cuda.empty_cache()
    return z_all

def training_step_ray_p3(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, stage: str, device: torch.device | str | None=None, num_targets: int=3) -> RayTrainingStepP3Output:
    del scheduler
    resolved_device = _resolve_batch_device(batch, device)
    target_views = resolve_p3_target_views(resolved_device, num_targets=num_targets)
    z_all = _prepare_z_all_for_p3(batch, components, resolved_device, target_views)
    sigma = sample_sigma_for_stage(z_all.shape[0], stage, resolved_device).to(dtype=z_all.dtype)
    streams = build_p3_streams_from_latents(z_all, target_views, sigma)
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E = batch['E'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    with torch.no_grad():
        plucker_all = compute_plucker_rays(K, E, h_tok=h_tok, w_tok=w_tok).to(dtype=z_all.dtype)
        plucker = plucker_all[:, streams.stream_view_ids]
        latent_frames = z_all.shape[3]
        if plucker.shape[2] < latent_frames:
            raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker.shape[2]}')
        plucker = plucker[:, :, :latent_frames]
    dit = _select_dit(components, stage)
    v_pred = dit(streams.z_streams, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), plucker, streams.stream_view_ids, streams.stream_role_ids)
    pred_targets = v_pred[:, streams.target_local_indices]
    loss = ((pred_targets - streams.v_target) ** 2).mean()
    metrics: dict[str, torch.Tensor | str | list[int]] = {'loss': loss.detach(), 'stage': stage, 'training_method': 'p3_source_front_random_targets', 'target_views': target_views, 'stream_view_ids': [int(item) for item in streams.stream_view_ids.detach().cpu().tolist()], 'stream_role_ids': [int(item) for item in streams.stream_role_ids.detach().cpu().tolist()]}
    for (local_idx, global_view_idx) in enumerate(target_views):
        metrics[f'flow_view_{global_view_idx}'] = ((pred_targets[:, local_idx] - streams.v_target[:, local_idx]) ** 2).mean().detach()
    return RayTrainingStepP3Output(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=streams.v_target)
