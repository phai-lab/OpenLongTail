from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.distributed as dist
from openlongtail.data.transforms import se3_inverse, se3_log_rotation_angle
from openlongtail.models.plucker import compute_plucker_rays
from openlongtail.models.dit_p5_vace import P5_ROLE_CONDITION, P5_ROLE_TARGET
from openlongtail.models.dit_p61_vace import P61_CONDITION_SLOTS, P61_TARGET_STREAM_INDEX, P61_TYPE_FRONT, P61_TYPE_NEIGHBOR, P61_TYPE_PAD, DiTP61VACE
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p3 import _prepare_z_all_for_p3, _resolve_batch_device
from openlongtail.training.forward_ray_p5 import FRONT_IDX
from openlongtail.training.forward_wan21 import sample_sigma_unified
from openlongtail.training.schedulers import FlowMatchScheduler
P61_TARGET_VIEWS = (1, 2, 3, 4, 5)
P61_GRAPH_CONDITIONS: dict[int, tuple[int, int, int]] = {1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 1, 0), 4: (0, 2, 0), 5: (0, 3, 4)}
P61_GRAPH_AVAILABLE: dict[int, tuple[int, int, int]] = {1: (1, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (1, 1, 0), 5: (1, 1, 1)}

@dataclass(frozen=True)
class P61GraphLayout:
    target_view: int
    backbone_view_ids: torch.Tensor
    backbone_role_ids: torch.Tensor
    condition_view_ids: torch.Tensor
    condition_type_ids: torch.Tensor
    condition_available_mask: torch.Tensor

@dataclass
class P61StreamBatch:
    z_backbone: torch.Tensor
    condition_latents: torch.Tensor
    v_target: torch.Tensor
    target_clean: torch.Tensor
    target_noise: torch.Tensor
    layout: P61GraphLayout
    front_condition_noise_sigma: torch.Tensor
    neighbor_condition_noise_sigma: torch.Tensor

@dataclass
class RayTrainingStepP61Output:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | str | list[int] | float | int]
    sigma: torch.Tensor
    v_pred: torch.Tensor
    v_target: torch.Tensor

def canonical_p61_target_view(target_view: int | torch.Tensor | None=None) -> int:
    if target_view is None:
        raise ValueError('P6.1 target_view must be provided')
    view = int(target_view.item() if isinstance(target_view, torch.Tensor) else target_view)
    if view not in P61_TARGET_VIEWS:
        raise ValueError(f'P6.1 target view must be one of {P61_TARGET_VIEWS}, got {view}')
    return view

def sample_p61_target_view(generator: torch.Generator | None=None) -> int:
    idx = int(torch.randint(len(P61_TARGET_VIEWS), (1,), generator=generator).item())
    return P61_TARGET_VIEWS[idx]

def broadcast_p61_target_view(target_view: int | None, device: torch.device) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        if target_view is None:
            raise ValueError('target_view must be provided in single-process P6.1 target resolution')
        return canonical_p61_target_view(target_view)
    buffer = torch.zeros(1, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        if target_view is None:
            raise ValueError('rank 0 target_view must be provided for P6.1 broadcast')
        buffer[0] = canonical_p61_target_view(target_view)
    dist.broadcast(buffer, src=0)
    return canonical_p61_target_view(buffer[0])

def resolve_p61_target_view(device: torch.device) -> int:
    target_local = sample_p61_target_view() if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0 else None
    return broadcast_p61_target_view(target_local, device)

def p61_graph_layout(target_view: int, device: torch.device | str | None=None) -> P61GraphLayout:
    target = canonical_p61_target_view(target_view)
    condition_views = P61_GRAPH_CONDITIONS[target]
    available = P61_GRAPH_AVAILABLE[target]
    condition_types = [P61_TYPE_FRONT if idx == 0 else P61_TYPE_NEIGHBOR if bool(is_available) else P61_TYPE_PAD for (idx, is_available) in enumerate(available)]
    return P61GraphLayout(target_view=target, backbone_view_ids=torch.tensor([FRONT_IDX, target], device=device, dtype=torch.long), backbone_role_ids=torch.tensor([P5_ROLE_CONDITION, P5_ROLE_TARGET], device=device, dtype=torch.long), condition_view_ids=torch.tensor(condition_views, device=device, dtype=torch.long), condition_type_ids=torch.tensor(condition_types, device=device, dtype=torch.long), condition_available_mask=torch.tensor(available, device=device, dtype=torch.bool))

def _sample_condition_sigma(batch: int, slots: int, device: torch.device, dtype: torch.dtype, noise_prob: float, noise_min: float, noise_max: float, available_mask: torch.Tensor) -> torch.Tensor:
    if not 0.0 <= float(noise_prob) <= 1.0:
        raise ValueError(f'expected noise probability in [0, 1], got {noise_prob}')
    if noise_min < 0 or noise_max < noise_min:
        raise ValueError(f'expected 0 <= noise_min <= noise_max, got {noise_min}, {noise_max}')
    sigma = torch.zeros(batch, slots, device=device, dtype=dtype)
    if noise_prob == 0.0 or noise_max == 0.0:
        return sigma
    apply = torch.rand(batch, slots, device=device) < float(noise_prob)
    apply = apply & available_mask.view(1, slots).to(device=device, dtype=torch.bool)
    sampled = noise_min + (noise_max - noise_min) * torch.rand(batch, slots, device=device, dtype=dtype)
    return torch.where(apply, sampled, sigma)

def build_p61_streams_from_latents(z_all: torch.Tensor, sigma: torch.Tensor, target_view: int, front_condition_noise_prob: float=0.05, front_condition_noise_min: float=0.005, front_condition_noise_max: float=0.03, neighbor_condition_noise_prob: float=0.3, neighbor_condition_noise_min: float=0.02, neighbor_condition_noise_max: float=0.15, target_noise: torch.Tensor | None=None, front_noise: torch.Tensor | None=None, condition_noise: torch.Tensor | None=None) -> P61StreamBatch:
    if z_all.ndim != 6 or z_all.shape[1] != 6 or z_all.shape[2] != 16:
        raise ValueError(f'expected z_all shape (B,6,16,T,H,W), got {tuple(z_all.shape)}')
    batch = z_all.shape[0]
    if sigma.shape != (batch,):
        raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
    layout = p61_graph_layout(target_view, device=z_all.device)
    target_clean = z_all[:, layout.target_view:layout.target_view + 1]
    if target_noise is None:
        target_noise = torch.randn_like(target_clean)
    if tuple(target_noise.shape) != tuple(target_clean.shape):
        raise ValueError(f'expected target_noise shape {tuple(target_clean.shape)}, got {tuple(target_noise.shape)}')
    target_noise = target_noise.to(device=z_all.device, dtype=z_all.dtype)
    sigma_b = sigma.view(batch, 1, 1, 1, 1, 1).to(dtype=z_all.dtype)
    target_noisy = (1.0 - sigma_b) * target_clean + sigma_b * target_noise
    v_target = target_noise - target_clean
    if front_noise is None:
        front_noise = torch.randn_like(z_all[:, FRONT_IDX])
    if tuple(front_noise.shape) != tuple(z_all[:, FRONT_IDX].shape):
        raise ValueError(f'expected front_noise shape {tuple(z_all[:, FRONT_IDX].shape)}, got {tuple(front_noise.shape)}')
    front_sigma = _sample_condition_sigma(batch, 1, z_all.device, z_all.dtype, front_condition_noise_prob, front_condition_noise_min, front_condition_noise_max, torch.ones(1, device=z_all.device, dtype=torch.bool))
    front_sigma_b = front_sigma.view(batch, 1, 1, 1, 1)
    front_condition = (1.0 - front_sigma_b) * z_all[:, FRONT_IDX] + front_sigma_b * front_noise.to(device=z_all.device, dtype=z_all.dtype)
    condition_latents = z_all.new_zeros(batch, P61_CONDITION_SLOTS, *z_all.shape[2:])
    condition_latents[:, 0] = front_condition
    neighbor_available = layout.condition_available_mask & (layout.condition_type_ids == P61_TYPE_NEIGHBOR)
    neighbor_sigma = _sample_condition_sigma(batch, P61_CONDITION_SLOTS, z_all.device, z_all.dtype, neighbor_condition_noise_prob, neighbor_condition_noise_min, neighbor_condition_noise_max, neighbor_available)
    if condition_noise is None:
        condition_noise = torch.randn(batch, P61_CONDITION_SLOTS, *z_all.shape[2:], device=z_all.device, dtype=z_all.dtype)
    expected_condition_noise = (batch, P61_CONDITION_SLOTS, *z_all.shape[2:])
    if tuple(condition_noise.shape) != expected_condition_noise:
        raise ValueError(f'expected condition_noise shape {expected_condition_noise}, got {tuple(condition_noise.shape)}')
    condition_noise = condition_noise.to(device=z_all.device, dtype=z_all.dtype)
    for slot_idx in range(1, P61_CONDITION_SLOTS):
        if not bool(neighbor_available[slot_idx].item()):
            continue
        view_id = int(layout.condition_view_ids[slot_idx].item())
        cond_sigma = neighbor_sigma[:, slot_idx].view(batch, 1, 1, 1, 1)
        condition_latents[:, slot_idx] = (1.0 - cond_sigma) * z_all[:, view_id] + cond_sigma * condition_noise[:, slot_idx]
    z_backbone = torch.stack([front_condition, target_noisy[:, 0]], dim=1)
    return P61StreamBatch(z_backbone=z_backbone, condition_latents=condition_latents, v_target=v_target, target_clean=target_clean, target_noise=target_noise, layout=layout, front_condition_noise_sigma=front_sigma, neighbor_condition_noise_sigma=neighbor_sigma)

def build_p61_anchor_camera_transforms(E_rig: torch.Tensor, T_anchor_front: torch.Tensor) -> torch.Tensor:
    if E_rig.ndim != 4 or E_rig.shape[1:] != (6, 4, 4):
        raise ValueError(f'expected E_rig shape (B,6,4,4), got {tuple(E_rig.shape)}')
    if T_anchor_front.ndim != 4 or T_anchor_front.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor_front shape (B,T,4,4), got {tuple(T_anchor_front.shape)}')
    E_front = E_rig[:, FRONT_IDX:FRONT_IDX + 1].to(dtype=torch.float32)
    E_front_cam = se3_inverse(E_front) @ E_rig.to(dtype=torch.float32)
    T = T_anchor_front.to(dtype=torch.float32)
    return (T[:, :, None] @ E_front_cam[:, None]).permute(0, 2, 1, 3, 4).contiguous()

def _so3_log_vector(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack([rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]], dim=-1)
    sin_theta = torch.sin(theta)
    scale = torch.where(theta < 1e-05, torch.full_like(theta, 0.5), theta / (2.0 * sin_theta.clamp_min(1e-08)))
    return vee * scale.unsqueeze(-1)

def build_p61_time_window_relative_pose_features(T_anchor_cam: torch.Tensor, target_view: int, condition_view_ids: torch.Tensor, dtype: torch.dtype | None=None) -> torch.Tensor:
    if T_anchor_cam.ndim != 5 or T_anchor_cam.shape[1] != 6 or T_anchor_cam.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor_cam shape (B,6,T,4,4), got {tuple(T_anchor_cam.shape)}')
    if condition_view_ids.shape != (P61_CONDITION_SLOTS,):
        raise ValueError(f'expected condition_view_ids shape ({P61_CONDITION_SLOTS},), got {tuple(condition_view_ids.shape)}')
    target = canonical_p61_target_view(target_view)
    T_target_inv = se3_inverse(T_anchor_cam[:, target])
    T_cond = T_anchor_cam[:, condition_view_ids.to(T_anchor_cam.device)]
    T_rel = T_target_inv[:, :, None, None] @ T_cond[:, None]
    trans = T_rel[..., :3, 3]
    rot = _so3_log_vector(T_rel[..., :3, :3])
    features = torch.cat([trans, rot], dim=-1)
    return features if dtype is None else features.to(dtype=dtype)

def build_p61_dynamic_plucker_streams(K: torch.Tensor, E_rig: torch.Tensor, T_anchor_front: torch.Tensor, view_ids: torch.Tensor, latent_frames: int, h_tok: int, w_tok: int, dtype: torch.dtype | None=None) -> torch.Tensor:
    if K.ndim != 4 or K.shape[1:] != (6, 3, 3):
        raise ValueError(f'expected K shape (B,6,3,3), got {tuple(K.shape)}')
    if view_ids.ndim != 1:
        raise ValueError(f'expected view_ids shape (S,), got {tuple(view_ids.shape)}')
    T_anchor_cam = build_p61_anchor_camera_transforms(E_rig, T_anchor_front)
    E_dynamic = T_anchor_cam.permute(0, 2, 1, 3, 4).contiguous()
    plucker_all = compute_plucker_rays(K.to(dtype=torch.float32), E_dynamic, h_tok=h_tok, w_tok=w_tok)
    plucker = plucker_all[:, view_ids.to(plucker_all.device), :latent_frames]
    return plucker if dtype is None else plucker.to(dtype=dtype)

def training_step_ray_p61(batch: dict[str, Any], components: RayTrainingComponents, scheduler: FlowMatchScheduler, device: torch.device | str | None=None, target_view: int | None=None, front_condition_noise_prob: float=0.05, front_condition_noise_min: float=0.005, front_condition_noise_max: float=0.03, neighbor_condition_noise_prob: float=0.3, neighbor_condition_noise_min: float=0.02, neighbor_condition_noise_max: float=0.15) -> RayTrainingStepP61Output:
    resolved_device = _resolve_batch_device(batch, device)
    target = canonical_p61_target_view(target_view) if target_view is not None else resolve_p61_target_view(resolved_device)
    layout = p61_graph_layout(target, device=resolved_device)
    needed_views = sorted({target, *[int(view_id) for (view_id, available) in zip(layout.condition_view_ids.detach().cpu().tolist(), layout.condition_available_mask.detach().cpu().tolist()) if bool(available) and int(view_id) != FRONT_IDX]})
    z_all = _prepare_z_all_for_p3(batch, components, resolved_device, needed_views)
    sigma = sample_sigma_unified(z_all.shape[0], scheduler, resolved_device).to(dtype=z_all.dtype)
    streams = build_p61_streams_from_latents(z_all, sigma, target, front_condition_noise_prob=front_condition_noise_prob, front_condition_noise_min=front_condition_noise_min, front_condition_noise_max=front_condition_noise_max, neighbor_condition_noise_prob=neighbor_condition_noise_prob, neighbor_condition_noise_min=neighbor_condition_noise_min, neighbor_condition_noise_max=neighbor_condition_noise_max)
    if 'T_anchor_front' not in batch:
        raise KeyError("P6.1 training requires batch['T_anchor_front']")
    K = batch['K'].to(device=resolved_device, dtype=torch.float32)
    E_rig = batch['E'].to(device=resolved_device, dtype=torch.float32)
    T_anchor_front = batch['T_anchor_front'].to(device=resolved_device, dtype=torch.float32)
    h_tok = z_all.shape[-2] // 2
    w_tok = z_all.shape[-1] // 2
    latent_frames = z_all.shape[3]
    with torch.no_grad():
        backbone_plucker = build_p61_dynamic_plucker_streams(K, E_rig, T_anchor_front, streams.layout.backbone_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_all.dtype)
        condition_plucker = build_p61_dynamic_plucker_streams(K, E_rig, T_anchor_front, streams.layout.condition_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_all.dtype)
        T_anchor_cam = build_p61_anchor_camera_transforms(E_rig, T_anchor_front)
        relative_pose_features = build_p61_time_window_relative_pose_features(T_anchor_cam, target, streams.layout.condition_view_ids, dtype=z_all.dtype)
    dit = components.low_dit
    if dit is None:
        raise ValueError('P6.1 requires components.low_dit')
    v_pred = dit(streams.z_backbone, sigma, batch['text_emb'].to(resolved_device), batch['text_mask'].to(resolved_device), backbone_plucker, streams.layout.backbone_view_ids, streams.layout.backbone_role_ids, T_anchor_front, streams.condition_latents, condition_plucker, streams.layout.condition_view_ids, streams.layout.condition_type_ids, streams.layout.condition_available_mask, relative_pose_features)
    pred_target = v_pred[:, P61_TARGET_STREAM_INDEX:P61_TARGET_STREAM_INDEX + 1]
    loss = ((pred_target - streams.v_target) ** 2).mean()
    front_translation_norm = T_anchor_front[:, :latent_frames, :3, 3].norm(dim=-1)
    metrics: dict[str, torch.Tensor | str | list[int] | float | int] = {'loss': loss.detach(), 'stage': 'unified', 'training_method': 'p61_graph_autoregressive_single_target_vace', 'target_view': int(target), 'condition_view_ids': [int(item) for item in streams.layout.condition_view_ids.detach().cpu().tolist()], 'condition_available_mask': [int(item) for item in streams.layout.condition_available_mask.detach().cpu().tolist()], 'backbone_view_ids': [int(item) for item in streams.layout.backbone_view_ids.detach().cpu().tolist()], 'sigma_mean': sigma.detach().float().mean(), 'sigma_min': sigma.detach().float().min(), 'sigma_max': sigma.detach().float().max(), 'front_condition_noise_sigma_mean': streams.front_condition_noise_sigma.detach().float().mean(), 'front_condition_noise_sigma_max': streams.front_condition_noise_sigma.detach().float().max(), 'neighbor_condition_noise_sigma_mean': streams.neighbor_condition_noise_sigma.detach().float().mean(), 'neighbor_condition_noise_sigma_max': streams.neighbor_condition_noise_sigma.detach().float().max(), 'front_translation_norm_mean': front_translation_norm.mean().detach(), 'front_translation_norm_max': front_translation_norm.max().detach(), 'front_rotation_angle_max': se3_log_rotation_angle(T_anchor_front[:, :latent_frames]).abs().max().detach()}
    if isinstance(dit, DiTP61VACE):
        metrics['sync_temporal_window'] = int(dit.sync_temporal_window)
    else:
        sync_window = getattr(dit, 'sync_temporal_window', None)
        if sync_window is not None:
            metrics['sync_temporal_window'] = int(sync_window)
    metrics[f'flow_view_{target}'] = loss.detach()
    return RayTrainingStepP61Output(loss=loss, metrics=metrics, sigma=sigma.detach(), v_pred=v_pred, v_target=streams.v_target)
