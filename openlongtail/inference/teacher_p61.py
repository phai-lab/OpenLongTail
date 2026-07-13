from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import torch
from openlongtail.inference.teacher_p3 import _decode_targets, _encode_front, _ensure_batched_mask, _ensure_batched_text, sigmas_descending_from_start
from openlongtail.models.dit_p61_vace import P61_TARGET_STREAM_INDEX, P61_TYPE_PAD
from openlongtail.training.forward_ray_p61 import P61_CONDITION_SLOTS, P61_TARGET_VIEWS, build_p61_anchor_camera_transforms, build_p61_dynamic_plucker_streams, build_p61_time_window_relative_pose_features, p61_graph_layout
from openlongtail.training.schedulers import FlowMatchScheduler
P61_GEOMETRY_MODES = ('correct', 'zero_plucker', 'zero_geometry', 'front_geometry', 'swap_rear_cross_left')

class TextEncoderLike(Protocol):

    def encode_cached(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def null_cached(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...

@dataclass
class P61InferenceBundle:
    vae: object
    dit: torch.nn.Module
    text_encoder: TextEncoderLike
    scheduler: FlowMatchScheduler
    last_condition_preserved: bool = False
    front_vace_target_condition: bool = False

@dataclass(frozen=True)
class P61InferenceInputs:
    z_backbone: torch.Tensor
    condition_latents: torch.Tensor
    backbone_view_ids: torch.Tensor
    backbone_role_ids: torch.Tensor
    condition_view_ids: torch.Tensor
    condition_type_ids: torch.Tensor
    condition_available_mask: torch.Tensor

def _as_batched_latent(z: torch.Tensor, name: str) -> torch.Tensor:
    if z.ndim == 4:
        z_batched = z.unsqueeze(0)
    elif z.ndim == 5:
        z_batched = z
    else:
        raise ValueError(f'expected {name} shape (16,T,H,W) or (B,16,T,H,W), got {tuple(z.shape)}')
    if z_batched.shape[1] != 16:
        raise ValueError(f'expected {name} latent channel dim 16, got {tuple(z_batched.shape)}')
    return z_batched

def build_p61_inference_inputs(z_front: torch.Tensor, target_view: int, condition_latents_by_view: dict[int, torch.Tensor] | None=None, target_noise: torch.Tensor | None=None, generator: torch.Generator | None=None) -> P61InferenceInputs:
    z_front_batched = _as_batched_latent(z_front, 'z_front')
    layout = p61_graph_layout(target_view, device=z_front_batched.device)
    condition_latents_by_view = {} if condition_latents_by_view is None else condition_latents_by_view
    batch = z_front_batched.shape[0]
    target_shape = (batch, *z_front_batched.shape[1:])
    if target_noise is None:
        target_noise = torch.randn(target_shape, device=z_front_batched.device, dtype=z_front_batched.dtype, generator=generator)
    if tuple(target_noise.shape) != target_shape:
        raise ValueError(f'expected target_noise shape {target_shape}, got {tuple(target_noise.shape)}')
    z_backbone = torch.stack([z_front_batched, target_noise.to(device=z_front_batched.device, dtype=z_front_batched.dtype)], dim=1)
    condition_latents = z_front_batched.new_zeros(batch, P61_CONDITION_SLOTS, *z_front_batched.shape[1:])
    condition_latents[:, 0] = z_front_batched
    available = layout.condition_available_mask.clone()
    type_ids = layout.condition_type_ids.clone()
    for slot_idx in range(1, P61_CONDITION_SLOTS):
        view_id = int(layout.condition_view_ids[slot_idx].item())
        if not bool(layout.condition_available_mask[slot_idx].item()) or view_id not in condition_latents_by_view:
            available[slot_idx] = False
            type_ids[slot_idx] = P61_TYPE_PAD
            continue
        z_cond = _as_batched_latent(condition_latents_by_view[view_id].to(device=z_front_batched.device), f'condition_latents_by_view[{view_id}]')
        if tuple(z_cond.shape) != tuple(z_front_batched.shape):
            raise ValueError(f'expected condition view {view_id} latent shape {tuple(z_front_batched.shape)}, got {tuple(z_cond.shape)}')
        condition_latents[:, slot_idx] = z_cond
    return P61InferenceInputs(z_backbone=z_backbone, condition_latents=condition_latents, backbone_view_ids=layout.backbone_view_ids, backbone_role_ids=layout.backbone_role_ids, condition_view_ids=layout.condition_view_ids, condition_type_ids=type_ids, condition_available_mask=available)

def _reset_condition_inputs(inputs: P61InferenceInputs, reference_backbone: torch.Tensor, reference_conditions: torch.Tensor) -> None:
    inputs.z_backbone[:, 0] = reference_backbone[:, 0]
    inputs.condition_latents.copy_(reference_conditions)

def _front_as_target_vace_context(dit: torch.nn.Module, z_backbone: torch.Tensor) -> torch.Tensor:
    if not hasattr(dit, 'build_vace_context'):
        raise TypeError('front_vace_target_condition requires a DIT with build_vace_context')
    vace_context = dit.build_vace_context(z_backbone)
    front = z_backbone[:, 0]
    vace_context[:, P61_TARGET_STREAM_INDEX, :16] = 0
    vace_context[:, P61_TARGET_STREAM_INDEX, 16:32] = front
    vace_context[:, P61_TARGET_STREAM_INDEX, 32:] = 1.0
    return vace_context

@dataclass(frozen=True)
class P61GeometryOverride:
    backbone_plucker_view_ids: torch.Tensor
    condition_plucker_view_ids: torch.Tensor
    dit_backbone_view_ids: torch.Tensor
    dit_condition_view_ids: torch.Tensor
    relative_pose_target_view: int
    zero_plucker: bool = False
    zero_relative_pose: bool = False

def _resolve_geometry_override(inputs: P61InferenceInputs, target_view: int, geometry_mode: str) -> P61GeometryOverride:
    if geometry_mode not in P61_GEOMETRY_MODES:
        raise ValueError(f'geometry_mode must be one of {P61_GEOMETRY_MODES}, got {geometry_mode!r}')
    if geometry_mode == 'correct':
        return P61GeometryOverride(backbone_plucker_view_ids=inputs.backbone_view_ids, condition_plucker_view_ids=inputs.condition_view_ids, dit_backbone_view_ids=inputs.backbone_view_ids, dit_condition_view_ids=inputs.condition_view_ids, relative_pose_target_view=target_view)
    if geometry_mode == 'zero_plucker':
        return P61GeometryOverride(backbone_plucker_view_ids=inputs.backbone_view_ids, condition_plucker_view_ids=inputs.condition_view_ids, dit_backbone_view_ids=inputs.backbone_view_ids, dit_condition_view_ids=inputs.condition_view_ids, relative_pose_target_view=target_view, zero_plucker=True)
    if geometry_mode == 'zero_geometry':
        return P61GeometryOverride(backbone_plucker_view_ids=inputs.backbone_view_ids, condition_plucker_view_ids=inputs.condition_view_ids, dit_backbone_view_ids=inputs.backbone_view_ids, dit_condition_view_ids=inputs.condition_view_ids, relative_pose_target_view=target_view, zero_plucker=True, zero_relative_pose=True)
    if geometry_mode == 'front_geometry':
        return P61GeometryOverride(backbone_plucker_view_ids=torch.zeros_like(inputs.backbone_view_ids), condition_plucker_view_ids=torch.zeros_like(inputs.condition_view_ids), dit_backbone_view_ids=torch.zeros_like(inputs.backbone_view_ids), dit_condition_view_ids=torch.zeros_like(inputs.condition_view_ids), relative_pose_target_view=target_view, zero_relative_pose=True)
    backbone_ids = inputs.backbone_view_ids.clone()
    dit_backbone_ids = inputs.backbone_view_ids.clone()
    relative_target = target_view
    if int(target_view) == 5:
        backbone_ids[1] = 1
        dit_backbone_ids[1] = 1
        relative_target = 1
    return P61GeometryOverride(backbone_plucker_view_ids=backbone_ids, condition_plucker_view_ids=inputs.condition_view_ids, dit_backbone_view_ids=dit_backbone_ids, dit_condition_view_ids=inputs.condition_view_ids, relative_pose_target_view=relative_target)

@torch.no_grad()
def inference_p61_single_target(z_front: torch.Tensor, K_all: torch.Tensor, E_all: torch.Tensor, T_anchor_front: torch.Tensor, text_prompt: str, bundle: P61InferenceBundle, target_view: int, condition_latents_by_view: dict[int, torch.Tensor] | None=None, target_noise: torch.Tensor | None=None, num_steps: int=40, guide_scale: float=3.5, start_sigma: float=1.0, generator: torch.Generator | None=None, geometry_mode: str='correct') -> torch.Tensor:
    device = z_front.device
    inputs = build_p61_inference_inputs(z_front, target_view, condition_latents_by_view=condition_latents_by_view, target_noise=target_noise, generator=generator)
    z_backbone = inputs.z_backbone
    reference_backbone = z_backbone.clone()
    reference_conditions = inputs.condition_latents.clone()
    latent_frames = z_backbone.shape[3]
    h_tok = z_backbone.shape[-2] // 2
    w_tok = z_backbone.shape[-1] // 2
    K_b = K_all.to(device=device, dtype=torch.float32).unsqueeze(0)
    E_b = E_all.to(device=device, dtype=torch.float32).unsqueeze(0)
    T_b = T_anchor_front.to(device=device, dtype=torch.float32).unsqueeze(0)
    geometry = _resolve_geometry_override(inputs, target_view, geometry_mode)
    backbone_plucker = build_p61_dynamic_plucker_streams(K_b, E_b, T_b, geometry.backbone_plucker_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_backbone.dtype)
    condition_plucker = build_p61_dynamic_plucker_streams(K_b, E_b, T_b, geometry.condition_plucker_view_ids, latent_frames=latent_frames, h_tok=h_tok, w_tok=w_tok, dtype=z_backbone.dtype)
    if geometry.zero_plucker:
        backbone_plucker = torch.zeros_like(backbone_plucker)
        condition_plucker = torch.zeros_like(condition_plucker)
    T_anchor_cam = build_p61_anchor_camera_transforms(E_b, T_b)
    if geometry.zero_relative_pose:
        relative_pose_features = z_backbone.new_zeros(T_anchor_cam.shape[0], T_anchor_cam.shape[2], P61_CONDITION_SLOTS, T_anchor_cam.shape[2], 6)
    else:
        relative_pose_features = build_p61_time_window_relative_pose_features(T_anchor_cam, geometry.relative_pose_target_view, geometry.condition_plucker_view_ids, dtype=z_backbone.dtype)
    (text_emb, text_mask) = bundle.text_encoder.encode_cached(text_prompt)
    (null_emb, null_mask) = bundle.text_encoder.null_cached()
    text_emb = _ensure_batched_text(text_emb, 'text_emb').to(device=device)
    text_mask = _ensure_batched_mask(text_mask, 'text_mask').to(device=device)
    null_emb = _ensure_batched_text(null_emb, 'null_emb').to(device=device)
    null_mask = _ensure_batched_mask(null_mask, 'null_mask').to(device=device)
    sigmas = sigmas_descending_from_start(bundle.scheduler, num_steps, start_sigma, device=device, dtype=z_backbone.dtype)
    for (sigma_value, sigma_next) in zip(sigmas[:-1], sigmas[1:]):
        sigma = sigma_value.view(1)
        _reset_condition_inputs(inputs, reference_backbone, reference_conditions)
        vace_context = _front_as_target_vace_context(bundle.dit, z_backbone) if bundle.front_vace_target_condition else None
        extra_kwargs = {'vace_context': vace_context} if vace_context is not None else {}
        v_cond = bundle.dit(z_backbone, sigma, text_emb, text_mask, backbone_plucker, geometry.dit_backbone_view_ids, inputs.backbone_role_ids, T_b, inputs.condition_latents, condition_plucker, geometry.dit_condition_view_ids, inputs.condition_type_ids, inputs.condition_available_mask, relative_pose_features, **extra_kwargs)
        v_uncond = bundle.dit(z_backbone, sigma, null_emb, null_mask, backbone_plucker, geometry.dit_backbone_view_ids, inputs.backbone_role_ids, T_b, inputs.condition_latents, condition_plucker, geometry.dit_condition_view_ids, inputs.condition_type_ids, inputs.condition_available_mask, relative_pose_features, **extra_kwargs)
        v_pred = v_uncond + guide_scale * (v_cond - v_uncond)
        z_backbone[:, P61_TARGET_STREAM_INDEX] = z_backbone[:, P61_TARGET_STREAM_INDEX] + (sigma_next - sigma_value).view(1, 1, 1, 1, 1) * v_pred[:, P61_TARGET_STREAM_INDEX]
        _reset_condition_inputs(inputs, reference_backbone, reference_conditions)
    bundle.last_condition_preserved = bool(torch.equal(z_backbone[:, 0], reference_backbone[:, 0]) and torch.equal(inputs.condition_latents, reference_conditions))
    return z_backbone[0, P61_TARGET_STREAM_INDEX]

@torch.no_grad()
def inference_p61(front_rgb: torch.Tensor, K_all: torch.Tensor, E_all: torch.Tensor, T_anchor_front: torch.Tensor, text_prompt: str, bundle: P61InferenceBundle, num_steps: int=40, guide_scale: float=3.5, start_sigma: float=1.0, shared_noise_alpha: float=0.5, generator: torch.Generator | None=None, geometry_mode: str='correct') -> torch.Tensor:
    z_front = _encode_front(bundle.vae, front_rgb).to(device=front_rgb.device)
    batch_shape = z_front.unsqueeze(0).shape
    shared = torch.randn(batch_shape, device=z_front.device, dtype=z_front.dtype, generator=generator)
    private = {view_id: torch.randn(batch_shape, device=z_front.device, dtype=z_front.dtype, generator=generator) for view_id in P61_TARGET_VIEWS}
    alpha = float(shared_noise_alpha)
    target_noises = {view_id: alpha * shared + max(0.0, 1.0 - alpha * alpha) ** 0.5 * noise for (view_id, noise) in private.items()}
    generated: dict[int, torch.Tensor] = {}
    for target_view in P61_TARGET_VIEWS:
        generated[target_view] = inference_p61_single_target(z_front, K_all, E_all, T_anchor_front, text_prompt, bundle, target_view, condition_latents_by_view=generated, target_noise=target_noises[target_view], num_steps=num_steps, guide_scale=guide_scale, start_sigma=start_sigma, generator=generator, geometry_mode=geometry_mode)
    targets = torch.stack([generated[view_id] for view_id in P61_TARGET_VIEWS], dim=0)
    return _decode_targets(bundle.vae, targets)
inference = inference_p61
