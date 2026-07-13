from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol
import torch
from openlongtail.inference.teacher_p3 import _decode_targets, _encode_front, _ensure_batched_mask, _ensure_batched_text, sigmas_descending_from_start
from openlongtail.inference.teacher_p61 import P61_GEOMETRY_MODES, _reset_condition_inputs, _resolve_geometry_override, build_p61_inference_inputs
from openlongtail.models.dit_p61_vace import P61_TARGET_STREAM_INDEX
from openlongtail.training.forward_ray_p61 import P61_TARGET_VIEWS, build_p61_anchor_camera_transforms, build_p61_dynamic_plucker_streams
from openlongtail.training.forward_ray_p62 import build_p62_target_pose_features, build_p62_time_window_relative_pose_features
from openlongtail.training.schedulers import FlowMatchScheduler

class TextEncoderLike(Protocol):

    def encode_cached(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def null_cached(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...
WarpProvider = Callable[[int], tuple[torch.Tensor, torch.Tensor]]

@dataclass
class InferenceBundleWarp:
    vae: object
    dit: torch.nn.Module
    text_encoder: TextEncoderLike
    scheduler: FlowMatchScheduler
    warp_provider: WarpProvider
    last_condition_preserved: bool = False

def make_sidecar_warp_provider(sidecar_v0_path: str, sidecar_v1_path: str | None, device: torch.device) -> WarpProvider:
    s0 = torch.load(sidecar_v0_path, map_location='cpu', weights_only=True)
    warped_v0 = s0['warped_target_latents']
    vis_v0 = s0['warped_target_visibility']
    if sidecar_v1_path is not None:
        s1 = torch.load(sidecar_v1_path, map_location='cpu', weights_only=True)
        warped_v1 = s1['adjacent_warped_target_latents']
        vis_v1 = s1['adjacent_warped_target_visibility']
        use_adj = vis_v1.float() > vis_v0.float()
        warped_merged = torch.where(use_adj.expand_as(warped_v0), warped_v1, warped_v0)
        vis_merged = torch.maximum(vis_v0.float(), vis_v1.float())
    else:
        warped_merged = warped_v0
        vis_merged = vis_v0
    warped_merged = warped_merged.to(device=device, dtype=torch.bfloat16)
    vis_merged = vis_merged.to(device=device, dtype=torch.float16)

    def provider(target_view: int) -> tuple[torch.Tensor, torch.Tensor]:
        slot = int(target_view) - 1
        return (warped_merged[slot:slot + 1], vis_merged[slot:slot + 1])
    return provider

def make_zero_warp_provider(latent_shape: tuple[int, int, int, int], device: torch.device) -> WarpProvider:
    (T, H, W) = latent_shape
    z = torch.zeros(1, 16, T, H, W, dtype=torch.bfloat16, device=device)
    v = torch.zeros(1, 1, T, H, W, dtype=torch.float16, device=device)

    def provider(target_view: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (z, v)
    return provider

@torch.no_grad()
def inference_single_target(z_front: torch.Tensor, K_all: torch.Tensor, E_all: torch.Tensor, T_anchor_front: torch.Tensor, text_prompt: str, bundle: InferenceBundleWarp, target_view: int, condition_latents_by_view: dict[int, torch.Tensor] | None=None, target_noise: torch.Tensor | None=None, num_steps: int=40, guide_scale: float=3.5, start_sigma: float=1.0, generator: torch.Generator | None=None, geometry_mode: str='correct') -> torch.Tensor:
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
        relative_pose_features = z_backbone.new_zeros(T_anchor_cam.shape[0], T_anchor_cam.shape[2], 3, T_anchor_cam.shape[2], 6)
        target_pose_features = z_backbone.new_zeros(T_anchor_cam.shape[0], T_anchor_cam.shape[2], 6)
    else:
        relative_pose_features = build_p62_time_window_relative_pose_features(T_anchor_cam, geometry.relative_pose_target_view, geometry.condition_plucker_view_ids, dtype=z_backbone.dtype)
        target_pose_features = build_p62_target_pose_features(T_anchor_cam, geometry.relative_pose_target_view, dtype=z_backbone.dtype)
    (warp_t, vis_t) = bundle.warp_provider(target_view)
    warp_t = warp_t.to(device=device, dtype=torch.bfloat16)
    vis_t = vis_t.to(device=device, dtype=torch.float16)
    if warp_t.shape[0] != z_backbone.shape[0]:
        warp_t = warp_t.expand(z_backbone.shape[0], *warp_t.shape[1:]).contiguous()
        vis_t = vis_t.expand(z_backbone.shape[0], *vis_t.shape[1:]).contiguous()
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
        v_cond = bundle.dit(z_backbone, sigma, text_emb, text_mask, backbone_plucker, geometry.dit_backbone_view_ids, inputs.backbone_role_ids, T_b, inputs.condition_latents, condition_plucker, geometry.dit_condition_view_ids, inputs.condition_type_ids, inputs.condition_available_mask, relative_pose_features, target_pose_features, warped_target_latent=warp_t, warped_target_visibility=vis_t)
        v_uncond = bundle.dit(z_backbone, sigma, null_emb, null_mask, backbone_plucker, geometry.dit_backbone_view_ids, inputs.backbone_role_ids, T_b, inputs.condition_latents, condition_plucker, geometry.dit_condition_view_ids, inputs.condition_type_ids, inputs.condition_available_mask, relative_pose_features, target_pose_features, warped_target_latent=warp_t, warped_target_visibility=vis_t)
        v_pred = v_uncond + guide_scale * (v_cond - v_uncond)
        z_backbone[:, P61_TARGET_STREAM_INDEX] = z_backbone[:, P61_TARGET_STREAM_INDEX] + (sigma_next - sigma_value).view(1, 1, 1, 1, 1) * v_pred[:, P61_TARGET_STREAM_INDEX]
        _reset_condition_inputs(inputs, reference_backbone, reference_conditions)
    bundle.last_condition_preserved = bool(torch.equal(z_backbone[:, 0], reference_backbone[:, 0]) and torch.equal(inputs.condition_latents, reference_conditions))
    return z_backbone[0, P61_TARGET_STREAM_INDEX]

@torch.no_grad()
def inference_multiview(front_rgb: torch.Tensor, K_all: torch.Tensor, E_all: torch.Tensor, T_anchor_front: torch.Tensor, text_prompt: str, bundle: InferenceBundleWarp, num_steps: int=40, guide_scale: float=3.5, start_sigma: float=1.0, shared_noise_alpha: float=0.5, generator: torch.Generator | None=None, geometry_mode: str='correct') -> torch.Tensor:
    if geometry_mode not in P61_GEOMETRY_MODES:
        raise ValueError(f'geometry_mode must be one of {P61_GEOMETRY_MODES}, got {geometry_mode!r}')
    z_front = _encode_front(bundle.vae, front_rgb).to(device=front_rgb.device)
    batch_shape = z_front.unsqueeze(0).shape
    shared = torch.randn(batch_shape, device=z_front.device, dtype=z_front.dtype, generator=generator)
    private = {view_id: torch.randn(batch_shape, device=z_front.device, dtype=z_front.dtype, generator=generator) for view_id in P61_TARGET_VIEWS}
    alpha = float(shared_noise_alpha)
    target_noises = {view_id: alpha * shared + max(0.0, 1.0 - alpha * alpha) ** 0.5 * noise for (view_id, noise) in private.items()}
    generated: dict[int, torch.Tensor] = {}
    for target_view in P61_TARGET_VIEWS:
        generated[target_view] = inference_single_target(z_front, K_all, E_all, T_anchor_front, text_prompt, bundle, target_view, condition_latents_by_view=generated, target_noise=target_noises[target_view], num_steps=num_steps, guide_scale=guide_scale, start_sigma=start_sigma, generator=generator, geometry_mode=geometry_mode)
    targets = torch.stack([generated[v] for v in P61_TARGET_VIEWS], dim=0)
    return _decode_targets(bundle.vae, targets)
inference = inference_multiview
