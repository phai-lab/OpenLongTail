from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import torch
from openlongtail.models.plucker import compute_plucker_rays
from openlongtail.models.wan_vae import normalize_rgb_for_wan_vae
from openlongtail.training.schedulers import FlowMatchScheduler

class TextEncoderLike(Protocol):

    def encode_cached(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def null_cached(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...

@dataclass
class P3InferenceBundle:
    vae: object
    low_dit: torch.nn.Module
    high_dit: torch.nn.Module | None
    text_encoder: TextEncoderLike
    scheduler: FlowMatchScheduler
    last_front_preserved: bool = False

@dataclass(frozen=True)
class P3InferenceStreams:
    z_streams: torch.Tensor
    stream_view_ids: torch.Tensor
    stream_role_ids: torch.Tensor

def validate_p3_target_views(target_views: list[int] | tuple[int, ...]) -> list[int]:
    views = [int(view_id) for view_id in target_views]
    if len(views) != 3:
        raise ValueError(f'inference requires exactly 3 target views, got {views}')
    if len(set(views)) != 3:
        raise ValueError(f'target views must be unique, got {views}')
    if any((view_id < 0 or view_id >= 6 for view_id in views)):
        raise ValueError(f'target views must be global camera IDs in [0, 5], got {views}')
    return views

def canonical_p3_stream_ids(target_views: list[int] | tuple[int, ...], device: torch.device | str | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    views = validate_p3_target_views(target_views)
    stream_view_ids = torch.tensor([0, *views], device=device, dtype=torch.long)
    stream_role_ids = torch.tensor([0, 1, 1, 1], device=device, dtype=torch.long)
    return (stream_view_ids, stream_role_ids)

def build_canonical_p3_streams(z_front: torch.Tensor, target_views: list[int] | tuple[int, ...], noise: torch.Tensor | None=None, generator: torch.Generator | None=None) -> P3InferenceStreams:
    if z_front.ndim == 4:
        z_front_batched = z_front.unsqueeze(0)
    elif z_front.ndim == 5:
        z_front_batched = z_front
    else:
        raise ValueError(f'expected z_front shape (16,T,H,W) or (B,16,T,H,W), got {tuple(z_front.shape)}')
    if z_front_batched.shape[1] != 16:
        raise ValueError(f'expected z_front latent channel dim 16, got {tuple(z_front_batched.shape)}')
    batch = z_front_batched.shape[0]
    target_shape = (batch, 3, *z_front_batched.shape[1:])
    if noise is None:
        noise = torch.randn(target_shape, device=z_front_batched.device, dtype=z_front_batched.dtype, generator=generator)
    if tuple(noise.shape) != target_shape:
        raise ValueError(f'expected target noise shape {target_shape}, got {tuple(noise.shape)}')
    (stream_view_ids, stream_role_ids) = canonical_p3_stream_ids(target_views, device=z_front_batched.device)
    z_streams = torch.cat([z_front_batched.unsqueeze(1), noise], dim=1)
    return P3InferenceStreams(z_streams=z_streams, stream_view_ids=stream_view_ids, stream_role_ids=stream_role_ids)

def build_p3_plucker_streams(K_all: torch.Tensor, E_all: torch.Tensor, stream_view_ids: torch.Tensor, latent_frames: int, dtype: torch.dtype | None=None) -> torch.Tensor:
    if K_all.ndim != 3 or K_all.shape != (6, 3, 3):
        raise ValueError(f'expected K_all shape (6,3,3), got {tuple(K_all.shape)}')
    if E_all.ndim != 3 or E_all.shape != (6, 4, 4):
        raise ValueError(f'expected E_all shape (6,4,4), got {tuple(E_all.shape)}')
    if stream_view_ids.ndim != 1:
        raise ValueError(f'expected stream_view_ids shape (S,), got {tuple(stream_view_ids.shape)}')
    if latent_frames <= 0:
        raise ValueError(f'expected latent_frames > 0, got {latent_frames}')
    device = stream_view_ids.device
    plucker_all = compute_plucker_rays(K_all.unsqueeze(0).to(device=device, dtype=torch.float32), E_all.unsqueeze(0).to(device=device, dtype=torch.float32), h_tok=30, w_tok=52)
    plucker_streams = plucker_all[:, stream_view_ids]
    if plucker_streams.shape[2] < latent_frames:
        raise ValueError(f'expected plucker temporal dim >= {latent_frames}, got {plucker_streams.shape[2]}')
    plucker_streams = plucker_streams[:, :, :latent_frames]
    return plucker_streams if dtype is None else plucker_streams.to(dtype=dtype)

def _encode_front(vae: object, front_rgb: torch.Tensor) -> torch.Tensor:
    if front_rgb.ndim != 4 or front_rgb.shape[1] != 3:
        raise ValueError(f'expected front_rgb shape (T, 3, H, W), got {tuple(front_rgb.shape)}')
    video = normalize_rgb_for_wan_vae(front_rgb.permute(1, 0, 2, 3).contiguous(), source_range='raw_0_255')
    return vae.encode([video])[0]

def encode_target_first_frame_latents(vae: object, target_rgb: torch.Tensor) -> torch.Tensor:
    if target_rgb.ndim != 5 or target_rgb.shape[2] != 3:
        raise ValueError(f'expected target_rgb shape (N,T,3,H,W), got {tuple(target_rgb.shape)}')
    latents: list[torch.Tensor] = []
    for target_clip in target_rgb:
        static_first_frame_clip = target_clip[0:1].expand(target_clip.shape[0], -1, -1, -1).contiguous()
        latents.append(_encode_front(vae, static_first_frame_clip)[:, 0].detach())
    return torch.stack(latents, dim=0)

def _decode_targets(vae: object, z_targets: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode([item for item in z_targets])
    return torch.stack(decoded, dim=0)

def _ensure_batched_text(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f'expected {name} shape (B,L,D) or (L,D), got {tuple(tensor.shape)}')
    return tensor

def _ensure_batched_mask(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f'expected {name} shape (B,L) or (L,), got {tuple(tensor.shape)}')
    return tensor

def sigmas_descending_from_start(scheduler: FlowMatchScheduler, num_steps: int, start_sigma: float, device: torch.device | str | None=None, dtype: torch.dtype=torch.float32) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError(f'expected num_steps > 0, got {num_steps}')
    if not 0.0 <= start_sigma <= 1.0:
        raise ValueError(f'expected start_sigma in [0, 1], got {start_sigma}')
    start = torch.tensor(float(start_sigma), device=device, dtype=dtype)
    start_t = scheduler.t_for_sigma(start)
    t = torch.linspace(float(start_t), 0.0, num_steps + 1, device=device, dtype=dtype)
    return scheduler.sigmas_for(t)

def _select_p3_expert(bundle: P3InferenceBundle, sigma: torch.Tensor, high_switch_sigma: float, use_high_expert: bool) -> torch.nn.Module:
    if use_high_expert and bundle.high_dit is not None and (float(sigma.detach().float().cpu()) > high_switch_sigma):
        return bundle.high_dit
    return bundle.low_dit

def _normalize_target_first_latents(target_first_latents: torch.Tensor, z_streams: torch.Tensor) -> torch.Tensor:
    if target_first_latents.ndim == 4:
        target_first_latents = target_first_latents.unsqueeze(0)
    expected = (z_streams.shape[0], 3, z_streams.shape[2], z_streams.shape[4], z_streams.shape[5])
    if tuple(target_first_latents.shape) != expected:
        raise ValueError(f'expected target_first_latents shape {expected}, got {tuple(target_first_latents.shape)}')
    return target_first_latents.to(device=z_streams.device, dtype=z_streams.dtype)

@torch.no_grad()
def inference_p3(front_rgb: torch.Tensor, K_all: torch.Tensor, E_all: torch.Tensor, text_prompt: str, target_views: list[int], bundle: P3InferenceBundle, num_steps: int=40, guide_scale: float=3.5, start_sigma: float=0.9, high_switch_sigma: float=0.9, use_high_expert: bool=False, target_first_latents: torch.Tensor | None=None, generator: torch.Generator | None=None) -> torch.Tensor:
    device = front_rgb.device
    z_front = _encode_front(bundle.vae, front_rgb).to(device=device)
    streams = build_canonical_p3_streams(z_front, target_views, generator=generator)
    z_streams = streams.z_streams
    z_front_batched = z_front.unsqueeze(0)
    target_first_latents_batched = _normalize_target_first_latents(target_first_latents, z_streams) if target_first_latents is not None else None
    plucker_streams = build_p3_plucker_streams(K_all.to(device), E_all.to(device), streams.stream_view_ids, latent_frames=z_streams.shape[3], dtype=z_streams.dtype)
    (text_emb, text_mask) = bundle.text_encoder.encode_cached(text_prompt)
    (null_emb, null_mask) = bundle.text_encoder.null_cached()
    text_emb = _ensure_batched_text(text_emb, 'text_emb').to(device=device)
    text_mask = _ensure_batched_mask(text_mask, 'text_mask').to(device=device)
    null_emb = _ensure_batched_text(null_emb, 'null_emb').to(device=device)
    null_mask = _ensure_batched_mask(null_mask, 'null_mask').to(device=device)
    sigmas = sigmas_descending_from_start(bundle.scheduler, num_steps, start_sigma, device=device, dtype=z_streams.dtype)
    for (sigma_value, sigma_next) in zip(sigmas[:-1], sigmas[1:]):
        sigma = sigma_value.view(1)
        expert = _select_p3_expert(bundle, sigma_value, high_switch_sigma, use_high_expert)
        z_streams[:, 0] = z_front_batched
        if target_first_latents_batched is not None:
            z_streams[:, 1:4, :, 0] = target_first_latents_batched
        v_cond = expert(z_streams, sigma, text_emb, text_mask, plucker_streams, streams.stream_view_ids, streams.stream_role_ids)
        v_uncond = expert(z_streams, sigma, null_emb, null_mask, plucker_streams, streams.stream_view_ids, streams.stream_role_ids)
        v_pred = v_uncond + guide_scale * (v_cond - v_uncond)
        z_streams = z_streams + (sigma_next - sigma_value).view(1, 1, 1, 1, 1, 1) * v_pred
        z_streams[:, 0] = z_front_batched
        if target_first_latents_batched is not None:
            z_streams[:, 1:4, :, 0] = target_first_latents_batched
    bundle.last_front_preserved = bool(torch.equal(z_streams[:, 0], z_front_batched))
    return _decode_targets(bundle.vae, z_streams[0, 1:4])
inference = inference_p3
