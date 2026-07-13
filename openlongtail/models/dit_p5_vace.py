from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from openlongtail.data.transforms import se3_inverse
from openlongtail.models.camera_embed import CamIDEmbed, PluckerMLP
from openlongtail.models.ray_adjacent_view_attention import DEFAULT_CAMERA_GRAPH, RayAdjacentViewAttention
from openlongtail.models.ray_sync_front_attention import RaySynchronizedFrontAttention
from openlongtail.models.dit import sinusoidal_embedding_1d
P5_VACE_CONTEXT_CHANNELS = 96
P5_ROLE_CONDITION = 0
P5_ROLE_TARGET = 1

def _so3_log_vector(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f'expected rotation shape (..., 3, 3), got {tuple(rotation.shape)}')
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack([rotation[..., 2, 1] - rotation[..., 1, 2], rotation[..., 0, 2] - rotation[..., 2, 0], rotation[..., 1, 0] - rotation[..., 0, 1]], dim=-1)
    sin_theta = torch.sin(theta)
    scale = torch.where(theta < 1e-05, torch.full_like(theta, 0.5), theta / (2.0 * sin_theta.clamp_min(1e-08)))
    return vee * scale.unsqueeze(-1)

def trajectory_vectors_from_anchor_front(T_anchor_front: torch.Tensor, frames: int) -> torch.Tensor:
    if T_anchor_front.ndim != 4 or T_anchor_front.shape[-2:] != (4, 4):
        raise ValueError(f'expected T_anchor_front shape (B, T, 4, 4), got {tuple(T_anchor_front.shape)}')
    if T_anchor_front.shape[1] < frames:
        raise ValueError(f'expected T_anchor_front temporal dim >= {frames}, got {T_anchor_front.shape[1]}')
    T = T_anchor_front[:, :frames].float()
    delta = torch.eye(4, device=T.device, dtype=T.dtype).view(1, 1, 4, 4).expand(T.shape[0], frames, -1, -1).clone()
    if frames > 1:
        delta[:, 1:] = se3_inverse(T[:, :-1]) @ T[:, 1:]
    trans = delta[:, :, :3, 3]
    rotvec = _so3_log_vector(delta[:, :, :3, :3])
    return torch.cat([trans, rotvec], dim=-1)

class TrajectoryMLP(nn.Module):

    def __init__(self, dim: int, hidden_dim: int=256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, motion_vec: torch.Tensor) -> torch.Tensor:
        return self.net(motion_vec)

class DiTP5VACE(nn.Module):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(5, 11, 17, 23, 29, 35), sync_temporal_window: int=2, camera_graph: tuple[tuple[int, int], ...]=DEFAULT_CAMERA_GRAPH, enable_adjacent_attention: bool=True, enable_motion_embedding: bool=True) -> None:
        super().__init__()
        self.expert = expert
        self.dim = int(dim or getattr(expert, 'dim'))
        self.cross_view_blocks = tuple((int(idx) for idx in cross_view_blocks))
        self.sync_temporal_window = int(sync_temporal_window)
        self.enable_adjacent_attention = bool(enable_adjacent_attention)
        self.enable_motion_embedding = bool(enable_motion_embedding)
        self.gradient_checkpointing = False
        self.use_reentrant = True
        self.plucker_mlp = PluckerMLP(out_dim=self.dim)
        self.cam_id_embed = CamIDEmbed(dim=self.dim)
        self.role_embed = nn.Embedding(2, self.dim)
        self.trajectory_mlp = TrajectoryMLP(self.dim)
        nn.init.normal_(self.role_embed.weight, std=0.02)
        self.sync_front = nn.ModuleList([RaySynchronizedFrontAttention(dim=self.dim, heads=heads, temporal_window=self.sync_temporal_window) for _ in self.cross_view_blocks])
        self.adjacent_view = nn.ModuleList([RayAdjacentViewAttention(dim=self.dim, heads=heads, camera_graph=camera_graph) for _ in self.cross_view_blocks])
        self.last_patch_input_shape: tuple[int, ...] | None = None
        self.last_vace_context_shape: tuple[int, ...] | None = None
        self.last_t_per_stream: torch.Tensor | None = None
        self.force_fp32_time_and_head_modules()

    def enable_gradient_checkpointing(self, use_reentrant: bool=True) -> None:
        self.gradient_checkpointing = True
        self.use_reentrant = bool(use_reentrant)

    def force_fp32_time_and_head_modules(self) -> None:
        self.expert.time_embedding.float()
        self.expert.time_projection.float()
        self.expert.head.float()

    def build_vace_context(self, z_streams: torch.Tensor, stream_role_ids: torch.Tensor) -> torch.Tensor:
        if z_streams.ndim != 6 or z_streams.shape[2] != 16:
            raise ValueError(f'expected z_streams shape (B, S, 16, T, H, W), got {tuple(z_streams.shape)}')
        (batch, streams, _, frames, height, width) = z_streams.shape
        if stream_role_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream_role_ids, got {tuple(stream_role_ids.shape)}')
        inactive = torch.zeros_like(z_streams)
        reactive = torch.zeros_like(z_streams)
        mask = torch.zeros(batch, streams, 64, frames, height, width, device=z_streams.device, dtype=z_streams.dtype)
        role_ids = [int(item) for item in stream_role_ids.detach().cpu().tolist()]
        for (stream_idx, role_id) in enumerate(role_ids):
            if role_id == P5_ROLE_CONDITION:
                inactive[:, stream_idx] = z_streams[:, stream_idx]
            elif role_id == P5_ROLE_TARGET:
                mask[:, stream_idx] = 1.0
            else:
                raise ValueError(f'expected role id 0 or 1, got {role_id}')
        return torch.cat([inactive, reactive, mask], dim=2)

    def _patchify_base(self, z_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (batch, streams) = z_streams.shape[:2]
        patch_dtype = self.expert.patch_embedding.weight.dtype
        flat = z_streams.reshape(batch * streams, *z_streams.shape[2:]).to(dtype=patch_dtype)
        self.last_patch_input_shape = tuple(flat.shape)
        x = self.expert.patch_embedding(flat)
        grid_sizes = torch.tensor([x.shape[2:]] * (batch * streams), dtype=torch.long, device=x.device)
        hidden = x.flatten(2).transpose(1, 2)
        seq_lens = torch.full((batch * streams,), hidden.shape[1], dtype=torch.long, device=x.device)
        return (hidden, grid_sizes, seq_lens)

    def _patchify_vace_context(self, vace_context: torch.Tensor) -> torch.Tensor:
        if vace_context.ndim != 6 or vace_context.shape[2] != P5_VACE_CONTEXT_CHANNELS:
            raise ValueError(f'expected vace_context shape (B, S, {P5_VACE_CONTEXT_CHANNELS}, T, H, W), got {tuple(vace_context.shape)}')
        (batch, streams) = vace_context.shape[:2]
        patch_dtype = self.expert.vace_patch_embedding.weight.dtype
        flat = vace_context.reshape(batch * streams, *vace_context.shape[2:]).to(dtype=patch_dtype)
        self.last_vace_context_shape = tuple(flat.shape)
        c = self.expert.vace_patch_embedding(flat)
        return c.flatten(2).transpose(1, 2)

    def _condition_embedding(self, plucker: torch.Tensor, stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, batch: int) -> torch.Tensor:
        if plucker.ndim != 6 or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, S, T, 6, H, W), got {tuple(plucker.shape)}')
        streams = plucker.shape[1]
        frames = plucker.shape[2]
        height = plucker.shape[4]
        width = plucker.shape[5]
        if stream_view_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream_view_ids, got {tuple(stream_view_ids.shape)}')
        if stream_role_ids.numel() != streams:
            raise ValueError(f'expected {streams} stream_role_ids, got {tuple(stream_role_ids.shape)}')
        plucker_emb = self.plucker_mlp(plucker).permute(0, 1, 2, 4, 5, 3).reshape(batch * streams, -1, self.dim)
        cam_emb = self.cam_id_embed(stream_view_ids.to(plucker.device)).view(1, streams, 1, self.dim)
        role_emb = self.role_embed(stream_role_ids.to(plucker.device)).view(1, streams, 1, self.dim)
        tokens = frames * height * width
        cam_emb = cam_emb.expand(batch, -1, tokens, -1).reshape(batch * streams, tokens, self.dim)
        role_emb = role_emb.expand(batch, -1, tokens, -1).reshape(batch * streams, tokens, self.dim)
        cond = plucker_emb + cam_emb + role_emb
        if self.enable_motion_embedding:
            motion_vec = trajectory_vectors_from_anchor_front(T_anchor_front.to(plucker.device), frames)
            motion_emb = self.trajectory_mlp(motion_vec.to(dtype=plucker.dtype))
            motion_emb = motion_emb[:, None, :, None, :].expand(batch, streams, frames, height * width, self.dim).reshape(batch * streams, tokens, self.dim)
            cond = cond + motion_emb
        return cond

    def _prepare_time(self, sigma: torch.Tensor, batch: int, streams: int, device: torch.device, stream_role_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.shape != (batch,):
            raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
        t_real = sigma / (5.0 - 4.0 * sigma)
        t_per_stream = t_real.unsqueeze(1).expand(batch, streams).clone()
        role_ids = stream_role_ids.to(device=t_per_stream.device)
        t_per_stream[:, role_ids == P5_ROLE_CONDITION] = 0
        self.last_t_per_stream = t_per_stream.detach().cpu()
        flat_t = t_per_stream.reshape(batch * streams).to(device)
        freq_dim = int(getattr(self.expert, 'freq_dim', 256))
        self.force_fp32_time_and_head_modules()
        with torch.amp.autocast('cuda', enabled=False):
            time_input = sinusoidal_embedding_1d(freq_dim, flat_t).to(device=device, dtype=torch.float32)
            e = self.expert.time_embedding(time_input)
            e0 = self.expert.time_projection(e).unflatten(1, (6, self.dim))
        return (e, e0)

    def _prepare_context(self, text_emb: torch.Tensor, batch: int, streams: int) -> torch.Tensor:
        if text_emb.ndim != 3 or text_emb.shape[0] != batch or text_emb.shape[-1] != 4096:
            raise ValueError(f'expected text_emb shape (B, L, 4096), got {tuple(text_emb.shape)}')
        context = text_emb[:, None].expand(batch, streams, -1, -1).reshape(batch * streams, text_emb.shape[1], text_emb.shape[2])
        text_len = int(getattr(self.expert, 'text_len', 512))
        if context.shape[1] < text_len:
            pad = context.new_zeros(context.shape[0], text_len - context.shape[1], context.shape[2])
            context = torch.cat([context, pad], dim=1)
        elif context.shape[1] > text_len:
            context = context[:, :text_len]
        return self.expert.text_embedding(context)

    def _forward_vace_hints(self, c: torch.Tensor, x: torch.Tensor, seq_len: int, kwargs: dict[str, object]) -> list[torch.Tensor]:
        if not hasattr(self.expert, 'vace_blocks'):
            return []
        if c.shape[1] < seq_len:
            c = torch.cat([c, c.new_zeros(c.shape[0], seq_len - c.shape[1], c.shape[2])], dim=1)
        elif c.shape[1] > seq_len:
            c = c[:, :seq_len]
        hints: list[torch.Tensor] = []
        new_kwargs = dict(kwargs)
        new_kwargs['x'] = x
        for block in self.expert.vace_blocks:
            (c, c_skip) = block(c, **new_kwargs)
            hints.append(c_skip)
        return hints

    def _ensure_reentrant_checkpoint_input(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and (not hidden.requires_grad):
            return hidden.detach().requires_grad_(True)
        return hidden

    def forward(self, z_streams: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, plucker: torch.Tensor, stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, vace_context: torch.Tensor | None=None) -> torch.Tensor:
        del text_mask
        if z_streams.ndim != 6 or z_streams.shape[2] != 16:
            raise ValueError(f'expected z_streams shape (B, S, 16, T, H, W), got {tuple(z_streams.shape)}')
        (batch, streams) = z_streams.shape[:2]
        if streams < 2:
            raise ValueError(f'expected at least 2 P5 streams, got {streams}')
        if vace_context is None:
            vace_context = self.build_vace_context(z_streams, stream_role_ids)
        (hidden, grid_sizes, seq_lens) = self._patchify_base(z_streams)
        cond_emb = self._condition_embedding(plucker.to(hidden.device, hidden.dtype), stream_view_ids.to(hidden.device), stream_role_ids.to(hidden.device), T_anchor_front.to(hidden.device), batch)
        hidden = hidden + cond_emb
        vace_hidden = self._patchify_vace_context(vace_context.to(hidden.device, hidden.dtype)) + cond_emb
        hidden = self._ensure_reentrant_checkpoint_input(hidden)
        seq_len = hidden.shape[1]
        (e, e0) = self._prepare_time(sigma.to(hidden.device, hidden.dtype), batch, streams, hidden.device, stream_role_ids.to(hidden.device))
        context = self._prepare_context(text_emb.to(hidden.device, hidden.dtype), batch, streams)
        context_lens = None
        freqs = self.expert.freqs.to(hidden.device) if hasattr(self.expert, 'freqs') else None
        autocast_enabled = hidden.device.type == 'cuda'
        base_kwargs = {'e': e0, 'seq_lens': seq_lens, 'grid_sizes': grid_sizes, 'freqs': freqs, 'context': context, 'context_lens': context_lens}
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hints = self._forward_vace_hints(vace_hidden, hidden, seq_len, base_kwargs)
        cva_idx = 0
        for (block_idx, block) in enumerate(self.expert.blocks):
            kwargs = dict(base_kwargs)
            kwargs['hints'] = hints
            kwargs['context_scale'] = 1.0
            with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                if self.gradient_checkpointing and self.training:
                    hidden = checkpoint(lambda x, b=block, kw=kwargs: b(x, **kw), hidden, use_reentrant=self.use_reentrant)
                else:
                    hidden = block(hidden, **kwargs)
            if block_idx in self.cross_view_blocks:
                rays = plucker.to(hidden.device, hidden.dtype)
                with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    if self.gradient_checkpointing and self.training:
                        hidden = checkpoint(lambda x, p, svid, srid, module=self.sync_front[cva_idx]: x + module(x, p, svid, srid), hidden, rays, stream_view_ids.to(hidden.device), stream_role_ids.to(hidden.device), use_reentrant=self.use_reentrant)
                    else:
                        hidden = hidden + self.sync_front[cva_idx](hidden, rays, stream_view_ids.to(hidden.device), stream_role_ids.to(hidden.device))
                    if self.enable_adjacent_attention:
                        if self.gradient_checkpointing and self.training:
                            hidden = checkpoint(lambda x, p, svid, srid, module=self.adjacent_view[cva_idx]: x + module(x, p, svid, srid), hidden, rays, stream_view_ids.to(hidden.device), stream_role_ids.to(hidden.device), use_reentrant=self.use_reentrant)
                        else:
                            hidden = hidden + self.adjacent_view[cva_idx](hidden, rays, stream_view_ids.to(hidden.device), stream_role_ids.to(hidden.device))
                cva_idx += 1
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.expert.head(hidden, e)
        outputs = self.expert.unpatchify(hidden, grid_sizes)
        return torch.stack(outputs, dim=0).reshape(batch, streams, *outputs[0].shape)
