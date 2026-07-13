from __future__ import annotations
import math
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from openlongtail.models.camera_embed import CamIDEmbed, PluckerMLP
from openlongtail.models.dit import sinusoidal_embedding_1d
from openlongtail.models.dit_p5_vace import P5_ROLE_CONDITION, P5_ROLE_TARGET, P5_VACE_CONTEXT_CHANNELS, TrajectoryMLP, trajectory_vectors_from_anchor_front
P61_TYPE_FRONT = 0
P61_TYPE_NEIGHBOR = 1
P61_TYPE_PAD = 2
P61_BACKBONE_STREAMS = 2
P61_CONDITION_SLOTS = 3
P61_TARGET_STREAM_INDEX = 1
WAN_NATIVE_TIMESTEP_SCALE = 1000.0

class ConditionAdapterBlock(nn.Module):

    def __init__(self, dim: int, hidden_mult: int=2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * hidden_mult), nn.GELU(approximate='tanh'), nn.Linear(dim * hidden_mult, dim))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.ffn(self.norm(tokens))

class SDPACrossAttention(nn.Module):

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'expected dim divisible by heads, got dim={dim}, heads={heads}')
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, memory_available: torch.Tensor | None=None) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError(f'expected query/memory rank 3, got {tuple(query.shape)}, {tuple(memory.shape)}')
        (batch, q_len, _) = query.shape
        if memory.shape[0] != batch:
            raise ValueError(f'expected memory batch {batch}, got {memory.shape[0]}')
        q = self.to_q(query).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if memory_available is not None:
            if memory_available.shape != (batch, memory.shape[1]):
                raise ValueError(f'expected memory_available shape {(batch, memory.shape[1])}, got {tuple(memory_available.shape)}')
            attn_mask = torch.zeros(batch, 1, 1, memory.shape[1], device=query.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~memory_available[:, None, None, :].to(torch.bool), float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, q_len, self.dim)
        return self.out(out)

class SemanticResampler(nn.Module):

    def __init__(self, dim: int, heads: int, num_queries: int) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attn = SDPACrossAttention(dim, heads)

    def forward(self, dense_tokens: torch.Tensor) -> torch.Tensor:
        if dense_tokens.ndim != 5:
            raise ValueError(f'expected dense_tokens shape (B,S,T,N,D), got {tuple(dense_tokens.shape)}')
        (batch, slots, frames, _, dim) = dense_tokens.shape
        flat = dense_tokens.reshape(batch * slots * frames, dense_tokens.shape[3], dim)
        query = self.queries.to(device=dense_tokens.device, dtype=dense_tokens.dtype)
        query = query.view(1, self.num_queries, dim).expand(flat.shape[0], -1, -1)
        sem = self.attn(query, flat)
        return sem.reshape(batch, slots, frames, self.num_queries, dim)

class SigmaGate(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.0)

    def forward(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.sigmoid(self.net(sigma.float().view(-1, 1)))
        return (values[:, 0].view(-1, 1, 1), values[:, 1].view(-1, 1, 1))

class P61GraphMemoryAttention(nn.Module):

    def __init__(self, dim: int, heads: int=16, temporal_window: int=2, relative_pose_hidden: int=256) -> None:
        super().__init__()
        if temporal_window < 0:
            raise ValueError(f'expected temporal_window >= 0, got {temporal_window}')
        self.dim = int(dim)
        self.temporal_window = int(temporal_window)
        self.relative_pose_mlp = nn.Sequential(nn.Linear(6, relative_pose_hidden), nn.SiLU(), nn.Linear(relative_pose_hidden, dim))
        self.temporal_offset_embed = nn.Embedding(2 * temporal_window + 1, dim)
        self.semantic_attn = SDPACrossAttention(dim, heads)
        self.dense_attn = SDPACrossAttention(dim, heads)
        self.gate = SigmaGate()
        self.last_window_token_count: int | None = None

    def _window_memory(self, memory: torch.Tensor, relative_pose_features: torch.Tensor, condition_available_mask: torch.Tensor, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        (batch, slots, frames, tokens, dim) = memory.shape
        lo = max(0, frame_idx - self.temporal_window)
        hi = min(frames, frame_idx + self.temporal_window + 1)
        mem = memory[:, :, lo:hi]
        rel = relative_pose_features[:, frame_idx, :, lo:hi]
        rel_emb = self.relative_pose_mlp(rel.to(device=memory.device, dtype=memory.dtype))
        offsets = torch.arange(lo, hi, device=memory.device) - frame_idx + self.temporal_window
        off_emb = self.temporal_offset_embed(offsets).to(dtype=memory.dtype)
        rel_emb = rel_emb + off_emb.view(1, 1, hi - lo, dim)
        mem = mem + rel_emb.unsqueeze(3)
        mem = mem.reshape(batch, slots * (hi - lo) * tokens, dim)
        valid = condition_available_mask.to(device=memory.device, dtype=torch.bool)
        valid = valid.view(1, slots, 1, 1).expand(batch, slots, hi - lo, tokens)
        valid = valid.reshape(batch, slots * (hi - lo) * tokens)
        self.last_window_token_count = int(mem.shape[1])
        return (mem, valid)

    def forward(self, target_tokens: torch.Tensor, dense_memory: torch.Tensor, semantic_memory: torch.Tensor, relative_pose_features: torch.Tensor, condition_available_mask: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if target_tokens.ndim != 4:
            raise ValueError(f'expected target_tokens shape (B,T,N,D), got {tuple(target_tokens.shape)}')
        (batch, frames, tokens, dim) = target_tokens.shape
        if dense_memory.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected dense_memory shape {tuple(dense_memory.shape)}')
        if semantic_memory.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected semantic_memory shape {tuple(semantic_memory.shape)}')
        if relative_pose_features.shape[:4] != (batch, frames, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected relative_pose_features shape {tuple(relative_pose_features.shape)}')
        if condition_available_mask.shape != (P61_CONDITION_SLOTS,):
            raise ValueError(f'expected condition_available_mask shape ({P61_CONDITION_SLOTS},), got {tuple(condition_available_mask.shape)}')
        (g_sem, g_dense) = self.gate(sigma.to(target_tokens.device))
        delta = torch.zeros_like(target_tokens)
        for frame_idx in range(frames):
            query = target_tokens[:, frame_idx]
            (sem, sem_valid) = self._window_memory(semantic_memory, relative_pose_features, condition_available_mask, frame_idx)
            (dense, dense_valid) = self._window_memory(dense_memory, relative_pose_features, condition_available_mask, frame_idx)
            sem_out = self.semantic_attn(query, sem, sem_valid)
            dense_out = self.dense_attn(query, dense, dense_valid)
            delta[:, frame_idx] = g_sem.to(dtype=query.dtype) * sem_out + g_dense.to(dtype=query.dtype) * dense_out
        return delta.reshape(batch, frames * tokens, dim)

class DiTP61VACE(nn.Module):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(5, 11, 17, 23, 29, 35), sync_temporal_window: int=2, condition_encoder_layers: int=2, semantic_queries: int=64, enable_motion_embedding: bool=True) -> None:
        super().__init__()
        del dim_attn
        self.expert = expert
        self.dim = int(dim or getattr(expert, 'dim'))
        self.cross_view_blocks = tuple((int(idx) for idx in cross_view_blocks))
        self.sync_temporal_window = int(sync_temporal_window)
        self.enable_motion_embedding = bool(enable_motion_embedding)
        self.gradient_checkpointing = False
        self.use_reentrant = True
        self.plucker_mlp = PluckerMLP(out_dim=self.dim)
        self.cam_id_embed = CamIDEmbed(dim=self.dim)
        self.role_embed = nn.Embedding(2, self.dim)
        self.condition_type_embed = nn.Embedding(3, self.dim)
        self.availability_embed = nn.Embedding(2, self.dim)
        self.trajectory_mlp = TrajectoryMLP(self.dim)
        self.condition_encoder = nn.ModuleList([ConditionAdapterBlock(self.dim) for _ in range(condition_encoder_layers)])
        self.semantic_resampler = SemanticResampler(self.dim, heads, semantic_queries)
        self.graph_memory = nn.ModuleList([P61GraphMemoryAttention(self.dim, heads=heads, temporal_window=self.sync_temporal_window) for _ in self.cross_view_blocks])
        nn.init.normal_(self.role_embed.weight, std=0.02)
        nn.init.normal_(self.condition_type_embed.weight, std=0.02)
        nn.init.normal_(self.availability_embed.weight, std=0.02)
        self.last_patch_input_shape: tuple[int, ...] | None = None
        self.last_vace_context_shape: tuple[int, ...] | None = None
        self.last_condition_bank_shape: tuple[int, ...] | None = None
        self.last_t_per_stream: torch.Tensor | None = None
        self.collect_geometry_debug = False
        self.geometry_debug_records: list[dict[str, object]] = []
        self._last_backbone_embedding_debug: dict[str, float] = {}
        self._last_condition_memory_debug: dict[str, float] = {}
        self.force_fp32_time_and_head_modules()

    def enable_gradient_checkpointing(self, use_reentrant: bool=True) -> None:
        self.gradient_checkpointing = True
        self.use_reentrant = bool(use_reentrant)

    def set_geometry_debug(self, enabled: bool=True) -> None:
        self.collect_geometry_debug = bool(enabled)

    def clear_geometry_debug_records(self) -> None:
        self.geometry_debug_records.clear()

    @staticmethod
    def _debug_norm(tensor: torch.Tensor) -> float:
        return float(tensor.detach().float().norm().cpu())

    def force_fp32_time_and_head_modules(self) -> None:
        self.expert.time_embedding.float()
        self.expert.time_projection.float()
        self.expert.head.float()

    def build_vace_context(self, z_backbone: torch.Tensor) -> torch.Tensor:
        if z_backbone.ndim != 6 or z_backbone.shape[1] != P61_BACKBONE_STREAMS or z_backbone.shape[2] != 16:
            raise ValueError(f'expected z_backbone shape (B,2,16,T,H,W), got {tuple(z_backbone.shape)}')
        (batch, _, _, frames, height, width) = z_backbone.shape
        inactive = torch.zeros_like(z_backbone)
        reactive = torch.zeros_like(z_backbone)
        mask = torch.zeros(batch, P61_BACKBONE_STREAMS, 64, frames, height, width, device=z_backbone.device, dtype=z_backbone.dtype)
        inactive[:, 0] = z_backbone[:, 0]
        mask[:, P61_TARGET_STREAM_INDEX] = 1.0
        return torch.cat([inactive, reactive, mask], dim=2)

    def _patchify_base(self, z_backbone: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (batch, streams) = z_backbone.shape[:2]
        patch_dtype = self.expert.patch_embedding.weight.dtype
        flat = z_backbone.reshape(batch * streams, *z_backbone.shape[2:]).to(dtype=patch_dtype)
        self.last_patch_input_shape = tuple(flat.shape)
        x = self.expert.patch_embedding(flat)
        grid_sizes = torch.tensor([x.shape[2:]] * (batch * streams), dtype=torch.long, device=x.device)
        hidden = x.flatten(2).transpose(1, 2)
        seq_lens = torch.full((batch * streams,), hidden.shape[1], dtype=torch.long, device=x.device)
        return (hidden, grid_sizes, seq_lens)

    def _patchify_vace_context(self, vace_context: torch.Tensor) -> torch.Tensor:
        if vace_context.ndim != 6 or vace_context.shape[1] != P61_BACKBONE_STREAMS or vace_context.shape[2] != P5_VACE_CONTEXT_CHANNELS:
            raise ValueError(f'expected vace_context shape (B,2,96,T,H,W), got {tuple(vace_context.shape)}')
        (batch, streams) = vace_context.shape[:2]
        patch_dtype = self.expert.vace_patch_embedding.weight.dtype
        flat = vace_context.reshape(batch * streams, *vace_context.shape[2:]).to(dtype=patch_dtype)
        self.last_vace_context_shape = tuple(flat.shape)
        c = self.expert.vace_patch_embedding(flat)
        return c.flatten(2).transpose(1, 2)

    def _motion_embedding(self, T_anchor_front: torch.Tensor, frames: int, tokens_per_frame: int, streams: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        motion_vec = trajectory_vectors_from_anchor_front(T_anchor_front.to(device), frames)
        motion_emb = self.trajectory_mlp(motion_vec.to(dtype=dtype))
        return motion_emb[:, None, :, None, :].expand(T_anchor_front.shape[0], streams, frames, tokens_per_frame, self.dim).reshape(T_anchor_front.shape[0] * streams, frames * tokens_per_frame, self.dim)

    def _backbone_embedding(self, plucker: torch.Tensor, stream_view_ids: torch.Tensor, stream_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, batch: int) -> torch.Tensor:
        if plucker.ndim != 6 or plucker.shape[:2] != (batch, P61_BACKBONE_STREAMS):
            raise ValueError(f'expected plucker shape (B,2,T,6,H,W), got {tuple(plucker.shape)}')
        (frames, height, width) = (plucker.shape[2], plucker.shape[4], plucker.shape[5])
        tokens = frames * height * width
        plucker_emb = self.plucker_mlp(plucker).permute(0, 1, 2, 4, 5, 3).reshape(batch * P61_BACKBONE_STREAMS, tokens, self.dim)
        cam_emb = self.cam_id_embed(stream_view_ids.to(plucker.device)).view(1, P61_BACKBONE_STREAMS, 1, self.dim)
        role_emb = self.role_embed(stream_role_ids.to(plucker.device)).view(1, P61_BACKBONE_STREAMS, 1, self.dim)
        cond = plucker_emb
        cond = cond + cam_emb.expand(batch, -1, tokens, -1).reshape(batch * P61_BACKBONE_STREAMS, tokens, self.dim)
        cond = cond + role_emb.expand(batch, -1, tokens, -1).reshape(batch * P61_BACKBONE_STREAMS, tokens, self.dim)
        motion_emb = None
        if self.enable_motion_embedding:
            motion_emb = self._motion_embedding(T_anchor_front, frames, height * width, P61_BACKBONE_STREAMS, plucker.dtype, plucker.device)
            cond = cond + motion_emb
        if self.collect_geometry_debug:
            self._last_backbone_embedding_debug = {'backbone_plucker_emb_norm': self._debug_norm(plucker_emb), 'backbone_cam_emb_norm': self._debug_norm(cam_emb), 'backbone_role_emb_norm': self._debug_norm(role_emb), 'backbone_motion_emb_norm': self._debug_norm(motion_emb) if motion_emb is not None else 0.0, 'backbone_cond_emb_norm': self._debug_norm(cond)}
        return cond

    def _condition_bank_memory(self, condition_latents: torch.Tensor, condition_plucker: torch.Tensor, condition_view_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_available_mask: torch.Tensor, T_anchor_front: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition_latents.ndim != 6 or condition_latents.shape[1] != P61_CONDITION_SLOTS or condition_latents.shape[2] != 16:
            raise ValueError(f'expected condition_latents shape (B,3,16,T,H,W), got {tuple(condition_latents.shape)}')
        (batch, slots, _, frames, _, _) = condition_latents.shape
        if condition_plucker.ndim != 6 or condition_plucker.shape[:2] != (batch, slots):
            raise ValueError(f'expected condition_plucker shape (B,3,T,6,H,W), got {tuple(condition_plucker.shape)}')
        if condition_view_ids.shape != (slots,):
            raise ValueError(f'expected condition_view_ids shape ({slots},), got {tuple(condition_view_ids.shape)}')
        if condition_type_ids.shape != (slots,):
            raise ValueError(f'expected condition_type_ids shape ({slots},), got {tuple(condition_type_ids.shape)}')
        if condition_available_mask.shape != (slots,):
            raise ValueError(f'expected condition_available_mask shape ({slots},), got {tuple(condition_available_mask.shape)}')
        patch_dtype = self.expert.patch_embedding.weight.dtype
        flat = condition_latents.reshape(batch * slots, *condition_latents.shape[2:]).to(dtype=patch_dtype)
        patch = self.expert.patch_embedding(flat)
        (h_tok, w_tok) = patch.shape[-2:]
        tokens_per_frame = h_tok * w_tok
        hidden = patch.flatten(2).transpose(1, 2)
        plucker_emb = self.plucker_mlp(condition_plucker.to(device=hidden.device, dtype=hidden.dtype))
        plucker_emb = plucker_emb.permute(0, 1, 2, 4, 5, 3).reshape(batch * slots, frames * tokens_per_frame, self.dim)
        cam_emb = self.cam_id_embed(condition_view_ids.to(hidden.device)).view(1, slots, 1, self.dim)
        type_emb = self.condition_type_embed(condition_type_ids.to(hidden.device)).view(1, slots, 1, self.dim)
        avail_ids = condition_available_mask.to(device=hidden.device, dtype=torch.long).clamp(0, 1)
        avail_emb = self.availability_embed(avail_ids).view(1, slots, 1, self.dim)
        hidden = hidden + plucker_emb
        hidden = hidden + cam_emb.expand(batch, -1, frames * tokens_per_frame, -1).reshape(batch * slots, frames * tokens_per_frame, self.dim)
        hidden = hidden + type_emb.expand(batch, -1, frames * tokens_per_frame, -1).reshape(batch * slots, frames * tokens_per_frame, self.dim)
        hidden = hidden + avail_emb.expand(batch, -1, frames * tokens_per_frame, -1).reshape(batch * slots, frames * tokens_per_frame, self.dim)
        motion_emb = None
        if self.enable_motion_embedding:
            motion_emb = self._motion_embedding(T_anchor_front, frames, tokens_per_frame, slots, hidden.dtype, hidden.device)
            hidden = hidden + motion_emb
        for block in self.condition_encoder:
            hidden = block(hidden)
        dense = hidden.reshape(batch, slots, frames, tokens_per_frame, self.dim)
        self.last_condition_bank_shape = tuple(dense.shape)
        semantic = self.semantic_resampler(dense)
        if self.collect_geometry_debug:
            self._last_condition_memory_debug = {'condition_plucker_emb_norm': self._debug_norm(plucker_emb), 'condition_cam_emb_norm': self._debug_norm(cam_emb), 'condition_type_emb_norm': self._debug_norm(type_emb), 'condition_availability_emb_norm': self._debug_norm(avail_emb), 'condition_motion_emb_norm': self._debug_norm(motion_emb) if motion_emb is not None else 0.0, 'condition_dense_memory_norm': self._debug_norm(dense), 'condition_semantic_memory_norm': self._debug_norm(semantic)}
        return (dense, semantic)

    def _prepare_time(self, sigma: torch.Tensor, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma.shape != (batch,):
            raise ValueError(f'expected sigma shape ({batch},), got {tuple(sigma.shape)}')
        t_wan = sigma.float() * WAN_NATIVE_TIMESTEP_SCALE
        t_per_stream = torch.stack([torch.zeros_like(t_wan), t_wan], dim=1)
        self.last_t_per_stream = t_per_stream.detach().cpu()
        flat_t = t_per_stream.reshape(batch * P61_BACKBONE_STREAMS).to(device)
        freq_dim = int(getattr(self.expert, 'freq_dim', 256))
        self.force_fp32_time_and_head_modules()
        with torch.amp.autocast('cuda', enabled=False):
            time_input = sinusoidal_embedding_1d(freq_dim, flat_t).to(device=device, dtype=torch.float32)
            e = self.expert.time_embedding(time_input)
            e0 = self.expert.time_projection(e).unflatten(1, (6, self.dim))
        return (e, e0)

    def _prepare_context(self, text_emb: torch.Tensor, batch: int) -> torch.Tensor:
        if text_emb.ndim != 3 or text_emb.shape[0] != batch or text_emb.shape[-1] != 4096:
            raise ValueError(f'expected text_emb shape (B, L, 4096), got {tuple(text_emb.shape)}')
        context = text_emb[:, None].expand(batch, P61_BACKBONE_STREAMS, -1, -1).reshape(batch * P61_BACKBONE_STREAMS, text_emb.shape[1], text_emb.shape[2])
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

    def forward(self, z_backbone: torch.Tensor, sigma: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, backbone_plucker: torch.Tensor, backbone_view_ids: torch.Tensor, backbone_role_ids: torch.Tensor, T_anchor_front: torch.Tensor, condition_latents: torch.Tensor, condition_plucker: torch.Tensor, condition_view_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_available_mask: torch.Tensor, relative_pose_features: torch.Tensor, vace_context: torch.Tensor | None=None) -> torch.Tensor:
        del text_mask
        if z_backbone.ndim != 6 or z_backbone.shape[1] != P61_BACKBONE_STREAMS or z_backbone.shape[2] != 16:
            raise ValueError(f'expected z_backbone shape (B,2,16,T,H,W), got {tuple(z_backbone.shape)}')
        batch = z_backbone.shape[0]
        if backbone_view_ids.shape != (P61_BACKBONE_STREAMS,):
            raise ValueError(f'expected backbone_view_ids shape (2,), got {tuple(backbone_view_ids.shape)}')
        if backbone_role_ids.shape != (P61_BACKBONE_STREAMS,):
            raise ValueError(f'expected backbone_role_ids shape (2,), got {tuple(backbone_role_ids.shape)}')
        if vace_context is None:
            vace_context = self.build_vace_context(z_backbone)
        (dense_memory, semantic_memory) = self._condition_bank_memory(condition_latents, condition_plucker, condition_view_ids, condition_type_ids, condition_available_mask, T_anchor_front)
        (hidden, grid_sizes, seq_lens) = self._patchify_base(z_backbone)
        pre_cond_hidden_norm = self._debug_norm(hidden) if self.collect_geometry_debug else 0.0
        cond_emb = self._backbone_embedding(backbone_plucker.to(hidden.device, hidden.dtype), backbone_view_ids.to(hidden.device), backbone_role_ids.to(hidden.device), T_anchor_front.to(hidden.device), batch)
        hidden = hidden + cond_emb
        post_cond_hidden_norm = self._debug_norm(hidden) if self.collect_geometry_debug else 0.0
        vace_hidden = self._patchify_vace_context(vace_context.to(hidden.device, hidden.dtype)) + cond_emb
        hidden = self._ensure_reentrant_checkpoint_input(hidden)
        seq_len = hidden.shape[1]
        (e, e0) = self._prepare_time(sigma.to(hidden.device, hidden.dtype), batch, hidden.device)
        context = self._prepare_context(text_emb.to(hidden.device, hidden.dtype), batch)
        freqs = self.expert.freqs.to(hidden.device) if hasattr(self.expert, 'freqs') else None
        autocast_enabled = hidden.device.type == 'cuda'
        base_kwargs = {'e': e0, 'seq_lens': seq_lens, 'grid_sizes': grid_sizes, 'freqs': freqs, 'context': context, 'context_lens': None}
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hints = self._forward_vace_hints(vace_hidden, hidden, seq_len, base_kwargs)
        target_tokens_per_stream = hidden.shape[1]
        frames = backbone_plucker.shape[2]
        tokens_per_frame = target_tokens_per_stream // frames
        debug_record: dict[str, object] | None = None
        if self.collect_geometry_debug:
            debug_record = {'sigma': float(sigma.detach().float().mean().cpu()), 'backbone_view_ids': [int(item) for item in backbone_view_ids.detach().cpu().tolist()], 'condition_view_ids': [int(item) for item in condition_view_ids.detach().cpu().tolist()], 'condition_available_mask': [int(item) for item in condition_available_mask.detach().cpu().to(torch.long).tolist()], 'pre_cond_hidden_norm': pre_cond_hidden_norm, 'post_cond_hidden_norm': post_cond_hidden_norm, 'cond_emb_to_hidden_ratio': self._debug_norm(cond_emb) / max(pre_cond_hidden_norm, 1e-12), **self._last_backbone_embedding_debug, **self._last_condition_memory_debug, 'graph_blocks': []}
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
                target_hidden = hidden.view(batch, P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)[:, P61_TARGET_STREAM_INDEX]
                target_frames = target_hidden.reshape(batch, frames, tokens_per_frame, self.dim)
                with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    if self.gradient_checkpointing and self.training:
                        delta = checkpoint(lambda tgt, dense, sem, rel, avail, sig, module=self.graph_memory[cva_idx]: module(tgt, dense, sem, rel, avail, sig), target_frames, dense_memory.to(hidden.device, hidden.dtype), semantic_memory.to(hidden.device, hidden.dtype), relative_pose_features.to(hidden.device, hidden.dtype), condition_available_mask.to(hidden.device), sigma.to(hidden.device), use_reentrant=self.use_reentrant)
                    else:
                        delta = self.graph_memory[cva_idx](target_frames, dense_memory.to(hidden.device, hidden.dtype), semantic_memory.to(hidden.device, hidden.dtype), relative_pose_features.to(hidden.device, hidden.dtype), condition_available_mask.to(hidden.device), sigma.to(hidden.device))
                if debug_record is not None:
                    target_norm = self._debug_norm(target_hidden)
                    delta_norm = self._debug_norm(delta)
                    gate_module = self.graph_memory[cva_idx].gate
                    with torch.no_grad(), torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        (g_sem, g_dense) = gate_module(sigma.to(hidden.device))
                    debug_record['graph_blocks'].append({'block_idx': int(block_idx), 'target_hidden_norm': target_norm, 'graph_delta_norm': delta_norm, 'graph_delta_to_target_ratio': delta_norm / max(target_norm, 1e-12), 'g_sem_mean': float(g_sem.detach().float().mean().cpu()), 'g_dense_mean': float(g_dense.detach().float().mean().cpu())})
                hidden_view = hidden.view(batch, P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)
                front_stream = hidden_view[:, 0]
                target_stream = hidden_view[:, P61_TARGET_STREAM_INDEX] + delta
                hidden = torch.stack([front_stream, target_stream], dim=1).reshape(batch * P61_BACKBONE_STREAMS, target_tokens_per_stream, self.dim)
                cva_idx += 1
        if debug_record is not None:
            self.geometry_debug_records.append(debug_record)
        with torch.amp.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden = self.expert.head(hidden, e)
        outputs = self.expert.unpatchify(hidden, grid_sizes)
        return torch.stack(outputs, dim=0).reshape(batch, P61_BACKBONE_STREAMS, *outputs[0].shape)
