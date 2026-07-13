from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from torch import nn
from openlongtail.models.dit_p61_vace import P61_CONDITION_SLOTS
from openlongtail.models.dit_p62_vace import P62SigmaGate, DiTP62VACE

class P63SigmaGate(P62SigmaGate):

    def __init__(self, init_bias: float=-1.4) -> None:
        super().__init__(init_bias=init_bias)

class GeometrySDPACrossAttentionP63(nn.Module):

    def __init__(self, dim: int, heads: int, query_geo_dim: int, memory_geo_dim: int, geo_head_dim: int=16, zero_init_out: bool=False, geo_projection_temperature: float=4.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'expected dim divisible by heads, got dim={dim}, heads={heads}')
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.geo_head_dim = int(geo_head_dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q_geo = nn.Linear(query_geo_dim, heads * self.geo_head_dim)
        self.to_k_geo = nn.Linear(memory_geo_dim, heads * self.geo_head_dim)
        self.geo_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.geo_input_clip = 256.0
        self.geo_projection_temperature = float(geo_projection_temperature)
        self.out = nn.Linear(dim, dim)
        if zero_init_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, query_geo: torch.Tensor, memory_geo: torch.Tensor, memory_available: torch.Tensor | None=None) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError(f'expected query/memory rank 3, got {tuple(query.shape)}, {tuple(memory.shape)}')
        if query_geo.shape[:2] != query.shape[:2]:
            raise ValueError(f'expected query_geo prefix {query.shape[:2]}, got {tuple(query_geo.shape)}')
        if memory_geo.shape[:2] != memory.shape[:2]:
            raise ValueError(f'expected memory_geo prefix {memory.shape[:2]}, got {tuple(memory_geo.shape)}')
        (batch, q_len, _) = query.shape
        if memory.shape[0] != batch:
            raise ValueError(f'expected memory batch {batch}, got {memory.shape[0]}')
        q = self.to_q(query).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
        query_geo = query_geo.to(device=query.device, dtype=query.dtype)
        memory_geo = memory_geo.to(device=query.device, dtype=query.dtype)
        query_geo = torch.nan_to_num(query_geo, nan=0.0, posinf=self.geo_input_clip, neginf=-self.geo_input_clip).clamp(-self.geo_input_clip, self.geo_input_clip)
        memory_geo = torch.nan_to_num(memory_geo, nan=0.0, posinf=self.geo_input_clip, neginf=-self.geo_input_clip).clamp(-self.geo_input_clip, self.geo_input_clip)
        q_geo = self.to_q_geo(query_geo)
        k_geo = self.to_k_geo(memory_geo)
        q_geo = torch.tanh(q_geo.float() / self.geo_projection_temperature).to(dtype=query.dtype)
        k_geo = torch.tanh(k_geo.float() / self.geo_projection_temperature).to(dtype=query.dtype)
        q_geo = q_geo.view(batch, q_len, self.heads, self.geo_head_dim).transpose(1, 2)
        k_geo = k_geo.view(batch, memory.shape[1], self.heads, self.geo_head_dim).transpose(1, 2)
        scale = self.geo_logit_scale.to(device=query.device, dtype=query.dtype).clamp(0.0, 4.0)
        q = torch.cat([q, scale * q_geo], dim=-1)
        k = torch.cat([k, scale * k_geo], dim=-1)
        attn_mask = None
        if memory_available is not None:
            if memory_available.shape != (batch, memory.shape[1]):
                raise ValueError(f'expected memory_available shape {(batch, memory.shape[1])}, got {tuple(memory_available.shape)}')
            attn_mask = torch.zeros(batch, 1, 1, memory.shape[1], device=query.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~memory_available[:, None, None, :].to(torch.bool), float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, q_len, self.dim)
        return self.out(out)

class P63GraphMemoryAttention(nn.Module):

    def __init__(self, dim: int, heads: int=16, temporal_window: int=2, relative_pose_hidden: int=256, graph_gate_init_bias: float=-1.4, geo_head_dim: int=16, geo_projection_temperature: float=4.0) -> None:
        super().__init__()
        if temporal_window < 0:
            raise ValueError(f'expected temporal_window >= 0, got {temporal_window}')
        self.dim = int(dim)
        self.temporal_window = int(temporal_window)
        self.relative_pose_mlp = nn.Sequential(nn.Linear(6, relative_pose_hidden), nn.SiLU(), nn.Linear(relative_pose_hidden, dim))
        self.temporal_offset_embed = nn.Embedding(2 * temporal_window + 1, dim)
        self.semantic_attn = GeometrySDPACrossAttentionP63(dim, heads, query_geo_dim=6, memory_geo_dim=13, geo_head_dim=geo_head_dim, geo_projection_temperature=geo_projection_temperature)
        self.dense_attn = GeometrySDPACrossAttentionP63(dim, heads, query_geo_dim=6, memory_geo_dim=13, geo_head_dim=geo_head_dim, geo_projection_temperature=geo_projection_temperature)
        self.gate = P63SigmaGate(init_bias=graph_gate_init_bias)
        self.last_window_token_count: int | None = None

    @staticmethod
    def _target_query_geo(target_plucker: torch.Tensor, frame_idx: int, dtype: torch.dtype) -> torch.Tensor:
        query_geo = target_plucker[:, frame_idx].permute(0, 2, 3, 1).reshape(target_plucker.shape[0], -1, 6)
        return query_geo.to(dtype=dtype)

    def _window_memory(self, memory: torch.Tensor, condition_plucker: torch.Tensor, relative_pose_features: torch.Tensor, condition_available_mask: torch.Tensor, frame_idx: int, semantic: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if semantic:
            cond_geo = condition_plucker[:, :, lo:hi].mean(dim=(-1, -2))
            cond_geo = cond_geo[:, :, :, None, :].expand(batch, slots, hi - lo, tokens, 6)
        else:
            cond_geo = condition_plucker[:, :, lo:hi].permute(0, 1, 2, 4, 5, 3).reshape(batch, slots, hi - lo, tokens, 6)
        rel_geo = rel[:, :, :, None, :].expand(batch, slots, hi - lo, tokens, 6)
        denom = max(1, self.temporal_window)
        offset_geo = (torch.arange(lo, hi, device=memory.device, dtype=memory.dtype) - frame_idx) / float(denom)
        offset_geo = offset_geo.view(1, 1, hi - lo, 1, 1).expand(batch, slots, hi - lo, tokens, 1)
        geo = torch.cat([cond_geo.to(memory.dtype), rel_geo.to(memory.dtype), offset_geo], dim=-1)
        mem = mem.reshape(batch, slots * (hi - lo) * tokens, dim)
        geo = geo.reshape(batch, slots * (hi - lo) * tokens, 13)
        valid = condition_available_mask.to(device=memory.device, dtype=torch.bool)
        valid = valid.view(1, slots, 1, 1).expand(batch, slots, hi - lo, tokens)
        valid = valid.reshape(batch, slots * (hi - lo) * tokens)
        self.last_window_token_count = int(mem.shape[1])
        return (mem, valid, geo)

    def forward(self, target_tokens: torch.Tensor, dense_memory: torch.Tensor, semantic_memory: torch.Tensor, target_plucker: torch.Tensor, condition_plucker: torch.Tensor, relative_pose_features: torch.Tensor, condition_available_mask: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if target_tokens.ndim != 4:
            raise ValueError(f'expected target_tokens shape (B,T,N,D), got {tuple(target_tokens.shape)}')
        (batch, frames, tokens, dim) = target_tokens.shape
        if dense_memory.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected dense_memory shape {tuple(dense_memory.shape)}')
        if semantic_memory.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected semantic_memory shape {tuple(semantic_memory.shape)}')
        if target_plucker.shape[:3] != (batch, frames, 6):
            raise ValueError(f'unexpected target_plucker shape {tuple(target_plucker.shape)}')
        if condition_plucker.shape[:3] != (batch, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected condition_plucker shape {tuple(condition_plucker.shape)}')
        if target_plucker.shape[-2] * target_plucker.shape[-1] != tokens:
            raise ValueError(f'target plucker token count does not match target tokens: {target_plucker.shape[-2]}*{target_plucker.shape[-1]} vs {tokens}')
        if relative_pose_features.shape[:4] != (batch, frames, P61_CONDITION_SLOTS, frames):
            raise ValueError(f'unexpected relative_pose_features shape {tuple(relative_pose_features.shape)}')
        if condition_available_mask.shape != (P61_CONDITION_SLOTS,):
            raise ValueError(f'expected condition_available_mask shape ({P61_CONDITION_SLOTS},), got {tuple(condition_available_mask.shape)}')
        (g_sem, g_dense) = self.gate(sigma.to(target_tokens.device))
        delta = torch.zeros_like(target_tokens)
        target_plucker = target_plucker.to(device=target_tokens.device, dtype=target_tokens.dtype)
        condition_plucker = condition_plucker.to(device=target_tokens.device, dtype=target_tokens.dtype)
        for frame_idx in range(frames):
            query = target_tokens[:, frame_idx]
            query_geo = self._target_query_geo(target_plucker, frame_idx, query.dtype)
            (sem, sem_valid, sem_geo) = self._window_memory(semantic_memory, condition_plucker, relative_pose_features, condition_available_mask, frame_idx, semantic=True)
            (dense, dense_valid, dense_geo) = self._window_memory(dense_memory, condition_plucker, relative_pose_features, condition_available_mask, frame_idx, semantic=False)
            sem_out = self.semantic_attn(query, sem, query_geo, sem_geo, sem_valid)
            dense_out = self.dense_attn(query, dense, query_geo, dense_geo, dense_valid)
            delta[:, frame_idx] = g_sem.to(dtype=query.dtype) * sem_out + g_dense.to(dtype=query.dtype) * dense_out
        return delta.reshape(batch, frames * tokens, dim)

class DiTP63VACE(DiTP62VACE):

    def __init__(self, expert: nn.Module, dim: int | None=None, dim_attn: int=2048, heads: int=16, cross_view_blocks: Sequence[int]=(5, 11, 17, 23, 29, 35), sync_temporal_window: int=2, condition_encoder_layers: int=2, semantic_queries: int=64, enable_motion_embedding: bool=True, graph_gate_init_bias: float=-1.4, geo_head_dim: int=16, geo_projection_temperature: float=4.0) -> None:
        super().__init__(expert, dim=dim, dim_attn=dim_attn, heads=heads, cross_view_blocks=cross_view_blocks, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding, graph_gate_init_bias=graph_gate_init_bias, geo_head_dim=geo_head_dim)
        self.geo_projection_temperature = float(geo_projection_temperature)
        self.graph_memory = nn.ModuleList([P63GraphMemoryAttention(self.dim, heads=heads, temporal_window=self.sync_temporal_window, graph_gate_init_bias=self.graph_gate_init_bias, geo_head_dim=self.geo_head_dim, geo_projection_temperature=self.geo_projection_temperature) for _ in self.cross_view_blocks])
