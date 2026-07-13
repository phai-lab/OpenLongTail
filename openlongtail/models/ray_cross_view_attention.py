from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
GRAPH_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 3), (3, 5), (5, 4), (4, 2))

class RayCrossViewAttention(nn.Module):

    def __init__(self, dim_in: int=5120, dim_attn: int=2048, heads: int=16, view_edges: tuple[tuple[int, int], ...]=GRAPH_EDGES) -> None:
        super().__init__()
        if dim_attn % heads != 0:
            raise ValueError(f'expected dim_attn divisible by heads, got dim_attn={dim_attn}, heads={heads}')
        self.dim_in = dim_in
        self.dim_attn = dim_attn
        self.heads = heads
        self.head_dim = dim_attn // heads
        self.view_edges = view_edges
        self.proj_in = nn.Linear(dim_in, dim_attn)
        self.norm = nn.LayerNorm(dim_attn)
        self.to_q = nn.Linear(dim_attn, dim_attn, bias=False)
        self.to_k = nn.Linear(dim_attn, dim_attn, bias=False)
        self.to_v = nn.Linear(dim_attn, dim_attn, bias=False)
        self.to_out = nn.Linear(dim_attn, dim_attn)
        self.proj_out = nn.Linear(dim_attn, dim_in)
        for module in (self.proj_in, self.to_q, self.to_k, self.to_v, self.to_out, self.proj_out):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.gate = nn.Parameter(torch.zeros(1))

    def _active_directed_edges(self, active_views: list[int] | tuple[int, ...]) -> list[tuple[int, int]]:
        active_set = set((int(view_id) for view_id in active_views))
        pos_in_active = {int(view_id): idx for (idx, view_id) in enumerate(active_views)}
        edges: list[tuple[int, int]] = []
        for (view_a, view_b) in self.view_edges:
            if view_a in active_set and view_b in active_set:
                edges.append((pos_in_active[view_a], pos_in_active[view_b]))
        return edges

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        (batch, tokens, _) = x.shape
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _flatten_plucker(plucker: torch.Tensor) -> torch.Tensor:
        return plucker.permute(0, 1, 2, 4, 5, 3).reshape(plucker.shape[0], plucker.shape[1], -1, 6)

    @staticmethod
    def _pairwise_ray_distance(q_rays: torch.Tensor, k_rays: torch.Tensor) -> torch.Tensor:
        q_dir = q_rays[..., :3]
        q_moment = q_rays[..., 3:]
        k_dir = k_rays[..., :3]
        k_moment = k_rays[..., 3:]
        numerator = torch.einsum('bqd,bkd->bqk', q_dir, k_moment).add_(torch.einsum('bkd,bqd->bqk', k_dir, q_moment)).abs()
        cross = torch.cross(q_dir[:, :, None, :], k_dir[:, None, :, :], dim=-1)
        denom = torch.linalg.vector_norm(cross, dim=-1).clamp_min(1e-06)
        return numerator / denom

    def _directed_message(self, hidden: torch.Tensor, rays: torch.Tensor, src_view: int, dst_view: int) -> torch.Tensor:
        q = self._split_heads(self.to_q(hidden[:, dst_view]))
        k = self._split_heads(self.to_k(hidden[:, src_view]))
        v = self._split_heads(self.to_v(hidden[:, src_view]))
        ray_distance = self._pairwise_ray_distance(rays[:, dst_view], rays[:, src_view])
        attn_mask = -self.alpha.view(1, 1, 1, 1) * ray_distance[:, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(hidden.shape[0], q.shape[2], self.dim_attn)
        return self.to_out(out)

    def forward(self, hidden: torch.Tensor, plucker: torch.Tensor, num_views: int | None=None, active_views: list[int] | None=None) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.dim_in:
            raise ValueError(f'expected hidden shape (B*6, S, {self.dim_in}), got {tuple(hidden.shape)}')
        if active_views is not None:
            num_views = len(active_views)
        elif num_views is None:
            num_views = 6
        if hidden.shape[0] % num_views != 0:
            raise ValueError(f'expected hidden batch {hidden.shape[0]} divisible by num_views={num_views}')
        if plucker.ndim != 6 or plucker.shape[1] != num_views or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, {num_views}, T, 6, H, W), got {tuple(plucker.shape)}')
        (batch_views, tokens, _) = hidden.shape
        batch = batch_views // num_views
        rays = self._flatten_plucker(plucker).to(device=hidden.device, dtype=hidden.dtype)
        if rays.shape[0] != batch or rays.shape[2] != tokens:
            raise ValueError(f'expected plucker flattened shape ({batch}, {num_views}, {tokens}, 6), got {tuple(rays.shape)}')
        h = self.norm(self.proj_in(hidden)).reshape(batch, num_views, tokens, self.dim_attn)
        out = torch.zeros_like(h)
        degree = torch.zeros(num_views, device=hidden.device, dtype=hidden.dtype)
        edges = self.view_edges if active_views is None else self._active_directed_edges(active_views)
        if not edges:
            return torch.zeros_like(hidden)
        for (view_a, view_b) in edges:
            out[:, view_a] += self._directed_message(h, rays, src_view=view_b, dst_view=view_a)
            out[:, view_b] += self._directed_message(h, rays, src_view=view_a, dst_view=view_b)
            degree[view_a] += 1
            degree[view_b] += 1
        out = out / degree.clamp_min(1.0).view(1, num_views, 1, 1)
        out = self.proj_out(out.reshape(batch_views, tokens, self.dim_attn))
        return torch.tanh(self.gate).view(1, 1, 1) * out

    def forward_local(self, hidden_by_view: torch.Tensor, plucker: torch.Tensor, dst_view: int, active_views: list[int] | None=None) -> torch.Tensor:
        if hidden_by_view.ndim != 4 or hidden_by_view.shape[-1] != self.dim_in:
            raise ValueError(f'expected hidden_by_view shape (B, V, S, {self.dim_in}), got {tuple(hidden_by_view.shape)}')
        num_views = hidden_by_view.shape[1]
        if active_views is None:
            active_views = list(range(num_views))
        pos_in_active = {int(view_id): idx for (idx, view_id) in enumerate(active_views)}
        if dst_view not in pos_in_active:
            raise ValueError(f'expected dst_view {dst_view} to be in active_views={active_views}')
        dst_pos = pos_in_active[dst_view]
        if plucker.ndim != 6 or plucker.shape[1] != hidden_by_view.shape[1] or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, {num_views}, T, 6, H, W), got {tuple(plucker.shape)}')
        (batch, num_views, tokens, _) = hidden_by_view.shape
        rays = self._flatten_plucker(plucker).to(device=hidden_by_view.device, dtype=hidden_by_view.dtype)
        if rays.shape[0] != batch or rays.shape[2] != tokens:
            raise ValueError(f'expected plucker flattened shape ({batch}, {num_views}, {tokens}, 6), got {tuple(rays.shape)}')
        h = self.norm(self.proj_in(hidden_by_view.reshape(batch * num_views, tokens, self.dim_in)))
        h = h.reshape(batch, num_views, tokens, self.dim_attn)
        out = torch.zeros(batch, tokens, self.dim_attn, device=hidden_by_view.device, dtype=hidden_by_view.dtype)
        degree = 0
        for (view_a, view_b) in self._active_directed_edges(active_views):
            if view_a == dst_pos:
                out = out + self._directed_message(h, rays, src_view=view_b, dst_view=view_a)
                degree += 1
            if view_b == dst_pos:
                out = out + self._directed_message(h, rays, src_view=view_a, dst_view=view_b)
                degree += 1
        if degree == 0:
            return hidden_by_view.new_zeros(batch, tokens, self.dim_in)
        if degree > 0:
            out = out / float(degree)
        out = self.proj_out(out)
        return torch.tanh(self.gate).view(1, 1, 1) * out
