from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class RayCrossViewAttentionP3(nn.Module):

    def __init__(self, dim_in: int=5120, dim_attn: int=2048, heads: int=16) -> None:
        super().__init__()
        if dim_attn % heads != 0:
            raise ValueError(f'expected dim_attn divisible by heads, got dim_attn={dim_attn}, heads={heads}')
        self.dim_in = dim_in
        self.dim_attn = dim_attn
        self.heads = heads
        self.head_dim = dim_attn // heads
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

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        (batch, tokens, _) = x.shape
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def _directed_message(self, hidden: torch.Tensor, rays: torch.Tensor, src_stream: int, dst_stream: int) -> torch.Tensor:
        q = self._split_heads(self.to_q(hidden[:, dst_stream]))
        k = self._split_heads(self.to_k(hidden[:, src_stream]))
        v = self._split_heads(self.to_v(hidden[:, src_stream]))
        ray_distance = self._pairwise_ray_distance(rays[:, dst_stream], rays[:, src_stream])
        attn_mask = -self.alpha.view(1, 1, 1, 1) * ray_distance[:, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(hidden.shape[0], q.shape[2], self.dim_attn)
        return self.to_out(out)

    def forward(self, hidden: torch.Tensor, plucker: torch.Tensor, num_streams: int | None=None) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.dim_in:
            raise ValueError(f'expected hidden shape (B*S, T, {self.dim_in}), got {tuple(hidden.shape)}')
        if num_streams is None:
            num_streams = plucker.shape[1]
        if num_streams < 2:
            return torch.zeros_like(hidden)
        if hidden.shape[0] % num_streams != 0:
            raise ValueError(f'expected hidden batch {hidden.shape[0]} divisible by num_streams={num_streams}')
        if plucker.ndim != 6 or plucker.shape[1] != num_streams or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, {num_streams}, T, 6, H, W), got {tuple(plucker.shape)}')
        (batch_streams, tokens, _) = hidden.shape
        batch = batch_streams // num_streams
        rays = self._flatten_plucker(plucker).to(device=hidden.device, dtype=hidden.dtype)
        if rays.shape[0] != batch or rays.shape[2] != tokens:
            raise ValueError(f'expected plucker flattened shape ({batch}, {num_streams}, {tokens}, 6), got {tuple(rays.shape)}')
        h = self.norm(self.proj_in(hidden)).reshape(batch, num_streams, tokens, self.dim_attn)
        out = torch.zeros_like(h)
        degree = torch.zeros(num_streams, device=hidden.device, dtype=hidden.dtype)
        for src_stream in range(num_streams):
            for dst_stream in range(num_streams):
                if src_stream == dst_stream:
                    continue
                out[:, dst_stream] += self._directed_message(h, rays, src_stream=src_stream, dst_stream=dst_stream)
                degree[dst_stream] += 1
        out = out / degree.clamp_min(1.0).view(1, num_streams, 1, 1)
        out = self.proj_out(out.reshape(batch_streams, tokens, self.dim_attn))
        return torch.tanh(self.gate).view(1, 1, 1) * out
