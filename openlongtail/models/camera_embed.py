from __future__ import annotations
import torch
from torch import nn

class PluckerMLP(nn.Module):

    def __init__(self, in_dim: int=6, hidden: int=256, out_dim: int=5120) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = nn.SiLU()
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        if plucker.ndim != 6 or plucker.shape[3] != 6:
            raise ValueError(f'expected plucker shape (B, 6, T, 6, H, W), got {tuple(plucker.shape)}')
        x = plucker.permute(0, 1, 2, 4, 5, 3).contiguous()
        x = self.fc2(self.act(self.fc1(x)))
        return x.permute(0, 1, 2, 5, 3, 4).contiguous()

class CamIDEmbed(nn.Module):

    def __init__(self, num_views: int=6, dim: int=5120) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_views, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, view_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(view_ids)
