from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class FlowMatchScheduler:
    shift: float = 5.0

    def sigmas_for(self, t: torch.Tensor) -> torch.Tensor:
        return self.shift * t / (1.0 + (self.shift - 1.0) * t)

    def t_for_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma / (self.shift - (self.shift - 1.0) * sigma)

    def sigmas_descending(self, num_steps: int, device: torch.device | str | None=None, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f'expected num_steps > 0, got {num_steps}')
        t = torch.linspace(1.0, 0.0, num_steps, device=device, dtype=dtype)
        return self.sigmas_for(t)
