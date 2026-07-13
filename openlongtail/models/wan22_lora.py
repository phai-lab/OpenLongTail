from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
LORA_TARGETS: tuple[str, ...] = ('self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o')

@dataclass(frozen=True)
class LoRAInjectionResult:
    replaced: int
    trainable_params: int
    frozen_params: int

class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, rank: int=32, alpha: int=16) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f'expected rank > 0, got {rank}')
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling

def expected_lora_trainable_params(num_layers: int=40, dim: int=5120, rank: int=32, targets_per_layer: int=4) -> int:
    return num_layers * targets_per_layer * (rank * dim + dim * rank)

def _parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return (parent, parts[-1])

def _matches_target(name: str, targets: tuple[str, ...]) -> bool:
    return any((name.endswith(target) for target in targets))

def inject_lora_into_wan22_expert(expert: nn.Module, rank: int=32, alpha: int=16, targets: tuple[str, ...]=LORA_TARGETS, assert_wan22_count: bool=True) -> LoRAInjectionResult:
    for param in expert.parameters():
        param.requires_grad = False
    to_replace = [(name, module) for (name, module) in expert.named_modules() if _matches_target(name, targets) and isinstance(module, nn.Linear)]
    if not to_replace:
        raise ValueError(f'found no nn.Linear modules matching LoRA targets {targets}')
    for (name, module) in to_replace:
        (parent, child_name) = _parent_module(expert, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
    trainable = sum((param.numel() for param in expert.parameters() if param.requires_grad))
    frozen = sum((param.numel() for param in expert.parameters() if not param.requires_grad))
    expected = sum((module.in_features * rank + rank * module.out_features for (_, module) in to_replace))
    if trainable != expected:
        raise AssertionError(f'expected LoRA trainable params {expected}, got {trainable}')
    wan22_expected = expected_lora_trainable_params(rank=rank)
    if assert_wan22_count and len(to_replace) == 160 and (trainable != wan22_expected):
        raise AssertionError(f'expected Wan2.2 LoRA trainable params {wan22_expected}, got {trainable}')
    return LoRAInjectionResult(replaced=len(to_replace), trainable_params=trainable, frozen_params=frozen)
