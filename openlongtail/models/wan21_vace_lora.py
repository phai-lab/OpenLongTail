from __future__ import annotations
from torch import nn
from openlongtail.models.wan22_lora import LORA_TARGETS, LoRAInjectionResult, inject_lora_into_wan22_expert

def inject_lora_into_wan21_vace_expert(expert: nn.Module, rank: int=32, alpha: int=16, targets: tuple[str, ...]=LORA_TARGETS) -> LoRAInjectionResult:
    return inject_lora_into_wan22_expert(expert, rank=rank, alpha=alpha, targets=targets, assert_wan22_count=False)
