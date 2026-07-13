from __future__ import annotations
import importlib
import os
from pathlib import Path
import torch
from torch import nn
from openlongtail.models.wan21_backbone import _ensure_wan21_package, patch_wan21_flash_attention_fallback
from openlongtail.training.distributed import NodeLocalContext, broadcast_module_state_dict_from_node_src

def default_wan21_vace_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get('OPENLONGTAIL_WAN21_VACE_DIR', str(repo_root / 'checkpoints' / 'Wan2.1-VACE-14B')))

def default_wan21_vace_1p3b_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get('OPENLONGTAIL_WAN21_VACE_1P3B_DIR', str(repo_root / 'checkpoints' / 'Wan2.1-VACE-1.3B')))

def _wan21_vace_model_class():
    patch_wan21_flash_attention_fallback()
    package = _ensure_wan21_package()
    return importlib.import_module(f'{package}.modules.vace_model').VaceWanModel

def _reset_wan21_vace_rope_freqs(expert: nn.Module, device: torch.device | str) -> None:
    package = _ensure_wan21_package()
    rope_params = importlib.import_module(f'{package}.modules.model').rope_params
    dim = int(getattr(expert, 'dim'))
    num_heads = int(getattr(expert, 'num_heads'))
    if dim % num_heads != 0:
        raise ValueError(f'expected dim divisible by num_heads, got dim={dim}, num_heads={num_heads}')
    head_dim = dim // num_heads
    expert.freqs = torch.cat([rope_params(1024, head_dim - 4 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6))], dim=1).to(device=device)

def _assert_wan21_vace(expert: nn.Module) -> None:
    if getattr(expert, 'model_type', None) != 'vace':
        raise ValueError(f"expected Wan2.1-VACE model_type='vace', got {getattr(expert, 'model_type', None)!r}")
    if int(getattr(expert, 'in_dim', -1)) != 16:
        raise ValueError(f"expected Wan2.1-VACE in_dim=16, got {getattr(expert, 'in_dim', None)}")
    if int(getattr(expert, 'out_dim', -1)) != 16:
        raise ValueError(f"expected Wan2.1-VACE out_dim=16, got {getattr(expert, 'out_dim', None)}")
    if int(getattr(expert, 'vace_in_dim', -1)) != 96:
        raise ValueError(f"expected Wan2.1-VACE vace_in_dim=96, got {getattr(expert, 'vace_in_dim', None)}")

def load_wan21_vace_expert(ckpt_dir: Path | None=None, dtype: torch.dtype=torch.bfloat16, device: torch.device | str='cpu', freeze_backbone: bool=True) -> nn.Module:
    VaceWanModel = _wan21_vace_model_class()
    ckpt_dir = Path(ckpt_dir or default_wan21_vace_dir())
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'expected Wan2.1-VACE checkpoint dir to exist, got {ckpt_dir}')
    expert = VaceWanModel.from_pretrained(str(ckpt_dir))
    expert.to(device=device, dtype=dtype)
    _reset_wan21_vace_rope_freqs(expert, device)
    if freeze_backbone:
        expert.eval()
        for param in expert.parameters():
            param.requires_grad = False
    else:
        expert.train()
        for param in expert.parameters():
            param.requires_grad = True
    _assert_wan21_vace(expert)
    return expert

def _empty_wan21_vace_expert_from_config(ckpt_dir: Path | None=None, dtype: torch.dtype=torch.bfloat16, device: torch.device | str='cpu', freeze_backbone: bool=True) -> nn.Module:
    VaceWanModel = _wan21_vace_model_class()
    ckpt_dir = Path(ckpt_dir or default_wan21_vace_dir())
    config = VaceWanModel.load_config(str(ckpt_dir))
    with torch.device('meta'):
        expert = VaceWanModel.from_config(config)
    expert.to_empty(device=device)
    expert.to(dtype=dtype)
    _reset_wan21_vace_rope_freqs(expert, device)
    if freeze_backbone:
        expert.eval()
        for param in expert.parameters():
            param.requires_grad = False
    else:
        expert.train()
        for param in expert.parameters():
            param.requires_grad = True
    _assert_wan21_vace(expert)
    return expert

def load_wan21_vace_expert_node_broadcast(ckpt_dir: Path | None=None, dtype: torch.dtype=torch.bfloat16, device: torch.device | str='cpu', node_context: NodeLocalContext | None=None, freeze_backbone: bool=True) -> nn.Module:
    if node_context is None or node_context.world_size == 1:
        return load_wan21_vace_expert(ckpt_dir, dtype=dtype, device=device, freeze_backbone=freeze_backbone)
    if node_context.is_local_src:
        expert = load_wan21_vace_expert(ckpt_dir, dtype=dtype, device=device, freeze_backbone=freeze_backbone)
    else:
        expert = _empty_wan21_vace_expert_from_config(ckpt_dir, dtype=dtype, device=device, freeze_backbone=freeze_backbone)
    broadcast_module_state_dict_from_node_src(expert, node_context)
    _reset_wan21_vace_rope_freqs(expert, device)
    return expert
