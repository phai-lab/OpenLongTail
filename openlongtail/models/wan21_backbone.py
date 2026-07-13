from __future__ import annotations
import os
import sys
import types
import warnings
from pathlib import Path
import torch
from torch import nn
from openlongtail.training.distributed import NodeLocalContext, broadcast_module_state_dict_from_node_src
_WAN21_PACKAGE = '_openlongtail_external_wan21'

def default_wan21_i2v_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get('OPENLONGTAIL_WAN21_I2V_DIR', str(repo_root / 'checkpoints' / 'Wan2.1-I2V-14B-480P')))

def _wan21_code_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    longtail_root = repo_root.parent
    return Path(os.environ.get('OPENLONGTAIL_WAN21_CODE_ROOT', str(longtail_root / 'BEV_WAN' / 'Wan2.1')))

def _ensure_wan21_package() -> str:
    wan_root = _wan21_code_root()
    if not (wan_root / 'wan' / 'modules' / 'model.py').exists():
        raise FileNotFoundError(f'expected Wan2.1 code at {wan_root}')
    pkg = sys.modules.get(_WAN21_PACKAGE)
    if pkg is None or not hasattr(pkg, '__path__'):
        pkg = types.ModuleType(_WAN21_PACKAGE)
        pkg.__path__ = [str(wan_root / 'wan')]
        sys.modules[_WAN21_PACKAGE] = pkg
    modules_name = f'{_WAN21_PACKAGE}.modules'
    modules_pkg = sys.modules.get(modules_name)
    if modules_pkg is None or not hasattr(modules_pkg, '__path__'):
        modules_pkg = types.ModuleType(modules_name)
        modules_pkg.__path__ = [str(wan_root / 'wan' / 'modules')]
        sys.modules[modules_name] = modules_pkg
    return _WAN21_PACKAGE

def _sdpa_flash_attention_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_lens: torch.Tensor | None=None, k_lens: torch.Tensor | None=None, dropout_p: float=0.0, softmax_scale: float | None=None, q_scale: float | None=None, causal: bool=False, window_size: tuple[int, int]=(-1, -1), deterministic: bool=False, dtype: torch.dtype=torch.bfloat16, version: int | None=None) -> torch.Tensor:
    del q_lens, deterministic, version
    if k_lens is not None:
        warnings.warn("Padding lengths are ignored by the SDPA fallback, matching Wan's own fallback limitation.")
    if window_size != (-1, -1):
        warnings.warn('Windowed attention is ignored by the SDPA fallback.')
    if q_scale is not None:
        q = q * q_scale
    target_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else dtype
    q = q.transpose(1, 2).to(target_dtype)
    k = k.transpose(1, 2).to(target_dtype)
    v = v.transpose(1, 2).to(target_dtype)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()

@torch.amp.autocast('cuda', enabled=False)
def _memory_efficient_rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    (num_heads, complex_channels) = (x.size(2), x.size(3) // 2)
    split_freqs = freqs.split([complex_channels - 2 * (complex_channels // 3), complex_channels // 3, complex_channels // 3], dim=1)
    out = torch.empty_like(x)
    for (idx, (frames, height, width)) in enumerate(grid_sizes.tolist()):
        seq_len = frames * height * width
        x_i = torch.view_as_complex(x[idx, :seq_len].to(torch.float64).reshape(seq_len, num_heads, -1, 2))
        freqs_i = torch.cat([split_freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1), split_freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1), split_freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1)], dim=-1).reshape(seq_len, 1, -1)
        rotated = torch.view_as_real(x_i * freqs_i).flatten(2)
        out[idx, :seq_len] = rotated.to(dtype=x.dtype)
        if seq_len < x.shape[1]:
            out[idx, seq_len:] = x[idx, seq_len:]
    return out

def patch_wan21_flash_attention_fallback() -> bool:
    import importlib
    package = _ensure_wan21_package()
    attention_module = importlib.import_module(f'{package}.modules.attention')
    model_module = importlib.import_module(f'{package}.modules.model')
    modules_package = importlib.import_module(f'{package}.modules')
    has_flash = bool(getattr(attention_module, 'FLASH_ATTN_2_AVAILABLE', False) or getattr(attention_module, 'FLASH_ATTN_3_AVAILABLE', False))
    if has_flash:
        patched_attention = False
    else:
        attention_module.flash_attention = _sdpa_flash_attention_fallback
        attention_module.attention = _sdpa_flash_attention_fallback
        modules_package.flash_attention = _sdpa_flash_attention_fallback
        model_module.flash_attention = _sdpa_flash_attention_fallback
        patched_attention = True
    model_module.rope_apply = _memory_efficient_rope_apply
    return patched_attention

def _wan21_model_class():
    import importlib
    patch_wan21_flash_attention_fallback()
    package = _ensure_wan21_package()
    return importlib.import_module(f'{package}.modules.model').WanModel

def _reset_wan21_rope_freqs(expert: nn.Module, device: torch.device | str) -> None:
    import importlib
    package = _ensure_wan21_package()
    rope_params = importlib.import_module(f'{package}.modules.model').rope_params
    dim = int(getattr(expert, 'dim'))
    num_heads = int(getattr(expert, 'num_heads'))
    if dim % num_heads != 0:
        raise ValueError(f'expected dim divisible by num_heads, got dim={dim}, num_heads={num_heads}')
    head_dim = dim // num_heads
    expert.freqs = torch.cat([rope_params(1024, head_dim - 4 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6))], dim=1).to(device=device)

def _assert_wan21_i2v(expert: nn.Module) -> None:
    if getattr(expert, 'model_type', None) != 'i2v':
        raise ValueError(f"expected Wan2.1-I2V model_type='i2v', got {getattr(expert, 'model_type', None)!r}")
    if int(getattr(expert, 'in_dim', -1)) != 36:
        raise ValueError(f"expected Wan2.1-I2V in_dim=36, got {getattr(expert, 'in_dim', None)}")
    if int(getattr(expert, 'out_dim', -1)) != 16:
        raise ValueError(f"expected Wan2.1-I2V out_dim=16, got {getattr(expert, 'out_dim', None)}")

def load_wan21_i2v_expert(ckpt_dir: Path | None=None, dtype: torch.dtype=torch.bfloat16, device: torch.device | str='cpu') -> nn.Module:
    WanModel = _wan21_model_class()
    ckpt_dir = Path(ckpt_dir or default_wan21_i2v_dir())
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'expected Wan2.1-I2V checkpoint dir to exist, got {ckpt_dir}')
    expert = WanModel.from_pretrained(str(ckpt_dir))
    expert.to(device=device, dtype=dtype)
    _reset_wan21_rope_freqs(expert, device)
    expert.eval()
    for param in expert.parameters():
        param.requires_grad = False
    _assert_wan21_i2v(expert)
    return expert

def _empty_wan21_i2v_expert_from_config(ckpt_dir: Path | None=None, dtype: torch.dtype=torch.bfloat16, device: torch.device | str='cpu') -> nn.Module:
    WanModel = _wan21_model_class()
    ckpt_dir = Path(ckpt_dir or default_wan21_i2v_dir())
    config = WanModel.load_config(str(ckpt_dir))
    with torch.device('meta'):
        expert = WanModel.from_config(config)
    expert.to_empty(device=device)
    expert.to(dtype=dtype)
    _reset_wan21_rope_freqs(expert, device)
    expert.eval()
    for param in expert.parameters():
        param.requires_grad = False
    _assert_wan21_i2v(expert)
    return expert

def load_wan21_i2v_expert_node_broadcast(ckpt_dir: Path | None=None, dtype: torch.dtype=torch.bfloat16, device: torch.device | str='cpu', node_context: NodeLocalContext | None=None) -> nn.Module:
    if node_context is None or node_context.world_size == 1:
        return load_wan21_i2v_expert(ckpt_dir, dtype=dtype, device=device)
    if node_context.is_local_src:
        expert = load_wan21_i2v_expert(ckpt_dir, dtype=dtype, device=device)
    else:
        expert = _empty_wan21_i2v_expert_from_config(ckpt_dir, dtype=dtype, device=device)
    broadcast_module_state_dict_from_node_src(expert, node_context)
    _reset_wan21_rope_freqs(expert, device)
    return expert
