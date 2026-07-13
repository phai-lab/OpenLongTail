from __future__ import annotations
import sys
import types
from pathlib import Path
from typing import Literal
import torch
from openlongtail.configs.default import CheckpointConfig
_BLANK_LATENT_CACHE: dict[tuple[int, int, int, str], torch.Tensor] = {}

def normalize_rgb_for_wan_vae(rgb: torch.Tensor, source_range: Literal['raw_0_255', 'normalized_minus_one_one', 'auto']='auto') -> torch.Tensor:
    if not torch.is_floating_point(rgb):
        rgb = rgb.float()
    if source_range == 'raw_0_255':
        return rgb / 127.5 - 1.0
    if source_range == 'normalized_minus_one_one':
        return rgb
    if source_range != 'auto':
        raise ValueError(f'expected source_range raw_0_255, normalized_minus_one_one, or auto, got {source_range!r}')
    min_value = float(rgb.detach().amin().cpu())
    max_value = float(rgb.detach().amax().cpu())
    if min_value < -1.05:
        raise ValueError(f'expected RGB tensor min >= -1.05 for Wan VAE auto-normalization, got {min_value}')
    if max_value > 2.0:
        return rgb / 127.5 - 1.0
    if max_value > 1.05:
        raise ValueError(f'expected RGB tensor max <= 1.05 or raw 0..255 for Wan VAE auto-normalization, got {max_value}')
    return rgb

def _ensure_wan22_on_path() -> None:
    wan_root = Path(__file__).resolve().parents[2] / 'Wan2.2'
    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    wan_pkg = sys.modules.get('wan')
    if wan_pkg is None or not hasattr(wan_pkg, '__path__'):
        wan_pkg = types.ModuleType('wan')
        wan_pkg.__path__ = [str(wan_root / 'wan')]
        sys.modules['wan'] = wan_pkg
    modules_pkg = sys.modules.get('wan.modules')
    if modules_pkg is None or not hasattr(modules_pkg, '__path__'):
        modules_pkg = types.ModuleType('wan.modules')
        modules_pkg.__path__ = [str(wan_root / 'wan' / 'modules')]
        sys.modules['wan.modules'] = modules_pkg

def load_wan21_vae(vae_path: Path | None=None, dtype: torch.dtype=torch.bfloat16, device: torch.device | str='cuda'):
    _ensure_wan22_on_path()
    from wan.modules.vae2_1 import Wan2_1_VAE
    ckpt = CheckpointConfig()
    vae = Wan2_1_VAE(vae_pth=str(vae_path or ckpt.vae_path), dtype=dtype, device=str(device))
    vae.model.eval().requires_grad_(False)
    return vae

@torch.no_grad()
def precompute_blank_image_cond_latent(vae, output_size: tuple[int, int]=(480, 832), clip_length: int=81, device: torch.device | str='cuda', dtype: torch.dtype=torch.bfloat16) -> torch.Tensor:
    key = (clip_length, output_size[0], output_size[1], str(dtype))
    cached = _BLANK_LATENT_CACHE.get(key)
    if cached is not None:
        return cached.to(device=device, dtype=dtype)
    blank = torch.zeros(3, clip_length, output_size[0], output_size[1], device=device, dtype=torch.float32)
    encoded = vae.encode([blank])[0].detach().to('cpu', dtype=dtype)
    _BLANK_LATENT_CACHE[key] = encoded
    return encoded.to(device=device, dtype=dtype)
