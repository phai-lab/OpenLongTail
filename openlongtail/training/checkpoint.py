from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from openlongtail.training.forward_ray import RayTrainingComponents

def _unwrap_ddp(module: torch.nn.Module | None) -> torch.nn.Module | None:
    if module is not None and hasattr(module, 'module'):
        return module.module
    return module

def _lora_state_dict(module: torch.nn.Module | None) -> dict[str, torch.Tensor]:
    module = _unwrap_ddp(module)
    if module is None:
        return {}
    return {key: value.detach().cpu() for (key, value) in module.state_dict().items() if 'lora_' in key}

def _module_state_dict(module: torch.nn.Module | None) -> dict[str, torch.Tensor]:
    module = _unwrap_ddp(module)
    if module is None:
        return {}
    return {key: value.detach().cpu() for (key, value) in module.state_dict().items()}

def save_checkpoint(components: RayTrainingComponents, optimizer: torch.optim.Optimizer | None, step: int, output_dir: Path, stage: str) -> Path:
    if stage not in ('A.0', 'A.1', 'B'):
        raise ValueError(f'expected stage one of A.0, A.1, B, got {stage!r}')
    ckpt_dir = Path(output_dir) / f'step_{step:08d}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'step': step, 'stage': stage}
    (ckpt_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True))
    if stage in ('A.0', 'A.1'):
        torch.save(_lora_state_dict(components.low_dit), ckpt_dir / 'low_lora.pt')
        torch.save(_module_state_dict(components.shared_modules), ckpt_dir / 'shared_modules.pt')
    else:
        torch.save(_lora_state_dict(components.high_dit), ckpt_dir / 'high_lora.pt')
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_dir / 'optimizer.pt')
    return ckpt_dir

def load_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f'expected checkpoint metadata at {metadata_path}')
    payload: dict[str, Any] = {'metadata': json.loads(metadata_path.read_text())}
    for name in ('low_lora', 'high_lora', 'shared_modules', 'optimizer'):
        path = checkpoint_dir / f'{name}.pt'
        if path.exists():
            payload[name] = torch.load(path, map_location='cpu', weights_only=True)
    return payload
