from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from openlongtail.training.checkpoint import _lora_state_dict, _module_state_dict
from openlongtail.training.forward_ray import RayTrainingComponents

def save_checkpoint_p61(components: RayTrainingComponents, optimizer: torch.optim.Optimizer | None, step: int, output_dir: Path) -> Path:
    ckpt_dir = Path(output_dir) / f'step_{step:08d}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'step': int(step), 'stage': 'unified', 'backbone': 'wan2.1-vace-14b', 'checkpoint_format': 'p61_vace_graph_autoregressive_single_target'}
    (ckpt_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(_lora_state_dict(components.low_dit), ckpt_dir / 'p61_lora.pt')
    torch.save(_module_state_dict(components.shared_modules), ckpt_dir / 'shared_modules.pt')
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_dir / 'optimizer.pt')
    return ckpt_dir

def load_checkpoint_p61(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f'expected checkpoint metadata at {metadata_path}')
    payload: dict[str, Any] = {'metadata': json.loads(metadata_path.read_text())}
    for name in ('p61_lora', 'shared_modules', 'optimizer'):
        path = checkpoint_dir / f'{name}.pt'
        if path.exists():
            payload[name] = torch.load(path, map_location='cpu', weights_only=True)
    return payload
