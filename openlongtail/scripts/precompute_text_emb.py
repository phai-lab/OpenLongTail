from __future__ import annotations
import argparse
from pathlib import Path
import torch
from openlongtail.configs.default import CheckpointConfig
from openlongtail.data.rig_parquet import discover_uuid_dirs
from openlongtail.models.wan_vae import _ensure_wan22_on_path

def _load_t5(device: str):
    _ensure_wan22_on_path()
    from wan.modules.t5 import T5EncoderModel
    ckpt = CheckpointConfig()
    return T5EncoderModel(text_len=256, dtype=torch.bfloat16, device=torch.device(device), checkpoint_path=str(ckpt.umt5_path), tokenizer_path=str(ckpt.umt5_tokenizer))

def _encode_prompt(text_encoder, prompt: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = text_encoder([prompt], torch.device(device))[0].detach().to('cpu', dtype=torch.bfloat16)
    text_emb = torch.zeros(256, 4096, dtype=torch.bfloat16)
    text_mask = torch.zeros(256, dtype=torch.bool)
    length = min(encoded.shape[0], 256)
    text_emb[:length] = encoded[:length]
    text_mask[:length] = True
    return (text_emb, text_mask)

def precompute_text_embeddings(data_root: Path, output_root: Path, device: str) -> None:
    text_encoder = _load_t5(device)
    per_uuid_root = output_root / 'per_uuid'
    per_uuid_root.mkdir(parents=True, exist_ok=True)
    (null_emb, null_mask) = _encode_prompt(text_encoder, '', device)
    torch.save({'text_emb': null_emb, 'text_mask': null_mask}, output_root / 'null.pt')
    for uuid_dir in discover_uuid_dirs(data_root):
        caption_path = uuid_dir / 'vlm_caption.txt'
        if not caption_path.exists():
            raise FileNotFoundError(f'expected VLM caption at {caption_path}')
        prompt = caption_path.read_text().strip()
        (text_emb, text_mask) = _encode_prompt(text_encoder, prompt, device)
        torch.save({'text_emb': text_emb, 'text_mask': text_mask}, per_uuid_root / f'{uuid_dir.name}.pt')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    precompute_text_embeddings(args.data_root, args.output_root, args.device)
if __name__ == '__main__':
    main()
