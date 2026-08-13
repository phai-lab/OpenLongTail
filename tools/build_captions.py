#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
from openlongtail.scripts.qwen_rear_style_judge import DEFAULT_MODEL, resolve_model_class, sample_frames
_CAPTION_PROMPT = 'You are captioning a driving-scene dashcam video for a generative model. Look at these frames (all the same clip) and write ONE concise sentence describing the overall scene: road/location type, time of day, weather, and the most salient surroundings (buildings, vehicles, vegetation). Be factual and specific; no camera talk, no lists, no quotes. Respond with ONLY the sentence.'

class QwenCaptioner:

    def __init__(self, model_dir: str, device: str='cuda'):
        from transformers import AutoProcessor
        (cls, cls_name, note) = resolve_model_class(model_dir)
        if cls is None:
            raise ImportError(f'cannot import a Qwen VL model class for {model_dir}: {note}')
        self.cls_name = cls_name
        self.device = device
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.model = cls.from_pretrained(model_dir, torch_dtype=dtype, device_map=device if device == 'cuda' else None, trust_remote_code=True)
        self.model.eval()

    def caption(self, frames) -> str:
        content = [{'type': 'image', 'image': img} for img in frames]
        content.append({'type': 'text', 'text': _CAPTION_PROMPT})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=frames, padding=True, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for (k, v) in inputs.items()}
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=80, do_sample=False, temperature=0.0)
        trimmed = gen[:, inputs['input_ids'].shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return _clean(out)

def _clean(text: str) -> str:
    s = ' '.join(text.replace('```', ' ').split()).strip().strip('"').strip("'").strip()
    for end in ('. ', '? ', '! '):
        i = s.find(end)
        if i != -1:
            s = s[:i + 1]
            break
    return s

def collect_clip_dirs(test_data_root: Path, max_clips: int, shard_index: int, num_shards: int) -> list[Path]:
    manifest = test_data_root / 'manifest_clips.jsonl'
    clips: list[Path] = []
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            clips.append(test_data_root / f"chunk_{int(item['chunk']):03d}" / str(item['uuid']) / str(item['clip_id']))
    else:
        clips = sorted(test_data_root.glob('chunk_*/*/clip_*'))
    required = ('front.mp4', 'front_depth.pt', 'pose.pt', 'meta.pt')
    existing = [c for c in clips if all(((c / n).exists() for n in required))]
    sharded = existing[shard_index::num_shards]
    if max_clips > 0:
        sharded = sharded[:max_clips]
    return sharded

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Qwen-VL dashcam front-video caption builder')
    ap.add_argument('--test-data-root', type=Path)
    ap.add_argument('--out-root', type=Path, help='caption cache root; captions written to <out-root>/per_uuid/<uuid>.txt')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--max-clips', type=int, default=-1)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        (cls, cls_name, note) = resolve_model_class(a.model)
        print(f'chosen model class: {cls_name}  importable={cls is not None}')
        print(f'note: {note}')
        assert _clean('```\nA city street at night in the rain. Extra text.') == 'A city street at night in the rain.'
        assert _clean('  "A quiet suburban road on a clear day."  ') == 'A quiet suburban road on a clear day.'
        print('clean() ok')
        print('SELF-TEST OK' if cls is not None else 'SELF-TEST WARN (class not importable here)')
        return 0 if cls is not None else 2
    if not a.test_data_root or not a.out_root:
        print('--test-data-root and --out-root are required (unless --self-test)', file=sys.stderr)
        return 1
    per_uuid = a.out_root / 'per_uuid'
    per_uuid.mkdir(parents=True, exist_ok=True)
    clips = collect_clip_dirs(a.test_data_root, a.max_clips, a.shard_index, a.num_shards)
    print(f'[dashcam-caption] {len(clips)} clips (shard {a.shard_index}/{a.num_shards}) -> {per_uuid}', flush=True)
    captioner = QwenCaptioner(a.model, device=a.device)
    print(f'[dashcam-caption] loaded {captioner.cls_name} from {a.model}', flush=True)
    index = a.out_root / f'captions_shard{a.shard_index}.jsonl'
    n_done = 0
    with open(index, 'w') as fh:
        for (i, clip_dir) in enumerate(clips, start=1):
            meta = torch.load(clip_dir / 'meta.pt', map_location='cpu', weights_only=False)
            uuid = str(meta.get('uuid', clip_dir.parent.name))
            out_path = per_uuid / f'{uuid}.txt'
            if out_path.exists() and (not a.overwrite):
                cap = out_path.read_text().strip()
                print(f'  [{i}/{len(clips)}] {uuid}: skip (exists)', flush=True)
            else:
                try:
                    frames = sample_frames(str(clip_dir / 'front.mp4'), a.frames)
                    cap = captioner.caption(frames)
                except Exception as exc:
                    print(f'  [{i}/{len(clips)}] {uuid}: FAILED {type(exc).__name__}: {exc}', flush=True)
                    cap = ''
                out_path.write_text(cap)
                print(f'  [{i}/{len(clips)}] {uuid}: {cap!r}', flush=True)
            fh.write(json.dumps({'uuid': uuid, 'caption': cap}) + '\n')
            n_done += 1
    print(f'[dashcam-caption] wrote {n_done} captions -> {per_uuid}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
