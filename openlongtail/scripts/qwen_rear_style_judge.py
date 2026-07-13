#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
AXES: dict[str, list[str]] = {'location': ['city', 'suburb', 'highway', 'rural', 'tunnel'], 'time': ['day', 'dusk_dawn', 'night'], 'weather': ['clear', 'overcast', 'rain', 'fog_snow']}
AXIS_ORDER = ['location', 'time', 'weather']
DEFAULT_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'

def _read_architectures(model_dir: str) -> list[str]:
    cfg_path = Path(model_dir) / 'config.json'
    if not cfg_path.is_file():
        return []
    try:
        with open(cfg_path) as fh:
            return list(json.load(fh).get('architectures', []) or [])
    except Exception:
        return []

def resolve_model_class(model_dir: str) -> tuple[type | None, str, str]:
    import transformers
    archs = _read_architectures(model_dir)
    wants_25 = any(('Qwen2_5_VL' in a or 'Qwen2.5' in a for a in archs))
    wants_2 = any((a.startswith('Qwen2VL') for a in archs))
    candidates: list[str] = []
    if wants_2 and (not wants_25):
        candidates = ['Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration']
    else:
        candidates = ['Qwen2_5_VLForConditionalGeneration', 'Qwen2VLForConditionalGeneration']
    tried: list[str] = []
    for name in candidates:
        try:
            cls = getattr(__import__('transformers', fromlist=[name]), name)
            return (cls, name, f'imported {name}; config.architectures={archs}')
        except Exception as exc:
            tried.append(f'{name}: {type(exc).__name__}')
    return (None, candidates[0], f"could not import any of {candidates}; tried [{'; '.join(tried)}]")

def sample_frames(mp4_path: str, n_frames: int):
    from PIL import Image
    p = Path(mp4_path)
    if not p.is_file():
        raise FileNotFoundError(f'video not found: {mp4_path}')
    frames = []
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(str(p))
        try:
            total = reader.count_frames()
        except Exception:
            total = None
        if not total or total <= 0:
            all_frames = [f for f in reader]
            total = len(all_frames)
            if total == 0:
                raise RuntimeError(f'no frames decoded from {mp4_path}')
            idxs = _even_indices(total, n_frames)
            return [Image.fromarray(all_frames[i]).convert('RGB') for i in idxs]
        idxs = _even_indices(total, n_frames)
        for i in idxs:
            frames.append(Image.fromarray(reader.get_data(i)).convert('RGB'))
        reader.close()
        return frames
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(p))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        grabbed = []
        if total <= 0:
            while True:
                (ok, fr) = cap.read()
                if not ok:
                    break
                grabbed.append(fr)
            total = len(grabbed)
            if total == 0:
                cap.release()
                raise RuntimeError(f'no frames decoded from {mp4_path}')
            idxs = _even_indices(total, n_frames)
            out = [grabbed[i] for i in idxs]
        else:
            idxs = _even_indices(total, n_frames)
            out = []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                (ok, fr) = cap.read()
                if ok:
                    out.append(fr)
            cap.release()
        return [Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) for fr in out]

def _even_indices(total: int, n: int) -> list[int]:
    n = max(1, min(n, total))
    if n == 1:
        return [total // 2]
    step = (total - 1) / (n - 1)
    return [min(total - 1, int(round(i * step))) for i in range(n)]
_PROMPT = 'You are a strict driving-scene STYLE annotator. Look at these frames from a single dashcam video clip (all frames are the same scene) and classify the overall scene STYLE on exactly three axes. Choose EXACTLY ONE option per axis:\n- location: one of [city, suburb, highway, rural, tunnel]\n    city=dense urban buildings/intersections; suburb=residential low-density with houses; highway=multi-lane freeway/expressway; rural=open countryside/fields/two-lane roads; tunnel=inside an enclosed tunnel.\n- time: one of [day, dusk_dawn, night]\n- weather: one of [clear, overcast, rain, fog_snow]\nRespond with ONLY a compact JSON object, no prose, no code fences, exactly:\n{"location": "...", "time": "...", "weather": "..."}'

class QwenStyleJudge:

    def __init__(self, model_dir: str, device: str='cuda'):
        import torch
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
        self._torch = torch

    def annotate(self, frames) -> dict[str, str]:
        content = [{'type': 'image', 'image': img} for img in frames]
        content.append({'type': 'text', 'text': _PROMPT})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=frames, padding=True, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for (k, v) in inputs.items()}
        with self._torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=96, do_sample=False, temperature=0.0)
        trimmed = gen[:, inputs['input_ids'].shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return parse_annotation(out)

def parse_annotation(text: str) -> dict[str, str]:
    result: dict[str, str] = {'location': 'unknown', 'time': 'unknown', 'weather': 'unknown'}
    obj: dict[str, Any] = {}
    m = re.search('\\{.*\\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = {}
    for (axis, options) in AXES.items():
        val = str(obj.get(axis, '')).strip().lower()
        if val in options:
            result[axis] = val
        else:
            hay = (val + ' ' + text).lower()
            for opt in options:
                if re.search('\\b' + re.escape(opt) + '\\b', hay):
                    result[axis] = opt
                    break
    return result

def read_manifest(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'clip', 'arm', 'view', 'mp4'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'manifest missing columns {missing}; got {reader.fieldnames}')
        for r in reader:
            rows.append({'clip': str(r['clip']).strip(), 'arm': str(r['arm']).strip(), 'view': str(r['view']).strip(), 'mp4': str(r['mp4']).strip()})
    return rows

def run_judge(args) -> int:
    rows = read_manifest(args.manifest)
    if not rows:
        print('empty manifest', file=sys.stderr)
        return 1
    judge = QwenStyleJudge(args.model, device=args.device)
    print(f'[judge] loaded {judge.cls_name} from {args.model}', flush=True)
    ann_cache: dict[str, dict[str, str]] = {}
    for r in rows:
        mp4 = r['mp4']
        if mp4 in ann_cache:
            continue
        try:
            frames = sample_frames(mp4, args.frames)
            ann_cache[mp4] = judge.annotate(frames)
        except Exception as exc:
            print(f'[warn] failed {mp4}: {type(exc).__name__}: {exc}', file=sys.stderr)
            ann_cache[mp4] = {a: 'unknown' for a in AXIS_ORDER}
        a = ann_cache[mp4]
        print(f"[ann] {r['arm']:>6} clip={r['clip']} view={r['view']}  {a['location']}/{a['time']}/{a['weather']}", flush=True)
    gt: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        if r['arm'].lower() == 'gt':
            gt[r['clip'], r['view']] = ann_cache[r['mp4']]
    per_arm_axis_hit: dict[str, dict[str, int]] = defaultdict(lambda : defaultdict(int))
    per_arm_axis_tot: dict[str, dict[str, int]] = defaultdict(lambda : defaultdict(int))
    per_arm_stylematch_sum: dict[str, float] = defaultdict(float)
    per_arm_stylematch_n: dict[str, int] = defaultdict(int)
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        arm = r['arm']
        ann = ann_cache[r['mp4']]
        gt_ann = gt.get((r['clip'], r['view']))
        row_out = {'clip': r['clip'], 'arm': arm, 'view': r['view'], 'mp4': r['mp4'], 'location': ann['location'], 'time': ann['time'], 'weather': ann['weather']}
        if arm.lower() != 'gt' and gt_ann is not None:
            hits = 0
            for axis in AXIS_ORDER:
                match = int(ann[axis] == gt_ann[axis] and ann[axis] != 'unknown')
                row_out[f'{axis}_match'] = match
                per_arm_axis_hit[arm][axis] += match
                per_arm_axis_tot[arm][axis] += 1
                hits += match
            sm = hits / len(AXIS_ORDER)
            row_out['style_match'] = round(sm, 4)
            per_arm_stylematch_sum[arm] += sm
            per_arm_stylematch_n[arm] += 1
        else:
            for axis in AXIS_ORDER:
                row_out[f'{axis}_match'] = ''
            row_out['style_match'] = ''
        out_rows.append(row_out)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['clip', 'arm', 'view', 'mp4', 'location', 'time', 'weather', 'location_match', 'time_match', 'weather_match', 'style_match']
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f'[judge] wrote {len(out_rows)} rows -> {out_path}', flush=True)
    _print_summary(per_arm_axis_hit, per_arm_axis_tot, per_arm_stylematch_sum, per_arm_stylematch_n)
    return 0

def _print_summary(hit, tot, sm_sum, sm_n) -> None:
    arms = sorted(set(list(hit.keys()) + list(sm_n.keys())))
    if not arms:
        print('\n[summary] no non-GT arms with a matching GT-rear clip found.')
        return
    print('\n=== STYLE-MATCH vs GT-rear (fraction of axes agreeing) ===')
    header = f"{'arm':>10} | " + ' | '.join((f'{a:>10}' for a in AXIS_ORDER)) + f" | {'STYLE-MATCH':>11} | {'n':>4}"
    print(header)
    print('-' * len(header))
    for arm in arms:
        cells = []
        for axis in AXIS_ORDER:
            t = tot[arm][axis]
            acc = hit[arm][axis] / t if t else float('nan')
            cells.append(f'{acc:>10.3f}')
        n = sm_n[arm]
        overall = sm_sum[arm] / n if n else float('nan')
        print(f'{arm:>10} | ' + ' | '.join(cells) + f' | {overall:>11.3f} | {n:>4}')

def self_test(model_dir: str) -> int:
    import transformers
    print(f'transformers version : {transformers.__version__}')
    p = Path(model_dir)
    ok_path = p.is_dir() and (p / 'config.json').is_file()
    print(f'weight dir           : {model_dir}')
    print(f'weight dir exists    : {ok_path}')
    archs = _read_architectures(model_dir)
    print(f'config.architectures : {archs}')
    (cls, cls_name, note) = resolve_model_class(model_dir)
    print(f'chosen model class   : {cls_name}')
    print(f'class importable     : {cls is not None}')
    print(f'note                 : {note}')
    model_id = archs[0] if archs else 'unknown'
    print(f'model id             : {model_id}')
    if not ok_path:
        print('SELF-TEST FAILED: weight path missing', file=sys.stderr)
        return 1
    if cls is None:
        print('SELF-TEST WARNING: this transformers cannot import the preferred class; run the GPU judge under the autovla env (transformers>=4.49).', file=sys.stderr)
        return 2
    print('SELF-TEST OK')
    return 0

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Qwen VLM rear-STYLE judge (rear style).')
    ap.add_argument('--manifest', help="CSV with columns clip,arm,view,mp4 (arm=='gt' is GT-rear).")
    ap.add_argument('--out-csv', default='qwen_rear_style_judge.csv')
    ap.add_argument('--model', default=DEFAULT_MODEL, help='Qwen VL weight directory.')
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--self-test', action='store_true', help='Verify weight path + model-class import only (no GPU).')
    return ap

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args.model)
    if not args.manifest:
        print('--manifest is required (unless --self-test)', file=sys.stderr)
        return 1
    return run_judge(args)
if __name__ == '__main__':
    raise SystemExit(main())
