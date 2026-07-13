from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
from openlongtail.configs.default import TrainConfig
from openlongtail.configs.cfg_p4 import P4_CONFIG
from openlongtail.data.latent_cache_dataset import RayLatentCacheDataConfig, RayLatentCacheDataset
from openlongtail.data.multiview_dataset import RayMultiViewDataConfig, RaySixCamDataset
from openlongtail.training.forward_ray_p4 import training_step_ray_p4
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.train_ray import HybridLRScheduler, HybridOptimizer, build_dataloader, total_steps_for_stage, trainable_params
from openlongtail.training.train_ray_p3 import _mean_loss_for_logging, build_components_p3, enable_p3_static_graph_for_ddp
from openlongtail.training.distributed import setup_distributed, wrap_ddp
CONFIGS: dict[str, TrainConfig] = {'p4': P4_CONFIG}

def build_dataset_p4(config: TrainConfig) -> RaySixCamDataset | RayLatentCacheDataset:
    if config.latent_cache.use_latent_cache:
        if config.latent_cache.latent_cache_root is None:
            raise ValueError('latent_cache_root is required when use_latent_cache=True')
        data_clip_length = config.data.clip_length
        cache_clip_length = config.latent_cache.clip_length
        if data_clip_length != cache_clip_length:
            raise ValueError(f'expected data.clip_length == latent_cache.clip_length, got {data_clip_length} and {cache_clip_length}')
        data_cfg = RayLatentCacheDataConfig(latent_cache_root=config.latent_cache.latent_cache_root, text_emb_cache_root=config.data.text_emb_cache_root, cache_version=config.latent_cache.cache_version, text_drop_prob=config.data.text_drop_prob, max_items=config.data.max_items)
        return RayLatentCacheDataset(data_cfg, require_p4_sidecar=True)
    data_cfg = RayMultiViewDataConfig(data_root=config.data.data_root, text_emb_cache_root=config.data.text_emb_cache_root, uuid_allowlist_json=config.data.uuid_allowlist_json, clip_length=config.data.clip_length, output_size=config.data.output_size, target_fps=config.data.target_fps, clip_anchor_seconds=config.data.clip_anchor_seconds, clip_jitter_seconds=config.data.clip_jitter_seconds, use_undistorted_simplecalib=config.data.use_undistorted_simplecalib, use_offline_extrinsics=config.data.use_offline_extrinsics, text_drop_prob=config.data.text_drop_prob, max_items=config.data.max_items, num_workers=config.data.num_workers, include_p4_front_pose=True)
    return RaySixCamDataset(data_cfg)

def run_training_p4(config: TrainConfig, stage: str, output_dir: Path, resume_from: Path | None=None, num_steps: int | None=None, device: str | None=None) -> None:
    context = setup_distributed(device)
    components = build_components_p3(config, context.device, stage, resume_from=resume_from)
    dataset = build_dataset_p4(config)
    loader = build_dataloader(config, dataset, distributed=context.is_distributed)
    dit = components.low_dit if stage in ('A.0', 'A.1') else components.high_dit
    if dit is None:
        raise ValueError('expected active P4 DiT module')
    new_params = [] if stage == 'B' else trainable_params(dit.plucker_mlp) + trainable_params(dit.cam_id_embed) + trainable_params(dit.role_embed) + trainable_params(dit.cross_view)
    lora_params = [param for (name, param) in dit.named_parameters() if 'lora_' in name and param.requires_grad]
    optimizer = HybridOptimizer(new_params, lora_params, config)
    if context.is_distributed:
        ddp_dit = wrap_ddp(dit, context.local_rank)
        enable_p3_static_graph_for_ddp(ddp_dit)
        if stage in ('A.0', 'A.1'):
            components.low_dit = ddp_dit
        else:
            components.high_dit = ddp_dit
    total_steps = total_steps_for_stage(config, stage, num_steps)
    lr_scheduler = HybridLRScheduler(optimizer, total_steps, config.optim.warmup_steps, config.optim.cosine_min_lr_ratio)
    flow_scheduler = FlowMatchScheduler()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.run.train_log_name
    iterator = iter(loader)
    for step in range(1, total_steps + 1):
        step_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        result = training_step_ray_p4(batch, components, flow_scheduler, stage, context.device)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params(dit), config.optim.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        loss_mean_all_ranks = _mean_loss_for_logging(result.loss)
        if context.device.type == 'cuda':
            torch.cuda.synchronize(context.device)
        step_time_sec = time.perf_counter() - step_start
        if context.rank == 0:
            with log_path.open('a') as handle:
                serializable = {key: float(value.item()) if isinstance(value, torch.Tensor) else value for (key, value) in result.metrics.items()}
                serializable['loss_mean_all_ranks'] = loss_mean_all_ranks
                serializable['step'] = step
                serializable['step_time_sec'] = step_time_sec
                handle.write(json.dumps(serializable, sort_keys=True) + '\n')
            if step % config.run.save_every == 0 or step == total_steps:
                from openlongtail.training.checkpoint import save_checkpoint
                save_checkpoint(components, optimizer, step, output_dir, stage)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(CONFIGS), default='p4')
    parser.add_argument('--stage', choices=('A.0', 'A.1', 'B'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--latent-cache-root', type=Path, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]
    if args.num_workers is not None:
        config = replace(config, data=replace(config.data, num_workers=args.num_workers))
    if args.latent_cache_root is not None:
        config = replace(config, latent_cache=replace(config.latent_cache, use_latent_cache=True, latent_cache_root=args.latent_cache_root))
    run_training_p4(config, args.stage, args.output_dir, resume_from=args.resume_from, num_steps=args.num_steps, device=args.device)
if __name__ == '__main__':
    main()
