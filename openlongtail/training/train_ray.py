from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from openlongtail.configs.default import TrainConfig
from openlongtail.configs.smoke import SMOKE_CONFIG
from openlongtail.configs.smoke_41 import SMOKE_41_CONFIG
from openlongtail.configs.smoke_49 import SMOKE_49_CONFIG
from openlongtail.configs.smoke_61 import SMOKE_61_CONFIG
from openlongtail.configs.smoke_a1 import SMOKE_A1_CONFIG
from openlongtail.configs.stage_a0 import STAGE_A0_CONFIG
from openlongtail.configs.stage_a1 import STAGE_A1_CONFIG
from openlongtail.configs.cfg_2d_warmup100_vdrop import CFG_2D_WARMUP100_VDROP_CONFIG
from openlongtail.configs.cfg_12h_warmup30_vdrop import CFG_12H_WARMUP30_VDROP_CONFIG
from openlongtail.configs.cfg_latent_cache_vdrop import LATENT_CACHE_VDROP_CONFIG
from openlongtail.configs.stage_a49_12h import STAGE_A49_12H_CONFIG
from openlongtail.configs.stage_a49_12h_warmup30 import STAGE_A49_12H_WARMUP30_CONFIG
from openlongtail.configs.stage_b import STAGE_B_CONFIG
from openlongtail.data.latent_cache_dataset import RayLatentCacheDataConfig, RayLatentCacheDataset, ray_latent_cache_collate
from openlongtail.data.multiview_dataset import RayMultiViewDataConfig, RaySixCamDataset, ray_six_cam_collate
from openlongtail.models.dit import DiT
from openlongtail.models.wan22_backbone import load_wan22_expert, load_wan22_expert_node_broadcast
from openlongtail.models.wan22_lora import inject_lora_into_wan22_expert
from openlongtail.models.wan_vae import load_wan21_vae, precompute_blank_image_cond_latent
from openlongtail.training.checkpoint import save_checkpoint
from openlongtail.training.distributed import NodeLocalContext, broadcast_tensor_from_node_src, build_node_local_context, setup_distributed, wrap_ddp
from openlongtail.training.forward_ray import RayTrainingComponents, training_step_ray, training_step_ray_svt_recompute, training_step_ray_view_parallel
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.view_parallel import ViewParallelContext, build_view_parallel_context, create_node_view_group
CONFIGS: dict[str, TrainConfig] = {'stage_a0': STAGE_A0_CONFIG, 'stage_a1': STAGE_A1_CONFIG, '2d_warmup100_vdrop': CFG_2D_WARMUP100_VDROP_CONFIG, '12h_warmup30_vdrop': CFG_12H_WARMUP30_VDROP_CONFIG, 'latent_cache_vdrop': LATENT_CACHE_VDROP_CONFIG, 'stage_a49_12h': STAGE_A49_12H_CONFIG, 'stage_a49_12h_warmup30': STAGE_A49_12H_WARMUP30_CONFIG, 'stage_b': STAGE_B_CONFIG, 'smoke': SMOKE_CONFIG, 'smoke_41': SMOKE_41_CONFIG, 'smoke_49': SMOKE_49_CONFIG, 'smoke_61': SMOKE_61_CONFIG, 'smoke_a1': SMOKE_A1_CONFIG}

class HybridOptimizer:

    def __init__(self, new_module_params: Iterable[torch.nn.Parameter], lora_params: Iterable[torch.nn.Parameter], config: TrainConfig) -> None:
        import bitsandbytes as bnb
        new_param_list = list(new_module_params)
        self.new_optimizer = bnb.optim.AdamW8bit(new_param_list, lr=config.optim.new_module_lr, betas=config.optim.betas, weight_decay=config.optim.new_module_weight_decay) if new_param_list else None
        self.lora_optimizer = torch.optim.AdamW(list(lora_params), lr=config.optim.lora_lr, betas=config.optim.betas, weight_decay=config.optim.lora_weight_decay)

    def zero_grad(self, set_to_none: bool=True) -> None:
        if self.new_optimizer is not None:
            self.new_optimizer.zero_grad(set_to_none=set_to_none)
        self.lora_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.new_optimizer is not None:
            self.new_optimizer.step()
        self.lora_optimizer.step()

    def state_dict(self) -> dict[str, object]:
        return {'new_optimizer': self.new_optimizer.state_dict() if self.new_optimizer is not None else None, 'lora_optimizer': self.lora_optimizer.state_dict()}

class HybridLRScheduler:

    def __init__(self, optimizer: HybridOptimizer, total_steps: int, warmup_steps: int, min_ratio: float) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_ratio = min_ratio
        self.step_idx = 0
        self.base_lrs = [[group['lr'] for group in optimizer.new_optimizer.param_groups] if optimizer.new_optimizer is not None else [], [group['lr'] for group in optimizer.lora_optimizer.param_groups]]

    def _scale(self) -> float:
        if self.step_idx < self.warmup_steps:
            return float(self.step_idx + 1) / float(max(1, self.warmup_steps))
        progress = (self.step_idx - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return self.min_ratio + (1.0 - self.min_ratio) * cosine

    def step(self) -> None:
        scale = self._scale()
        pairs = [(self.optimizer.lora_optimizer, self.base_lrs[1])]
        if self.optimizer.new_optimizer is not None:
            pairs.insert(0, (self.optimizer.new_optimizer, self.base_lrs[0]))
        for (opt, base_lrs) in pairs:
            for (group, base_lr) in zip(opt.param_groups, base_lrs):
                group['lr'] = base_lr * scale
        self.step_idx += 1

def trainable_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]

def maybe_enable_svt_static_graph(module: object, training_mode: str, num_active_targets: int=5) -> None:
    if training_mode == 'standard':
        pass
    elif training_mode == 'sequential_view_recompute' and num_active_targets == 5:
        pass
    else:
        return
    set_static_graph = getattr(module, '_set_static_graph', None)
    if set_static_graph is None:
        raise ValueError(f'{training_mode} DDP requires _set_static_graph support')
    set_static_graph()

def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False

def ray_shared_modules(dit: DiT) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict({'plucker_mlp': dit.plucker_mlp, 'cam_id_embed': dit.cam_id_embed, 'cross_view': dit.cross_view})

def use_node_broadcast_load() -> bool:
    value = os.environ.get('OPENLONGTAIL_NODE_BROADCAST_LOAD', '0').strip().lower()
    return value in ('1', 'true', 'yes', 'on')

def blank_latent_shape(config: TrainConfig) -> tuple[int, int, int, int]:
    (height, width) = config.data.output_size
    latent_frames = (config.data.clip_length - 1) // config.model.vae_stride[0] + 1
    return (config.model.latent_channels, latent_frames, height // config.model.vae_stride[1], width // config.model.vae_stride[2])

def build_blank_image_cond_latent(config: TrainConfig, vae: object, device: torch.device | str, node_context: NodeLocalContext | None=None) -> torch.Tensor:
    if use_node_broadcast_load() and node_context is not None and (node_context.world_size > 1):
        if node_context.is_local_src:
            blank = precompute_blank_image_cond_latent(vae, output_size=config.data.output_size, clip_length=config.data.clip_length, device=device, dtype=torch.bfloat16)
        else:
            blank = torch.empty(blank_latent_shape(config), device=device, dtype=torch.bfloat16)
        return broadcast_tensor_from_node_src(blank, node_context)
    return precompute_blank_image_cond_latent(vae, output_size=config.data.output_size, clip_length=config.data.clip_length, device=device, dtype=torch.bfloat16)

def load_stage_expert(ckpt_dir: Path, device: torch.device | str, node_context: NodeLocalContext | None=None) -> torch.nn.Module:
    if use_node_broadcast_load():
        return load_wan22_expert_node_broadcast(ckpt_dir, dtype=torch.bfloat16, device=device, node_context=node_context)
    return load_wan22_expert(ckpt_dir, dtype=torch.bfloat16, device=device)

def build_components(config: TrainConfig, device: torch.device | str, stage: str, resume_from: Path | None=None, node_context: NodeLocalContext | None=None) -> RayTrainingComponents:
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    blank = build_blank_image_cond_latent(config, vae, device, node_context=node_context)
    if stage in ('A.0', 'A.1'):
        low = load_stage_expert(config.checkpoints.wan22_low_dir, device=device, node_context=node_context)
        inject_lora_into_wan22_expert(low, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
        low_dit = DiT(low, blank_image_cond_latent=blank).to(device=device, dtype=torch.bfloat16)
        low_dit.enable_gradient_checkpointing(use_reentrant=True)
        return RayTrainingComponents(vae=vae, low_dit=low_dit, shared_modules=ray_shared_modules(low_dit))
    high = load_stage_expert(config.checkpoints.wan22_high_dir, device=device, node_context=node_context)
    inject_lora_into_wan22_expert(high, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    high_dit = DiT(high, blank_image_cond_latent=blank).to(device=device, dtype=torch.bfloat16)
    high_dit.enable_gradient_checkpointing(use_reentrant=True)
    shared_modules = ray_shared_modules(high_dit)
    if resume_from is not None:
        payload = torch.load(Path(resume_from) / 'shared_modules.pt', map_location='cpu', weights_only=True)
        shared_modules.load_state_dict(payload, strict=False)
    freeze_module(shared_modules)
    return RayTrainingComponents(vae=vae, low_dit=high_dit, high_dit=high_dit, shared_modules=shared_modules)

def build_dataset(config: TrainConfig) -> RaySixCamDataset | RayLatentCacheDataset:
    if config.latent_cache.use_latent_cache:
        if config.latent_cache.latent_cache_root is None:
            raise ValueError('latent_cache_root is required when use_latent_cache=True')
        data_clip_length = config.data.clip_length
        cache_clip_length = config.latent_cache.clip_length
        if data_clip_length != cache_clip_length:
            raise ValueError(f'expected data.clip_length == latent_cache.clip_length, got {data_clip_length} and {cache_clip_length}')
        data_cfg = RayLatentCacheDataConfig(latent_cache_root=config.latent_cache.latent_cache_root, text_emb_cache_root=config.data.text_emb_cache_root, cache_version=config.latent_cache.cache_version, text_drop_prob=config.data.text_drop_prob, max_items=config.data.max_items)
        return RayLatentCacheDataset(data_cfg)
    data_cfg = RayMultiViewDataConfig(data_root=config.data.data_root, text_emb_cache_root=config.data.text_emb_cache_root, uuid_allowlist_json=config.data.uuid_allowlist_json, clip_length=config.data.clip_length, output_size=config.data.output_size, target_fps=config.data.target_fps, clip_anchor_seconds=config.data.clip_anchor_seconds, clip_jitter_seconds=config.data.clip_jitter_seconds, use_undistorted_simplecalib=config.data.use_undistorted_simplecalib, use_offline_extrinsics=config.data.use_offline_extrinsics, text_drop_prob=config.data.text_drop_prob, max_items=config.data.max_items, num_workers=config.data.num_workers)
    return RaySixCamDataset(data_cfg)

def build_dataloader(config: TrainConfig, dataset: RaySixCamDataset | RayLatentCacheDataset, distributed: bool=False, num_replicas: int | None=None, rank: int | None=None) -> DataLoader:
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank) if distributed else None
    collate_fn = ray_latent_cache_collate if isinstance(dataset, RayLatentCacheDataset) else ray_six_cam_collate
    return DataLoader(dataset, batch_size=config.run.batch_size_per_gpu, sampler=sampler, shuffle=sampler is None, num_workers=config.data.num_workers, collate_fn=collate_fn)

def total_steps_for_stage(config: TrainConfig, stage: str, override: int | None=None) -> int:
    if override is not None:
        return override
    return {'A.0': config.run.stage_a0_steps, 'A.1': config.run.stage_a1_steps, 'B': config.run.stage_b_steps}[stage]

def _mean_loss_for_logging(loss: torch.Tensor) -> float:
    value = loss.detach().float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return float(value.item())

def run_training(config: TrainConfig, stage: str, output_dir: Path, resume_from: Path | None=None, num_steps: int | None=None, device: str | None=None, view_parallel: bool=False, training_mode: str='standard') -> None:
    context = setup_distributed(device)
    if view_parallel:
        training_mode = 'view_parallel'
    if training_mode not in ('standard', 'view_parallel', 'sequential_view_recompute'):
        raise ValueError(f'expected training_mode standard/view_parallel/sequential_view_recompute, got {training_mode!r}')
    view_context: ViewParallelContext | None = None
    if training_mode == 'view_parallel':
        if not context.is_distributed:
            raise ValueError('view-parallel training requires distributed launch with 6 ranks per node')
        view_context = create_node_view_group(build_view_parallel_context(rank=context.rank, world_size=context.world_size, local_rank=context.local_rank, views_per_node=config.model.num_views))
    node_context = build_node_local_context() if use_node_broadcast_load() else None
    components = build_components(config, context.device, stage, resume_from=resume_from, node_context=node_context)
    dataset = build_dataset(config)
    loader = build_dataloader(config, dataset, distributed=context.is_distributed, num_replicas=view_context.data_parallel_size if view_context is not None else None, rank=view_context.data_parallel_rank if view_context is not None else None)
    dit = components.low_dit if stage in ('A.0', 'A.1') else components.high_dit
    if dit is None:
        raise ValueError('expected active DiT module')
    new_params = [] if stage == 'B' else trainable_params(dit.plucker_mlp) + trainable_params(dit.cam_id_embed) + trainable_params(dit.cross_view)
    lora_params = [param for (name, param) in dit.named_parameters() if 'lora_' in name and param.requires_grad]
    optimizer = HybridOptimizer(new_params, lora_params, config)
    if context.is_distributed:
        ddp_dit = wrap_ddp(dit, context.local_rank)
        maybe_enable_svt_static_graph(ddp_dit, training_mode, config.model.view_dropout_active_targets)
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
        if training_mode == 'view_parallel':
            if view_context is None:
                raise ValueError('view_parallel mode requires view_context')
            result = training_step_ray_view_parallel(batch, components, flow_scheduler, stage, view_context, context.device, num_active_targets=config.model.view_dropout_active_targets)
        elif training_mode == 'sequential_view_recompute':
            result = training_step_ray_svt_recompute(batch, components, flow_scheduler, stage, context.device, num_active_targets=config.model.view_dropout_active_targets)
        else:
            result = training_step_ray(batch, components, flow_scheduler, stage, context.device, num_active_targets=config.model.view_dropout_active_targets)
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
                save_checkpoint(components, optimizer, step, output_dir, stage)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(CONFIGS), default='smoke')
    parser.add_argument('--stage', choices=('A.0', 'A.1', 'B'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--training-mode', choices=('standard', 'view_parallel', 'sequential_view_recompute'), default='standard')
    parser.add_argument('--view-parallel', action='store_true')
    parser.add_argument('--view-dropout-active-targets', type=int, choices=(3, 5), default=None)
    parser.add_argument('--latent-cache-root', type=Path, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]
    if args.num_workers is not None:
        config = replace(config, data=replace(config.data, num_workers=args.num_workers))
    if args.view_dropout_active_targets is not None:
        config = replace(config, model=replace(config.model, view_dropout_active_targets=args.view_dropout_active_targets))
    if args.latent_cache_root is not None:
        config = replace(config, latent_cache=replace(config.latent_cache, use_latent_cache=True, latent_cache_root=args.latent_cache_root))
    run_training(config, args.stage, args.output_dir, resume_from=args.resume_from, num_steps=args.num_steps, device=args.device, view_parallel=args.view_parallel, training_mode=args.training_mode)
if __name__ == '__main__':
    main()
