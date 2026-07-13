from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
import torch.distributed as dist
from openlongtail.configs.default import TrainConfig
from openlongtail.configs.cfg_p3 import P3_CONFIG
from openlongtail.models.dit_p3 import DiTP3
from openlongtail.models.wan22_lora import inject_lora_into_wan22_expert
from openlongtail.training.checkpoint import load_checkpoint, save_checkpoint
from openlongtail.training.distributed import build_node_local_context, setup_distributed, wrap_ddp
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p3 import training_step_ray_p3
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.train_ray import HybridLRScheduler, HybridOptimizer, build_blank_image_cond_latent, build_dataloader, build_dataset, freeze_module, load_stage_expert, total_steps_for_stage, trainable_params, use_node_broadcast_load
CONFIGS: dict[str, TrainConfig] = {'p3': P3_CONFIG}

def enable_p3_static_graph_for_ddp(module: object) -> None:
    set_static_graph = getattr(module, '_set_static_graph', None)
    if set_static_graph is None:
        raise ValueError('DDP with reentrant checkpointing requires _set_static_graph support')
    set_static_graph()

def ray_shared_modules_p3(dit: DiTP3) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict({'plucker_mlp': dit.plucker_mlp, 'cam_id_embed': dit.cam_id_embed, 'role_embed': dit.role_embed, 'cross_view': dit.cross_view})

def initialize_low_p3_from_checkpoint(dit: DiTP3, checkpoint_dir: Path) -> dict[str, list[str]]:
    payload = load_checkpoint(checkpoint_dir)
    load_info: dict[str, list[str]] = {'missing_low_keys': [], 'unexpected_low_keys': [], 'missing_shared_keys': [], 'unexpected_shared_keys': []}
    if 'low_lora' in payload:
        (missing, unexpected) = dit.load_state_dict(payload['low_lora'], strict=False)
        load_info['missing_low_keys'] = list(missing)
        load_info['unexpected_low_keys'] = list(unexpected)
    if 'shared_modules' in payload:
        (missing, unexpected) = ray_shared_modules_p3(dit).load_state_dict(payload['shared_modules'], strict=False)
        load_info['missing_shared_keys'] = list(missing)
        load_info['unexpected_shared_keys'] = list(unexpected)
    return load_info

def build_components_p3(config: TrainConfig, device: torch.device | str, stage: str, resume_from: Path | None=None) -> RayTrainingComponents:
    from openlongtail.models.wan_vae import load_wan21_vae
    node_context = build_node_local_context() if use_node_broadcast_load() else None
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    blank = build_blank_image_cond_latent(config, vae, device, node_context=node_context)
    if stage in ('A.0', 'A.1'):
        low = load_stage_expert(config.checkpoints.wan22_low_dir, device=device, node_context=node_context)
        inject_lora_into_wan22_expert(low, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
        low_dit = DiTP3(low, blank_image_cond_latent=blank, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks).to(device=device, dtype=torch.bfloat16)
        if resume_from is not None:
            initialize_low_p3_from_checkpoint(low_dit, Path(resume_from))
        low_dit.enable_gradient_checkpointing(use_reentrant=True)
        return RayTrainingComponents(vae=vae, low_dit=low_dit, shared_modules=ray_shared_modules_p3(low_dit))
    if stage != 'B':
        raise ValueError(f'expected stage one of A.0, A.1, B, got {stage!r}')
    high = load_stage_expert(config.checkpoints.wan22_high_dir, device=device, node_context=node_context)
    inject_lora_into_wan22_expert(high, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    high_dit = DiTP3(high, blank_image_cond_latent=blank, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks).to(device=device, dtype=torch.bfloat16)
    high_dit.enable_gradient_checkpointing(use_reentrant=True)
    shared_modules = ray_shared_modules_p3(high_dit)
    if resume_from is not None:
        payload = torch.load(Path(resume_from) / 'shared_modules.pt', map_location='cpu', weights_only=True)
        shared_modules.load_state_dict(payload, strict=False)
    freeze_module(shared_modules)
    return RayTrainingComponents(vae=vae, low_dit=high_dit, high_dit=high_dit, shared_modules=shared_modules)

def _mean_loss_for_logging(loss: torch.Tensor) -> float:
    value = loss.detach().float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return float(value.item())

def run_training_p3(config: TrainConfig, stage: str, output_dir: Path, resume_from: Path | None=None, num_steps: int | None=None, device: str | None=None) -> None:
    context = setup_distributed(device)
    components = build_components_p3(config, context.device, stage, resume_from=resume_from)
    dataset = build_dataset(config)
    loader = build_dataloader(config, dataset, distributed=context.is_distributed)
    dit = components.low_dit if stage in ('A.0', 'A.1') else components.high_dit
    if dit is None:
        raise ValueError('expected active DiT module')
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
        result = training_step_ray_p3(batch, components, flow_scheduler, stage, context.device)
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
    parser.add_argument('--config', choices=sorted(CONFIGS), default='p3')
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
    run_training_p3(config, args.stage, args.output_dir, resume_from=args.resume_from, num_steps=args.num_steps, device=args.device)
if __name__ == '__main__':
    main()
