from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable
import torch
import torch.distributed as dist
from openlongtail.configs.default import TrainConfig
from openlongtail.configs.cfg_p61_vace import P61_CONDITION_ENCODER_LAYERS, P61_FRONT_CONDITION_NOISE_MAX, P61_FRONT_CONDITION_NOISE_MIN, P61_FRONT_CONDITION_NOISE_PROB, P61_NEIGHBOR_CONDITION_NOISE_MAX, P61_NEIGHBOR_CONDITION_NOISE_MIN, P61_NEIGHBOR_CONDITION_NOISE_PROB, P61_SEMANTIC_QUERIES, P61_SYNC_TEMPORAL_WINDOW, P61_VACE_1P3B_CONFIG, P61_VACE_CONFIG
from openlongtail.models.dit_p61_vace import DiTP61VACE
from openlongtail.models.wan21_vace_backbone import default_wan21_vace_dir, load_wan21_vace_expert_node_broadcast
from openlongtail.models.wan21_vace_lora import inject_lora_into_wan21_vace_expert
from openlongtail.training.checkpoint_p61 import load_checkpoint_p61, save_checkpoint_p61
from openlongtail.training.distributed import build_node_local_context, setup_distributed, wrap_ddp
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p61 import P61_TARGET_VIEWS, canonical_p61_target_view, training_step_ray_p61
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.train_ray import HybridLRScheduler, build_dataloader, total_steps_for_stage, trainable_params, use_node_broadcast_load
from openlongtail.training.train_ray_p3 import enable_p3_static_graph_for_ddp
from openlongtail.training.train_ray_p4 import build_dataset_p4
CONFIGS: dict[str, TrainConfig] = {'p61_vace': P61_VACE_CONFIG, 'p61_vace_1p3b': P61_VACE_1P3B_CONFIG}

def _mean_loss_for_logging_p61(loss: torch.Tensor) -> float:
    value = loss.detach().float().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return float(value.item())

def parse_target_view_sequence(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == '':
        return None
    sequence = tuple((canonical_p61_target_view(item.strip()) for item in value.split(',') if item.strip()))
    if not sequence:
        raise ValueError('target view sequence must not be empty')
    return sequence

class P61HybridOptimizer:

    def __init__(self, new_module_params: Iterable[torch.nn.Parameter], lora_params: Iterable[torch.nn.Parameter], config: TrainConfig) -> None:
        new_param_list = list(new_module_params)
        self.new_optimizer = torch.optim.AdamW(new_param_list, lr=config.optim.new_module_lr, betas=config.optim.betas, weight_decay=config.optim.new_module_weight_decay) if new_param_list else None
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

def ray_shared_modules_p61(dit: DiTP61VACE) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict({'plucker_mlp': dit.plucker_mlp, 'cam_id_embed': dit.cam_id_embed, 'role_embed': dit.role_embed, 'condition_type_embed': dit.condition_type_embed, 'availability_embed': dit.availability_embed, 'trajectory_mlp': dit.trajectory_mlp, 'condition_encoder': dit.condition_encoder, 'semantic_resampler': dit.semantic_resampler, 'graph_memory': dit.graph_memory})

def initialize_p61_from_checkpoint(dit: DiTP61VACE, checkpoint_dir: Path) -> dict[str, list[str]]:
    payload = load_checkpoint_p61(checkpoint_dir)
    load_info: dict[str, list[str]] = {'missing_p61_lora_keys': [], 'unexpected_p61_lora_keys': [], 'missing_shared_keys': [], 'unexpected_shared_keys': []}
    if 'p61_lora' in payload:
        (missing, unexpected) = dit.load_state_dict(payload['p61_lora'], strict=False)
        load_info['missing_p61_lora_keys'] = list(missing)
        load_info['unexpected_p61_lora_keys'] = list(unexpected)
    if 'shared_modules' in payload:
        (missing, unexpected) = ray_shared_modules_p61(dit).load_state_dict(payload['shared_modules'], strict=False)
        load_info['missing_shared_keys'] = list(missing)
        load_info['unexpected_shared_keys'] = list(unexpected)
    return load_info

def build_components_p61(config: TrainConfig, device: torch.device | str, resume_from: Path | None=None, wan21_vace_dir: Path | None=None, sync_temporal_window: int=P61_SYNC_TEMPORAL_WINDOW, condition_encoder_layers: int=P61_CONDITION_ENCODER_LAYERS, semantic_queries: int=P61_SEMANTIC_QUERIES, enable_motion_embedding: bool=True) -> RayTrainingComponents:
    from openlongtail.models.wan_vae import load_wan21_vae
    node_context = build_node_local_context() if use_node_broadcast_load() else None
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    expert = load_wan21_vace_expert_node_broadcast(wan21_vace_dir or default_wan21_vace_dir(), dtype=torch.bfloat16, device=device, node_context=node_context)
    inject_lora_into_wan21_vace_expert(expert, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    dit = DiTP61VACE(expert, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding).to(device=device, dtype=torch.bfloat16)
    if resume_from is not None:
        initialize_p61_from_checkpoint(dit, Path(resume_from))
    dit.enable_gradient_checkpointing(use_reentrant=False)
    return RayTrainingComponents(vae=vae, low_dit=dit, shared_modules=ray_shared_modules_p61(dit))

def run_training_p61(config: TrainConfig, output_dir: Path, resume_from: Path | None=None, num_steps: int | None=None, device: str | None=None, wan21_vace_dir: Path | None=None, front_condition_noise_prob: float=P61_FRONT_CONDITION_NOISE_PROB, front_condition_noise_min: float=P61_FRONT_CONDITION_NOISE_MIN, front_condition_noise_max: float=P61_FRONT_CONDITION_NOISE_MAX, neighbor_condition_noise_prob: float=P61_NEIGHBOR_CONDITION_NOISE_PROB, neighbor_condition_noise_min: float=P61_NEIGHBOR_CONDITION_NOISE_MIN, neighbor_condition_noise_max: float=P61_NEIGHBOR_CONDITION_NOISE_MAX, sync_temporal_window: int=P61_SYNC_TEMPORAL_WINDOW, condition_encoder_layers: int=P61_CONDITION_ENCODER_LAYERS, semantic_queries: int=P61_SEMANTIC_QUERIES, enable_motion_embedding: bool=True, target_view_sequence: tuple[int, ...] | None=None) -> None:
    context = setup_distributed(device)
    components = build_components_p61(config, context.device, resume_from=resume_from, wan21_vace_dir=wan21_vace_dir, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding)
    dataset = build_dataset_p4(config)
    loader = build_dataloader(config, dataset, distributed=context.is_distributed)
    dit = components.low_dit
    new_params = trainable_params(components.shared_modules)
    lora_params = [param for (name, param) in dit.named_parameters() if 'lora_' in name and param.requires_grad]
    optimizer = P61HybridOptimizer(new_params, lora_params, config)
    if context.is_distributed:
        ddp_dit = wrap_ddp(dit, context.local_rank)
        enable_p3_static_graph_for_ddp(ddp_dit)
        components.low_dit = ddp_dit
    total_steps = int(num_steps or total_steps_for_stage(config, 'A.1'))
    lr_scheduler = HybridLRScheduler(optimizer, total_steps, config.optim.warmup_steps, config.optim.cosine_min_lr_ratio)
    flow_scheduler = FlowMatchScheduler(shift=config.model.sigma_shift)
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
        target_view = None
        if target_view_sequence is not None:
            target_view = target_view_sequence[(step - 1) % len(target_view_sequence)]
        result = training_step_ray_p61(batch, components, flow_scheduler, context.device, target_view=target_view, front_condition_noise_prob=front_condition_noise_prob, front_condition_noise_min=front_condition_noise_min, front_condition_noise_max=front_condition_noise_max, neighbor_condition_noise_prob=neighbor_condition_noise_prob, neighbor_condition_noise_min=neighbor_condition_noise_min, neighbor_condition_noise_max=neighbor_condition_noise_max)
        if not torch.isfinite(result.loss.detach()).all():
            raise RuntimeError(f'P6.1 training produced non-finite loss at step {step}: {result.loss.detach().float().item()}')
        result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params(dit), config.optim.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        lr_scheduler.step()
        loss_mean_all_ranks = _mean_loss_for_logging_p61(result.loss)
        if context.device.type == 'cuda':
            torch.cuda.synchronize(context.device)
        step_time_sec = time.perf_counter() - step_start
        if context.rank == 0:
            with log_path.open('a') as handle:
                serializable = {key: float(value.item()) if isinstance(value, torch.Tensor) else value for (key, value) in result.metrics.items()}
                serializable['loss_mean_all_ranks'] = loss_mean_all_ranks
                serializable['grad_norm'] = float(grad_norm.detach().float().item())
                serializable['flow_shift'] = float(flow_scheduler.shift)
                serializable['new_module_optimizer'] = 'torch.optim.AdamW'
                serializable['step'] = step
                serializable['step_time_sec'] = step_time_sec
                handle.write(json.dumps(serializable, sort_keys=True) + '\n')
            if step % config.run.save_every == 0 or step == total_steps:
                save_checkpoint_p61(components, optimizer, step, output_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(CONFIGS), default='p61_vace')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--warmup-steps', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=None)
    parser.add_argument('--new-module-lr', type=float, default=None)
    parser.add_argument('--lora-lr', type=float, default=None)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--latent-cache-root', type=Path, default=None)
    parser.add_argument('--wan21-vace-dir', type=Path, default=None)
    parser.add_argument('--front-condition-noise-prob', type=float, default=P61_FRONT_CONDITION_NOISE_PROB)
    parser.add_argument('--front-condition-noise-min', type=float, default=P61_FRONT_CONDITION_NOISE_MIN)
    parser.add_argument('--front-condition-noise-max', type=float, default=P61_FRONT_CONDITION_NOISE_MAX)
    parser.add_argument('--neighbor-condition-noise-prob', type=float, default=P61_NEIGHBOR_CONDITION_NOISE_PROB)
    parser.add_argument('--neighbor-condition-noise-min', type=float, default=P61_NEIGHBOR_CONDITION_NOISE_MIN)
    parser.add_argument('--neighbor-condition-noise-max', type=float, default=P61_NEIGHBOR_CONDITION_NOISE_MAX)
    parser.add_argument('--sync-temporal-window', type=int, default=P61_SYNC_TEMPORAL_WINDOW)
    parser.add_argument('--condition-encoder-layers', type=int, default=P61_CONDITION_ENCODER_LAYERS)
    parser.add_argument('--semantic-queries', type=int, default=P61_SEMANTIC_QUERIES)
    parser.add_argument('--target-view-sequence', type=str, default=None, help=f'Comma-separated P6.1 target views to cycle through, e.g. 1,2,3,4,5. Valid targets: {P61_TARGET_VIEWS}.')
    parser.add_argument('--disable-motion-embedding', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]
    if args.num_workers is not None:
        config = replace(config, data=replace(config.data, num_workers=args.num_workers))
    if args.warmup_steps is not None:
        config = replace(config, optim=replace(config.optim, warmup_steps=args.warmup_steps))
    if args.save_every is not None:
        config = replace(config, run=replace(config.run, save_every=args.save_every))
    if args.new_module_lr is not None or args.lora_lr is not None or args.grad_clip is not None:
        config = replace(config, optim=replace(config.optim, new_module_lr=config.optim.new_module_lr if args.new_module_lr is None else args.new_module_lr, lora_lr=config.optim.lora_lr if args.lora_lr is None else args.lora_lr, grad_clip=config.optim.grad_clip if args.grad_clip is None else args.grad_clip))
    if args.latent_cache_root is not None:
        config = replace(config, latent_cache=replace(config.latent_cache, use_latent_cache=True, latent_cache_root=args.latent_cache_root))
    run_training_p61(config, args.output_dir, resume_from=args.resume_from, num_steps=args.num_steps, device=args.device, wan21_vace_dir=args.wan21_vace_dir, front_condition_noise_prob=args.front_condition_noise_prob, front_condition_noise_min=args.front_condition_noise_min, front_condition_noise_max=args.front_condition_noise_max, neighbor_condition_noise_prob=args.neighbor_condition_noise_prob, neighbor_condition_noise_min=args.neighbor_condition_noise_min, neighbor_condition_noise_max=args.neighbor_condition_noise_max, sync_temporal_window=args.sync_temporal_window, condition_encoder_layers=args.condition_encoder_layers, semantic_queries=args.semantic_queries, enable_motion_embedding=not args.disable_motion_embedding, target_view_sequence=parse_target_view_sequence(args.target_view_sequence))
if __name__ == '__main__':
    main()
