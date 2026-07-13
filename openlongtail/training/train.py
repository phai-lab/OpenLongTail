from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
import torch.distributed as dist
from openlongtail.configs.default import TrainConfig
from openlongtail.configs.cfg_p61_vace import P61_CONDITION_ENCODER_LAYERS, P61_FRONT_CONDITION_NOISE_MAX, P61_FRONT_CONDITION_NOISE_MIN, P61_FRONT_CONDITION_NOISE_PROB, P61_NEIGHBOR_CONDITION_NOISE_MAX, P61_NEIGHBOR_CONDITION_NOISE_MIN, P61_NEIGHBOR_CONDITION_NOISE_PROB, P61_SEMANTIC_QUERIES, P61_SYNC_TEMPORAL_WINDOW
from openlongtail.configs.openlongtail_vace import GEO_HEAD_DIM, GEO_PROJECTION_TEMPERATURE, GRAPH_GATE_INIT_BIAS, OPENLONGTAIL_14B_CONFIG, OPENLONGTAIL_1P3B_CONFIG, OPENLONGTAIL_BASE_CONFIG
from torch.utils.data import DataLoader, DistributedSampler
from openlongtail.data.latent_cache_dataset import RayLatentCacheDataConfig
from openlongtail.data.warp_dataset import RayLatentCacheDatasetWarp, ray_latent_cache_collate_warp
from openlongtail.models.dit_vace import DiTVACEWarp
from openlongtail.models.wan21_vace_backbone import default_wan21_vace_dir, load_wan21_vace_expert_node_broadcast
from openlongtail.models.wan21_vace_lora import inject_lora_into_wan21_vace_expert
from openlongtail.training.checkpoint_warp import load_warp_checkpoint, save_warp_checkpoint, save_warp_checkpoint_fulltune
from openlongtail.training.distributed import build_node_local_context, setup_distributed, wrap_ddp
from openlongtail.training.forward_ray import RayTrainingComponents
from openlongtail.training.forward_ray_p61 import P61_TARGET_VIEWS
from openlongtail.training.forward_ray_warp import training_step_ray_warp
from openlongtail.training.schedulers import FlowMatchScheduler
from openlongtail.training.train_ray import HybridLRScheduler, total_steps_for_stage, trainable_params, use_node_broadcast_load
from openlongtail.training.train_ray_p3 import enable_p3_static_graph_for_ddp
from openlongtail.training.train_ray_p61 import P61HybridOptimizer, _mean_loss_for_logging_p61, parse_target_view_sequence
CONFIGS: dict[str, TrainConfig] = {'openlongtail_1p3b': OPENLONGTAIL_1P3B_CONFIG, 'openlongtail_14b': OPENLONGTAIL_14B_CONFIG, 'openlongtail_base': OPENLONGTAIL_BASE_CONFIG}

def _nonfinite_gradient_summary(module: torch.nn.Module, limit: int=32) -> dict[str, object]:
    bad_params: list[dict[str, object]] = []
    total_nonfinite = 0
    for (name, param) in module.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if bool(finite.all().item()):
            continue
        nonfinite_count = int((~finite).sum().detach().cpu().item())
        total_nonfinite += nonfinite_count
        finite_abs = grad.detach()[finite].float().abs()
        max_abs_finite = float(finite_abs.max().cpu().item()) if finite_abs.numel() > 0 else None
        bad_params.append({'name': name, 'shape': list(param.shape), 'dtype': str(param.dtype), 'nonfinite_count': nonfinite_count, 'max_abs_finite_grad': max_abs_finite})
    return {'total_nonfinite_grad_values': total_nonfinite, 'bad_param_count': len(bad_params), 'bad_params': bad_params[:limit], 'truncated': len(bad_params) > limit}

def ray_shared_modules(dit: DiTVACEWarp) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict({'plucker_mlp': dit.plucker_mlp, 'cam_id_embed': dit.cam_id_embed, 'role_embed': dit.role_embed, 'condition_type_embed': dit.condition_type_embed, 'availability_embed': dit.availability_embed, 'trajectory_mlp': dit.trajectory_mlp, 'condition_encoder': dit.condition_encoder, 'semantic_resampler': dit.semantic_resampler, 'graph_memory': dit.graph_memory, 'target_pose_mlp': dit.target_pose_mlp, 'target_view_mod_embed': dit.target_view_mod_embed})

def initialize_from_checkpoint(dit: DiTVACEWarp, checkpoint_dir: Path) -> dict[str, list[str]]:
    payload = load_warp_checkpoint(checkpoint_dir)
    info: dict[str, list[str]] = {'missing_lora_keys': [], 'unexpected_lora_keys': [], 'missing_shared_keys': [], 'unexpected_shared_keys': [], 'missing_backbone_keys': [], 'unexpected_backbone_keys': []}
    if 'lora' in payload:
        (m, u) = dit.load_state_dict(payload['lora'], strict=False)
        info['missing_lora_keys'] = list(m)
        info['unexpected_lora_keys'] = list(u)
    if 'backbone' in payload:
        (m, u) = dit.load_state_dict(payload['backbone'], strict=False)
        info['missing_backbone_keys'] = list(m)
        info['unexpected_backbone_keys'] = list(u)
    if 'shared_modules' in payload:
        (m, u) = ray_shared_modules(dit).load_state_dict(payload['shared_modules'], strict=False)
        info['missing_shared_keys'] = list(m)
        info['unexpected_shared_keys'] = list(u)
    return info

def build_components(config: TrainConfig, device: torch.device | str, resume_from: Path | None=None, wan21_vace_dir: Path | None=None, sync_temporal_window: int=P61_SYNC_TEMPORAL_WINDOW, condition_encoder_layers: int=P61_CONDITION_ENCODER_LAYERS, semantic_queries: int=P61_SEMANTIC_QUERIES, enable_motion_embedding: bool=True, graph_gate_init_bias: float=GRAPH_GATE_INIT_BIAS, geo_head_dim: int=GEO_HEAD_DIM, geo_projection_temperature: float=GEO_PROJECTION_TEMPERATURE, unfreeze_backbone: bool=False, inject_lora: bool=True) -> RayTrainingComponents:
    from openlongtail.models.wan_vae import load_wan21_vae
    node_context = build_node_local_context() if use_node_broadcast_load() else None
    vae = load_wan21_vae(config.checkpoints.vae_path, dtype=torch.bfloat16, device=device)
    expert = load_wan21_vace_expert_node_broadcast(wan21_vace_dir or default_wan21_vace_dir(), dtype=torch.bfloat16, device=device, node_context=node_context, freeze_backbone=not unfreeze_backbone)
    if inject_lora:
        inject_lora_into_wan21_vace_expert(expert, rank=config.model.lora_rank, alpha=config.model.lora_alpha)
    dit = DiTVACEWarp(expert, dim_attn=config.model.cross_view_dim_attn, heads=config.model.cross_view_heads, cross_view_blocks=config.model.cross_view_blocks, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding, graph_gate_init_bias=graph_gate_init_bias, geo_head_dim=geo_head_dim, geo_projection_temperature=geo_projection_temperature).to(device=device, dtype=torch.bfloat16)
    if resume_from is not None:
        initialize_from_checkpoint(dit, Path(resume_from))
    dit.enable_gradient_checkpointing(use_reentrant=False)
    if unfreeze_backbone:
        for fp32_mod_name in ('time_embedding', 'time_projection', 'head'):
            fp32_mod = getattr(dit.expert, fp32_mod_name, None)
            if fp32_mod is not None:
                for p in fp32_mod.parameters():
                    p.requires_grad = False
    return RayTrainingComponents(vae=vae, low_dit=dit, shared_modules=ray_shared_modules(dit))

def build_warp_dataloader(config: TrainConfig, dataset: RayLatentCacheDatasetWarp, distributed: bool=False, num_replicas: int | None=None, rank: int | None=None) -> DataLoader:
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank) if distributed else None
    return DataLoader(dataset, batch_size=config.run.batch_size_per_gpu, sampler=sampler, shuffle=sampler is None, num_workers=config.data.num_workers, collate_fn=ray_latent_cache_collate_warp)

def build_dataset(config: TrainConfig, restrict_to_existing_sidecars: bool=False, require_lookback_sidecar: bool=False, index_filename: str='index.jsonl') -> RayLatentCacheDatasetWarp:
    if not config.latent_cache.use_latent_cache:
        raise ValueError('requires latent_cache.use_latent_cache=True (sidecar lives next to latent cache)')
    if config.latent_cache.latent_cache_root is None:
        raise ValueError('requires latent_cache_root to be set')
    data_cfg = RayLatentCacheDataConfig(latent_cache_root=config.latent_cache.latent_cache_root, text_emb_cache_root=config.data.text_emb_cache_root, cache_version=config.latent_cache.cache_version, cache_versions=config.latent_cache.cache_versions, text_drop_prob=config.data.text_drop_prob, max_items=config.data.max_items, index_filename=index_filename)
    return RayLatentCacheDatasetWarp(data_cfg, require_p4_sidecar=True, restrict_to_existing_sidecars=restrict_to_existing_sidecars, require_lookback_sidecar=require_lookback_sidecar)

def run_training(config: TrainConfig, output_dir: Path, resume_from: Path | None=None, num_steps: int | None=None, device: str | None=None, wan21_vace_dir: Path | None=None, front_condition_noise_prob: float=P61_FRONT_CONDITION_NOISE_PROB, front_condition_noise_min: float=P61_FRONT_CONDITION_NOISE_MIN, front_condition_noise_max: float=P61_FRONT_CONDITION_NOISE_MAX, neighbor_condition_noise_prob: float=P61_NEIGHBOR_CONDITION_NOISE_PROB, neighbor_condition_noise_min: float=P61_NEIGHBOR_CONDITION_NOISE_MIN, neighbor_condition_noise_max: float=P61_NEIGHBOR_CONDITION_NOISE_MAX, sync_temporal_window: int=P61_SYNC_TEMPORAL_WINDOW, condition_encoder_layers: int=P61_CONDITION_ENCODER_LAYERS, semantic_queries: int=P61_SEMANTIC_QUERIES, enable_motion_embedding: bool=True, target_view_sequence: tuple[int, ...] | None=None, graph_gate_init_bias: float=GRAPH_GATE_INIT_BIAS, geo_head_dim: int=GEO_HEAD_DIM, geo_projection_temperature: float=GEO_PROJECTION_TEMPERATURE, rear_loss_weight: float=1.0, restrict_to_existing_sidecars: bool=False, require_lookback_sidecar: bool=False, index_filename: str='index.jsonl', unfreeze_backbone: bool=False, backbone_lr: float=1e-06) -> None:
    context = setup_distributed(device)
    if unfreeze_backbone:
        config = replace(config, optim=replace(config.optim, lora_lr=backbone_lr))
    components = build_components(config, context.device, resume_from=resume_from, wan21_vace_dir=wan21_vace_dir, sync_temporal_window=sync_temporal_window, condition_encoder_layers=condition_encoder_layers, semantic_queries=semantic_queries, enable_motion_embedding=enable_motion_embedding, graph_gate_init_bias=graph_gate_init_bias, geo_head_dim=geo_head_dim, geo_projection_temperature=geo_projection_temperature, unfreeze_backbone=unfreeze_backbone, inject_lora=not unfreeze_backbone)
    dataset = build_dataset(config, restrict_to_existing_sidecars=restrict_to_existing_sidecars, require_lookback_sidecar=require_lookback_sidecar, index_filename=index_filename)
    loader = build_warp_dataloader(config, dataset, distributed=context.is_distributed, num_replicas=context.world_size if context.is_distributed else None, rank=context.rank if context.is_distributed else None)
    dit = components.low_dit
    new_params = trainable_params(components.shared_modules)
    if unfreeze_backbone:
        shared_param_ids = {id(p) for p in new_params}
        backbone_or_lora_params = [p for (n, p) in dit.named_parameters() if p.requires_grad and id(p) not in shared_param_ids]
    else:
        backbone_or_lora_params = [p for (n, p) in dit.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = P61HybridOptimizer(new_params, backbone_or_lora_params, config)
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
    start_step = 1
    if resume_from is not None:
        try:
            payload = load_warp_checkpoint(Path(resume_from))
            start_step = int(payload['metadata']['step']) + 1
            if context.rank == 0:
                print(f'[train] resuming GLOBAL step counter at {start_step} (from {resume_from})')
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if context.rank == 0:
                print(f'[train] could not read resume step (defaulting to 1): {exc!r}')
    for step in range(start_step, total_steps + 1):
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
        result = training_step_ray_warp(batch, components, flow_scheduler, context.device, target_view=target_view, front_condition_noise_prob=front_condition_noise_prob, front_condition_noise_min=front_condition_noise_min, front_condition_noise_max=front_condition_noise_max, neighbor_condition_noise_prob=neighbor_condition_noise_prob, neighbor_condition_noise_min=neighbor_condition_noise_min, neighbor_condition_noise_max=neighbor_condition_noise_max, rear_loss_weight=rear_loss_weight)
        if not torch.isfinite(result.loss.detach()).all():
            raise RuntimeError(f'non-finite loss at step {step}: {result.loss.detach().float().item()}')
        result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params(dit), config.optim.grad_clip, error_if_nonfinite=False)
        if not torch.isfinite(grad_norm.detach()).all():
            summary = _nonfinite_gradient_summary(dit)
            summary.update({'step': step, 'rank': context.rank, 'loss': float(result.loss.detach().float().item()), 'target_view': int(result.metrics['target_view']), 'grad_norm': float(grad_norm.detach().float().item())})
            with (output_dir / f'nonfinite_grad_rank{context.rank}.jsonl').open('a') as h:
                h.write(json.dumps(summary, sort_keys=True) + '\n')
            first_bad = summary['bad_params'][0]['name'] if summary['bad_params'] else '<unknown>'
            raise RuntimeError(f'non-finite gradients at step {step}; first_bad_param={first_bad}')
        optimizer.step()
        lr_scheduler.step()
        loss_mean_all_ranks = _mean_loss_for_logging_p61(result.loss)
        if context.device.type == 'cuda':
            torch.cuda.synchronize(context.device)
        step_time_sec = time.perf_counter() - step_start
        if context.rank == 0:
            with log_path.open('a') as h:
                ser = {k: float(v.item()) if isinstance(v, torch.Tensor) else v for (k, v) in result.metrics.items()}
                ser['loss_mean_all_ranks'] = loss_mean_all_ranks
                ser['grad_norm'] = float(grad_norm.detach().float().item())
                ser['flow_shift'] = float(flow_scheduler.shift)
                ser['new_module_optimizer'] = 'torch.optim.AdamW'
                ser['step'] = step
                ser['step_time_sec'] = step_time_sec
                h.write(json.dumps(ser, sort_keys=True) + '\n')
            if step % config.run.save_every == 0 or step == total_steps:
                if unfreeze_backbone:
                    save_warp_checkpoint_fulltune(components, optimizer, step, output_dir)
                else:
                    save_warp_checkpoint(components, optimizer, step, output_dir)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', choices=sorted(CONFIGS), default='openlongtail_1p3b')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume-from', type=Path, default=None)
    p.add_argument('--num-steps', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--warmup-steps', type=int, default=None)
    p.add_argument('--save-every', type=int, default=None)
    p.add_argument('--new-module-lr', type=float, default=None)
    p.add_argument('--lora-lr', type=float, default=None)
    p.add_argument('--grad-clip', type=float, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--latent-cache-root', type=Path, default=None)
    p.add_argument('--wan21-vace-dir', type=Path, default=None)
    p.add_argument('--front-condition-noise-prob', type=float, default=P61_FRONT_CONDITION_NOISE_PROB)
    p.add_argument('--front-condition-noise-min', type=float, default=P61_FRONT_CONDITION_NOISE_MIN)
    p.add_argument('--front-condition-noise-max', type=float, default=P61_FRONT_CONDITION_NOISE_MAX)
    p.add_argument('--neighbor-condition-noise-prob', type=float, default=P61_NEIGHBOR_CONDITION_NOISE_PROB)
    p.add_argument('--neighbor-condition-noise-min', type=float, default=P61_NEIGHBOR_CONDITION_NOISE_MIN)
    p.add_argument('--neighbor-condition-noise-max', type=float, default=P61_NEIGHBOR_CONDITION_NOISE_MAX)
    p.add_argument('--sync-temporal-window', type=int, default=P61_SYNC_TEMPORAL_WINDOW)
    p.add_argument('--condition-encoder-layers', type=int, default=P61_CONDITION_ENCODER_LAYERS)
    p.add_argument('--semantic-queries', type=int, default=P61_SEMANTIC_QUERIES)
    p.add_argument('--graph-gate-init-bias', type=float, default=GRAPH_GATE_INIT_BIAS)
    p.add_argument('--geo-head-dim', type=int, default=GEO_HEAD_DIM)
    p.add_argument('--geo-projection-temperature', type=float, default=GEO_PROJECTION_TEMPERATURE)
    p.add_argument('--rear-loss-weight', type=float, default=1.0)
    p.add_argument('--target-view-sequence', type=str, default=None, help=f'Comma-separated target views to cycle through, e.g. 1,2,3,4,5. Valid targets: {P61_TARGET_VIEWS}.')
    p.add_argument('--disable-motion-embedding', action='store_true')
    p.add_argument('--restrict-to-existing-sidecars', action='store_true', help='Only use clips whose warp sidecar (clip_<id>_warp.pt) is already on disk.')
    p.add_argument('--require-lookback-sidecar', action='store_true', help='Also load and merge the temporal-lookback adjacent sidecar clip_<id>_lookback.pt (visibility-max merge in the dataset).')
    p.add_argument('--max-items', type=int, default=None, help='Cap dataset to N clips (after sidecar filtering). For overfit experiments.')
    p.add_argument('--index-filename', type=str, default='index.jsonl', help='Index file name inside latent_cache_root. Use to point at a filtered training set (e.g. index_zzh_20260517_203414_min1m.jsonl) without overwriting the canonical index.')
    p.add_argument('--cross-view-blocks', type=str, default=None, help="Comma-separated block indices for cross-view / graph_memory insertion. Overrides config.model.cross_view_blocks. Use for 14B memory diagnostics, e.g. '19,39'.")
    p.add_argument('--unfreeze-backbone', action='store_true', help="FULLTUNE mode: do NOT inject LoRA and do NOT freeze the Wan2.1-VACE backbone. Train all backbone params at --backbone-lr alongside the new shared modules. Ckpt format becomes 'vace_fulltune' (backbone.pt + shared_modules.pt). 1.3B fits H200; 14B needs FSDP.")
    p.add_argument('--backbone-lr', type=float, default=1e-06, help='LR for the unfrozen backbone in --unfreeze-backbone mode. Defaults to 1e-6 (10x smaller than typical lora_lr=1e-5).')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]
    if args.num_workers is not None:
        config = replace(config, data=replace(config.data, num_workers=args.num_workers))
    if args.max_items is not None:
        config = replace(config, data=replace(config.data, max_items=int(args.max_items)))
    if args.warmup_steps is not None:
        config = replace(config, optim=replace(config.optim, warmup_steps=args.warmup_steps))
    if args.save_every is not None:
        config = replace(config, run=replace(config.run, save_every=args.save_every))
    if args.new_module_lr is not None or args.lora_lr is not None or args.grad_clip is not None:
        config = replace(config, optim=replace(config.optim, new_module_lr=config.optim.new_module_lr if args.new_module_lr is None else args.new_module_lr, lora_lr=config.optim.lora_lr if args.lora_lr is None else args.lora_lr, grad_clip=config.optim.grad_clip if args.grad_clip is None else args.grad_clip))
    if args.latent_cache_root is not None:
        config = replace(config, latent_cache=replace(config.latent_cache, use_latent_cache=True, latent_cache_root=args.latent_cache_root))
    if args.cross_view_blocks is not None:
        cvb = tuple((int(s) for s in args.cross_view_blocks.split(',') if s.strip()))
        config = replace(config, model=replace(config.model, cross_view_blocks=cvb))
    run_training(config, args.output_dir, resume_from=args.resume_from, num_steps=args.num_steps, device=args.device, wan21_vace_dir=args.wan21_vace_dir, front_condition_noise_prob=args.front_condition_noise_prob, front_condition_noise_min=args.front_condition_noise_min, front_condition_noise_max=args.front_condition_noise_max, neighbor_condition_noise_prob=args.neighbor_condition_noise_prob, neighbor_condition_noise_min=args.neighbor_condition_noise_min, neighbor_condition_noise_max=args.neighbor_condition_noise_max, sync_temporal_window=args.sync_temporal_window, condition_encoder_layers=args.condition_encoder_layers, semantic_queries=args.semantic_queries, enable_motion_embedding=not args.disable_motion_embedding, target_view_sequence=parse_target_view_sequence(args.target_view_sequence), graph_gate_init_bias=args.graph_gate_init_bias, geo_head_dim=args.geo_head_dim, geo_projection_temperature=args.geo_projection_temperature, rear_loss_weight=args.rear_loss_weight, restrict_to_existing_sidecars=args.restrict_to_existing_sidecars, require_lookback_sidecar=args.require_lookback_sidecar, index_filename=args.index_filename, unfreeze_backbone=args.unfreeze_backbone, backbone_lr=args.backbone_lr)
if __name__ == '__main__':
    main()
