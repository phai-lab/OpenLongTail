from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.distributed as dist
VIEW_PARALLEL_SENSOR_ORDER: tuple[str, ...] = ('front_wide', 'cross_left', 'cross_right', 'rear_left', 'rear_right', 'rear_tele')

@dataclass(frozen=True)
class ViewParallelContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    views_per_node: int
    view_id: int
    view_name: str
    data_parallel_rank: int
    data_parallel_size: int
    node_ranks: tuple[int, ...]
    group: object | None = None

def build_view_parallel_context(rank: int, world_size: int, local_rank: int, views_per_node: int=6) -> ViewParallelContext:
    if views_per_node != len(VIEW_PARALLEL_SENSOR_ORDER):
        raise ValueError(f'expected views_per_node=6, got {views_per_node}')
    if world_size % views_per_node != 0:
        raise ValueError(f'expected world_size divisible by {views_per_node}, got {world_size}')
    if local_rank < 0 or local_rank >= views_per_node:
        raise ValueError(f'expected local_rank in [0, {views_per_node - 1}] for view-parallel, got {local_rank}')
    if rank < 0 or rank >= world_size:
        raise ValueError(f'expected rank in [0, {world_size - 1}], got {rank}')
    data_parallel_rank = rank // views_per_node
    data_parallel_size = world_size // views_per_node
    node_start = data_parallel_rank * views_per_node
    node_ranks = tuple(range(node_start, node_start + views_per_node))
    view_id = local_rank
    return ViewParallelContext(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank, views_per_node=views_per_node, view_id=view_id, view_name=VIEW_PARALLEL_SENSOR_ORDER[view_id], data_parallel_rank=data_parallel_rank, data_parallel_size=data_parallel_size, node_ranks=node_ranks)

def create_node_view_group(context: ViewParallelContext) -> ViewParallelContext:
    if not dist.is_available() or not dist.is_initialized():
        return context
    group = None
    for start in range(0, context.world_size, context.views_per_node):
        ranks = list(range(start, start + context.views_per_node))
        candidate = dist.new_group(ranks=ranks)
        if context.rank in ranks:
            group = candidate
    if group is None:
        raise ValueError(f'rank {context.rank} did not match any view-parallel node group')
    return ViewParallelContext(enabled=context.enabled, rank=context.rank, world_size=context.world_size, local_rank=context.local_rank, views_per_node=context.views_per_node, view_id=context.view_id, view_name=context.view_name, data_parallel_rank=context.data_parallel_rank, data_parallel_size=context.data_parallel_size, node_ranks=context.node_ranks, group=group)

def gather_view_tensor(local: torch.Tensor, view_id: int, group: object | None, views_per_node: int=6) -> torch.Tensor:
    if view_id < 0 or view_id >= views_per_node:
        raise ValueError(f'expected view_id in [0, {views_per_node - 1}], got {view_id}')
    local_for_gather = local.detach().contiguous()
    if group is not None and dist.is_available() and dist.is_initialized():
        gathered = [torch.empty_like(local_for_gather) for _ in range(views_per_node)]
        dist.all_gather(gathered, local_for_gather, group=group)
    else:
        gathered = [torch.zeros_like(local_for_gather) for _ in range(views_per_node)]
    gathered[view_id] = local
    return torch.stack(gathered, dim=1)

def broadcast_sigma_within_node(sigma: torch.Tensor, context: ViewParallelContext) -> torch.Tensor:
    if context.group is not None and dist.is_available() and dist.is_initialized():
        dist.broadcast(sigma, src=context.node_ranks[0], group=context.group)
    return sigma
