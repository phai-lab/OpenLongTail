from __future__ import annotations
import os
import socket
import subprocess
from dataclasses import dataclass
from typing import Iterable
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from openlongtail.configs.default import DistributedConfig

@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    is_distributed: bool

@dataclass(frozen=True)
class NodeLocalContext:
    rank: int
    node_ranks: tuple[int, ...]
    local_src_rank: int
    group: dist.ProcessGroup | None

    @property
    def is_local_src(self) -> bool:
        return self.rank == self.local_src_rank

    @property
    def world_size(self) -> int:
        return len(self.node_ranks)

def apply_nccl_env(config: DistributedConfig | None=None) -> None:
    cfg = config or DistributedConfig()
    env = {'NCCL_DEBUG': cfg.nccl_debug, 'NCCL_DEBUG_SUBSYS': cfg.nccl_debug_subsys, 'NCCL_ASYNC_ERROR_HANDLING': cfg.nccl_async_error_handling, 'TORCH_NCCL_ASYNC_ERROR_HANDLING': cfg.torch_nccl_async_error_handling, 'TORCH_DISTRIBUTED_DEBUG': cfg.torch_distributed_debug, 'TORCH_NCCL_TRACE_BUFFER_SIZE': cfg.torch_nccl_trace_buffer_size, 'TORCH_NCCL_DUMP_ON_TIMEOUT': cfg.torch_nccl_dump_on_timeout, 'PYTORCH_CUDA_ALLOC_CONF': cfg.pytorch_cuda_alloc_conf, 'PYTHONUNBUFFERED': cfg.python_unbuffered}
    for (key, value) in env.items():
        os.environ.setdefault(key, value)

def default_master_port() -> str:
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id is not None and job_id.isdigit():
        return str(10000 + int(job_id) % 50000)
    return '29500'

def ensure_master_port() -> None:
    os.environ.setdefault('MASTER_PORT', default_master_port())

def setup_distributed(device: str | torch.device | None=None) -> DistributedContext:
    apply_nccl_env()
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    is_distributed = world_size > 1
    if is_distributed:
        os.environ.setdefault('RANK', str(rank))
        os.environ.setdefault('WORLD_SIZE', str(world_size))
        os.environ.setdefault('LOCAL_RANK', str(local_rank))
        if 'MASTER_ADDR' not in os.environ:
            nodelist = os.environ.get('SLURM_JOB_NODELIST')
            if nodelist:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', nodelist], text=True)
                os.environ['MASTER_ADDR'] = hostnames.splitlines()[0]
            else:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
        ensure_master_port()
    if device is None:
        resolved = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    else:
        resolved = torch.device(device)
    if resolved.type == 'cuda':
        torch.cuda.set_device(resolved)
    if is_distributed and (not dist.is_initialized()):
        backend = 'nccl' if resolved.type == 'cuda' else 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank, device=resolved, is_distributed=is_distributed)

def ddp_kwargs(local_rank: int) -> dict[str, object]:
    return {'device_ids': [local_rank], 'output_device': local_rank, 'broadcast_buffers': False, 'find_unused_parameters': True, 'gradient_as_bucket_view': True, 'static_graph': False}

def wrap_ddp(module: nn.Module, local_rank: int) -> DistributedDataParallel:
    return DistributedDataParallel(module, **ddp_kwargs(local_rank))

def build_node_local_context() -> NodeLocalContext | None:
    if not dist.is_available() or not dist.is_initialized():
        return None
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()
    hostnames: list[str | None] = [None for _ in range(world_size)]
    dist.all_gather_object(hostnames, hostname)
    groups: list[tuple[int, ...]] = []
    seen: set[str] = set()
    for node_name in hostnames:
        if node_name is None or node_name in seen:
            continue
        seen.add(node_name)
        groups.append(tuple((idx for (idx, item) in enumerate(hostnames) if item == node_name)))
    current_group: dist.ProcessGroup | None = None
    current_ranks: tuple[int, ...] | None = None
    for node_ranks in groups:
        group = dist.new_group(ranks=list(node_ranks))
        if rank in node_ranks:
            current_group = group
            current_ranks = node_ranks
    if current_ranks is None:
        raise RuntimeError(f'rank {rank} did not match any node-local group from hosts {hostnames}')
    return NodeLocalContext(rank=rank, node_ranks=current_ranks, local_src_rank=current_ranks[0], group=current_group)

@torch.no_grad()
def broadcast_module_state_dict_from_node_src(module: nn.Module, context: NodeLocalContext | None) -> None:
    if context is None or context.world_size == 1:
        return
    if context.group is None:
        raise RuntimeError('expected node-local process group for module broadcast')
    tensors = list(module.named_parameters()) + list(module.named_buffers())
    for (name, tensor) in tensors:
        if not torch.is_tensor(tensor):
            raise TypeError(f'expected tensor module entry for {name}, got {type(tensor).__name__}')
        dist.broadcast(tensor, src=context.local_src_rank, group=context.group)

@torch.no_grad()
def broadcast_tensor_from_node_src(tensor: torch.Tensor, context: NodeLocalContext | None) -> torch.Tensor:
    if context is None or context.world_size == 1:
        return tensor
    if context.group is None:
        raise RuntimeError('expected node-local process group for tensor broadcast')
    dist.broadcast(tensor, src=context.local_src_rank, group=context.group)
    return tensor

def _barrier(local_rank: int | None=None) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device() if local_rank is None else local_rank
        dist.barrier(device_ids=[device_id])
    else:
        dist.barrier()

def all_gather_floats(values: dict[str, float] | Iterable[float]) -> list[list[float]]:
    if isinstance(values, dict):
        ordered = [float(values[key]) for key in sorted(values)]
    else:
        ordered = [float(value) for value in values]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.tensor(ordered, dtype=torch.float32, device=device)
    if not dist.is_available() or not dist.is_initialized():
        return [tensor.cpu().tolist()]
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [item.cpu().tolist() for item in gathered]

def grad_norm(parameters: Iterable[nn.Parameter], norm_type: float=2.0) -> float:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(g.to(device), ord=norm_type) for g in grads])
    total = torch.linalg.vector_norm(norms, ord=norm_type)
    return float(total.item())
