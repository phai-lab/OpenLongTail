from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
LONGTAIL_ROOT = REPO_ROOT.parent

@dataclass(frozen=True)
class MultiViewModelConfig:
    resolution: tuple[int, int] = (480, 832)
    fps: int = 16
    clip_length: int = 81
    warmup_clip_length: int = 33
    latent_channels: int = 16
    latent_frames: int = 21
    warmup_latent_frames: int = 9
    latent_hw: tuple[int, int] = (60, 104)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    vae_stride: tuple[int, int, int] = (4, 8, 8)
    token_grid: tuple[int, int, int] = (21, 30, 52)
    warmup_token_grid: tuple[int, int, int] = (9, 30, 52)
    num_views: int = 6
    front_idx: int = 0
    wan_dim: int = 5120
    wan_num_heads: int = 40
    wan_num_layers: int = 40
    wan_ffn_dim: int = 13824
    text_length: int = 256
    text_dim: int = 4096
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_targets: tuple[str, ...] = ('self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o')
    cross_view_blocks: tuple[int, ...] = (7, 15, 23, 31)
    cross_view_dim_attn: int = 2048
    cross_view_heads: int = 16
    cross_view_edges: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 3), (3, 5), (5, 4), (4, 2))
    plucker_in_dim: int = 6
    plucker_hidden_dim: int = 256
    cam_id_init_std: float = 0.02
    module_init_std: float = 0.02
    sigma_shift: float = 5.0
    moe_boundary_sigma: float = 0.9
    moe_boundary_t: float = 0.643
    stage_a_t_min: float = 0.0
    stage_a_t_max: float = 0.643
    stage_b_t_min: float = 0.643
    stage_b_t_max: float = 1.0
    text_drop_prob: float = 0.1
    sample_steps: int = 40
    sample_guide_scale_low: float = 3.5
    sample_guide_scale_high: float = 3.5
    sample_shift: float = 5.0
    view_dropout_active_targets: int = 5

    def __post_init__(self) -> None:
        if self.view_dropout_active_targets not in (3, 5):
            raise ValueError(f'view_dropout_active_targets must be 3 or 5 in V1.1, got {self.view_dropout_active_targets}')

@dataclass(frozen=True)
class MultiViewOptimConfig:
    new_module_lr: float = 0.0001
    lora_lr: float = 2e-05
    warmup_steps: int = 2000
    cosine_min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    new_module_weight_decay: float = 0.1
    lora_weight_decay: float = 0.0
    new_module_optimizer: str = 'bnb.optim.AdamW8bit'
    lora_optimizer: str = 'torch.optim.AdamW'

@dataclass(frozen=True)
class DistributedConfig:
    find_unused_parameters: bool = True
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = False
    static_graph: bool = False
    nccl_debug: str = 'INFO'
    nccl_debug_subsys: str = 'COLL,INIT'
    nccl_async_error_handling: str = '1'
    torch_nccl_async_error_handling: str = '1'
    torch_distributed_debug: str = 'DETAIL'
    torch_nccl_trace_buffer_size: str = '2000'
    torch_nccl_dump_on_timeout: str = '1'
    pytorch_cuda_alloc_conf: str = 'expandable_segments:True'
    python_unbuffered: str = '1'

@dataclass(frozen=True)
class DataConfig:
    data_root: Path = Path(os.environ.get('OPENLONGTAIL_DATA_ROOT', str(LONGTAIL_ROOT / 'data_ft' / 'by_uuid')))
    text_emb_cache_root: Path = Path(os.environ.get('OPENLONGTAIL_TEXT_EMB_ROOT', str(LONGTAIL_ROOT / 'BEV_WAN' / 'text_cache')))
    uuid_allowlist_json: Path | None = None
    clip_length: int = 81
    output_size: tuple[int, int] = (480, 832)
    target_fps: int = 16
    clip_anchor_seconds: float = 10.0
    clip_jitter_seconds: float = 2.0
    use_undistorted_simplecalib: bool = True
    use_offline_extrinsics: bool = True
    text_drop_prob: float = 0.1
    max_items: int | None = None
    num_workers: int = 4

def _parse_cache_versions_env() -> tuple[str, ...]:
    raw = os.environ.get('OPENLONGTAIL_CACHE_VERSIONS', '')
    return tuple((v.strip() for v in raw.split(',') if v.strip()))

@dataclass(frozen=True)
class LatentCacheConfig:
    use_latent_cache: bool = False
    latent_cache_root: Path | None = None
    cache_version: str = os.environ.get('OPENLONGTAIL_CACHE_VERSION', 'latent_t41_stride4_v1')
    cache_versions: tuple[str, ...] = field(default_factory=_parse_cache_versions_env)
    clip_length: int = 41
    front_stride: int = 4
    clips_per_uuid: int = 3

@dataclass(frozen=True)
class RunConfig:
    batch_size_per_gpu: int = 1
    stage_a0_steps: int = 5000
    stage_a1_steps: int = 20000
    stage_b_steps: int = 5000
    save_every: int = 2000
    precision: str = 'bf16'
    compute_aux_steps: tuple[str, ...] = ('step_1', 'save_every', 'last_step')
    train_log_name: str = 'train_log.jsonl'

@dataclass(frozen=True)
class InferenceConfig:
    num_steps: int = 40
    guide_scale: float = 3.5
    low_guide_scale: float = 3.5
    high_guide_scale: float = 3.5

@dataclass(frozen=True)
class CheckpointConfig:
    vae_path: Path = Path(os.environ.get('OPENLONGTAIL_VAE_PATH', str(LONGTAIL_ROOT / 'BEV_WAN' / 'checkpoints' / 'Wan2.1-T2V-14B' / 'Wan2.1_VAE.pth')))
    umt5_path: Path = Path(os.environ.get('OPENLONGTAIL_UMT5_PATH', str(LONGTAIL_ROOT / 'BEV_WAN' / 'checkpoints' / 'Wan2.1-T2V-14B' / 'models_t5_umt5-xxl-enc-bf16.pth')))
    umt5_tokenizer: Path = Path(os.environ.get('OPENLONGTAIL_UMT5_TOKENIZER', str(LONGTAIL_ROOT / 'BEV_WAN' / 'checkpoints' / 'Wan2.1-T2V-14B' / 'google' / 'umt5-xxl')))
    wan22_low_dir: Path = Path(os.environ.get('OPENLONGTAIL_WAN22_LOW_DIR', str(REPO_ROOT / 'checkpoints' / 'Wan2.2-I2V-A14B' / 'low_noise_model')))
    wan22_high_dir: Path = Path(os.environ.get('OPENLONGTAIL_WAN22_HIGH_DIR', str(REPO_ROOT / 'checkpoints' / 'Wan2.2-I2V-A14B' / 'high_noise_model')))

@dataclass(frozen=True)
class TrainConfig:
    model: MultiViewModelConfig = field(default_factory=MultiViewModelConfig)
    optim: MultiViewOptimConfig = field(default_factory=MultiViewOptimConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    latent_cache: LatentCacheConfig = field(default_factory=LatentCacheConfig)
    run: RunConfig = field(default_factory=RunConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
