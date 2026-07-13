#!/bin/bash
# ===========================================================================
# Multi-node SLURM training launcher
# ===========================================================================
# Self-requeue safe: auto-resumes from the latest step_* checkpoint under
# OUTPUT_DIR. The trainer reads RANK/WORLD_SIZE/LOCAL_RANK and falls back to
# SLURM_PROCID/NTASKS/LOCALID, so `srun` wires up DDP with no torchrun needed.
#
# Submit:
#   sbatch scripts/train.slurm
# ===========================================================================
#SBATCH --job-name=openlongtail-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --qos=high
#SBATCH --requeue
#SBATCH --exclude=dgx002
#SBATCH --output=logs/openlongtail_train_%j.out
#SBATCH --error=logs/openlongtail_train_%j.err

set -euo pipefail

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OPENLONGTAIL_NODE_BROADCAST_LOAD=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Resolve repo root from this script's location, then load shared env.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd )"
source "${SCRIPT_DIR}/common.sh"
mkdir -p logs outputs

# --- data / output (override via --export) ------------------------------------
CONFIG="${CONFIG:-openlongtail_1p3b}"
LATENT_CACHE_ROOT="${LATENT_CACHE_ROOT:-${LONGTAIL_ROOT}/openlongtail_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/openlongtail_train}"
export OPENLONGTAIL_TEXT_EMB_ROOT="${OPENLONGTAIL_TEXT_EMB_ROOT:-${LATENT_CACHE_ROOT}/text_cache}"

NUM_STEPS="${NUM_STEPS:-40000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TARGET_VIEW_SEQUENCE="${TARGET_VIEW_SEQUENCE:-1,2,3,4,5}"
FRONT_NOISE_PROB="${FRONT_NOISE_PROB:-0.05}"
NEIGHBOR_NOISE_PROB="${NEIGHBOR_NOISE_PROB:-0.3}"
SEMANTIC_QUERIES="${SEMANTIC_QUERIES:-64}"
GRAPH_GATE_INIT_BIAS="${GRAPH_GATE_INIT_BIAS:--1.4}"
GEO_HEAD_DIM="${GEO_HEAD_DIM:-16}"
GEO_PROJECTION_TEMPERATURE="${GEO_PROJECTION_TEMPERATURE:-4.0}"
NEW_MODULE_LR="${NEW_MODULE_LR:-1e-5}"
LORA_LR="${LORA_LR:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
REAR_LOSS_WEIGHT="${REAR_LOSS_WEIGHT:-0.5}"

mkdir -p "${OUTPUT_DIR}"

# --- auto-resume from latest step_* checkpoint --------------------------------
RESUME_ARGS=()
LATEST_CKPT="$(ls -d "${OUTPUT_DIR}"/step_* 2>/dev/null | sort | tail -1 || true)"
if [[ -n "${LATEST_CKPT}" && -d "${LATEST_CKPT}" ]]; then
  RESUME_ARGS+=(--resume-from "${LATEST_CKPT}")
  echo "AUTO-RESUME from ${LATEST_CKPT}"
else
  echo "FRESH start (no step_* under ${OUTPUT_DIR})"
fi

echo "OpenLongTail train — 32 GPU x ${NUM_STEPS} steps  config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}  latent_cache=${LATENT_CACHE_ROOT}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"

srun "${PYTHON}" \
  -m openlongtail.training.train \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --latent-cache-root "${LATENT_CACHE_ROOT}" \
  --wan21-vace-dir "${WAN21_VACE_DIR}" \
  "${RESUME_ARGS[@]}" \
  --num-steps "${NUM_STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --new-module-lr "${NEW_MODULE_LR}" \
  --lora-lr "${LORA_LR}" \
  --grad-clip "${GRAD_CLIP}" \
  --num-workers "${NUM_WORKERS}" \
  --front-condition-noise-prob "${FRONT_NOISE_PROB}" \
  --neighbor-condition-noise-prob "${NEIGHBOR_NOISE_PROB}" \
  --semantic-queries "${SEMANTIC_QUERIES}" \
  --graph-gate-init-bias "${GRAPH_GATE_INIT_BIAS}" \
  --geo-head-dim "${GEO_HEAD_DIM}" \
  --geo-projection-temperature "${GEO_PROJECTION_TEMPERATURE}" \
  --rear-loss-weight "${REAR_LOSS_WEIGHT}" \
  --target-view-sequence "${TARGET_VIEW_SEQUENCE}" \
  --restrict-to-existing-sidecars \
  --require-lookback-sidecar
