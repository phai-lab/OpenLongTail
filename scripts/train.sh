#!/usr/bin/env bash
# ===========================================================================
# Single-node (multi-GPU) training launcher
# ===========================================================================
# Trains the multi-view driving-video model with torchrun on one node with
# NUM_GPUS GPUs. For multi-node SLURM training use scripts/train.slurm.
#
# Quick start:
#   ./train.sh
#
# Override anything via the environment, e.g.:
#   NUM_GPUS=4 NUM_STEPS=5000 OUTPUT_DIR=outputs/my_run ./train.sh
# ===========================================================================
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
source "${HERE}/common.sh"

# ---- launch / data -----------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-openlongtail_1p3b}"                 # *_1p3b or *_14b
LATENT_CACHE_ROOT="${LATENT_CACHE_ROOT:-${LONGTAIL_ROOT}/openlongtail_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/openlongtail_train}"
export OPENLONGTAIL_TEXT_EMB_ROOT="${OPENLONGTAIL_TEXT_EMB_ROOT:-${LATENT_CACHE_ROOT}/text_cache}"

# ---- schedule ----------------------------------------------------------------
NUM_STEPS="${NUM_STEPS:-20000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-1}"

# ---- conditioning / loss -----------------------------------------------------
TARGET_VIEW_SEQUENCE="${TARGET_VIEW_SEQUENCE:-1,2,3,4,5}"   # all 5 target cameras
FRONT_NOISE_PROB="${FRONT_NOISE_PROB:-0.05}"
NEIGHBOR_NOISE_PROB="${NEIGHBOR_NOISE_PROB:-0.3}"
SEMANTIC_QUERIES="${SEMANTIC_QUERIES:-64}"
GRAPH_GATE_INIT_BIAS="${GRAPH_GATE_INIT_BIAS:--1.4}"        # sigmoid(-1.4)=0.197
GEO_HEAD_DIM="${GEO_HEAD_DIM:-16}"
GEO_PROJECTION_TEMPERATURE="${GEO_PROJECTION_TEMPERATURE:-4.0}"
REAR_LOSS_WEIGHT="${REAR_LOSS_WEIGHT:-0.5}"

# ---- optimization ------------------------------------------------------------
NEW_MODULE_LR="${NEW_MODULE_LR:-1e-5}"
LORA_LR="${LORA_LR:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_DIR}" logs

# ---- auto-resume from the latest checkpoint in OUTPUT_DIR ---------------------
RESUME_ARGS=()
LATEST_CKPT="$(ls -d "${OUTPUT_DIR}"/step_* 2>/dev/null | sort | tail -1 || true)"
if [[ -n "${LATEST_CKPT}" && -d "${LATEST_CKPT}" ]]; then
  RESUME_ARGS+=(--resume-from "${LATEST_CKPT}")
  echo "[train] auto-resume from ${LATEST_CKPT}"
else
  echo "[train] fresh start (no step_* under ${OUTPUT_DIR})"
fi

echo "[train] config=${CONFIG}  gpus=${NUM_GPUS}  steps=${NUM_STEPS}"
echo "[train] latent_cache=${LATENT_CACHE_ROOT}"
echo "[train] text_emb=${OPENLONGTAIL_TEXT_EMB_ROOT}"
echo "[train] output=${OUTPUT_DIR}"

# torchrun via the venv interpreter so we never depend on PATH.
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
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
