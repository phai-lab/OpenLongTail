#!/usr/bin/env bash
# ===========================================================================
# Warp-conditioning sidecar builder (offline, one-time data prep)
# ===========================================================================
# Builds the per-clip conditioning sidecars on top of an existing Wan-VAE
# latent cache. Both sidecar files must exist before training.
#
# The released training-ready caches already ship with these sidecars; you
# only need this to build a fresh cache.
#
# Usage (single GPU, one shard of NUM_SHARDS):
#   NUM_SHARDS=8 SHARD_INDEX=0 ./scripts/prepare_data.sh
#   # ...run SHARD_INDEX=0..7 in parallel (e.g. one per GPU / array task).
# ===========================================================================
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
source "${HERE}/common.sh"

LATENT_CACHE_ROOT="${LATENT_CACHE_ROOT:-${LONGTAIL_ROOT}/openlongtail_cache}"
DATA_ROOT="${DATA_ROOT:-${LONGTAIL_ROOT}/data_ft/by_uuid}"
VAE_PATH="${VAE_PATH:-${OPENLONGTAIL_VAE_PATH}}"
SPLAT_RADIUS="${SPLAT_RADIUS:-1}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
OVERWRITE="${OVERWRITE:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/outputs/sidecar_manifest}"
mkdir -p "${MANIFEST_DIR}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then EXTRA_ARGS+=(--overwrite); fi

echo "[prepare] latent_cache=${LATENT_CACHE_ROOT}"
echo "[prepare] data_root=${DATA_ROOT}"
echo "[prepare] shard ${SHARD_INDEX} of ${NUM_SHARDS}  splat_radius=${SPLAT_RADIUS}"

echo "[prepare] (1/2) building side-view conditioning ..."
"${PYTHON}" -m openlongtail.scripts.build_warp \
  --latent-cache-root "${LATENT_CACHE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --vae-path "${VAE_PATH}" \
  --splat-radius "${SPLAT_RADIUS}" \
  --uuid-stride "${NUM_SHARDS}" \
  --uuid-offset "${SHARD_INDEX}" \
  --max-clips -1 \
  --max-uuids -1 \
  --manifest "${MANIFEST_DIR}/v0_shard_${SHARD_INDEX}.jsonl" \
  "${EXTRA_ARGS[@]}"

echo "[prepare] (2/2) building rear-view conditioning ..."
"${PYTHON}" -m openlongtail.scripts.build_lookback_warp \
  --latent-cache-root "${LATENT_CACHE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --vae-path "${VAE_PATH}" \
  --splat-radius "${SPLAT_RADIUS}" \
  --vae-batch-size "${VAE_BATCH_SIZE}" \
  --uuid-stride "${NUM_SHARDS}" \
  --uuid-offset "${SHARD_INDEX}" \
  --max-clips -1 \
  --max-uuids -1 \
  --manifest "${MANIFEST_DIR}/v1_shard_${SHARD_INDEX}.jsonl" \
  "${EXTRA_ARGS[@]}"

echo "[prepare] shard ${SHARD_INDEX} done."
