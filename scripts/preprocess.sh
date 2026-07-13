#!/usr/bin/env bash
# ===========================================================================
# OpenLongTail — data preprocessing
# ---------------------------------------------------------------------------
# Turn raw front-camera clips into a ready-to-train multi-view cache. Two
# stages:
#   1. Metric pose recovery  — estimate a metric-scale ego-trajectory for each
#      clip (skip with --have-poses if your source already ships poses).
#   2. Multi-view conditioning — reproject the front-view evidence into every
#      target camera and cache it alongside the latents.
#
#   bash preprocess.sh --input <RAW_CLIPS> --output <CACHE_DIR>
#   # shard across GPUs:  --shards N --shard i   (run i = 0..N-1 in parallel)
# ===========================================================================
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
source "${HERE}/common.sh"

INPUT=""; OUTPUT="data/openlongtail_cache"; SHARDS=1; SHARD=0; HAVE_POSES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)   INPUT="$2";  shift 2 ;;
    --output)  OUTPUT="$2"; shift 2 ;;
    --shards)  SHARDS="$2"; shift 2 ;;
    --shard)   SHARD="$2";  shift 2 ;;
    --have-poses) HAVE_POSES=1; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done
[[ -z "${INPUT}" ]] && { echo "usage: bash preprocess.sh --input <raw_clips> --output <cache_dir>"; exit 1; }
mkdir -p "${OUTPUT}"

# Stage 1 — metric pose recovery (MapAnything) for in-the-wild monocular clips.
if [[ "${HAVE_POSES}" != "1" ]]; then
  echo "[preprocess] (1/2) recovering metric camera pose ..."
  "${PYTHON}" "${REPO_ROOT}/tools/run_mapanything_poses.py" \
    --images "${INPUT}" --output "${OUTPUT}/poses.pt"
fi

# Stage 2 — multi-view warp conditioning cache.
echo "[preprocess] (2/2) building multi-view conditioning cache ..."
LATENT_CACHE_ROOT="${OUTPUT}" DATA_ROOT="${INPUT}" \
NUM_SHARDS="${SHARDS}" SHARD_INDEX="${SHARD}" \
  bash "${HERE}/prepare_data.sh"

echo "[OpenLongTail] preprocessing done -> ${OUTPUT}"
