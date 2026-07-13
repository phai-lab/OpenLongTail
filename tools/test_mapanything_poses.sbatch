#!/bin/bash
#SBATCH --job-name=mapanything_val
#SBATCH --qos=standard
#SBATCH --gres=gpu:H200:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=${REPO_ROOT}/P65V1/tools/_slurm_%j.out
#SBATCH --error=${REPO_ROOT}/P65V1/tools/_slurm_%j.err

set -eo pipefail

BASE=${LONGTAIL_ROOT}
TOOLS=$BASE/Ray_WAN/P65V1/tools

# ---- caches on SCRATCH only, never $HOME ----
export HF_HOME=$BASE/.hf_cache
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN to your Hugging Face token}
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TORCH_HOME=$BASE/.torch_cache
export HF_HUB_OFFLINE=1          # all weights (map-anything + dinov2 vitg14) pre-cached
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=$BASE/mapanything_venv/bin/python

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -c "import torch;print('torch',torch.__version__,'cuda avail',torch.cuda.is_available())"

FRAMES=$TOOLS/_nexar_val_frames
OUT=$TOOLS/_nexar_val_poses.pt

echo "=== RUN run_mapanything_poses.py ==="
$PY $TOOLS/run_mapanything_poses.py --images $FRAMES --output $OUT

echo "=== ANALYZE ==="
$PY $TOOLS/_analyze_val.py $OUT

echo "=== DONE ==="
