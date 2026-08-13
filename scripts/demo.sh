#!/usr/bin/env bash
# ============================================================================
# OpenLongTail — ONE-CLICK DEMO.
#
# From a single front-camera driving video, generate the 5 surrounding rig
# cameras (2 cross + 3 rear), then build a side-by-side PRED-vs-GT comparison.
#
#   bash scripts/demo.sh                      # cached tier: the bundled demo clips
#   bash scripts/demo.sh --gpus 0,1,2         # parallel: one clip per GPU
#   bash scripts/demo.sh --raw my_front.mp4   # raw tier: your own dashcam video
#   bash scripts/demo.sh --gpu 3              # pick a single CUDA device (default 0)
#
# CACHED tier (default): runs the bundled demo clips from their
#   pre-built latent cache (bundled in demo/cached). No depth/pose models
#   needed. Bit-exact to the released demos. ~11 min/clip on one H200 (serial),
#   or use --gpus 0,1,2 (clips shard one-per-GPU).
#   Each clip has real ground-truth side/rear cameras -> full PRED-vs-GT viz.
#
# RAW tier (--raw): the true end-to-end path for YOUR video —
#   DepthAnythingV2 (depth) + MapAnything (ego-pose) -> generate. No GT, so the
#   output is the 5 generated views + a preview grid.
#
# Outputs -> demo/outputs/<name>/  (5 view mp4s, preview grid, comparison mp4,
#   and an index.html gallery).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source "$HERE/scripts/env.sh"

MODE=cached RAW_VIDEO="" GPU=0 GPUS=""
while [ $# -gt 0 ]; do case "$1" in
  --raw)  MODE=raw; RAW_VIDEO="${2:?--raw needs a video path}"; shift 2 ;;
  --gpu)  GPU="${2:?}"; shift 2 ;;
  --gpus) GPUS="${2:?--gpus needs a comma list, e.g. 0,1,2}"; shift 2 ;;
  -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1"; exit 2 ;;
esac; done

PY="${OLT_PYTHON:-python}"
OUT="$HERE/demo/outputs"; mkdir -p "$OUT"

# ---- preflight: weights present? ----
if [ ! -f "$OLT_WM_DIR/checkpoint/shared_modules.pt" ] || [ ! -f "$OPENLONGTAIL_VAE_PATH" ]; then
  echo "Weights not found. Run:  bash scripts/download.sh"; exit 1
fi

if [ "$MODE" = cached ]; then
  RENDERS="$OUT/cached_renders"

  # one sharded worker per GPU: clip i -> GPU (i mod N). inference_cached shards
  # by clips[shard_index::num_shards], so N shards spread the clips one per GPU.
  run_shard() {  # $1=gpu  $2=shard_index  $3=num_shards
    CUDA_VISIBLE_DEVICES="$1" $PY -m openlongtail.scripts.inference_cached \
      --v3-root        "$HERE/demo/cached" \
      --output-dir     "$RENDERS" \
      --checkpoint-dir "$OLT_WM_DIR/checkpoint" \
      --config         openlongtail_1p3b \
      --wan21-vace-dir "$OLT_WAN_VACE_DIR" \
      --first-clip-per-uuid --max-clips -1 \
      --num-shards "$3" --shard-index "$2" \
      $OLT_RECIPE --device cuda
  }

  if [ -n "$GPUS" ]; then
    IFS=',' read -ra GPU_ARR <<< "$GPUS"
    N="${#GPU_ARR[@]}"
    echo "############################################################"
    echo "#  OpenLongTail demo — CACHED tier, $N-GPU parallel"
    echo "#  GPUs=$GPUS  recipe: $OLT_RECIPE"
    echo "############################################################"
    pids=()
    for i in "${!GPU_ARR[@]}"; do
      g="${GPU_ARR[$i]}"
      echo "  -> shard $i/$N on GPU $g  (log: $RENDERS/shard_$i.log)"
      mkdir -p "$RENDERS"
      run_shard "$g" "$i" "$N" > "$RENDERS/shard_$i.log" 2>&1 &
      pids+=("$!")
    done
    fail=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then echo "  shard $i done"; else echo "  !! shard $i FAILED (see $RENDERS/shard_$i.log)"; fail=1; fi
    done
    [ "$fail" = 0 ] || { echo "one or more shards failed; not all clips rendered."; exit 1; }
  else
    echo "############################################################"
    echo "#  OpenLongTail demo — CACHED tier (bundled clips)"
    echo "#  GPU=$GPU  recipe: $OLT_RECIPE  (use --gpus 0,1,2 to parallelize)"
    echo "############################################################"
    run_shard "$GPU" 0 1
  fi

  echo "== building PRED-vs-GT comparison gallery =="
  $PY "$HERE/tools/make_demo_viz.py" \
    --renders "$RENDERS" --gt "$HERE/demo/gt" \
    --out "$OUT/gallery" --with-gt
  echo
  echo "DONE.  Open:  $OUT/gallery/index.html"

else
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "############################################################"
  echo "#  OpenLongTail demo — RAW tier (your own front video)"
  echo "#  input: $RAW_VIDEO   GPU=$GPU"
  echo "############################################################"
  [ -f "$RAW_VIDEO" ] || { echo "no such file: $RAW_VIDEO"; exit 1; }
  NAME="$(basename "${RAW_VIDEO%.*}")"
  WORK="$OUT/$NAME"; TD="$WORK/testdata"; mkdir -p "$WORK/raw"
  cp "$RAW_VIDEO" "$WORK/raw/$NAME.mp4"

  echo "== [1/4] depth (DepthAnythingV2) + frame dump =="
  $PY "$HERE/tools/nexar_to_testdata.py" \
    --nexar-root "$WORK/raw" --out-root "$TD" --num-clips 1 \
    --dump-frames --pose-mode mapanything --device cuda

  CLIP="$TD/chunk_900/nexar_${NAME}/clip_000000"
  echo "== [2/4] ego-pose (MapAnything) =="
  ( cd "${OLT_MAPANYTHING_REPO:-$HERE/third_party/mapanything}" 2>/dev/null || cd "$HERE"
    "${OLT_MAPANYTHING_PYTHON:-$PY}" "$HERE/tools/run_mapanything_poses.py" \
      --images "$CLIP/frames" --output "$CLIP/mapanything_raw.pt" --amp_dtype bf16 )
  $PY "$HERE/tools/pose_from_mapanything.py" \
    --mapanything-pt "$CLIP/mapanything_raw.pt" --out "$CLIP/pose.pt"

  echo "== [3/4] scene caption (Qwen2.5-VL) =="
  $PY "$HERE/tools/build_captions.py" --test-data-root "$TD" \
    --out-root "$TD/captions" || echo "  (caption step optional; continuing)"

  echo "== [4/4] generate 5 surround views =="
  RENDERS="$WORK/renders"
  $PY -m openlongtail.scripts.inference \
    --test-data-root "$TD" --output-dir "$RENDERS" \
    --checkpoint-dir "$OLT_WM_DIR/checkpoint" --config openlongtail_1p3b \
    --wan21-vace-dir "$OLT_WAN_VACE_DIR" \
    --latent-cache-root "$HERE/demo/cached" \
    --caption-cache "$TD/captions" --treat-as-nv --max-clips -1 \
    $OLT_RECIPE $OLT_SPLAT --device cuda

  echo "== building surround gallery (no GT for raw video) =="
  $PY "$HERE/tools/make_demo_viz.py" --renders "$RENDERS" --out "$WORK/gallery"
  echo
  echo "DONE.  Open:  $WORK/gallery/index.html"
fi
