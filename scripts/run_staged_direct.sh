#!/bin/bash
# Stage-batched end-to-end run, outside Slurm:
#   ./run_staged_direct.sh 0 0 1 2      # gpu 0, folds 0 1 2
#   ./run_staged_direct.sh 1 3 4        # gpu 1, folds 3 4
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}

gpu_id=${1:?usage: run_staged_direct.sh <gpu_id> <fold...>}
shift
folds=("$@")
(( ${#folds[@]} )) || { echo "[ERROR] give at least one fold" >&2; exit 1; }
extra=${EXTRA_ARGS:-}

cache=${ROOT}/work/cache/staged_gpu_${gpu_id}
mkdir -p "${ROOT}/logs" "${cache}"/{ultralytics,triton,torchinductor,matplotlib,xdg}

echo "[START] gpu=${gpu_id} folds=${folds[*]} $(date '+%F %T')"
# Without this, enroot bind-mounts every /dev/nvidia* node and NVIDIA_VISIBLE_DEVICES
# is ignored, so CUDA_VISIBLE_DEVICES=0 lands every job on physical GPU 0.
export ENROOT_RESTRICT_DEV=yes
enroot start --root --mount /data:/data --mount "${YOLO_ENV}:${YOLO_ENV}" \
  --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
  "${IMAGE}" bash -lc "
set -euo pipefail
export NUMEXPR_MAX_THREADS=64
export NNUNETV2_PREDICT_BIN=/opt/conda/envs/nnunetv2/bin/nnUNetv2_predict
export YOLO_CONFIG_DIR='${cache}/ultralytics'
export TRITON_CACHE_DIR='${cache}/triton'
export TORCHINDUCTOR_CACHE_DIR='${cache}/torchinductor'
export MPLCONFIGDIR='${cache}/matplotlib'
export XDG_CACHE_HOME='${cache}/xdg'
exec '${YOLO_ENV}/bin/python' '${ROOT}/code/16_run_staged_pipeline.py' \"\$@\"
" -- --folds "${folds[@]}" --yolo-device 0 ${extra}
echo "[DONE] gpu=${gpu_id} folds=${folds[*]} $(date '+%F %T')"
