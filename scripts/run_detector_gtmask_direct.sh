#!/bin/bash
# Standalone localisation (fold-matched YOLO on the GT L3 mask), outside Slurm:
#   ./run_detector_gtmask_direct.sh 1
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}

gpu_id=${1:?usage: run_detector_gtmask_direct.sh <gpu_id>}
cache=${ROOT}/work/cache/detector_gtmask_gpu_${gpu_id}
mkdir -p "${ROOT}/logs" "${cache}"/{ultralytics,matplotlib,xdg}

echo "[START] gpu=${gpu_id} $(date '+%F %T')"
# Without this, enroot bind-mounts every /dev/nvidia* node and NVIDIA_VISIBLE_DEVICES
# is ignored, so CUDA_VISIBLE_DEVICES=0 lands every job on physical GPU 0.
export ENROOT_RESTRICT_DEV=yes
enroot start --root --mount /data:/data --mount "${YOLO_ENV}:${YOLO_ENV}" \
  --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
  "${IMAGE}" bash -lc "
set -euo pipefail
export NUMEXPR_MAX_THREADS=64
export YOLO_CONFIG_DIR='${cache}/ultralytics'
export MPLCONFIGDIR='${cache}/matplotlib'
export XDG_CACHE_HOME='${cache}/xdg'
exec '${YOLO_ENV}/bin/python' '${ROOT}/code/17_evaluate_detector_on_gt_mask.py' --yolo-device 0
"
echo "[DONE] gpu=${gpu_id} $(date '+%F %T')"
