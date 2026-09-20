#!/bin/bash
# GT-anchored +-1/+-2 slice offset experiment, outside Slurm:
#   ./run_offset_direct.sh prepare              # build the 3000-row manifest (needs the e2e evaluation CSV)
#   ./run_offset_direct.sh folds 0 0 1 2        # gpu 0, folds 0 1 2
#   ./run_offset_direct.sh evaluate             # CPU-only scoring into statistics/
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HERE=${ROOT}/analysis/offset
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}

mode=${1:?usage: run_offset_direct.sh prepare | folds <gpu_id> <fold...> | evaluate}
shift || true
mkdir -p "${HERE}"/{manifests,predictions,statistics,work,logs}

case "${mode}" in
  prepare)
    "${YOLO_ENV}/bin/python" "${HERE}/01_prepare_gt_offset_manifest.py"
    ;;
  folds)
    gpu_id=${1:?usage: run_offset_direct.sh folds <gpu_id> <fold...>}
    shift
    folds=("$@")
    (( ${#folds[@]} )) || { echo "[ERROR] give at least one fold" >&2; exit 1; }
    cache=${ROOT}/work/cache/offset_gpu_${gpu_id}
    mkdir -p "${cache}"/{triton,torchinductor,matplotlib,xdg}
    for fold in "${folds[@]}"; do
      echo "[START] offset fold=${fold} gpu=${gpu_id} $(date '+%F %T')"
# Without this, enroot bind-mounts every /dev/nvidia* node and NVIDIA_VISIBLE_DEVICES
# is ignored, so CUDA_VISIBLE_DEVICES=0 lands every job on physical GPU 0.
export ENROOT_RESTRICT_DEV=yes
      enroot start --root --mount /data:/data --mount "${YOLO_ENV}:${YOLO_ENV}" \
        --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
        "${IMAGE}" bash -lc "
set -euo pipefail
export NUMEXPR_MAX_THREADS=64
export TRITON_CACHE_DIR='${cache}/triton'
export TORCHINDUCTOR_CACHE_DIR='${cache}/torchinductor'
export MPLCONFIGDIR='${cache}/matplotlib'
export XDG_CACHE_HOME='${cache}/xdg'
exec '${YOLO_ENV}/bin/python' '${HERE}/02_run_offset_inference_fold.py' --fold ${fold}
"
      echo "[DONE] offset fold=${fold} $(date '+%F %T')"
    done
    ;;
  evaluate)
    # SimpleITK only exists inside the container image.
    enroot start --root --mount /data:/data "${IMAGE}" bash -lc "
set -euo pipefail
export NUMEXPR_MAX_THREADS=64
exec python '${HERE}/03_evaluate_gt_offsets.py'
"
    ;;
  *)
    echo "[ERROR] unknown mode ${mode}" >&2; exit 1 ;;
esac
