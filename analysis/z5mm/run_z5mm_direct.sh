#!/bin/bash
# 5 mm normalised localisation analysis, run outside Slurm:
#   ./run_z5mm_direct.sh prepare
#   ./run_z5mm_direct.sh folds <gpu_id> <fold> [fold ...]
#   ./run_z5mm_direct.sh summarize
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}
NNUNET_PY=/opt/conda/envs/nnunetv2/bin/python
mode=${1:?usage: run_z5mm_direct.sh <prepare|folds|summarize> ...}
shift || true

run_in_container() {
  local gpu_id=$1; shift
# Without this, enroot bind-mounts every /dev/nvidia* node and NVIDIA_VISIBLE_DEVICES
# is ignored, so CUDA_VISIBLE_DEVICES=0 lands every job on physical GPU 0.
export ENROOT_RESTRICT_DEV=yes
  enroot start --root --mount /data:/data --mount "${YOLO_ENV}:${YOLO_ENV}" \
    --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
    "${IMAGE}" bash -lc "$*"
}

case "${mode}" in
  prepare)
    run_in_container "" "${NNUNET_PY} '${ROOT}/code/01_prepare_z5mm.py'"
    ;;
  folds)
    gpu_id=${1:?usage: run_z5mm_direct.sh folds <gpu_id> <fold...>}
    shift
    (( $# )) || { echo "[ERROR] give at least one fold" >&2; exit 1; }
    cache=${ROOT}/work/cache/gpu_${gpu_id}
    mkdir -p "${cache}"/{ultralytics,xdg,matplotlib}
    for fold in "$@"; do
      echo "[FOLD ${fold}] L3 inference on the 5 mm volumes  $(date '+%F %T')"
      run_in_container "${gpu_id}" "${NNUNET_PY} '${ROOT}/code/02_run_fold.py' --fold ${fold}"
      echo "[FOLD ${fold}] detection  $(date '+%F %T')"
      run_in_container "${gpu_id}" "
export YOLO_CONFIG_DIR='${cache}/ultralytics'
export XDG_CACHE_HOME='${cache}/xdg'
export MPLCONFIGDIR='${cache}/matplotlib'
exec '${YOLO_ENV}/bin/python' '${ROOT}/code/03_run_yolo.py' --fold ${fold} --device 0"
    done
    ;;
  summarize)
    run_in_container "" "${NNUNET_PY} '${ROOT}/code/04_summarize.py'"
    ;;
  *)
    echo "[ERROR] unknown mode: ${mode}" >&2; exit 1 ;;
esac
