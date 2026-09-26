#!/bin/bash
# Run the end-to-end pipeline outside Slurm, e.g.:
#   ./run_end_to_end_direct.sh 0 both 0 1 2
#   ./run_end_to_end_direct.sh 1 both 3 4
# Arguments: <gpu_id> <inferior|middle|both> <fold> [fold ...]
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}
# Where nnU-Net resolves <dataset>/<trainer>__<plans>__<config>/fold_k/. Point this at the
# directory the released weights were downloaded into, e.g. weights/nnUNet_results.
NNUNET_RESULTS=${NNUNET_RESULTS:-${ROOT}/work/nnUNet_results}

gpu_id=${1:?usage: run_end_to_end_direct.sh <gpu_id> <branches> <fold...>}
branches=${2:?usage: run_end_to_end_direct.sh <gpu_id> <branches> <fold...>}
shift 2
folds=("$@")
(( ${#folds[@]} )) || { echo "[ERROR] give at least one fold" >&2; exit 1; }

cache=${ROOT}/work/cache/e2e_gpu_${gpu_id}
mkdir -p "${ROOT}/logs" "${cache}"/{ultralytics,triton,torchinductor,matplotlib,xdg}

echo "[START] gpu=${gpu_id} branches=${branches} folds=${folds[*]} $(date '+%F %T')"
enroot start --root --mount /data:/data --mount "${YOLO_ENV}:${YOLO_ENV}" \
  --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
  "${IMAGE}" bash -lc "
set -euo pipefail
export NUMEXPR_MAX_THREADS=32
export NNUNET_RESULTS='${NNUNET_RESULTS}'
export NNUNETV2_PREDICT_BIN=/opt/conda/envs/nnunetv2/bin/nnUNetv2_predict
export YOLO_CONFIG_DIR='${cache}/ultralytics'
export TRITON_CACHE_DIR='${cache}/triton'
export TORCHINDUCTOR_CACHE_DIR='${cache}/torchinductor'
export MPLCONFIGDIR='${cache}/matplotlib'
export XDG_CACHE_HOME='${cache}/xdg'
exec '${YOLO_ENV}/bin/python' '${ROOT}/code/13_run_end_to_end_5fold.py' \"\$@\"
" -- --folds "${folds[@]}" --branches "${branches}" --yolo-device 0
echo "[DONE] gpu=${gpu_id} folds=${folds[*]} $(date '+%F %T')"
