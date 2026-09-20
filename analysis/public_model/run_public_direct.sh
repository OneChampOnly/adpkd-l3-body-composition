#!/bin/bash
# Inferior public-model comparison on the reannotated 300-case cohort, outside Slurm:
#   ./run_public_direct.sh prepare       # stage the 300 reference-slice CTs + manifest
#   ./run_public_direct.sh infer 0       # run the public model on all 300, gpu 0
#   ./run_public_direct.sh tables        # Supplementary Tables 5A / 5B
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE=${ROOT}/analysis/public_model
REPO=${PUBLIC_MODEL_REPO:?clone Chen et al. and set PUBLIC_MODEL_REPO}
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}

mode=${1:?usage: run_public_direct.sh prepare | infer <gpu_id> | tables}
shift || true
mkdir -p "${BASE}"/{manifest,public_input,public_pred,tables,logs,cache/matplotlib,cache/xdg}

case "${mode}" in
  prepare)
    "${YOLO_ENV}/bin/python" "${BASE}/scripts/01_prepare_manifest.py"
    ;;
  infer)
    gpu_id=${1:?usage: run_public_direct.sh infer <gpu_id>}
    echo "[START] public model on 300 cases, gpu=${gpu_id} $(date '+%F %T')"
# Without this, enroot bind-mounts every /dev/nvidia* node and NVIDIA_VISIBLE_DEVICES
# is ignored, so CUDA_VISIBLE_DEVICES=0 lands every job on physical GPU 0.
export ENROOT_RESTRICT_DEV=yes
    enroot start --root --mount /data:/data \
      --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
      "${IMAGE}" bash -lc "
set -euo pipefail
export MPLCONFIGDIR='${BASE}/cache/matplotlib'
export XDG_CACHE_HOME='${BASE}/cache/xdg'
cd '${REPO}'
exec /opt/conda/envs/nnunetv2/bin/python predict_muscle_fat.py \
  --input '${BASE}/public_input' \
  --output '${BASE}/public_pred' \
  --checkpoint_type best \
  --body_composition_type None \
  --maxprocesses 2 \
  --overwrite True
"
    echo "[DONE] public model $(date '+%F %T')"
    ;;
  tables)
    "${YOLO_ENV}/bin/python" "${BASE}/scripts/02_compute_supplement_tables.py"
    ;;
  *)
    echo "[ERROR] unknown mode ${mode}" >&2; exit 1 ;;
esac
