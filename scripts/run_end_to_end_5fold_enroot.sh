#!/bin/bash
# Submit explicitly, for example:
#   sbatch --array=0-4 run_end_to_end_5fold_enroot.sh both
#   sbatch --array=0-4 run_end_to_end_5fold_enroot.sh inferior
#SBATCH --job-name=e2e_0917
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=${ADPKD_ROOT}/logs/e2e_%A_fold_%a.out
#SBATCH --error=${ADPKD_ROOT}/logs/e2e_%A_fold_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch --array=0-4}"
: "${SLURM_JOB_GPUS:?Slurm did not set SLURM_JOB_GPUS}"

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}
BRANCHES=${1:-both}
CASE_LIMIT=${CASE_LIMIT:-0}
fold=${SLURM_ARRAY_TASK_ID}
gpu_id=${SLURM_JOB_GPUS%%,*}

[[ "${fold}" =~ ^[0-4]$ ]] || { echo "[ERROR] invalid fold=${fold}" >&2; exit 1; }
[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "[ERROR] unsupported SLURM_JOB_GPUS=${SLURM_JOB_GPUS}" >&2; exit 1; }
[[ "${BRANCHES}" =~ ^(inferior|middle|both)$ ]] || { echo "[ERROR] branches must be inferior, middle, or both" >&2; exit 1; }

cache=${ROOT}/work/cache/e2e_fold_${fold}
mkdir -p "${ROOT}/logs" "${cache}/ultralytics" "${cache}/triton" "${cache}/torchinductor" "${cache}/matplotlib" "${cache}/xdg"

echo "JOB=${SLURM_JOB_ID} FOLD=${fold} BRANCHES=${BRANCHES} NODE=$(hostname)"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader

args=(--folds "${fold}" --branches "${BRANCHES}" --yolo-device 0)
if (( CASE_LIMIT > 0 )); then args+=(--case-limit "${CASE_LIMIT}"); fi

export ENROOT_RESTRICT_DEV=yes
srun --exclusive --ntasks=1 --gres=gpu:1 --gpu-bind=single:1 \
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
exec '${YOLO_ENV}/bin/python' '${ROOT}/code/13_run_end_to_end_5fold.py' \"\$@\"
" -- "${args[@]}"
