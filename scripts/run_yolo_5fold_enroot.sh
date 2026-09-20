#!/bin/bash
# Submit explicitly, for example:
#   sbatch --array=0-4 run_yolo_5fold_enroot.sh inferior
#   sbatch --array=0-4 run_yolo_5fold_enroot.sh middle
#SBATCH --job-name=yolo_0917
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=${ADPKD_ROOT}/logs/yolo_%A_fold_%a.out
#SBATCH --error=${ADPKD_ROOT}/logs/yolo_%A_fold_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch --array=0-4}"
: "${SLURM_JOB_GPUS:?Slurm did not set SLURM_JOB_GPUS}"

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE=${IMAGE:?set IMAGE to the container image}
YOLO_ENV=${YOLO_ENV:?set YOLO_ENV to the ultralytics conda env}
BASE_WEIGHTS=${BASE_WEIGHTS:-${ADPKD_ROOT}/external_inputs/yolov8n.pt}
BRANCH=${1:?usage: sbatch --array=0-4 run_yolo_5fold_enroot.sh <inferior|middle>}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-16}
IMGSZ=${IMGSZ:-512}
WORKERS=${WORKERS:-4}
fold=${SLURM_ARRAY_TASK_ID}
gpu_id=${SLURM_JOB_GPUS%%,*}

[[ "${BRANCH}" =~ ^(inferior|middle)$ ]] || { echo "[ERROR] branch must be inferior or middle" >&2; exit 1; }
[[ "${fold}" =~ ^[0-4]$ ]] || { echo "[ERROR] invalid fold=${fold}" >&2; exit 1; }
[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "[ERROR] unsupported SLURM_JOB_GPUS=${SLURM_JOB_GPUS}" >&2; exit 1; }

DATA="${ROOT}/yolo/datasets/${BRANCH}/fold_${fold}/data.yaml"
[[ -f "${DATA}" ]] || { echo "[ERROR] dataset missing: ${DATA}" >&2; exit 1; }
[[ -f "${BASE_WEIGHTS}" ]] || { echo "[ERROR] base weights missing: ${BASE_WEIGHTS}" >&2; exit 1; }

cache=${ROOT}/workspace/cache/yolo_${BRANCH}_fold_${fold}
mkdir -p "${ROOT}/logs" "${cache}/ultralytics" "${cache}/xdg" "${cache}/matplotlib"

echo "JOB=${SLURM_JOB_ID} BRANCH=${BRANCH} FOLD=${fold} NODE=$(hostname)"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader

export ENROOT_RESTRICT_DEV=yes
srun --exclusive --ntasks=1 --gres=gpu:1 --gpu-bind=single:1 \
  enroot start --root --mount /data:/data --mount "${YOLO_ENV}:${YOLO_ENV}" \
  --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
  --env YOLO_CONFIG_DIR="${cache}/ultralytics" --env XDG_CACHE_HOME="${cache}/xdg" \
  --env MPLCONFIGDIR="${cache}/matplotlib" \
  "${IMAGE}" "${YOLO_ENV}/bin/yolo" detect train \
    model="${BASE_WEIGHTS}" \
    data="${DATA}" \
    imgsz="${IMGSZ}" epochs="${EPOCHS}" batch="${BATCH}" \
    device=0 workers="${WORKERS}" \
    project="${ROOT}/yolo/runs/${BRANCH}" name="fold_${fold}"
