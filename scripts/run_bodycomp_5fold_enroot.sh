#!/bin/bash
# nnU-Net 2d 5-fold training for the body-composition branches, on the re-annotated GT.
#
# Submit both branches:
#   bash run_bodycomp_5fold_enroot.sh submit inferior
#   bash run_bodycomp_5fold_enroot.sh submit middle
#
# The folds come from splits/splits_final.json, the same 240/60 partition the L3 and YOLO
# stages use, so a case is out-of-fold everywhere at once and the end-to-end evaluation stays
# honest. The file is copied in after preprocessing, which is the only point at which nnU-Net
# will not overwrite it.
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=${ROOT}/workspace
NNUNET_BASE=${WORK}/nnunet
LOG_DIR=${ROOT}/logs
CACHE_BASE=${WORK}/cache
IMAGE=${IMAGE:?set IMAGE to the container image}
CONFIG_JSON=${ROOT}/config.json
PREPARE_SCRIPT=${ROOT}/code/10_prepare_branch_nnunet.py
SPLITS=${ROOT}/splits/splits_final.json

MODE=${1:?usage: run_bodycomp_5fold_enroot.sh <submit|prepare|train> <inferior|middle>}
BRANCH=${2:?usage: run_bodycomp_5fold_enroot.sh <submit|prepare|train> <inferior|middle>}
[[ "${BRANCH}" =~ ^(inferior|middle)$ ]] || { echo "[ERROR] branch must be inferior or middle" >&2; exit 1; }

read -r DATASET_ID DATASET_NAME CONFIG TRAINER PLANS < <(
    python3 -c "
import json, sys
cfg = json.load(open('${CONFIG_JSON}'))
print(cfg['${BRANCH}_dataset_id'], cfg['${BRANCH}_dataset_name'],
      cfg['bodycomp_config'], cfg['nnunet_trainer'], cfg['nnunet_plans'])")
DATASET_DIR=Dataset$(printf '%03d' "${DATASET_ID}")_${DATASET_NAME}

mkdir -p "${LOG_DIR}"

if [[ "${MODE}" == submit ]]; then
    command -v sbatch >/dev/null || { echo "[ERROR] sbatch not found" >&2; exit 1; }
    prep=$(sbatch --parsable --job-name="bc_${DATASET_ID}_prepare" \
        --partition=gpu --cpus-per-task=12 --mem=40G --gres=none --time=1-00:00:00 \
        --output="${LOG_DIR}/bodycomp_${BRANCH}_prepare_%j.out" \
        --error="${LOG_DIR}/bodycomp_${BRANCH}_prepare_%j.err" "$0" prepare "${BRANCH}")
    train=$(sbatch --parsable --dependency="afterok:${prep}" --array=0-4 \
        --job-name="bc_${DATASET_ID}_cv5" --partition=gpu --gres=gpu:1 --cpus-per-task=12 \
        --mem=40G --time=7-00:00:00 \
        --output="${LOG_DIR}/bodycomp_${BRANCH}_train_%A_fold_%a.out" \
        --error="${LOG_DIR}/bodycomp_${BRANCH}_train_%A_fold_%a.err" "$0" train "${BRANCH}")
    echo "[OK] ${BRANCH}: prepare=${prep}, train=${train} (folds 0-4, ${DATASET_DIR})"
    exit 0
fi

[[ -f "${IMAGE}" ]] || { echo "[ERROR] enroot image not found: ${IMAGE}" >&2; exit 1; }

if [[ "${MODE}" == prepare ]]; then
    mkdir -p "${NNUNET_BASE}/nnUNet_raw" "${NNUNET_BASE}/nnUNet_preprocessed" \
             "${NNUNET_BASE}/nnUNet_results" "${CACHE_BASE}/bodycomp_${BRANCH}_prepare"

    enroot start --root --mount /data:/data "${IMAGE}" bash -lc "
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate nnunetv2
export nnUNet_raw='${NNUNET_BASE}/nnUNet_raw'
export nnUNet_preprocessed='${NNUNET_BASE}/nnUNet_preprocessed'
export nnUNet_results='${NNUNET_BASE}/nnUNet_results'
export XDG_CACHE_HOME='${CACHE_BASE}/bodycomp_${BRANCH}_prepare'
export MPLCONFIGDIR='${CACHE_BASE}/bodycomp_${BRANCH}_prepare/mpl'
python '${PREPARE_SCRIPT}' --config '${CONFIG_JSON}' --branches '${BRANCH}' \
    --nnunet-raw \"\${nnUNet_raw}\"
nnUNetv2_plan_and_preprocess -d '${DATASET_ID}' -c '${CONFIG}' \
    --verify_dataset_integrity -np '${SLURM_CPUS_PER_TASK:-12}'
cp '${SPLITS}' \"\${nnUNet_preprocessed}/${DATASET_DIR}/splits_final.json\"
"
    echo "[OK] ${BRANCH} prepared: ${DATASET_DIR}"
    exit 0
fi

if [[ "${MODE}" == train ]]; then
    : "${SLURM_ARRAY_TASK_ID:?train mode must be submitted with --array=0-4}"
    : "${SLURM_JOB_GPUS:?Slurm did not set SLURM_JOB_GPUS}"
    fold=${SLURM_ARRAY_TASK_ID}
    cache=${CACHE_BASE}/bodycomp_${BRANCH}_fold_${fold}
    gpu_id=${SLURM_JOB_GPUS%%,*}
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "[ERROR] unsupported SLURM_JOB_GPUS=${SLURM_JOB_GPUS}" >&2; exit 1; }
    [[ -d "${NNUNET_BASE}/nnUNet_preprocessed/${DATASET_DIR}" ]] || {
        echo "[ERROR] preprocessed dataset missing: ${DATASET_DIR}" >&2; exit 1; }
    mkdir -p "${cache}/triton" "${cache}/torchinductor" "${cache}/mpl" "${cache}/xdg"

    echo "JOB=${SLURM_JOB_ID} BRANCH=${BRANCH} FOLD=${fold} NODE=$(hostname)"
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader

    export ENROOT_RESTRICT_DEV=yes
    srun --exclusive --ntasks=1 --gres=gpu:1 --gpu-bind=single:1 \
      enroot start --root --mount /data:/data \
      --env NVIDIA_VISIBLE_DEVICES="${gpu_id}" --env CUDA_VISIBLE_DEVICES=0 \
      "${IMAGE}" bash -lc "
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate nnunetv2
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
export nnUNet_raw='${NNUNET_BASE}/nnUNet_raw'
export nnUNet_preprocessed='${NNUNET_BASE}/nnUNet_preprocessed'
export nnUNet_results='${NNUNET_BASE}/nnUNet_results'
export TRITON_CACHE_DIR='${cache}/triton'
export TORCHINDUCTOR_CACHE_DIR='${cache}/torchinductor'
export MPLCONFIGDIR='${cache}/mpl'
export XDG_CACHE_HOME='${cache}/xdg'
export MPLBACKEND=Agg
nnUNetv2_train '${DATASET_ID}' '${CONFIG}' '${fold}' -tr '${TRAINER}' -p '${PLANS}'
"
    echo "[OK] ${BRANCH} fold ${fold} finished"
    exit 0
fi

echo "[ERROR] mode must be submit, prepare, or train" >&2
exit 2
