#!/bin/bash
# Submit CPU preparation followed by GPU folds 0-4:
#   bash run_l3_5fold_enroot.sh
set -euo pipefail

ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=${ROOT}/workspace
NNUNET_BASE=${WORK}/nnunet
LOG_DIR=${ROOT}/logs
CACHE_BASE=${WORK}/cache
IMAGE=${IMAGE:?set IMAGE to the container image}
CONFIG_JSON=${ROOT}/config.json
PREPARE_SCRIPT=${ROOT}/code/02_prepare_l3_nnunet.py
DATASET_ID=920
DATASET_NAME=L3Vertebra0917
DATASET_DIR=Dataset${DATASET_ID}_${DATASET_NAME}
CONFIG=3d_fullres
TRAINER=nnUNetTrainer_500epochs
PLANS=nnUNetPlans
MODE=${1:-submit}

mkdir -p "${LOG_DIR}"

if [[ "${MODE}" == submit ]]; then
    prep=$(sbatch --parsable --job-name=l3_920_prepare \
        --partition=gpu --cpus-per-task=12 --mem=40G --gres=none --time=2-00:00:00 \
        --output="${LOG_DIR}/l3_prepare_%j.out" --error="${LOG_DIR}/l3_prepare_%j.err" "$0" prepare)
    train=$(sbatch --parsable --dependency="afterok:${prep}" --array=0-4 \
        --job-name=l3_920_cv5 --partition=gpu --gres=gpu:1 --cpus-per-task=12 --mem=40G \
        --time=7-00:00:00 \
        --output="${LOG_DIR}/l3_train_%A_fold_%a.out" \
        --error="${LOG_DIR}/l3_train_%A_fold_%a.err" "$0" train)
    echo "[OK] prepare=${prep}, train=${train} (folds 0-4)"
    exit 0
fi

if [[ "${MODE}" == prepare ]]; then
    mkdir -p "${NNUNET_BASE}/nnUNet_raw" "${NNUNET_BASE}/nnUNet_preprocessed" "${NNUNET_BASE}/nnUNet_results"

    enroot start --root --mount /data:/data "${IMAGE}" bash -lc "
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate nnunetv2
export nnUNet_raw='${NNUNET_BASE}/nnUNet_raw'
export nnUNet_preprocessed='${NNUNET_BASE}/nnUNet_preprocessed'
export nnUNet_results='${NNUNET_BASE}/nnUNet_results'
python '${PREPARE_SCRIPT}' --config '${CONFIG_JSON}' --nnunet-raw \"\${nnUNet_raw}\"
nnUNetv2_plan_and_preprocess -d '${DATASET_ID}' --verify_dataset_integrity -np '${SLURM_CPUS_PER_TASK:-12}'
cp '${ROOT}/splits/splits_final.json' \"\${nnUNet_preprocessed}/${DATASET_DIR}/splits_final.json\"
"
    exit 0
fi

if [[ "${MODE}" == train ]]; then
    : "${SLURM_ARRAY_TASK_ID:?train mode must be submitted with --array=0-4}"
    : "${SLURM_JOB_GPUS:?Slurm did not set SLURM_JOB_GPUS}"
    fold=${SLURM_ARRAY_TASK_ID}
    cache=${CACHE_BASE}/l3_fold_${fold}
    gpu_id=${SLURM_JOB_GPUS%%,*}
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "[ERROR] unsupported SLURM_JOB_GPUS=${SLURM_JOB_GPUS}" >&2; exit 1; }
    mkdir -p "${cache}/triton" "${cache}/torchinductor" "${cache}/mpl" "${cache}/xdg"

    echo "JOB=${SLURM_JOB_ID} FOLD=${fold} NODE=$(hostname)"
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
    exit 0
fi

echo "[ERROR] mode must be submit, prepare, or train" >&2
exit 2
