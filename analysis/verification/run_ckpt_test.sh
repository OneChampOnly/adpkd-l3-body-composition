#!/bin/bash
# Decisive test for the Table 2 / Supplementary Table 4 discrepancy:
# re-run inference on five fold-0 validation cases with checkpoint_best and with
# checkpoint_final, then compare both against nnU-Net's own validation export.
set -euo pipefail
ROOT="${ADPKD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
T=${ROOT}/analysis/ckj_verification/ckpt_test
IMAGE=${IMAGE:?set IMAGE to the container image}
export ENROOT_RESTRICT_DEV=yes
enroot start --root --mount /data:/data \
  --env NVIDIA_VISIBLE_DEVICES=1 --env CUDA_VISIBLE_DEVICES=0 "${IMAGE}" bash -lc "
set -e
export nnUNet_raw=${ROOT}/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=${ROOT}/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=${ROOT}/workspace/nnunet/nnUNet_results
for CK in checkpoint_best.pth checkpoint_final.pth; do
  OUT=${T}/out_\${CK%%.pth}
  OUT=\${OUT/checkpoint_/out_}
  done
for CK in best final; do
  /opt/conda/envs/nnunetv2/bin/nnUNetv2_predict \
    -i ${T}/in -o ${T}/out_\${CK} \
    -d Dataset917_InferiorSMA300Reann0917 -c 2d -f 0 \
    -tr nnUNetTrainer_500epochs -p nnUNetPlans \
    -chk checkpoint_\${CK}.pth --disable_progress_bar -npp 2 -nps 2 -device cuda
done
"
