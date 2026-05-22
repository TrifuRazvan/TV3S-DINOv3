#!/bin/bash
# Submit Oxford Spires 3DVC evaluation jobs for all trained models.
# Run from ~/TV3S-DINOv3/upstream on the HPC head node.
#
# Usage:
#   bash hpc/submit_oxford_spires_3dvc_all.sh
#   bash hpc/submit_oxford_spires_3dvc_all.sh christ-church-02 480x480

SEQUENCE=${1:-christ-church-02}
RESOLUTION=${2:-480x480}

BASE=local_configs/dinov3

submit() {
    local name=$1
    local work_dir=$2
    echo "Submitting: ${name}"
    WORK_DIR=${work_dir} SEQUENCE=${SEQUENCE} RESOLUTION=${RESOLUTION} \
        sbatch --job-name=${name}-3dvc hpc/oxford_spires_3dvc.sh
}

# --- DINOv3-S/16 + TV3S ---
submit vits16-frozen  dinov3_vits16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5
submit vits16-ft      dinov3_vits16_tv3s_finetune_2sample_2gpu_iter160k_lr3e-5
submit vits16-lpft    dinov3_vits16_tv3s_lpft_2sample_2gpu_iter160k_lr3e-5

# --- DINOv3-S/16+ + TV3S ---
submit vits16p-frozen dinov3_vits16plus_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5
submit vits16p-ft     dinov3_vits16plus_tv3s_finetune_2sample_2gpu_iter160k_lr3e-5
submit vits16p-lpft   dinov3_vits16plus_tv3s_lpft_2sample_2gpu_iter160k_lr3e-5

# --- DINOv3-B/16 + TV3S ---
submit vitb16-frozen  dinov3_vitb16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5
submit vitb16-ft      dinov3_vitb16_tv3s_finetune_2sample_2gpu_iter160k_lr3e-5
submit vitb16-lpft    dinov3_vitb16_tv3s_lpft_2sample_2gpu_iter160k_lr3e-5

# --- ConvNeXt-B + TV3S ---
submit cnxb-frozen    dinov3_convnext_base_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5
submit cnxb-ft        dinov3_convnext_base_tv3s_ft_2sample_2gpu_iter160k_lr3e-5
# cnxb-lpft skipped — training incomplete (only 80k iterations)

echo "All jobs submitted. Check with: squeue -u s2283921"
