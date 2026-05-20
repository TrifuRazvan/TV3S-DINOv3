#!/bin/bash
# Submit Oxford Spires inference jobs for all trained models.
# Run from ~/TV3S-DINOv3 on the HPC head node.
#
# Usage:
#   bash hpc/submit_oxford_spires_all.sh
#   bash hpc/submit_oxford_spires_all.sh christ-church-02 480x480

SEQUENCE=${1:-christ-church-02}
RESOLUTION=${2:-480x480}

BASE=local_configs/dinov3

submit() {
    local config=$1
    local work_dir=$2
    local checkpoint=${3:-work_dirs/${work_dir}/iter_160000.pth}
    echo "Submitting: ${work_dir}"
    CONFIG=${config} WORK_DIR=${work_dir} CHECKPOINT=${checkpoint} \
        SEQUENCE=${SEQUENCE} RESOLUTION=${RESOLUTION} \
        sbatch hpc/oxford_spires_infer.sh
}

# --- DINOv3-S/16 + TV3S ---
submit ${BASE}/dinov3_hf_vits16_tv3s_frozen.480x480.vspw2.160k.py \
       dinov3_vits16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5

submit ${BASE}/dinov3_hf_vits16_tv3s.480x480.vspw2.160k.py \
       dinov3_vits16_tv3s_finetune_2sample_2gpu_iter160k_lr3e-5

submit ${BASE}/dinov3_hf_vits16_tv3s_lpft.480x480.vspw2.160k.py \
       dinov3_vits16_tv3s_lpft_2sample_2gpu_iter160k_lr3e-5

# --- DINOv3-S/16+ + TV3S ---
submit ${BASE}/dinov3_hf_vits16plus_tv3s_frozen.480x480.vspw2.160k.py \
       dinov3_vits16plus_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5

submit ${BASE}/dinov3_hf_vits16plus_tv3s.480x480.vspw2.160k.py \
       dinov3_vits16plus_tv3s_finetune_2sample_2gpu_iter160k_lr3e-5

submit ${BASE}/dinov3_hf_vits16plus_tv3s_lpft.480x480.vspw2.160k.py \
       dinov3_vits16plus_tv3s_lpft_2sample_2gpu_iter160k_lr3e-5

# --- DINOv3-B/16 + TV3S ---
submit ${BASE}/dinov3_hf_vitb16_tv3s_frozen.480x480.vspw2.160k.py \
       dinov3_vitb16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5

submit ${BASE}/dinov3_hf_vitb16_tv3s.480x480.vspw2.160k.py \
       dinov3_vitb16_tv3s_finetune_2sample_2gpu_iter160k_lr3e-5

submit ${BASE}/dinov3_hf_vitb16_tv3s_lpft.480x480.vspw2.160k.py \
       dinov3_vitb16_tv3s_lpft_2sample_2gpu_iter160k_lr3e-5

# --- ConvNeXt-B + TV3S ---
submit ${BASE}/dinov3_hf_convnext_base_tv3s_frozen.480x480.vspw2.160k.py \
       dinov3_convnext_base_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5

submit ${BASE}/dinov3_hf_convnext_base_tv3s.480x480.vspw2.160k.py \
       dinov3_convnext_base_tv3s_ft_2sample_2gpu_iter160k_lr3e-5

submit ${BASE}/dinov3_hf_convnext_base_tv3s_lpft.480x480.vspw2.160k.py \
       dinov3_convnext_base_tv3s_lpft_2sample_2gpu_iter160k_lr3e-5

# --- TV3S baseline (MiT-B1) ---
submit local_configs/tv3s/B1/tv3s.b1.480x480.vspw2.160k.py \
       tv3s_b1_baseline \
       resources/checkpoints/B1/iter_160000.pth

echo "All jobs submitted. Check with: squeue -u s2283921"
