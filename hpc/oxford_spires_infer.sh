#!/bin/bash
#SBATCH --job-name=oxspires-infer
#SBATCH --partition=main-gpu,itc-gpu
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=hpc/logs/oxford_spires/%x_%j.log

# Usage:
#   CONFIG=... WORK_DIR=... SEQUENCE=... sbatch hpc/oxford_spires_infer.sh
#
# Example:
#   CONFIG=local_configs/dinov3/dinov3_hf_vits16_tv3s_frozen.480x480.vspw2.160k.py \
#   WORK_DIR=dinov3_vits16_tv3s_frozen_2sample_iter160k_lr6e-5 \
#   SEQUENCE=christ-church-02 \
#   RESOLUTION=480x480 \
#   sbatch hpc/oxford_spires_infer.sh

module load singularity/3.9.5

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

CONFIG=${CONFIG:-local_configs/dinov3/dinov3_hf_vits16_tv3s_frozen.480x480.vspw2.160k.py}
WORK_DIR=${WORK_DIR:-dinov3_vits16_tv3s_frozen_2sample_iter160k_lr6e-5}
CHECKPOINT=${CHECKPOINT:-work_dirs/${WORK_DIR}/iter_160000.pth}
SEQUENCE=${SEQUENCE:-christ-church-02}
RESOLUTION=${RESOLUTION:-480x480}

IMAGES_DIR=data/oxford_spires/${SEQUENCE}/cam0_${RESOLUTION}
OUT_DIR=results/oxford_spires/${SEQUENCE}/${WORK_DIR}/segs_${RESOLUTION}

mkdir -p hpc/logs/oxford_spires

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind /home/s2283921/TV3S-DINOv3/upstream:/workspace/TV3S \
    --bind /home/s2283921/.cache/huggingface:/root/.cache/huggingface \
    --pwd /workspace/TV3S \
    /home/s2283921/tv3s_sandbox_cu128/ \
    bash -c "
        set -e
        PYTHONPATH=/workspace/TV3S python3 tools/oxford_spires/infer_sequence.py \
            --config     ${CONFIG} \
            --checkpoint ${CHECKPOINT} \
            --images     ${IMAGES_DIR} \
            --out        ${OUT_DIR}
    "
