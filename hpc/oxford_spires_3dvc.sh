#!/bin/bash
#SBATCH --job-name=oxspires-3dvc
#SBATCH --partition=main
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=hpc/logs/oxford_spires/%x_%j.log

# Usage:
#   WORK_DIR=... SEQUENCE=... sbatch hpc/oxford_spires_3dvc.sh
#
# Example:
#   WORK_DIR=dinov3_vits16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5 \
#   SEQUENCE=christ-church-02 \
#   RESOLUTION=480x480 \
#   sbatch hpc/oxford_spires_3dvc.sh

module load singularity/3.9.5

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

WORK_DIR=${WORK_DIR:-dinov3_vits16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5}
SEQUENCE=${SEQUENCE:-christ-church-02}
RESOLUTION=${RESOLUTION:-480x480}

HOME_BASE=/home/s2283921/TV3S-DINOv3/upstream

SEGS_DIR=results/oxford_spires/${SEQUENCE}/${WORK_DIR}/segs_${RESOLUTION}
DEPTH_DIR=data/oxford_spires/${SEQUENCE}/depth_${RESOLUTION}
POSES=data/oxford_spires/${SEQUENCE}/gt-tum.txt
TIMESTAMPS=data/oxford_spires/${SEQUENCE}/cam0_${RESOLUTION}/timestamps.txt
K_FILE=data/oxford_spires/${SEQUENCE}/cam0_${RESOLUTION}/K_new.txt
OUT_JSON=results/oxford_spires/${SEQUENCE}/${WORK_DIR}/3dvc_${RESOLUTION}.json

mkdir -p hpc/logs/oxford_spires

singularity exec \
    --bind /dev/shm:/dev/shm \
    --bind ${HOME_BASE}:/workspace/TV3S \
    --pwd /workspace/TV3S \
    /home/s2283921/tv3s_sandbox_cu128/ \
    bash -c "
        set -e
        export PYTHONUNBUFFERED=1
        PYTHONPATH=/workspace/TV3S python3 tools/oxford_spires/compute_3dvc.py \
            --segs       ${SEGS_DIR} \
            --depth      ${DEPTH_DIR} \
            --poses      ${POSES} \
            --timestamps ${TIMESTAMPS} \
            --K          ${K_FILE} \
            --gaps       1 2 4 8 16 \
            --out        ${OUT_JSON}
    "
