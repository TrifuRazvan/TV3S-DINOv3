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
#   WORK_DIR=dinov3_vits16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5 \
#   SEQUENCE=christ-church-02 \
#   RESOLUTION=480x480 \
#   sbatch hpc/oxford_spires_infer.sh

module load singularity/3.9.5

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

CONFIG=${CONFIG:-local_configs/dinov3/dinov3_hf_vits16_tv3s_frozen.480x480.vspw2.160k.py}
WORK_DIR=${WORK_DIR:-dinov3_vits16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5}
CHECKPOINT=${CHECKPOINT:-work_dirs/${WORK_DIR}/iter_160000.pth}
SEQUENCE=${SEQUENCE:-christ-church-02}
RESOLUTION=${RESOLUTION:-480x480}

HOME_BASE=/home/s2283921/TV3S-DINOv3/upstream
IMAGES_DIR=data/oxford_spires/${SEQUENCE}/cam0_${RESOLUTION}
OUT_DIR=results/oxford_spires/${SEQUENCE}/${WORK_DIR}/segs_${RESOLUTION}

WORKDIR=${SLURM_TMPDIR:-/tmp/s2283921/${SLURM_JOB_ID}}
mkdir -p ${WORKDIR}

mkdir -p hpc/logs/oxford_spires

echo "=== Staging data to local disk (${WORKDIR}) ==="

# HF model cache (avoids thousands of small NFS reads at model load)
mkdir -p ${WORKDIR}/hf_cache
cp -r /home/s2283921/.cache/huggingface/. ${WORKDIR}/hf_cache/

# Input images
mkdir -p ${WORKDIR}/${IMAGES_DIR}
cp -r ${HOME_BASE}/${IMAGES_DIR}/. ${WORKDIR}/${IMAGES_DIR}/

# Checkpoint (single large file, copy to avoid NFS read during model load)
mkdir -p ${WORKDIR}/$(dirname ${CHECKPOINT})
cp ${HOME_BASE}/${CHECKPOINT} ${WORKDIR}/${CHECKPOINT}

# Local output dir (all PNG writes go here, not NFS)
mkdir -p ${WORKDIR}/${OUT_DIR}

echo "=== Running inference ==="

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind ${HOME_BASE}:/workspace/TV3S \
    --bind ${WORKDIR}/${IMAGES_DIR}:/workspace/TV3S/${IMAGES_DIR} \
    --bind ${WORKDIR}/$(dirname ${CHECKPOINT}):/workspace/TV3S/$(dirname ${CHECKPOINT}) \
    --bind ${WORKDIR}/${OUT_DIR}:/workspace/TV3S/${OUT_DIR} \
    --bind ${WORKDIR}/hf_cache:/root/.cache/huggingface \
    --pwd /workspace/TV3S \
    /home/s2283921/tv3s_sandbox_cu128/ \
    bash -c "
        set -e
        export TRANSFORMERS_OFFLINE=1
        export HF_DATASETS_OFFLINE=1
        export PYTHONUNBUFFERED=1
        PYTHONPATH=/workspace/TV3S python3 tools/oxford_spires/infer_sequence.py \
            --config     ${CONFIG} \
            --checkpoint ${CHECKPOINT} \
            --images     ${IMAGES_DIR} \
            --out        ${OUT_DIR}
    "

echo "=== Copying results back to /home ==="
mkdir -p ${HOME_BASE}/${OUT_DIR}
cp -r ${WORKDIR}/${OUT_DIR}/. ${HOME_BASE}/${OUT_DIR}/

echo "=== Done ==="
