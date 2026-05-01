#!/bin/bash
#SBATCH --job-name=tv3s-eval
#SBATCH --partition=main-gpu,itc-gpu
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=hpc/logs/eval/%x_%j.log

module load singularity/3.9.5

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

PORT=$((29500 + SLURM_JOB_ID % 1000))
JOBDIR=/local/s2283921/${SLURM_JOB_ID}
TMPDIR=${JOBDIR}/tmp
mkdir -p ${TMPDIR}
trap "rm -rf ${JOBDIR}" EXIT

CONFIG=local_configs/dinov3/dinov3_hf_vits16_tv3s_frozen.480x480.vspw2.160k.py
WORK_DIR=dinov3_vits16_tv3s_frozen_2sample_2gpu_iter160k_lr6e-5
CHECKPOINT=work_dirs/${WORK_DIR}/iter_160000.pth
RESULTS_DIR=results/${WORK_DIR}

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind /home/s2283921/TV3S-DINOv3/upstream:/workspace/TV3S \
    --bind /home/s2283921/.cache/huggingface:/root/.cache/huggingface \
    --bind ${JOBDIR}:${JOBDIR} \
    --pwd /workspace/TV3S \
    /home/s2283921/tv3s_sandbox_cu128/ \
    bash -c "
        set -e
        mkdir -p ${RESULTS_DIR}/images

        echo '=== Step 1: Inference + mIoU ==='
        PORT=${PORT} PYTHONPATH=/workspace/TV3S ./tools/dist_test.sh ${CONFIG} ${CHECKPOINT} 1 \
            --eval mIoU --out ${RESULTS_DIR}/predictions.pkl --tmpdir ${TMPDIR}

        echo '=== Step 2: Convert predictions to PNG images ==='
        PYTHONPATH=/workspace/TV3S python3 tools/format_predictions.py \
            ${RESULTS_DIR}/predictions.pkl ${CONFIG} ${RESULTS_DIR}/images/

        echo '=== Step 3: Temporal consistency (VC8, VC16) ==='
        PYTHONPATH=/workspace/TV3S python3 tools/VC_perclip.py \
            ${RESULTS_DIR}/images/result_submission data/vspw/VSPW_480p
    "
