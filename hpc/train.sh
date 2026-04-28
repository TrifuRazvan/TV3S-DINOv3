#!/bin/bash
#SBATCH --job-name=tv3s-train
#SBATCH --partition=main-gpu,itc-gpu
#SBATCH --nodes=1
#SBATCH --gpus=lovelace:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.log

module load singularity/3.9.5

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

CONFIG=local_configs/dinov3/dinov3_hf_vits16_tv3s_frozen.480x480.vspw2.160k.py
WORK_DIR=dinov3_vits16_tv3s_frozen_2sample_4gpu_iter160k_lr6e-5

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind /home/s2283921/TV3S-DINOv3/upstream:/workspace/TV3S \
    --bind /home/s2283921/.cache/huggingface:/root/.cache/huggingface \
    --pwd /workspace/TV3S \
    /home/s2283921/tv3s_sandbox/ \
    bash -c "PYTHONPATH=/workspace/TV3S ./tools/dist_train.sh ${CONFIG} 4 --work-dir work_dirs/${WORK_DIR}"
