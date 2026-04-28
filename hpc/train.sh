#!/bin/bash
#SBATCH --job-name=tv3s-train
#SBATCH --partition=main-gpu
#SBATCH --gpus=lovelace:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.log

module load singularity/3.9.5

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind /home/s2283921/TV3S-DINOv3/upstream:/workspace/TV3S \
    --bind /home/s2283921/.cache/huggingface:/root/.cache/huggingface \
    --pwd /workspace/TV3S \
    /home/s2283921/tv3s_sandbox/ \
    bash -c "PYTHONPATH=/workspace/TV3S ./tools/dist_train.sh ${CONFIG} 2 --work-dir work_dirs/${WORK_DIR}"
