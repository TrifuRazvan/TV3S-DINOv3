#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --partition=itc-gpu
#SBATCH --nodelist=hpc-node30
#SBATCH --gpus=2
#SBATCH --time=00:05:00
#SBATCH --output=hpc/logs/gpu_test_%j.log

module load singularity/3.9.5

echo "=== host nvidia-smi ==="
nvidia-smi

echo ""
echo "=== inside Singularity container ==="
singularity exec --nv \
    /home/s2283921/tv3s_sandbox_cu128/ \
    nvidia-smi
