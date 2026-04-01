#!/bin/bash

# TV3S Inference Script with Organized Output
# Creates timestamped folder for each run
# Usage: 
#   bash tools/run_inference.sh                    # Default folder name with timestamp
#   bash tools/run_inference.sh my_run_name        # Custom folder name

set -e

# ============ CONFIGURATION ============
# === B1 (SegFormer) - COMMENTED OUT ===
# CONFIG="local_configs/tv3s/B1/tv3s_realshift_w20_s10.b1.480x480.vspw2.160k.py"
# CHECKPOINT="resources/checkpoints/B1/iter_160000.pth"
# MODEL_NAME="b1"

# === Swin-S (COMMENTED OUT) ===
# CONFIG="local_configs/tv3s/Swin/swins_realshift_w20_s10.480x480.vspw2.160k.py"
# CHECKPOINT="resources/checkpoints/Swins/iter_160000.pth"
# MODEL_NAME="swins"

# === DINOv3 ViT-B16 (ACTIVE) ===
CONFIG="local_configs/tv3s/DINOv3/dinov3_hf_vits16.480x480.vspw2.160k.py"
CHECKPOINT="/workspace/TV3S/work_dirs/dinov3_hf_vitb16_4sample_2gpu_6e-4/iter_60000.pth"
MODEL_NAME="dinov3_vitb16"

NUM_GPUS=1  # Set to 1 for single GPU, 2+ for multi-GPU inference
# ======================================

# Parse arguments
CUSTOM_NAME="$1"

# Create run folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "$CUSTOM_NAME" ]; then
    RUN_FOLDER="results/${CUSTOM_NAME}"
else
    RUN_FOLDER="results/${MODEL_NAME}_run_${TIMESTAMP}"
fi
mkdir -p "$RUN_FOLDER/images"
mkdir -p "$RUN_FOLDER/predictions"  # Store .npy prediction files here

echo "═══════════════════════════════════════════════════════════"
echo "TV3S Inference Run"
echo "═══════════════════════════════════════════════════════════"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output folder: $RUN_FOLDER"
echo "GPUs: $NUM_GPUS"
echo "═══════════════════════════════════════════════════════════"

# Use run folder for temp predictions instead of global /workspace/tv3s_temp
export TMPDIR="$RUN_FOLDER/predictions"
rm -rf "$TMPDIR"/*
echo "✓ Using $TMPDIR for prediction files"

# Run inference
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU inference using distributed launcher
    PORT=${PORT:-29820}
    PYTHONPATH=/workspace/TV3S:$PYTHONPATH python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS --master_port=$PORT \
        tools/test.py \
        "$CONFIG" \
        "$CHECKPOINT" \
        --launcher pytorch \
        --format-only \
        --out "$RUN_FOLDER/${MODEL_NAME}_predictions.pkl"
else
    # Single GPU inference
    PYTHONPATH=/workspace/TV3S:$PYTHONPATH python3 tools/test.py \
        "$CONFIG" \
        "$CHECKPOINT" \
        --format-only \
        --out "$RUN_FOLDER/${MODEL_NAME}_predictions.pkl"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✓ Inference complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Evaluate semantic metrics (mIoU, mAcc, aAcc):"
echo "   python3 tools/eval_from_pkl.py \\"
echo "       '$RUN_FOLDER/${MODEL_NAME}_predictions.pkl' \\"
echo "       '$CONFIG'"
echo ""
echo "2. Evaluate temporal consistency (mVC8, mVC16):"
echo "   python3 tools/VC_perclip.py \\"
echo "       '$RUN_FOLDER/images' \\"
echo "       /workspace/TV3S/data/vspw/VSPW_480p"
echo ""
echo "       --metrics VC8 VC16 \\"
echo "       --vspw-root data/vspw/VSPW_480p"
echo ""
