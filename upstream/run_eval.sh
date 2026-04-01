#!/bin/bash
set -e
cd /workspace/TV3S
export PYTHONPATH=/workspace/TV3S:$PYTHONPATH

echo "=== Step 1: Inference + accuracy metrics ==="
python3 tools/test.py local_configs/tv3s/B1/tv3s.b1.480x480.vspw2.160k.py resources/checkpoints/B1/iter_160000.pth --eval mIoU --out results/b1_predictions.pkl

echo "=== Step 2: Format predictions to images ==="
python3 tools/format_predictions.py results/b1_predictions.pkl local_configs/tv3s/B1/tv3s.b1.480x480.vspw2.160k.py results/b1_images/

echo "=== Step 3: Temporal consistency (VC8, VC16) ==="
python3 tools/VC_perclip.py results/b1_images/result_submission data/vspw/VSPW_480p
