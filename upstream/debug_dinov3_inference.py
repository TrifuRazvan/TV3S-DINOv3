"""
Debug script: Analyze DINOv3 + TV3S behavior during inference vs training
Hypothesis: Loss decreases but eval results are nonsense = inference mode issue
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/workspace/TV3S')

# Test 1: Check backbone feature extraction consistency
print("=" * 80)
print("TEST 1: DINOv3 Feature Extraction Consistency")
print("=" * 80)

from utils.models.dinov3_backbone import DINOv3Backbone

backbone = DINOv3Backbone(
    model_name='dinov3_vits16',
    pretrained=True,
    checkpoint_path='pretrained/DinoV3/dinov3_vits16_pretrain.pth',
    out_indices=(3, 6, 9, 11),
    frozen_stages=-1
)

# Create dummy batch of 8 frames (2 clips × 4 frames)
dummy_input = torch.randn(8, 3, 480, 480)

backbone.eval()  # Set to eval mode
with torch.no_grad():
    features_eval = backbone(dummy_input)
    
print(f"Input shape: {dummy_input.shape}")
print(f"Output features (4 scales):")
for i, feat in enumerate(features_eval):
    print(f"  Scale {i}: {feat.shape} (stride {4 * (2**i)})")

# Check consistency: same input should give same output
with torch.no_grad():
    features_eval2 = backbone(dummy_input)
    
for i, (f1, f2) in enumerate(zip(features_eval, features_eval2)):
    mse = ((f1 - f2) ** 2).mean().item()
    print(f"  MSE between two forward passes (scale {i}): {mse:.2e}")

print("\n" + "=" * 80)
print("TEST 2: Check if DINOv3 Backbone is actually frozen")
print("=" * 80)

for name, param in backbone.named_parameters():
    if param.requires_grad:
        print(f"WARNING: Parameter '{name}' has requires_grad=True")
        break
else:
    print("✓ All backbone parameters frozen (requires_grad=False)")

print("\n" + "=" * 80)
print("TEST 3: Feature value statistics (are they reasonable?)")
print("=" * 80)

with torch.no_grad():
    features = backbone(dummy_input)
    
for i, feat in enumerate(features):
    mean_val = feat.mean().item()
    std_val = feat.std().item()
    min_val = feat.min().item()
    max_val = feat.max().item()
    print(f"Scale {i}: mean={mean_val:+.4f}, std={std_val:.4f}, min={min_val:+.4f}, max={max_val:+.4f}")

print("\n" + "=" * 80)
print("TEST 4: Compare DINOv3 with expected SegFormer features")
print("=" * 80)

print("SegFormer feature dims: [64, 128, 320, 512]")
print("DINOv3 feature dims:    [384, 384, 384, 384]")
print("\nIssue: All DINOv3 scales have the same channel dimension!")
print("This means the decoder MLP layers must project:")
print("  384 -> 256 (linear_c4, linear_c3, linear_c2, linear_c1)")
print("\nVs SegFormer which already has feature pyramid:")
print("  512->256, 320->256, 128->256, 64->256")

print("\n" + "=" * 80)
print("TEST 5: Load a checkpoint and check if dimensions match")
print("=" * 80)

# Try to simulate what happens in training
try:
    from mmseg.models import build_segmentor
    config_path = '/workspace/TV3S/local_configs/tv3s/DINOv3/dinov3_vits16.480x480.vspw2.160k.py'
    
    # This would need the full training setup
    print("(Skipping full model build - requires complete training environment)")
except Exception as e:
    print(f"Could not build full model: {e}")

print("\n" + "=" * 80)
print("TEST 6: Analyze potential inference issues")
print("=" * 80)

print("""
Key finding from tv3s_head.py line 979-980:
    if not self.training and num_clips!=self.num_clips:
        return x[:,-1]  # RETURNS ONLY THE LAST FRAME!
        
This means during evaluation:
1. All 4 video frames go through the decoder
2. But only the LAST frame prediction is returned
3. The Mamba temporal fusion might not be working correctly

Questions:
1. Is num_clips set correctly during evaluation?
2. Is self.num_clips initialized properly?
3. Should all frames be evaluated or just the last one?
""")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
1. **Check forward pass shapes during evaluation**:
   - Add debugging prints in TV3SHead_shift.forward()
   - Log what num_clips and batch_size values are

2. **Verify positional embeddings**:
   - DINOv3 uses its own pos_embed interpolation
   - Decoder adds another pos_embed on top
   - These might conflict or cause gradient issues

3. **Test inference mode**:
   - Run evaluation on a small validation set
   - Check if predictions make sense (not just top class everywhere)
   - Compare with SegFormer baseline

4. **Check loss computation**:
   - Is loss applied to correct frame during training?
   - Are targets properly aligned with video clips?

5. **Feature distribution comparison**:
   - Compare DINOv3 feature statistics with SegFormer
   - DINOv3 features might have very different ranges/distributions
   - May need layer normalization or scaling adjustments
""")
