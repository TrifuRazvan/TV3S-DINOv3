#!/usr/bin/env python3
"""
Test script to debug DINOv3 + TV3S forward pass for inference.
"""
import sys
import os
sys.path.insert(0, '/workspace/TV3S')
sys.path.insert(0, '/workspace/TV3S/3rdparty/mmcv')
sys.path.insert(0, '/workspace/TV3S/3rdparty/mamba')

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmseg.models import build_segmentor
from mmcv import Config

# Import custom models
from utils.models import DINOv3BackboneHF, TV3SHead_shift

def test_dinov3_backbone_output_range():
    """Test if DINOv3 backbone outputs are in reasonable ranges."""
    print("\n" + "="*80)
    print("TEST 1: DINOv3 Backbone Output Range Analysis")
    print("="*80)
    
    backbone = DINOv3BackboneHF(
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        pretrained=True,
        out_indices=(3, 6, 9, 11),
        freeze_backbone=True
    )
    backbone.eval()
    
    # Test input
    x = torch.randn(2, 3, 480, 480)
    
    with torch.no_grad():
        outputs = backbone(x)
    
    print(f"Input shape: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"Output is tuple with {len(outputs)} scales")
    
    for i, feat in enumerate(outputs):
        print(f"\nScale {i}:")
        print(f"  Shape: {feat.shape}")
        print(f"  dtype: {feat.dtype}")
        print(f"  Mean: {feat.mean():.6f}")
        print(f"  Std:  {feat.std():.6f}")
        print(f"  Min:  {feat.min():.6f}")
        print(f"  Max:  {feat.max():.6f}")
        
        # Check for NaN/Inf
        if torch.isnan(feat).any():
            print(f"  ⚠️  CONTAINS NaN!")
        if torch.isinf(feat).any():
            print(f"  ⚠️  CONTAINS Inf!")


def test_mlp_projections():
    """Test if MLP projections preserve information."""
    print("\n" + "="*80)
    print("TEST 2: MLP Projection Impact")
    print("="*80)
    
    from utils.models.tv3s_head import MLP
    
    # Simulate DINOv3 output
    feat = torch.randn(2, 768, 30, 30)
    print(f"Input to MLP: shape={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}")
    
    mlp = MLP(input_dim=768, embed_dim=256)
    
    with torch.no_grad():
        output = mlp(feat)
    
    print(f"Output from MLP: shape={output.shape}, mean={output.mean():.4f}, std={output.std():.4f}")
    
    # Reshape back
    output_reshaped = output.permute(0, 2, 1).reshape(2, -1, 30, 30)
    print(f"After reshape: shape={output_reshaped.shape}, mean={output_reshaped.mean():.4f}, std={output_reshaped.std():.4f}")


def test_with_training_data():
    """Load actual training data and test forward pass."""
    print("\n" + "="*80)
    print("TEST 3: Forward Pass with Actual Training Setup")
    print("="*80)
    
    # Load config
    config_path = '/workspace/TV3S/local_configs/tv3s/DINOv3/dinov3_hf_vits16.480x480.vspw2.160k.py'
    cfg = Config.fromfile(config_path)
    
    print(f"Config loaded: {config_path}")
    print(f"Model type: {cfg.model['type']}")
    print(f"Backbone type: {cfg.model['backbone']['type']}")
    print(f"Decoder type: {cfg.model['decode_head']['type']}")
    print(f"In channels: {cfg.model['decode_head']['in_channels']}")
    print(f"Model type (decoder): {cfg.model['decode_head']['decoder_params'].get('model_type', 'NOT SET')}")
    
    # Build model
    try:
        model = build_segmentor(cfg.model)
        print(f"✓ Model built successfully")
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    model.eval()
    
    # Test forward pass
    batch_size = 2
    num_clips = 4
    h, w = 480, 480
    
    # Simulate video batch: [batch, clips, 3, h, w]
    img = torch.randn(batch_size, num_clips, 3, h, w).to(next(model.parameters()).device)
    print(f"\nInput tensor: shape={img.shape}, dtype={img.dtype}, device={img.device}")
    
    # Dummy image metadata
    img_metas = [{'ori_shape': (h, w), 'img_shape': (h, w), 'pad_shape': (h, w), 
                  'scale_factor': 1.0, 'flip': False, 'reduce_zero_label': False} 
                 for _ in range(batch_size * num_clips)]
    
    # Forward pass
    with torch.no_grad():
        try:
            # The model expects imgs as list of tensors
            result = model.simple_test(list(img), img_metas)
            print(f"✓ Forward pass successful")
            print(f"  Output type: {type(result)}")
            if isinstance(result, list):
                print(f"  Output list length: {len(result)}")
                if len(result) > 0:
                    print(f"  First element shape: {result[0].shape}")
                    print(f"  First element dtype: {result[0].dtype}")
                    print(f"  First element min/max: {result[0].min()}/{result[0].max()}")
                    print(f"  First element mean: {result[0].mean()}")
                    
                    # Check if output has spatial coherence
                    # Random noise would have low spatial correlation
                    # Real segmentation would have high spatial correlation
                    first_map = result[0]
                    spatial_var = first_map.reshape(-1, first_map.shape[-1]).var(dim=0).mean()
                    print(f"  Spatial variance (higher = more coherent): {spatial_var:.4f}")
                    
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()


def test_backbone_feature_statistics():
    """Detailed analysis of backbone feature statistics."""
    print("\n" + "="*80)
    print("TEST 4: Backbone Feature Statistics (Similar to Training)")
    print("="*80)
    
    backbone = DINOv3BackboneHF(
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        pretrained=True,
        out_indices=(3, 6, 9, 11),
        freeze_backbone=True
    )
    backbone.eval()
    
    # Multiple forward passes to check consistency
    print("Running 5 forward passes with same input:")
    x = torch.ones(1, 3, 480, 480)  # Uniform input for consistency check
    
    outputs_per_scale = [[] for _ in range(4)]
    
    for i in range(5):
        with torch.no_grad():
            outputs = backbone(x)
        for j, feat in enumerate(outputs):
            outputs_per_scale[j].append(feat.clone())
    
    for scale_idx in range(4):
        print(f"\nScale {scale_idx} consistency check:")
        outputs = outputs_per_scale[scale_idx]
        
        # Check if outputs are identical (they should be since input is identical)
        for i in range(1, len(outputs)):
            diff = (outputs[i] - outputs[0]).abs().max().item()
            print(f"  Run {i} max diff from Run 0: {diff:.2e}")
    
    # Now test with random input
    print("\n\nRandom input test:")
    x_random = torch.randn(4, 3, 480, 480)
    
    with torch.no_grad():
        outputs = backbone(x_random)
    
    for i, feat in enumerate(outputs):
        print(f"\nScale {i}:")
        print(f"  Shape: {feat.shape}")
        print(f"  Mean: {feat.mean():.4f}")
        print(f"  Std:  {feat.std():.4f}")
        print(f"  Range: [{feat.min():.4f}, {feat.max():.4f}]")
        
        # Check sparsity
        num_zero = (feat == 0).sum().item()
        total = feat.numel()
        sparsity = 100 * num_zero / total
        print(f"  Sparsity: {sparsity:.2f}%")
        
        # Check for outliers
        abs_feat = feat.abs()
        mean_val = abs_feat.mean()
        std_val = abs_feat.std()
        outliers_3sigma = (abs_feat > mean_val + 3*std_val).sum().item()
        print(f"  Outliers (>3σ): {outliers_3sigma} ({100*outliers_3sigma/total:.2f}%)")


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)
    
    # Test 1: Output range
    try:
        test_dinov3_backbone_output_range()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: MLP projections
    try:
        test_mlp_projections()
    except Exception as e:
        print(f"Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Feature statistics (important)
    try:
        test_backbone_feature_statistics()
    except Exception as e:
        print(f"Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Full forward pass (requires config + model)
    try:
        test_with_training_data()
    except Exception as e:
        print(f"Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
