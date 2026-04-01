#!/usr/bin/env python3
"""
Test DINOv3 forward pass exactly as evaluation would do it.
"""
import sys
sys.path.insert(0, '/workspace/TV3S')
sys.path.insert(0, '/workspace/TV3S/3rdparty/mmcv')

import torch
import numpy as np
from mmcv import Config
from mmseg.models import build_segmentor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import custom models to register them
from utils.models import *
import utils.datasets.vspw

def test_full_pipeline():
    """Test the complete pipeline with realistic data."""
    print("\n" + "="*80)
    print("FULL PIPELINE TEST: DINOv3 + TV3S Segmentation")
    print("="*80)
    
    # Load config
    config_path = '/workspace/TV3S/local_configs/tv3s/DINOv3/dinov3_hf_vits16.480x480.vspw2.160k.py'
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = build_segmentor(cfg.model)
    print(f"Model built successfully")
    
    # Load checkpoint if available
    checkpoint_dir = '/workspace/TV3S/work_dirs/dinov3_hf_vitb16_1sample'
    import os
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    
    if checkpoints:
        latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"Loading checkpoint: {latest_ckpt}")
        state_dict = torch.load(latest_ckpt, map_location='cpu')
        if 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded")
    else:
        print("⚠️ No checkpoint found, using randomly initialized weights")
    
    model.eval()
    
    # Test with different random seeds to check for non-determinism
    print("\n" + "-"*80)
    print("Testing determinism with same input across 3 runs:")
    print("-"*80)
    
    # Create fixed input
    x_fixed = torch.randn(2, 4, 3, 480, 480)
    print(f"Input: shape={x_fixed.shape}, mean={x_fixed.mean():.4f}, std={x_fixed.std():.4f}")
    
    outputs_list = []
    for run in range(3):
        img_metas = [{'ori_shape': (480, 480), 'img_shape': (480, 480), 'pad_shape': (480, 480),
                      'scale_factor': 1.0, 'flip': False, 'reduce_zero_label': False}
                     for _ in range(8)]
        
        with torch.no_grad():
            result = model.simple_test(list(x_fixed), img_metas)
        
        outputs_list.append(result)
        
        if isinstance(result, list) and len(result) > 0:
            pred = result[0]
            print(f"\nRun {run+1}:")
            print(f"  Output shape: {pred.shape}")
            print(f"  Classes present: {np.unique(pred)}")
            print(f"  Class range: [{pred.min()}, {pred.max()}]")
            print(f"  Num unique classes: {len(np.unique(pred))}")
            
            # Check spatial coherence
            # Compute local smoothness
            h, w = pred.shape
            if h > 1 and w > 1:
                diffs = np.abs(pred[:-1, :].astype(int) - pred[1:, :].astype(int))
                avg_diff = diffs.mean()
                print(f"  Avg vertical pixel difference: {avg_diff:.2f}")
                print(f"  Percentage same-class neighbors: {100*(diffs==0).sum()/(h-1)/w:.1f}%")
    
    # Compare outputs from different runs
    print("\n" + "-"*80)
    print("Comparing outputs across runs:")
    print("-"*80)
    
    for i in range(1, len(outputs_list)):
        if isinstance(outputs_list[i], list) and isinstance(outputs_list[0], list):
            if len(outputs_list[i]) > 0 and len(outputs_list[0]) > 0:
                pred1 = outputs_list[0][0]
                pred2 = outputs_list[i][0]
                if pred1.shape == pred2.shape:
                    diff = (pred1 != pred2).sum()
                    total = pred1.size
                    match_pct = 100 * (1 - diff/total)
                    print(f"Run 1 vs Run {i+1}: {match_pct:.1f}% pixels match")


def test_inference_modes():
    """Test different inference modes."""
    print("\n" + "="*80)
    print("INFERENCE MODES TEST")
    print("="*80)
    
    config_path = '/workspace/TV3S/local_configs/tv3s/DINOv3/dinov3_hf_vits16.480x480.vspw2.160k.py'
    cfg = Config.fromfile(config_path)
    
    model = build_segmentor(cfg.model)
    model.eval()
    
    # Test with training mode (should not happen but let's check)
    print("\nWith training=False (normal eval mode):")
    x = torch.randn(2, 4, 3, 480, 480)
    img_metas = [{'ori_shape': (480, 480), 'img_shape': (480, 480), 'pad_shape': (480, 480),
                  'scale_factor': 1.0, 'flip': False, 'reduce_zero_label': False}
                 for _ in range(8)]
    
    with torch.no_grad():
        result = model.simple_test(list(x), img_metas)
    
    if isinstance(result, list) and len(result) > 0:
        pred = result[0]
        print(f"  Output shape: {pred.shape}")
        print(f"  Classes: {len(np.unique(pred))} unique")
        print(f"  All zeros?: {(pred==0).all()}")
        print(f"  All same class?: {len(np.unique(pred)) == 1}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(pred, cmap='tab20')
        axes[0].set_title('Predicted Segmentation')
        axes[0].axis('off')
        
        # Histogram
        axes[1].hist(pred.flatten(), bins=min(50, len(np.unique(pred))))
        axes[1].set_title('Class Distribution')
        axes[1].set_xlabel('Class ID')
        axes[1].set_ylabel('Pixel Count')
        
        plt.tight_layout()
        plt.savefig('/workspace/TV3S/test_output_segmentation.png', dpi=100, bbox_inches='tight')
        print(f"✓ Saved visualization to test_output_segmentation.png")


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)
    
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_inference_modes()
    except Exception as e:
        print(f"Inference modes test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
