#!/usr/bin/env python3
"""
Convert .npy prediction files to PNG images organized by video.
Uses the dataset's format_results method to ensure proper formatting.
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.insert(0, '/workspace/TV3S')

# Import utils to register custom datasets
import utils

from mmcv.runner import get_dist_info
import mmcv

def format_predictions(pkl_file, output_dir, config_file):
    """
    Load predictions from .pkl file and format them as images.
    
    Args:
        pkl_file: Path to predictions.pkl file
        output_dir: Directory to save formatted images
        config_file: Path to config file (to build dataset)
    """
    
    # Load predictions
    print(f"Loading predictions from: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        outputs = pickle.load(f)
    
    print(f"Loaded {len(outputs)} predictions")
    
    # Build dataset from config
    from mmcv import Config
    cfg = Config.fromfile(config_file)
    
    from mmseg.datasets import build_dataset
    dataset = build_dataset(cfg.data.test)
    
    # Format results
    print(f"Formatting results to: {output_dir}")
    dataset.format_results(outputs, output_dir)
    
    print(f"✓ Images saved to: {output_dir}/result_submission/")
    
    # Move images to proper location
    result_dir = os.path.join(output_dir, 'result_submission')
    if os.path.exists(result_dir):
        print(f"Found {len(os.listdir(result_dir))} items in result_submission")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 tools/format_predictions.py <predictions.pkl> <config.py> [output_dir]")
        print()
        print("Example:")
        print("  python3 tools/format_predictions.py \\")
        print("      results/swins_run_20251201_190250/swins_predictions.pkl \\")
        print("      local_configs/tv3s/Swin/swins_realshift_w20_s10.480x480.vspw2.160k.py \\")
        print("      results/swins_run_20251201_190250/")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    config_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(pkl_file)
    
    if not os.path.exists(pkl_file):
        print(f"Error: Pickle file not found: {pkl_file}")
        sys.exit(1)
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    format_predictions(pkl_file, output_dir, config_file)
