#!/usr/bin/env python3
"""
Convert raw .npy predictions to PNG images organized by video.
This is much faster than re-running inference.
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import mmcv

def npy_to_images(predictions_dir, images_output_dir, palette_file=None):
    """
    Convert .npy prediction files to PNG images organized by video.
    
    Args:
        predictions_dir: Path to folder containing .npy files (e.g., results/run/predictions/)
        images_output_dir: Path to output folder for PNG images (e.g., results/run/images/)
        palette_file: Optional path to color palette file
    """
    
    # Create output directory
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Default VSPW palette (124 classes)
    palette = None
    if palette_file and os.path.exists(palette_file):
        palette = np.load(palette_file)
    else:
        # Generate a simple palette if not provided
        palette = np.random.randint(0, 255, (124, 3), dtype=np.uint8)
    
    # Find all .npy files
    npy_files = sorted(Path(predictions_dir).glob('**/*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {predictions_dir}")
        return
    
    print(f"Found {len(npy_files)} prediction files")
    
    for i, npy_file in enumerate(npy_files):
        if (i + 1) % 1000 == 0:
            print(f"Processing {i + 1}/{len(npy_files)}...")
        
        # Get relative path structure (e.g., video_name/00000001.npy)
        rel_path = npy_file.relative_to(predictions_dir)
        video_name = rel_path.parent.name
        frame_name = rel_path.stem + '.png'
        
        # Create output video folder
        video_output_dir = os.path.join(images_output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Load prediction
        pred = np.load(npy_file)
        
        # Convert to uint8 for PIL
        pred_uint8 = pred.astype(np.uint8)
        
        # Create image with palette
        img = Image.fromarray(pred_uint8, mode='P')
        img.putpalette(palette.flatten().tolist())
        
        # Save PNG
        output_path = os.path.join(video_output_dir, frame_name)
        img.save(output_path)
    
    print(f"✓ Converted {len(npy_files)} predictions to PNG images")
    print(f"✓ Images saved to: {images_output_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 tools/predictions_to_images.py <predictions_dir> [images_output_dir]")
        print()
        print("Example:")
        print("  python3 tools/predictions_to_images.py \\")
        print("      results/swins_baseline/predictions \\")
        print("      results/swins_baseline/images")
        sys.exit(1)
    
    predictions_dir = sys.argv[1]
    images_output_dir = sys.argv[2] if len(sys.argv) > 2 else predictions_dir.replace('predictions', 'images')
    
    if not os.path.exists(predictions_dir):
        print(f"Error: Predictions directory not found: {predictions_dir}")
        sys.exit(1)
    
    npy_to_images(predictions_dir, images_output_dir)
