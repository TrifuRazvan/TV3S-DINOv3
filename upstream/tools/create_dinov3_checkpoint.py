#!/usr/bin/env python3
"""
Create a minimal checkpoint file for DINOv3 inference.
This checkpoint only contains metadata and will let the model load
the pretrained DINOv3 backbone weights via init_weights().
"""

import torch
import pickle

# Create minimal checkpoint with VSPW metadata
checkpoint = {
    'meta': {
        'CLASSES': tuple([f'class_{i}' for i in range(124)]),  # 124 VSPW classes
        'PALETTE': None  # Will be set by dataset
    },
    'state_dict': {}  # Empty - backbone weights will be loaded via init_weights()
}

# Save checkpoint
output_path = 'resources/checkpoints/DinoV3/dinov3_vits16_backbone_only.pth'
torch.save(checkpoint, output_path)
print(f"✓ Created checkpoint at: {output_path}")
print(f"  This checkpoint will load the pretrained DINOv3 backbone")
print(f"  The decoder head will be randomly initialized")
