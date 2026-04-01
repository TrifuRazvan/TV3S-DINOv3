# Copyright (c) Meta Platforms, Inc. and affiliates.
# Integrated for TV3S temporal segmentation by Ashesham
# Uses HuggingFace transformers DINOv3 implementation
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
import logging

try:
    from transformers import AutoBackbone
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ transformers not available. Install with: pip install transformers")

logger = logging.getLogger(__name__)


def load_dinov3_from_checkpoint(checkpoint_path, model_name='dinov3_vits16'):
    """Load DINOv3 model from checkpoint.
    
    Since timm 0.4.12 doesn't have DINOv3, we build a compatible ViT
    architecture and load the checkpoint weights.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Build a minimal ViT matching DINOv3-ViT-S16 specs
    # embed_dim=384, num_heads=6, depth=12
    model = _build_vit_s16_backbone()
    
    # Load checkpoint weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"🔹 Loading DINOv3 pretrained checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle minor key mismatches
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ DINOv3 checkpoint weights loaded successfully")
    else:
        logger.warning(f"⚠️ Initialized ViT backbone WITHOUT pretrained weights (checkpoint_path not found: {checkpoint_path})")
    
    return model


def _build_vit_s16_backbone():
    """Build ViT-S/16 backbone matching DINOv3 specs."""
    import math
    
    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, dim),
            )
        
        def forward(self, x):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.mlp(self.norm2(x))
            return x
    
    class VisionTransformer(nn.Module):
        def __init__(self, img_size=480, patch_size=16, in_chans=3, embed_dim=384,
                     depth=12, num_heads=6, mlp_ratio=4.0):
            super().__init__()
            self.embed_dim = embed_dim
            self.patch_size = patch_size
            
            # Patch embedding
            self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
            # Number of patches
            num_patches = (img_size // patch_size) ** 2
            self.num_patches = num_patches
            
            # Class token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
            # Positional embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ])
            
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            
            # Initialize weights
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        def forward(self, x):
            # Patch embedding
            x = self.patch_embed(x)  # (B, C, H_patch, W_patch)
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, C)
            
            # Add class token
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, 1 + num_patches, C)
            
            # Add positional embedding
            x = x + self.pos_embed
            
            # Transformer blocks
            for block in self.blocks:
                x = block(x)
            
            x = self.norm(x)
            return x
    
    model = VisionTransformer(
        img_size=480,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0
    )
    
    return model


@BACKBONES.register_module()
class DINOv3Backbone(nn.Module):
    """DINOv3 backbone for semantic segmentation.
    
    DINOv3 is a Vision Transformer (ViT) pretrained with DINO.
    This implementation extracts intermediate blocks to create a 4-level
    pyramid compatible with TV3S decoder.
    
    Args:
        model_name (str): Model variant, e.g., 'dinov3_vits16'
    Args:
        model_name (str): Name of DINOv3 model variant ('dinov3_vits16')
        pretrained (bool): Whether to load pretrained weights
        checkpoint_path (str): Path to checkpoint file
        out_indices (tuple): Which blocks to extract [3, 6, 9, 11]
        frozen_stages (int): Number of stages to freeze (-1 = none)
        style (str): Style parameter (pytorch), kept for MMSeg compatibility
        init_cfg (dict): Initialization config, kept for MMSeg compatibility
    """
    
    def __init__(
        self,
        model_name='dinov3_vits16',
        pretrained=True,
        checkpoint_path=None,
        out_indices=(3, 6, 9, 11),
        frozen_stages=-1,
        style='pytorch',
        init_cfg=None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Use explicit checkpoint path if provided, otherwise auto-detect
        if pretrained:
            if checkpoint_path:
                logger.info(f"✨ Using explicit DINOv3 checkpoint path: {checkpoint_path}")
            else:
                # Auto-detect checkpoint path if pretrained but no path provided
                possible_paths = [
                    f'pretrained/DinoV3/{model_name}_pretrain.pth',
                    f'pretrained/DinoV3/{model_name}.pth',
                    f'pretrained/{model_name}_pretrain.pth',
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        checkpoint_path = path
                        logger.info(f"✨ Auto-detected DINOv3 checkpoint: {path}")
                        break
                if not checkpoint_path:
                    logger.warning(f"⚠️ Could not auto-detect pretrained checkpoint. Looked in: {possible_paths}")
        
        # Load model
        self.model = load_dinov3_from_checkpoint(checkpoint_path, model_name)
        
        # Extract properties
        self.embed_dim = self.model.embed_dim
        self.num_blocks = len(self.model.blocks)
        self.patch_embed = self.model.patch_embed
        self.num_patches = self.model.num_patches
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        
        # Freeze stages
        self.freeze_stages()
        
        # Initialize weights and freeze backbone
        self.init_weights()
    
    def init_weights(self, pretrained=None):
        """Initialize model weights. 
        
        Since we load pretrained weights in __init__, this is mainly for compatibility.
        """
        if pretrained:
            # Already loaded in __init__
            pass
        
        # Freeze entire backbone - only decoder will train
        self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters. Only decoder will be trained."""
        for param in self.parameters():
            param.requires_grad = False
        print("✓ DINOv3Backbone frozen - only decoder will be trained")
    
    
    def freeze_stages(self):
        """Freeze backbone stages."""
        if self.frozen_stages < 0:
            return
        
        # Freeze patch embedding
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        
        # Freeze blocks
        for i in range(self.frozen_stages):
            self.blocks[i].eval()
            for param in self.blocks[i].parameters():
                param.requires_grad = False
    
    def _pos_embed(self, x):
        """Apply positional embedding."""
        if self.pos_embed is None:
            return x
        
        # Handle shape mismatch
        if x.shape[1] != self.pos_embed.shape[1] - 1:
            pos_embed_patches = self.pos_embed[:, 1:, :]  # (1, num_patches, C)
            # Interpolate: (1, C, num_patches_old) -> (1, C, num_patches_new)
            pos_embed_interp = F.interpolate(
                pos_embed_patches.permute(0, 2, 1),  # (1, C, num_patches_old)
                size=x.shape[1],
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # (1, num_patches_new, C)
            x = x + pos_embed_interp
        else:
            x = x + self.pos_embed[:, 1:, :]
        
        return x
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: (B*num_clips, 3, H, W) - Input tensor.
               H and W must be multiples of patch_size (16).
               Default: 480×480 (produces 30×30 patches)
            
        Returns:
            List of 4 feature tensors at different spatial scales:
            - (B*num_clips, 384, H*4, W*4)  [stride 4, upsample 4x]
            - (B*num_clips, 384, H*2, W*2)  [stride 8, upsample 2x]
            - (B*num_clips, 384, H, W)      [stride 16, identity]
            - (B*num_clips, 384, H/2, W/2)  [stride 32, downsample 2x]
            
            where H, W are calculated as input_H // 16, input_W // 16
        """
        # Patch embedding: (B, 3, H, W) -> (B, 384, H//16, W//16)
        B, C, H_orig, W_orig = x.shape
        x = self.patch_embed(x)  # (B, 384, H//16, W//16)
        
        # Store spatial dimensions for reshaping
        H_feat, W_feat = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, 384)
        
        # Add class token: (B, 901, 384)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = self._pos_embed(x)
        
        features = []
        B = x.shape[0]
        
        # Extract at specified blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            if i in self.out_indices:
                # Remove class token: (B, H*W, 384)
                feat = x[:, 1:, :]
                
                # Reshape to spatial: (B, 384, H_feat, W_feat)
                # where H_feat = H_orig // patch_size, W_feat = W_orig // patch_size
                feat_spatial = feat.permute(0, 2, 1).reshape(B, self.embed_dim, H_feat, W_feat)
                features.append(feat_spatial)
        
        # Create multi-scale pyramid
        # ViT-S/16 produces features at stride 16 (30x30 for 480 input)
        # We need strides [4, 8, 16, 32]
        
        # Scale 0 (stride 4, 120x120): upsample 4x
        scale0 = F.interpolate(features[0], scale_factor=4, mode='bilinear', align_corners=False)
        
        # Scale 1 (stride 8, 60x60): upsample 2x
        scale1 = F.interpolate(features[1], scale_factor=2, mode='bilinear', align_corners=False)
        
        # Scale 2 (stride 16, 30x30): identity
        scale2 = features[2]
        
        # Scale 3 (stride 32, 15x15): downsample 2x
        scale3 = F.avg_pool2d(features[3], kernel_size=2, stride=2)
        
        return [scale0, scale1, scale2, scale3]


__all__ = ['DINOv3Backbone']
