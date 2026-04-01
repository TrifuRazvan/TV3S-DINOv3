################################################################################
# DINOv3 Backbone for TV3S using HuggingFace Transformers
################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Integrated for TV3S temporal segmentation by Ashesham
#
# This module wraps Meta AI's official DINOv3 Vision Transformer from the
# HuggingFace transformers library, providing a backbone for video segmentation.
#
# Key Features:
# - Official Meta DINOv3 implementation (not custom)
# - Automatic pretrained weight loading from HuggingFace Hub
# - Multi-scale feature extraction from intermediate transformer layers
# - Support for all DINOv3 model variants (ViT-S/B/L/g)
# - Proper handling of register tokens (4 extra tokens in DINOv3)
#
# Requirements:
# - transformers >= 4.50.0
# - HuggingFace account with model access (models are gated)
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
import logging

try:
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
    from transformers import AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ DINOv3 not available in transformers. Install transformers>=4.50")

logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class DINOv3BackboneHF(nn.Module):
    """DINOv3 Vision Transformer backbone using HuggingFace transformers.
    
    This wrapper loads the official Meta AI DINOv3 model from HuggingFace
    and extracts multi-scale features from intermediate transformer layers
    for dense prediction tasks like semantic segmentation.
    
    Architecture:
        DINOv3 is a self-supervised Vision Transformer (ViT) trained with
        knowledge distillation. Key components:
        - Patch embedding (16x16 or 14x14 patches)
        - 12/24/40 transformer layers (depending on variant)
        - RoPE (Rotary Position Embeddings) instead of learned pos encoding
        - LayerScale for better training stability
        - 4 register tokens (special tokens for better feature learning)
    
    Available Models (HuggingFace Hub):
        ┌──────────────────┬───────────────────────────────────────────┬────────┬─────────┐
        │ Variant          │ HuggingFace Model ID                      │ Params │ Hidden  │
        ├──────────────────┼───────────────────────────────────────────┼────────┼─────────┤
        │ ViT-S/16 ⭐      │ facebook/dinov3-vits16-pretrain-lvd1689m  │ 21M    │ 384     │
        │ ViT-S/16+        │ facebook/dinov3-vits16plus-pretrain-...   │ 21M    │ 384     │
        │ ViT-B/16         │ facebook/dinov3-vitb16-pretrain-lvd1689m  │ 86M    │ 768     │
        │ ViT-L/16         │ facebook/dinov3-vitl16-pretrain-lvd1689m  │ 304M   │ 1024    │
        │ ViT-H/16+        │ facebook/dinov3-vith16plus-pretrain-...   │ 630M   │ 1280    │
        │ ViT-g/16 (7B)    │ facebook/dinov3-vit7b16-pretrain-lvd1689m │ 1.1B   │ 1536    │
        └──────────────────┴───────────────────────────────────────────┴────────┴─────────┘
        
        Pretraining Datasets:
        - LVD-1689M: Primary pretraining (recommended)
        - SAT-493M: Satellite imagery pretraining (alternative)
        
        Example full model IDs:
        - facebook/dinov3-vits16-pretrain-lvd1689m
        - facebook/dinov3-vitb16-pretrain-lvd1689m
        - facebook/dinov3-vitl16-pretrain-sat493m (satellite variant)
    
    Args:
        model_name (str): HuggingFace model identifier. See table above.
            Default: 'facebook/dinov3-vits16-pretrain-lvd1689m'
        
        pretrained (bool): Whether to load pretrained weights from HuggingFace.
            - True: Auto-download weights (requires HF authentication)
            - False: Random initialization (not recommended)
            Default: True
        
        out_indices (tuple): Transformer layer indices to extract features from.
            DINOv3-ViT-S/B has 12 layers, indexed 0-11.
            Typical choices:
            - (3, 6, 9, 11): Evenly spaced layers (recommended)
            - (2, 5, 8, 11): Alternative spacing
            - (8, 9, 10, 11): Late layers only (finer features)
            Default: (3, 6, 9, 11)
        
        freeze_backbone (bool): Whether to freeze all backbone parameters.
            - True: Only train decoder (faster, less VRAM, recommended)
            - False: Fine-tune entire model (slower, needs more VRAM)
            Default: True
        
        style (str): MMSeg compatibility parameter. Keep as 'pytorch'.
        
        init_cfg (dict): MMSeg initialization config. Not used (HF handles init).
    
    Example:
        >>> backbone = DINOv3BackboneHF(
        ...     model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',  # Switch to ViT-B
        ...     pretrained=True,
        ...     out_indices=(3, 6, 9, 11),
        ...     freeze_backbone=True
        ... )
        >>> x = torch.randn(1, 3, 480, 480)
        >>> features = backbone(x)  # List of 4 feature maps
        >>> print([f.shape for f in features])
        # Output: [torch.Size([1, 768, 120, 120]),  # Stride 4 (ViT-B hidden=768)
        #          torch.Size([1, 768, 60, 60]),    # Stride 8
        #          torch.Size([1, 768, 30, 30]),    # Stride 16
        #          torch.Size([1, 768, 15, 15])]    # Stride 32
    """
    
    def __init__(
        self,
        model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
        pretrained=True,
        out_indices=(3, 6, 9, 11),
        freeze_backbone=True,
        style='pytorch',
        init_cfg=None
    ):
        super().__init__()
        
        if not HF_AVAILABLE:
            raise ImportError(
                "DINOv3 not available in transformers. "
                "Upgrade with: pip install transformers>=4.50"
            )
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.freeze_backbone = freeze_backbone
        
        logger.info(f"✨ Loading DINOv3 from HuggingFace: {model_name}")
        logger.info(f"   Output layer indices: {out_indices}")
        
        # Load DINOv3 model
        try:
            if pretrained:
                self.backbone = DINOv3ViTModel.from_pretrained(model_name)
                logger.info(f"✅ Loaded pretrained DINOv3 weights from {model_name}")
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.backbone = DINOv3ViTModel(config)
                logger.info(f"⚠️ Initialized DINOv3 with random weights")
        except Exception as e:
            logger.error(f"❌ Failed to load DINOv3: {e}")
            logger.info(f"💡 Note: Some DINOv3 models are gated and require HuggingFace authentication")
            raise
        
        # Get feature dimensions
        self.embed_dim = self.backbone.config.hidden_size
        self.num_stages = len(out_indices)
        
        logger.info(f"   Feature dim: {self.embed_dim}")
        logger.info(f"   Num stages: {self.num_stages}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters so only decoder trains.
        
        This is the recommended approach for transfer learning:
        - Faster training (fewer parameters to update)
        - Less VRAM usage (no gradients for backbone)
        - Better generalization (pretrained features preserved)
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("✅ DINOv3 backbone frozen - only decoder will be trained")
    
    def init_weights(self, pretrained=None):
        """Initialize model weights.
        
        HuggingFace models handle their own initialization via from_pretrained().
        This method is kept for MMSeg compatibility but does nothing.
        
        Args:
            pretrained: Ignored (weights already loaded in __init__)
        """
        # Weights already loaded by HuggingFace in __init__
        if self.freeze_backbone:
            self._freeze_backbone()
    
    def forward(self, x):
        """Extract multi-scale features from input image.
        
        Process:
        1. Input image → DINOv3 transformer (gets all layer outputs)
        2. Extract features from specified layers (out_indices)
        3. Remove CLS and register tokens, keep only spatial patches
        4. Reshape from sequence (B, N, C) to spatial (B, C, H, W)
        5. Resize to target strides [4, 8, 16, 32] via bilinear interpolation
        
        DINOv3 Token Structure:
            Input: 480x480 image
            Patches: 480/16 = 30x30 = 900 patches
            Tokens: [CLS (1)] + [Registers (4)] + [Patches (900)] = 905 total
            
            We extract only the 900 patch tokens for spatial features.
        
        Args:
            x (Tensor): Input image tensor
                Shape: (B, 3, H, W) where H=W=480 typically
        
        Returns:
            list[Tensor]: Multi-scale feature maps (4 scales)
                Shapes for 480x480 input:
                - [0]: (B, embed_dim, 120, 120) - stride 4 (finest)
                - [1]: (B, embed_dim, 60, 60)   - stride 8
                - [2]: (B, embed_dim, 30, 30)   - stride 16
                - [3]: (B, embed_dim, 15, 15)   - stride 32 (coarsest)
                
                embed_dim varies by model:
                - ViT-S: 384
                - ViT-B: 768
                - ViT-L: 1024
                - ViT-H/16+: 1280
                - ViT-g (7B): 1536
        
        Example:
            >>> x = torch.randn(2, 3, 480, 480)  # Batch of 2 images
            >>> features = backbone(x)
            >>> len(features)  # 4 feature maps
            4
            >>> features[0].shape  # Finest resolution
            torch.Size([2, 384, 120, 120])
            >>> features[3].shape  # Coarsest resolution
            torch.Size([2, 384, 15, 15])
        """
        B, C, H, W = x.shape
        
        # Get intermediate hidden states from all transformer layers
        # output_hidden_states=True returns tuple of (layer_0, layer_1, ..., layer_12)
        # where layer_0 is patch embeddings, layer_1-12 are transformer outputs
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of 13 tensors
        
        # Calculate spatial dimensions
        patch_size = self.backbone.config.patch_size  # 16 for ViT-S/16
        feat_h = feat_w = H // patch_size  # 480 // 16 = 30
        num_patches = feat_h * feat_w      # 30 * 30 = 900
        
        # Extract features from specified transformer layers
        feat_list = []
        for idx in self.out_indices:
            # Get hidden state from layer idx
            # +1 offset because hidden_states[0] is patch embeddings, not transformer layer
            feat = hidden_states[idx + 1]  # Shape: (B, 905, embed_dim)
            
            # DINOv3 tokens: [CLS, reg1, reg2, reg3, reg4, patch1, ..., patch900]
            # Extract only the spatial patch tokens (last 900 tokens)
            feat = feat[:, -num_patches:, :]  # Shape: (B, 900, embed_dim)
            
            # Reshape from sequence to spatial grid
            # (B, num_patches, C) → (B, C, feat_h, feat_w)
            feat = feat.permute(0, 2, 1).reshape(B, self.embed_dim, feat_h, feat_w)
            feat_list.append(feat)
        
        # Resize features to multi-scale pyramid with strides [4, 8, 16, 32]
        # For 480x480 input: [120, 60, 30, 15]
        target_sizes = [120, 60, 30, 15]
        
        outputs = []
        for feat, target_size in zip(feat_list, target_sizes):
            # Bilinear interpolation to resize features
            resized = F.interpolate(
                feat,
                size=(target_size, target_size),
                mode='bilinear',  # Smooth interpolation (vs 'nearest' or 'bicubic')
                align_corners=False  # Don't align corner pixels (standard for segmentation)
            )
            outputs.append(resized)
        
        return outputs


################################################################################
# Module Export
################################################################################
__all__ = ['DINOv3BackboneHF']


################################################################################
# USAGE NOTES
################################################################################
# This backbone is designed for frozen-backbone transfer learning:
# 1. Pretrained DINOv3 features are preserved (frozen)
# 2. Only the task-specific decoder is trained
# 3. Much faster training and lower VRAM usage
#
# To switch models, just change model_name in config file:
#
# Examples:
#   backbone=dict(
#       type='DINOv3BackboneHF',
#       model_name='facebook/dinov3-vits16-pretrain-lvd1689m',  # ViT-S/16, 384 dim
#       pretrained=True,
#       freeze_backbone=True
#   )
#
#   backbone=dict(
#       type='DINOv3BackboneHF',
#       model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',  # ViT-B/16, 768 dim
#       pretrained=True,
#       freeze_backbone=True
#   )
#
#   backbone=dict(
#       type='DINOv3BackboneHF',
#       model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',  # ViT-L/16, 1024 dim
#       pretrained=True,
#       freeze_backbone=True
#   )
#
# Update in_channels in decode_head to match backbone's hidden_size:
#   ViT-S/16: in_channels=384
#   ViT-B/16: in_channels=768
#   ViT-L/16: in_channels=1024
#   ViT-H/16+: in_channels=1280
#   ViT-g/16: in_channels=1536
#
# HuggingFace Authentication (gated models):
# All DINOv3 models require approval. Before first use:
# 1. Visit: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
# 2. Click "Agree and access repository"
# 3. Get token: https://huggingface.co/settings/tokens (Read permission)
# 4. Login: huggingface-cli login
################################################################################
