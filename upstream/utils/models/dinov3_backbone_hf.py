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
    print("WARNING: DINOv3 not available. Install transformers>=4.50")

logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class DINOv3BackboneHF(nn.Module):
    """DINOv3 Vision Transformer backbone using HuggingFace transformers.

    Wraps the official Meta DINOv3 model and exposes 4 intermediate ViT layers
    as a multi-scale feature pyramid compatible with MMSeg decode heads.

    DINOv3 is isotropic (all layers output the same spatial resolution),
    so the multi-scale pyramid is created by bilinear interpolation.

    Token layout: [CLS (1)] + [Registers (4)] + [Patches (H/p * W/p)]
    We discard CLS and register tokens, keeping only the spatial patch tokens.

    Args:
        model_name (str): HuggingFace model ID.
            Default: 'facebook/dinov3-vits16-pretrain-lvd1689m'
        pretrained (bool): Load pretrained weights from HuggingFace. Default: True
        out_indices (tuple): 0-indexed transformer layer indices to extract.
            With +1 offset applied internally because hidden_states[0] is
            patch embeddings. Default: (3, 6, 9, 11)
        freeze_backbone (bool): Freeze all backbone parameters. Default: True
        style (str): MMSeg compatibility stub. Ignored.
        init_cfg: MMSeg compatibility stub. Ignored (HF handles init).

    Returns (from forward):
        list of 4 Tensors, each (B, embed_dim, H', W') at strides [4, 8, 16, 32].
        For 480x480 input: (B, 384, 120, 120), (B, 384, 60, 60),
                           (B, 384, 30, 30), (B, 384, 15, 15)
    """

    def __init__(
        self,
        model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
        pretrained=True,
        out_indices=(3, 6, 9, 11),
        freeze_backbone=True,
        style='pytorch',
        init_cfg=None,
    ):
        super().__init__()

        if not HF_AVAILABLE:
            raise ImportError(
                "DINOv3 not available. Install: pip install transformers>=4.50"
            )

        self.model_name = model_name
        self.out_indices = out_indices
        self.freeze_backbone = freeze_backbone

        logger.info(f"Loading DINOv3 from HuggingFace: {model_name}")
        logger.info(f"  Output layer indices: {out_indices}")

        if pretrained:
            self.backbone = DINOv3ViTModel.from_pretrained(model_name)
            logger.info(f"Loaded pretrained weights from {model_name}")
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = DINOv3ViTModel(config)
            logger.warning("DINOv3 initialized with RANDOM weights")

        self.embed_dim = self.backbone.config.hidden_size
        logger.info(f"  embed_dim={self.embed_dim}, num_layers={self.backbone.config.num_hidden_layers}")

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("DINOv3 backbone frozen — only decode head will be trained")

    def init_weights(self, pretrained=None):
        # HF handles weight loading in __init__; re-apply freeze in case
        # MMSeg calls this after construction.
        if self.freeze_backbone:
            self._freeze_backbone()

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: (B, 3, H, W)

        Returns:
            list of 4 tensors at strides [4, 8, 16, 32]:
                (B, embed_dim, H//4, W//4)
                (B, embed_dim, H//8, W//8)
                (B, embed_dim, H//16, W//16)
                (B, embed_dim, H//32, W//32)
        """
        B, C, H, W = x.shape

        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

        patch_size = self.backbone.config.patch_size   # 16 for ViT-S/16
        feat_h = H // patch_size                       # 30 for 480×480
        feat_w = W // patch_size
        num_patches = feat_h * feat_w                  # 900

        # Extract spatial patch tokens from each requested layer.
        # hidden_states[0] = patch embeddings; hidden_states[i+1] = after block i.
        # out_indices are 0-based block indices, so we shift by +1.
        feat_list = []
        for idx in self.out_indices:
            feat = hidden_states[idx + 1]          # (B, 1+num_reg+num_patches, C)
            feat = feat[:, -num_patches:, :]       # (B, num_patches, C)
            feat = feat.permute(0, 2, 1).reshape(B, self.embed_dim, feat_h, feat_w)
            feat_list.append(feat)

        # Interpolate to strides [4, 8, 16, 32] — creates fake pyramid from isotropic ViT.
        strides = [4, 8, 16, 32]
        result = []
        for feat, stride in zip(feat_list, strides):
            target_h = H // stride
            target_w = W // stride
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                )
            result.append(feat)

        return result


__all__ = ['DINOv3BackboneHF']
