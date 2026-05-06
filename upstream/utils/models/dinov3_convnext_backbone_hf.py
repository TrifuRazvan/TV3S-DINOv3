import torch
import torch.nn as nn
from mmseg.models.builder import BACKBONES
import logging

try:
    from transformers.models.dinov3_convnext.modeling_dinov3_convnext import DINOv3ConvNextModel
    from transformers import AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("WARNING: DINOv3 ConvNext not available. Install transformers>=4.50")

logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class DINOv3ConvNextBackboneHF(nn.Module):
    """DINOv3 ConvNext backbone using HuggingFace transformers.

    Unlike the ViT variant, ConvNext is hierarchical and produces a true
    4-stage feature pyramid at strides [4, 8, 16, 32] — no interpolation
    needed. Native fit for the TV3S/SegFormer-style decode head.

    For 480x480 input with ConvNext-Base (hidden_sizes=[128,256,512,1024]):
        stage1: (B, 128, 120, 120)   stride 4
        stage2: (B, 256,  60,  60)   stride 8
        stage3: (B, 512,  30,  30)   stride 16
        stage4: (B, 1024, 15,  15)   stride 32

    Args:
        model_name (str): HuggingFace model ID, e.g.
            'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'
            'facebook/dinov3-convnext-small-pretrain-lvd1689m'
            'facebook/dinov3-convnext-base-pretrain-lvd1689m'
            'facebook/dinov3-convnext-large-pretrain-lvd1689m'
        pretrained (bool): Load pretrained weights from HuggingFace.
        freeze_backbone (bool): Freeze all backbone parameters.
        style (str): MMSeg compatibility stub. Ignored.
        init_cfg: MMSeg compatibility stub. Ignored.
    """

    def __init__(
        self,
        model_name='facebook/dinov3-convnext-base-pretrain-lvd1689m',
        pretrained=True,
        freeze_backbone=True,
        style='pytorch',
        init_cfg=None,
    ):
        super().__init__()

        if not HF_AVAILABLE:
            raise ImportError(
                "DINOv3 ConvNext not available. Install: pip install transformers>=4.50"
            )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        logger.info(f"Loading DINOv3 ConvNext from HuggingFace: {model_name}")

        if pretrained:
            self.backbone = DINOv3ConvNextModel.from_pretrained(model_name)
            logger.info(f"Loaded pretrained weights from {model_name}")
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = DINOv3ConvNextModel(config)
            logger.warning("DINOv3 ConvNext initialized with RANDOM weights")

        self.hidden_sizes = self.backbone.config.hidden_sizes
        logger.info(f"  hidden_sizes={self.hidden_sizes}, num_stages={self.backbone.config.num_stages}")

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("DINOv3 ConvNext backbone frozen — only decode head will be trained")

    def init_weights(self, pretrained=None):
        if self.freeze_backbone:
            self._freeze_backbone()

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: (B, 3, H, W)

        Returns:
            list of 4 tensors at strides [4, 8, 16, 32]:
                (B, hidden_sizes[0], H//4,  W//4)
                (B, hidden_sizes[1], H//8,  W//8)
                (B, hidden_sizes[2], H//16, W//16)
                (B, hidden_sizes[3], H//32, W//32)
        """
        outputs = self.backbone(x, output_hidden_states=True)
        # hidden_states[0] = input pixel_values; [1..4] = after each stage.
        # ConvNeXt may output channels-last (NHWC) non-contiguous tensors; make
        # them contiguous so downstream CUDA kernels (e.g. SyncBN) don't crash.
        return [f.contiguous() for f in outputs.hidden_states[1:5]]


__all__ = ['DINOv3ConvNextBackboneHF']
