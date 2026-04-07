import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.losses import accuracy
from mmseg.ops import resize

from .decode_head import BaseDecodeHead_clips_flow


@HEADS.register_module()
class BNHead(BaseDecodeHead_clips_flow):
    """Linear multi-scale BNHead for DINOv3 (no temporal modeling).

    Replicates Meta's BNHead from the official DINOv3 segmentation benchmark:
    - Per-scale: SyncBN → 1x1 Conv → upsample to finest scale
    - Concat all scales → SyncBN → 1x1 Conv → num_classes

    No Mamba, no temporal state. Each frame processed independently.

    During training  → returns (B, T, num_classes, H', W') for TV3S loss infra
    During inference → returns (B, num_classes, H', W') for last clip frame

    Args:
        in_channels (list[int]): Feature dim per scale. All equal for DINOv3.
            e.g. [384, 384, 384, 384] for ViT-S/16.
        feature_strides (list[int]): Backbone strides — stored but not used
            (backbone already handles resizing).
        **kwargs: Passed to BaseDecodeHead_clips_flow (channels, dropout_ratio,
            num_classes, norm_cfg, align_corners, loss_decode, num_clips, …).
    """

    def __init__(self, in_channels, feature_strides, **kwargs):
        # Force multiple_select so _transform_inputs picks all 4 scales.
        # 'channels' from kwargs goes to parent's conv_seg; we override it below.
        super().__init__(
            in_channels=in_channels,
            input_transform='multiple_select',
            **kwargs,
        )
        assert len(feature_strides) == len(in_channels)

        embed_dims = in_channels  # e.g. [384, 384, 384, 384]
        num_scales = len(embed_dims)

        # Per-scale: BN → 1×1 conv (project each scale to its own dim)
        self.scale_bns = nn.ModuleList(
            [nn.SyncBatchNorm(c) for c in embed_dims]
        )
        self.scale_projs = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=1, bias=False) for c in embed_dims]
        )

        fused_dim = sum(embed_dims)  # 4 * 384 = 1536 for ViT-S
        self.fuse_bn = nn.SyncBatchNorm(fused_dim)

        # Override the parent's conv_seg (which expects 'channels' → num_classes)
        self.conv_seg = nn.Conv2d(fused_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, batch_size, num_clips, img=None):
        """Forward pass.

        Args:
            inputs: list of 4 feature tensors, each (B*T, embed_dim, H', W')
            batch_size (int): B
            num_clips (int): T

        Returns:
            Training:  (B, T, num_classes, H', W')
            Inference: (B, num_classes, H', W')  — last clip only
        """
        feats = self._transform_inputs(inputs)  # selects by in_index

        finest = feats[0].shape[2:]  # spatial size of finest scale

        processed = []
        for x, bn, proj in zip(feats, self.scale_bns, self.scale_projs):
            x = bn(x)
            x = proj(x)
            if x.shape[2:] != finest:
                x = F.interpolate(
                    x, size=finest, mode='bilinear', align_corners=False
                )
            processed.append(x)

        x = torch.cat(processed, dim=1)   # (B*T, fused_dim, H', W')
        x = self.fuse_bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        seg_logit = self.conv_seg(x)       # (B*T, num_classes, H', W')

        _, C, H, W = seg_logit.shape
        seg_logit = seg_logit.reshape(batch_size, num_clips, C, H, W)

        if not self.training:
            # Return last clip (current frame) for the inference pipeline
            return seg_logit[:, -1]  # (B, num_classes, H', W')

        return seg_logit  # (B, T, num_classes, H', W')

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute CE loss over all T clips independently.

        Args:
            seg_logit: (B, T, num_classes, H', W')
            seg_label: (B, T, 1, H_gt, W_gt)
        """
        assert seg_logit.dim() == 5 and seg_label.dim() == 5
        B, T, C, H, W = seg_logit.shape
        _, _, _, Hgt, Wgt = seg_label.shape

        seg_logit_flat = seg_logit.reshape(B * T, C, H, W)
        seg_logit_flat = resize(
            seg_logit_flat,
            size=(Hgt, Wgt),
            mode='bilinear',
            align_corners=self.align_corners,
        )
        seg_label_flat = seg_label.reshape(B * T, 1, Hgt, Wgt).squeeze(1)  # (B*T, H, W)

        loss = dict()
        loss['loss_seg'] = self.loss_decode(
            seg_logit_flat,
            seg_label_flat,
            ignore_index=self.ignore_index,
        )
        loss['acc_seg'] = accuracy(seg_logit_flat, seg_label_flat)
        return loss

    def init_weights(self):
        # SyncBN and Conv2d default init is fine; no special init needed.
        pass


__all__ = ['BNHead']
