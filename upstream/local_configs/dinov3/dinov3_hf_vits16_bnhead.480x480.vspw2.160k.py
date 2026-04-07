_base_ = [
    '../_base_/datasets/vspw_repeat2.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py',
]

# ---------------------------------------------------------------------------
# DINOv3 ViT-S/16 + BNHead (no temporal modeling) — VSPW baseline
#
# Purpose: isolate backbone quality from TV3S temporal contribution.
# Comparison table:
#   TV3S + MiT-B2         → mIoU ~40.0% (baseline)
#   DINOv3-S/16 + BNHead  → this run (no temporal)
#   TV3S + DINOv3-S/16    → future work
#
# Phase 2 (frozen, sanity check):
#   Set freeze_backbone=True below.  Run for ~20k iters to verify convergence.
#   Expected: loss dropping, acc_seg rising into 40-60% range within 10k iters.
#
# Phase 3 (fine-tuned, real comparison):
#   Set freeze_backbone=False (default below).  Train full 160k iters.
# ---------------------------------------------------------------------------

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder_clips',
    pretrained=None,
    backbone=dict(
        type='DINOv3BackboneHF',
        style='pytorch',
        model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
        pretrained=True,
        out_indices=(3, 6, 9, 11),   # → layers 4, 7, 10, 12 (1-indexed out of 12)
        freeze_backbone=True,         # ← set True for Phase 2 sanity check
    ),
    decode_head=dict(
        type='BNHead',
        in_channels=[384, 384, 384, 384],   # ViT-S/16 hidden_size
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=384,            # stored by base class; overridden by BNHead
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ---------------------------------------------------------------------------
# Optimizer — mirrors TV3S B2 config:
#   backbone LR:    6e-5   (low, to preserve pretrained features)
#   decode_head LR: 6e-4   (lr_mult=10 because 'head' matches 'decode_head')
#   norm layers:    weight decay = 0 (decay_mult=0)
# ---------------------------------------------------------------------------
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),   # matches 'decode_head' param names
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

evaluation = dict(interval=160000, metric='mIoU')
