_base_ = [
    '../_base_/datasets/vspw_repeat2.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py',
]

# ---------------------------------------------------------------------------
# DINOv3 ViT-S/16 + TV3S Head (Mamba temporal modeling) — VSPW
#
# Combines DINOv3's strong pretrained features with TV3S's temporal head.
# Hypothesis: DINOv3 features (high mIoU) + Mamba temporal propagation
#             (high VC) → best of both worlds.
#
# Comparison:
#   TV3S + MiT-B1          → mIoU 39.99, mVC16 84.52 (ours, mode 0)
#   DINOv3-S/16 + BNHead   → mIoU 50.38, mVC16 86.39 (ours, mode 0, finetune)
#   DINOv3-S/16 + TV3S     → this run (target: mIoU ≥50, mVC16 ≥87)
#
# Key design choices:
#   - in_channels=[384]×4: DINOv3 isotropic output (hidden_size=384)
#   - embed_dim=256: Mamba channel width, matches original TV3S
#   - window=20, shift=10, real_shift=True, n_mambas=8: standard TV3S settings
#   - backbone LR 3e-5 (lower than MiT's 6e-5): DINOv3 features are
#     stronger → less aggressive fine-tuning
#   - head LR 3e-4 (via lr_mult=10)
#   - samples_per_gpu=1: TV3S Mamba is memory-heavy, DINOv3 finetune adds
#     backbone grads → keep batch small to avoid OOM
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
        out_indices=(3, 6, 9, 11),
        freeze_backbone=False,
    ),
    decode_head=dict(
        type='TV3SHead_shift',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dim=256,
            window_w=20,
            window_h=20,
            shift_size=10,
            real_shift=True,
            model_type=0,
        ),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
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

evaluation = dict(interval=180000, metric='mIoU')
