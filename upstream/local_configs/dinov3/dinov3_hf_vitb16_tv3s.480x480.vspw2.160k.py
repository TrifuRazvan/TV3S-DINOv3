_base_ = [
    '../_base_/datasets/vspw_repeat2.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py',
]

# ---------------------------------------------------------------------------
# DINOv3 ViT-B/16 + TV3S Head (Mamba temporal modeling) — VSPW — finetune
#
# ViT-B/16: 86M params, hidden_dim=768, 12 layers.
# Larger backbone than ViT-S/16+ (~28M) — significant capacity jump.
# in_channels bumped to 768 to match hidden_dim; all other head params unchanged.
# ---------------------------------------------------------------------------

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder_clips',
    pretrained=None,
    backbone=dict(
        type='DINOv3BackboneHF',
        style='pytorch',
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        pretrained=True,
        out_indices=(3, 6, 9, 11),
        freeze_backbone=False,
    ),
    decode_head=dict(
        type='TV3SHead_shift',
        in_channels=[768, 768, 768, 768],
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
