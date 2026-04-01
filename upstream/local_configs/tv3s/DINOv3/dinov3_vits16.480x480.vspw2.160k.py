_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/vspw_repeat2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder_clips',
    pretrained=None,  # DINOv3 weights loaded within backbone
    backbone=dict(
        type='DINOv3Backbone',
        model_name='dinov3_vits16',
        checkpoint_path='pretrained/DinoV3/dinov3_vits16_pretrain.pth',  # Explicit checkpoint path
        pretrained=True,
        out_indices=(3, 6, 9, 11),
        frozen_stages=-1),
    decode_head=dict(
        type='TV3SHead_shift',
        in_channels=[384, 384, 384, 384],  # All 384 for ViT-S
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256, window_w=20, window_h=20, shift_size=10, real_shift=True, model_type=0),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
# DINOv3 backbone is FROZEN (requires_grad=False) - only decoder trains
# Base LR: 1e-4 (matches Swin-S effective decoder LR)
# Since backbone is frozen, all params get same LR
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
)





log_config = dict(_delete_=True, interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=10000)
# Set evaluation interval beyond max_iters to disable it during training
evaluation = dict(interval=170000, metric='mIoU')
