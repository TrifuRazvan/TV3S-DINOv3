################################################################################
# DINOv3 ViT-S/16 Configuration for TV3S Temporal Video Segmentation
# Dataset: VSPW (Video Scene Parsing in the Wild)
################################################################################

_base_ = [
    '../../_base_/models/segformer.py',           # Base model architecture
    '../../_base_/datasets/vspw_repeat2.py',      # VSPW dataset config
    '../../_base_/default_runtime.py',            # Runtime defaults
    '../../_base_/schedules/schedule_160k_adamw.py'  # Training schedule
]

# Model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder_clips',  # Encoder-decoder for video segmentation
    pretrained=None,  # HuggingFace handles pretrained weights, not MMSeg
    
    # ========================================================================
    # BACKBONE: DINOv3 Vision Transformer from Meta AI via HuggingFace
    # ========================================================================
    backbone=dict(
        type='DINOv3BackboneHF',  # Custom wrapper for HuggingFace DINOv3
        
        # MODEL OPTIONS (change model_name to switch):
        # ┌──────────────┬───────────────────────────────────────────┬────────┬────────┐
        # │ Model        │ HuggingFace ID                            │ Params │ Size   │
        # ├──────────────┼───────────────────────────────────────────┼────────┼────────┤
        # │ ViT-S/16 ⭐  │ facebook/dinov3-vits16-pretrain-lvd1689m  │ 21M    │ 86MB   │ CURRENT
        # │ ViT-S/16+    │ facebook/dinov3-vits16plus-pretrain-...   │ 21M    │ 87MB   │ Better variant
        # │ ViT-B/16     │ facebook/dinov3-vitb16-pretrain-lvd1689m  │ 86M    │ 345MB  │ Like Swin-S
        # │ ViT-L/16     │ facebook/dinov3-vitl16-pretrain-lvd1689m  │ 304M   │ 1.2GB  │ High quality
        # │ ViT-H/16+    │ facebook/dinov3-vith16plus-pretrain-...   │ 630M   │ 2.5GB  │ Very large
        # │ ViT-g (7B)   │ facebook/dinov3-vit7b16-pretrain-lvd1689m │ 1.1B   │ 4.5GB  │ Overkill, slow
        # └──────────────┴───────────────────────────────────────────┴────────┴────────┘
        # 
        # Pretraining variants: -lvd1689m (primary) or -sat493m (satellite data)
        # Example: facebook/dinov3-vitl16-pretrain-sat493m
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        
        pretrained=True,  # Auto-download weights from HuggingFace (requires login)
        
        # FEATURE EXTRACTION LAYERS:
        # DINOv3 has 12 transformer layers. Extract features from 4 intermediate layers.
        # - Lower indices (3, 6) → coarse features
        # - Higher indices (9, 11) → fine features
        # Options: Any 4 values from (1, 2, 3, ..., 11)
        out_indices=(3, 6, 9, 11),
        
        freeze_backbone=True,  # CRITICAL: Freeze backbone, only train decoder
        style='pytorch'  # MMSeg compatibility parameter
    ),
    
    # ========================================================================
    # DECODER HEAD: TV3S Temporal Video Segmentation with Mamba blocks
    # ========================================================================
    decode_head=dict(
        type='TV3SHead_shift',  # Temporal decoder with window shifting
        
        # INPUT CHANNELS: Must match backbone output dimensions
        # ViT-S/16: 384, ViT-B/16: 768, ViT-L/16: 1024, ViT-H/16+: 1280, ViT-g: 1536
        in_channels=[768, 768, 768, 768],  # 4 scales from DINOv3-ViT-B (change to 384 for ViT-S, 1024 for ViT-L)
        in_index=[0, 1, 2, 3],             # Use all 4 feature scales
        
        # FEATURE STRIDES: Multi-scale pyramid
        # Stride 4 = highest resolution, stride 32 = lowest resolution
        feature_strides=[4, 8, 16, 32],
        
        channels=256,       # Decoder hidden dimension (fixed)
        dropout_ratio=0.1,  # Dropout rate (0.0-0.3 typical)
        num_classes=124,    # VSPW has 124 semantic classes
        norm_cfg=norm_cfg,  # Batch normalization config
        align_corners=False,
        
        # TEMPORAL DECODER PARAMETERS:
        decoder_params=dict(
            embed_dim=256,     # Must match 'channels' above
            window_w=20,       # Temporal window width (frames)
            window_h=20,       # Spatial window height (pixels)
            shift_size=10,     # Window shift amount (half of window size)
            real_shift=True,   # Use actual shifting (not cyclic)
            model_type=0       # Mamba variant (0=standard, others experimental)
        ),
        
        # LOSS FUNCTION:
        loss_decode=dict(
            type='CrossEntropyLoss',  # Standard segmentation loss
            use_sigmoid=False,        # Use softmax instead
            loss_weight=1.0           # Loss multiplier
        ),
        
        num_clips=4  # Number of video frames per training sample
    ),
    
    train_cfg=dict(),             # Training-specific config (empty for now)
    test_cfg=dict(mode='whole')   # Inference mode: whole image (vs sliding window)
)

################################################################################
# OPTIMIZER CONFIGURATION
################################################################################
# Since backbone is FROZEN, only decoder parameters are optimized.
# Using higher LR (1e-4) to match Swin-S effective learning rate.
optimizer = dict(
    _delete_=True,      # Override base config
    type='AdamW',       # Adam with weight decay (better than SGD for transformers)
    
    # LEARNING RATE OPTIONS:
    # - 1e-4: Current setting (matches Swin-S, good for frozen backbone)
    # - 1e-5: More conservative (slower convergence)
    # - 2e-4: More aggressive (risk of instability)
    lr=3e-4,
    
    betas=(0.9, 0.999),  # Adam momentum parameters (standard)
    weight_decay=0.05,   # L2 regularization (0.01-0.1 typical)
    
    # PARAMETER-WISE CONFIGURATION:
    # Different learning rates for different parameter groups
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),  # No weight decay for positional encodings
            'norm': dict(decay_mult=0.),       # No weight decay for normalization layers
            # Note: No 'head' lr_mult needed since backbone is frozen (all params are decoder)
        }
    )
)

################################################################################
# LEARNING RATE SCHEDULE
################################################################################
lr_config = dict(
    _delete_=True,
    policy='poly',        # Polynomial decay: lr = base_lr * (1 - iter/max_iter)^power
    
    # WARMUP: Gradually increase LR from warmup_ratio*lr to lr
    warmup='linear',      # Linear warmup (vs exponential)
    warmup_iters=1500,    # Warmup for first 1500 iterations (~1% of training)
    warmup_ratio=1e-6,    # Start at lr*1e-6 = 1e-10 (very small)
    
    power=1.0,            # Polynomial power (1.0 = linear decay)
    min_lr=0.0,           # Minimum LR at end of training
    by_epoch=False        # LR schedule by iterations (not epochs)
)

################################################################################
# DATA LOADING
################################################################################
data = dict(

    samples_per_gpu=1,
    
    workers_per_gpu=6,  # DataLoader workers (4-8 typical)
)

################################################################################
# LOGGING AND CHECKPOINTING
################################################################################
log_config = dict(
    _delete_=True,
    interval=50,  # Log every 50 iterations
    hooks=[dict(type='TextLoggerHook', by_epoch=False)]
)

# TRAINING SCHEDULE:
runner = dict(
    type='IterBasedRunner',
    max_iters=160000  # Total training iterations (~80k for faster experimentation)
)

# CHECKPOINTS: Save model weights periodically
checkpoint_config = dict(
    by_epoch=False,
    interval=20000  # Save every 20000 iterations (~every 30 mins)
)

# EVALUATION: Validate on validation set
evaluation = dict(
    interval=170000,  # Set > max_iters to disable during training (faster)
    metric='mIoU'     # Mean Intersection over Union
)

################################################################################
# USAGE INSTRUCTIONS
################################################################################
## TRAIN:
## ./tools/dist_train.sh local_configs/tv3s/DINOv3/dinov3_hf_vits16.480x480.vspw2.160k.py 1 \
##     --work-dir work_dirs/dinov3_hf_vits16
##
## HUGGINGFACE SETUP (first time only):
## 1. Accept model license: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
## 2. Get access token: https://huggingface.co/settings/tokens (Read permission)
## 3. Login: huggingface-cli login
##
## SWITCH TO DIFFERENT MODEL:
## Change 'model_name' in backbone config (see table above)
## Adjust 'in_channels' if switching to ViT-L/g:
##   - ViT-S/B: [384, 384, 384, 384]
##   - ViT-L: [1024, 1024, 1024, 1024]
##   - ViT-g: [1536, 1536, 1536, 1536]
################################################################################
