
import torch
import torch.nn as nn
from utils.models.dinov3_backbone import DINOv3Backbone
from utils.models.tv3s_head import TV3SHead_shift

def check_compatibility():
    print("Checking DINOv3 Backbone <-> TV3SHead Compatibility...")
    
    # 1. Initialize Backbone
    print("\n1. Initializing Backbone...")
    backbone = DINOv3Backbone(
        model_name='dinov3_vits16',
        checkpoint_path='pretrained/DinoV3/dinov3_vits16_pretrain.pth',
        pretrained=True,
        out_indices=(3, 6, 9, 11)
    )
    backbone.eval()
    
    # 2. Create Dummy Input
    x = torch.randn(1, 3, 480, 480)
    print(f"\nInput shape: {x.shape}")
    
    # 3. Forward Pass Backbone
    print("\n2. Running Backbone Forward Pass...")
    with torch.no_grad():
        features = backbone(x)
        
    print(f"Backbone produced {len(features)} feature maps:")
    feature_shapes = []
    for i, feat in enumerate(features):
        print(f"  Scale {i}: {feat.shape} | Mean: {feat.mean():.4f} | Std: {feat.std():.4f}")
        feature_shapes.append(feat.shape)
        
    # 4. Initialize Decoder
    print("\n3. Initializing Decoder...")
    # Config from dinov3_vits16.480x480.vspw2.160k.py
    decoder = TV3SHead_shift(
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=dict(type='BN', requires_grad=True), # Use BN for CPU test
        align_corners=False,
        decoder_params=dict(
            embed_dim=256,
            window_w=20,
            window_h=20,
            shift_size=10,
            real_shift=True,
            model_type=0
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4
    )
    
    # 5. Forward Pass Decoder
    print("\n4. Running Decoder Forward Pass...")
    # Decoder expects a list of features, but for multiple clips.
    # TV3SHead_shift usually expects (B*num_clips, C, H, W)
    # Let's simulate 4 clips (B=1, num_clips=4 -> Batch=4)
    
    # Expand features to simulate 4 clips
    features_clips = [f.repeat(4, 1, 1, 1) for f in features]
    print(f"Simulating 4 clips. Input to decoder:")
    for i, f in enumerate(features_clips):
        print(f"  Scale {i}: {f.shape}")
        
    try:
        out = decoder(features_clips)
        print(f"\n✅ Decoder Forward Pass Successful!")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"\n❌ Decoder Forward Pass Failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_compatibility()
