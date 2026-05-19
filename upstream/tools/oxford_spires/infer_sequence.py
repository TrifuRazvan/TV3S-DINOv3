"""
Run TV3S inference on a directory of undistorted Oxford Spires frames and
save per-frame semantic segmentation maps (uint8 PNG, class indices 0-123).

The model receives 4-frame clips using the same dilation pattern as VSPW
evaluation (dilation=[-9,-6,-3] by default): three reference frames at
fixed temporal offsets before the current frame, plus the current frame.
The model outputs a prediction for the current frame.

Usage (inside Docker container from /workspace):
    python3 tools/oxford_spires/infer_sequence.py \
        --config   local_configs/dinov3/dinov3_hf_vits16_tv3s_frozen.480x480.vspw2.160k.py \
        --checkpoint work_dirs/dinov3_vits16_tv3s_frozen_2sample_iter160k_lr6e-5/latest.pth \
        --images   data/oxford_spires/christ-church-02/cam0_480x480 \
        --out      data/oxford_spires/christ-church-02/segs_480x480
"""

import argparse
import os
import glob
import sys

import numpy as np
import cv2
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor

import utils  # registers custom backbones, heads, datasets


MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


def preprocess(bgr_img):
    """BGR uint8 HWC → normalised float32 CHW tensor on CPU."""
    rgb = bgr_img[:, :, ::-1].astype(np.float32)   # BGR→RGB
    rgb = (rgb - MEAN) / STD
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(chw)


def build_model(config_path, checkpoint_path, device):
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    # Skip HF weight download — checkpoint provides all weights
    cfg.model.backbone.pretrained = False

    # Set test-time settings identical to tools/test.py (mode 0 = Normal)
    if hasattr(cfg.model.decode_head, 'decoder_params'):
        cfg.model.decode_head.decoder_params.test_mode = True
        cfg.model.decode_head.decoder_params.val_mode = 0   # Normal mode

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def infer_clip(model, frames_bgr, device, frame_path):
    """Run inference on a 4-frame clip. Returns (H,W) uint8 numpy array.

    frame_path is the path of the CURRENT (last) frame in the clip; it is
    stored in img_metas so the TV3S head can detect sequence boundaries and
    reset its Mamba state accordingly.
    """
    H, W = frames_bgr[0].shape[:2]

    # Build list of 4 tensors each (1, C, H, W) — format expected by simple_test
    clip = [preprocess(f).unsqueeze(0).to(device) for f in frames_bgr]

    img_meta = [{
        'filename':     frame_path,
        'ori_shape':    (H, W, 3),
        'img_shape':    (H, W, 3),
        'pad_shape':    (H, W, 3),
        'scale_factor': 1.0,
        'flip':         False,
        'flip_direction': 'horizontal',
    }]

    with torch.no_grad():
        result = model.simple_test(clip, img_meta, rescale=False)

    return result[0].astype(np.uint8)   # (H, W) class indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      required=True)
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--images',      required=True,
                        help='cam0_480x480 dir with 000000.png … frames')
    parser.add_argument('--out',         required=True,
                        help='Output dir for segmentation PNGs')
    parser.add_argument('--dilation',    type=int, nargs='+', default=[-9, -6, -3],
                        help='Temporal offsets for reference frames (default: -9 -6 -3)')
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame',   type=int, default=None,
                        help='Exclusive upper bound; default=all frames')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Discover all frames
    pngs = sorted(glob.glob(os.path.join(args.images, '*.png')))
    if not pngs:
        sys.exit(f'No PNG files in {args.images}')
    N = len(pngs)
    print(f'Found {N} frames in {args.images}')

    end = args.end_frame if args.end_frame is not None else N

    print(f'Building model from {args.config}')
    model = build_model(args.config, args.checkpoint, args.device)
    print('Model ready.')

    n_saved = n_skipped = 0
    for i in range(args.start_frame, end):
        out_path = os.path.join(args.out, f'{i:06d}.png')
        if os.path.exists(out_path):
            n_saved += 1
            continue

        # Reference frame indices (clamp to [0, N-1])
        ref_indices = [max(0, i + d) for d in args.dilation]
        clip_indices = ref_indices + [i]

        frames_bgr = []
        ok = True
        for idx in clip_indices:
            img = cv2.imread(pngs[idx])
            if img is None:
                print(f'  Warning: could not read {pngs[idx]}, skipping frame {i}')
                ok = False
                break
            frames_bgr.append(img)

        if not ok:
            n_skipped += 1
            continue

        seg = infer_clip(model, frames_bgr, args.device, pngs[i])
        cv2.imwrite(out_path, seg)
        n_saved += 1

        if i % 100 == 0:
            print(f'  frame {i:04d}/{end}  saved to {out_path}')

    print(f'\nDone. {n_saved} frames saved, {n_skipped} skipped.')
    print(f'Output: {args.out}')


if __name__ == '__main__':
    main()
