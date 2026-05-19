"""
Make a side-by-side RGB | segmentation video for visual inspection.

Usage (inside Docker from /workspace/TV3S):
    python3 tools/oxford_spires/make_seg_video.py \
        --images data/oxford_spires/christ-church-02/cam0_480x480 \
        --segs   data/oxford_spires/christ-church-02/segs_480x480 \
        --out    data/oxford_spires/christ-church-02/seg_video.mp4
"""

import argparse
import os
import glob
import numpy as np
import cv2

# VSPW 124-class palette (matches vspw.py)
PALETTE = [
    [120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],[120,120,80],
    [140,140,140],[204,5,255],[230,230,230],[4,250,7],[224,5,255],[235,255,7],
    [150,5,61],[120,120,70],[8,255,51],[255,6,82],[143,255,140],[204,255,4],
    [255,51,7],[204,70,3],[0,102,200],[61,230,250],[255,6,51],[11,102,255],
    [255,7,71],[255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],
    [8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],[7,255,255],
    [224,255,8],[102,8,255],[255,61,6],[255,194,7],[255,122,8],[0,255,20],
    [255,8,41],[255,5,153],[6,51,255],[235,12,255],[160,150,20],[0,163,255],
    [140,140,140],[250,10,15],[20,255,0],[31,255,0],[255,31,0],[255,224,0],
    [153,255,0],[0,0,255],[255,71,0],[0,235,255],[0,173,255],[31,0,255],
    [11,200,200],[255,82,0],[0,255,245],[0,61,255],[0,255,112],[0,255,133],
    [255,0,0],[255,163,0],[255,102,0],[194,255,0],[0,143,255],[51,255,0],
    [0,82,255],[0,255,41],[0,255,173],[10,0,255],[173,255,0],[0,255,153],
    [255,92,0],[255,0,255],[255,0,245],[255,0,102],[255,173,0],[255,0,20],
    [255,184,184],[0,31,255],[0,255,61],[0,71,255],[255,0,204],[0,255,194],
    [0,255,82],[0,10,255],[0,112,255],[51,0,255],[0,194,255],[0,122,255],
    [0,255,163],[255,153,0],[0,255,10],[255,112,0],[143,255,0],[82,0,255],
    [163,255,0],[255,235,0],[8,184,170],[133,0,255],[0,255,92],[184,0,255],
    [255,0,31],[0,184,255],[0,214,255],[255,0,112],[92,255,0],[0,224,255],
    [112,224,255],[70,184,160],[163,0,255],[153,0,255],[71,255,0],[255,0,163],
    [255,204,0],[255,0,143],[0,255,235],[133,255,0],
]
PALETTE_BGR = np.array([[b, g, r] for r, g, b in PALETTE], dtype=np.uint8)


def colorize(seg):
    """(H,W) uint8 class map → (H,W,3) BGR image."""
    seg = seg.clip(0, len(PALETTE_BGR) - 1)
    return PALETTE_BGR[seg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--segs',   required=True)
    parser.add_argument('--out',    required=True, help='Output .mp4 path')
    parser.add_argument('--fps',    type=float, default=10.0)
    parser.add_argument('--start',  type=int, default=0)
    parser.add_argument('--end',    type=int, default=None)
    args = parser.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.images, '*.png')))
    seg_paths = sorted(glob.glob(os.path.join(args.segs,   '*.png')))

    if not img_paths:
        raise RuntimeError(f'No PNGs in {args.images}')
    if not seg_paths:
        raise RuntimeError(f'No PNGs in {args.segs}')

    end = args.end if args.end is not None else len(img_paths)
    img_paths = img_paths[args.start:end]
    seg_paths = seg_paths[args.start:end]

    # Infer frame size from first image
    sample = cv2.imread(img_paths[0])
    H, W = sample.shape[:2]
    out_w = W * 2   # side-by-side

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, H))

    print(f'Writing {len(img_paths)} frames → {args.out}  ({out_w}x{H} @ {args.fps}fps)')
    for i, (ip, sp) in enumerate(zip(img_paths, seg_paths)):
        img = cv2.imread(ip)
        seg = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if img is None or seg is None:
            print(f'  Warning: skipping frame {i}')
            continue
        seg_color = colorize(seg)
        frame = np.hstack([img, seg_color])
        writer.write(frame)
        if i % 100 == 0:
            print(f'  {i}/{len(img_paths)}')

    writer.release()
    print(f'Done. Saved to {args.out}')


if __name__ == '__main__':
    main()
