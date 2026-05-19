"""
Overlay LiDAR depth hits onto the corresponding RGB frame to verify
that T_cam_lidar is correct (points should land on the right surfaces).

Saves a side-by-side image: RGB | depth colormap | RGB + overlay.

Usage:
    python3 tools/oxford_spires/visualize_depth.py \
        --images data/oxford_spires/christ-church-02/cam0_480x480 \
        --depth  data/oxford_spires/christ-church-02/depth_480x480 \
        --frames 0 300 700 1100 \
        --out    data/oxford_spires/christ-church-02/depth_check.jpg
"""

import argparse
import os
import numpy as np
import cv2


def colorize_depth(depth):
    """Map depth (metres) to a BGR colormap. Zero pixels are black."""
    valid = depth > 0
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        normed = np.zeros_like(depth)
        normed[valid] = (depth[valid] - d_min) / max(d_max - d_min, 1e-6)
        colored = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[valid] = colored[valid]
    return vis


def make_row(img_path, depth_path, dot_radius=2):
    img = cv2.imread(img_path)
    depth = np.load(depth_path)
    H, W = img.shape[:2]

    depth_vis = colorize_depth(depth)

    # Overlay: draw coloured dots at each LiDAR hit on the RGB image
    overlay = img.copy()
    ys, xs = np.where(depth > 0)
    if len(ys):
        d_min, d_max = depth[ys, xs].min(), depth[ys, xs].max()
        for y, x in zip(ys[::4], xs[::4]):  # thin out for readability
            t = (depth[y, x] - d_min) / max(d_max - d_min, 1e-6)
            color = cv2.applyColorMap(np.array([[[int(t * 255)]]], dtype=np.uint8),
                                      cv2.COLORMAP_JET)[0, 0].tolist()
            cv2.circle(overlay, (x, y), dot_radius, color, -1)

    frame_id = os.path.splitext(os.path.basename(img_path))[0]
    for panel in [img, depth_vis, overlay]:
        cv2.putText(panel, frame_id, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return np.hstack([img, depth_vis, overlay])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--depth',  required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 300, 700, 1100])
    parser.add_argument('--out',    required=True)
    args = parser.parse_args()

    rows = []
    for f in args.frames:
        img_path   = os.path.join(args.images, f'{f:06d}.png')
        depth_path = os.path.join(args.depth,  f'{f:06d}.npy')
        if not os.path.exists(img_path):
            print(f'Missing image: {img_path}'); continue
        if not os.path.exists(depth_path):
            print(f'Missing depth: {depth_path}'); continue
        rows.append(make_row(img_path, depth_path))
        print(f'Frame {f:04d} done')

    if rows:
        result = np.vstack(rows)
        cv2.imwrite(args.out, result)
        print(f'Saved {args.out}  ({result.shape[1]}x{result.shape[0]})')


if __name__ == '__main__':
    main()
