"""
Extract cam0 frames from an Oxford Spires ROS1 bag, undistort fisheye to
perspective, resize to the requested resolution, and save as PNGs with a
timestamps and K_new sidecar for downstream scripts.

Resolution must have both dimensions divisible by 16 (DINOv3 ViT patch size).

Usage:
    python3 tools/oxford_spires/preprocess_images.py \
        --bag    /path/to/sequence.bag \
        --calib  /path/to/cam0.yaml \
        --out    data/oxford_spires/christ-church-02/cam0_480x480 \
        --width  480 --height 480

    # Also save raw (distorted) frames for comparison:
        --raw-out data/oxford_spires/christ-church-02/cam0_raw
"""

import argparse
import os
import yaml
import cv2
import numpy as np
from rosbags.rosbag1 import Reader

TOPIC = '/alphasense_driver_ros/cam0/debayered/image/compressed'


def load_calibration(calib_path):
    with open(calib_path) as f:
        cal = yaml.safe_load(f)
    K = np.array(cal['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    D = np.array(cal['distortion_coefficients']['data'], dtype=np.float64)
    W = cal['image_width']
    H = cal['image_height']
    assert cal['distortion_model'] == 'equidistant', \
        f"Expected equidistant model, got {cal['distortion_model']}"
    return K, D, W, H


def build_undistort_maps(K, D, src_w, src_h, out_w, out_h):
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (src_w, src_h), np.eye(3), balance=0.0,
        new_size=(out_w, out_h), fov_scale=1.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (out_w, out_h), cv2.CV_16SC2,
    )
    return map1, map2, K_new


def decode_jpeg(rawdata):
    start = rawdata.find(b'\xff\xd8')
    if start == -1:
        return None
    return cv2.imdecode(np.frombuffer(rawdata[start:], dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',     required=True)
    parser.add_argument('--calib',   required=True)
    parser.add_argument('--out',     required=True,
                        help='Output dir — embed resolution in name, e.g. cam0_480x480')
    parser.add_argument('--width',   type=int, default=480)
    parser.add_argument('--height',  type=int, default=480)
    parser.add_argument('--raw-out', default=None,
                        help='If set, also save raw (distorted) frames here')
    args = parser.parse_args()

    assert args.width  % 16 == 0, f'--width {args.width} must be divisible by 16'
    assert args.height % 16 == 0, f'--height {args.height} must be divisible by 16'

    os.makedirs(args.out, exist_ok=True)
    if args.raw_out:
        os.makedirs(args.raw_out, exist_ok=True)

    K, D, src_w, src_h = load_calibration(args.calib)
    print(f'Source image:  {src_w}x{src_h}  fx={K[0,0]:.1f} fy={K[1,1]:.1f}')
    print(f'Output size:   {args.width}x{args.height}')

    map1, map2, K_new = build_undistort_maps(K, D, src_w, src_h, args.width, args.height)
    print(f'K_new: fx={K_new[0,0]:.1f} fy={K_new[1,1]:.1f} '
          f'cx={K_new[0,2]:.1f} cy={K_new[1,2]:.1f}')

    timestamps = []
    n_saved = 0

    with Reader(args.bag) as bag:
        conns = [c for c in bag.connections if c.topic == TOPIC]
        if not conns:
            raise RuntimeError(f'Topic {TOPIC} not found in bag')

        messages = list(bag.messages(connections=conns))
        total = len(messages)
        print(f'Found {total} cam0 frames')

        for i, (_, ts_ns, rawdata) in enumerate(messages):
            img = decode_jpeg(rawdata)
            if img is None:
                print(f'  Warning: failed to decode frame {i}')
                continue

            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(args.out, f'{i:06d}.png'), undistorted)

            if args.raw_out:
                cv2.imwrite(os.path.join(args.raw_out, f'{i:06d}.jpg'), img)

            timestamps.append(ts_ns / 1e9)
            n_saved += 1

            if i % 100 == 0:
                print(f'  {i}/{total}')

    np.savetxt(os.path.join(args.out, 'timestamps.txt'), timestamps, fmt='%.9f')
    np.savetxt(os.path.join(args.out, 'K_new.txt'), K_new, fmt='%.6f')
    print(f'\nDone. {n_saved} frames -> {args.out}')


if __name__ == '__main__':
    main()
