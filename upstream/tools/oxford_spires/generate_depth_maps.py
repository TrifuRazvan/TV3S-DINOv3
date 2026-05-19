"""
Generate sparse depth maps by projecting LiDAR scans onto the undistorted
camera frame for each camera frame in an Oxford Spires ROS1 bag.

For each camera frame, finds the nearest LiDAR scan by timestamp, transforms
points using T_cam_lidar, projects with K_new, and saves a float32 .npy depth
map (0.0 = no measurement).

Usage:
    python3 tools/oxford_spires/generate_depth_maps.py \
        --bag    /path/to/sequence.bag \
        --calib  /path/to/cam-lidar.yaml \
        --images data/oxford_spires/christ-church-02/cam0_480x480 \
        --out    data/oxford_spires/christ-church-02/depth_480x480
"""

import argparse
import os
import struct
import glob
import yaml
import cv2
import numpy as np
from rosbags.rosbag1 import Reader

LIDAR_TOPIC = '/hesai/pandar'
MAX_TIME_GAP = 0.06  # seconds — skip frame if nearest LiDAR scan is farther than this


def load_cam_lidar_extrinsic(calib_path):
    with open(calib_path) as f:
        cal = yaml.safe_load(f)
    return np.array(cal['cam0']['T_cam_lidar'], dtype=np.float64)


def parse_pointcloud2(rawdata):
    """Parse a ROS1 PointCloud2 binary message.
    Returns (stamp_seconds: float, points: (N,3) float32 array).
    """
    pos = 0

    # Header: seq, stamp (sec + nsec), frame_id string
    pos += 4
    sec  = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    nsec = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    frame_id_len = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    pos += frame_id_len

    stamp = sec + nsec * 1e-9

    height = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    width  = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4

    # Field descriptors — find byte offsets of x, y, z
    n_fields = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    field_offsets = {}
    for _ in range(n_fields):
        name_len = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
        name     = rawdata[pos:pos+name_len].decode();        pos += name_len
        offset   = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
        pos += 1  # datatype (uint8)
        pos += 4  # count (uint32)
        field_offsets[name] = offset

    pos += 1  # is_bigendian
    point_step = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    pos += 4  # row_step
    data_len   = struct.unpack_from('<I', rawdata, pos)[0]; pos += 4
    data = rawdata[pos:pos+data_len]

    n_points = height * width
    x_off = field_offsets['x']
    y_off = field_offsets['y']
    z_off = field_offsets['z']

    # Reshape to (N, point_step) and extract float32 fields
    d = np.frombuffer(data, dtype=np.uint8).reshape(n_points, point_step)
    x = d[:, x_off:x_off+4].copy().view(np.float32).reshape(-1)
    y = d[:, y_off:y_off+4].copy().view(np.float32).reshape(-1)
    z = d[:, z_off:z_off+4].copy().view(np.float32).reshape(-1)

    pts = np.stack([x, y, z], axis=1)
    return stamp, pts[np.isfinite(pts).all(axis=1)]


def project_to_depth(pts_lidar, T_cam_lidar, K_new, width, height):
    """Project LiDAR points into the undistorted camera frame.
    Returns (H, W) float32 depth map in metres (0.0 = no measurement).
    When multiple points map to the same pixel, the nearest is kept.
    """
    R = T_cam_lidar[:3, :3]
    t = T_cam_lidar[:3, 3]
    pts_cam = (R @ pts_lidar.T).T + t          # (N, 3)

    mask = pts_cam[:, 2] > 0.1                 # discard points behind camera
    pts_cam = pts_cam[mask]

    fx, fy = K_new[0, 0], K_new[1, 1]
    cx, cy = K_new[0, 2], K_new[1, 2]
    u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx)
    v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy)
    z = pts_cam[:, 2].astype(np.float32)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z = ui[in_bounds], vi[in_bounds], z[in_bounds]

    depth = np.zeros((height, width), dtype=np.float32)
    # Process far-to-near so nearest point wins
    order = np.argsort(z)[::-1]
    depth[vi[order], ui[order]] = z[order]

    return depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',    required=True)
    parser.add_argument('--calib',  required=True, help='cam-lidar.yaml')
    parser.add_argument('--images', required=True,
                        help='cam0_WxH dir (contains timestamps.txt and K_new.txt)')
    parser.add_argument('--out',    required=True, help='depth_WxH output dir')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    T_cam_lidar    = load_cam_lidar_extrinsic(args.calib)
    K_new          = np.loadtxt(os.path.join(args.images, 'K_new.txt'))
    cam_timestamps = np.loadtxt(os.path.join(args.images, 'timestamps.txt'))

    # Get image dimensions from first PNG in the images directory
    pngs = sorted(glob.glob(os.path.join(args.images, '*.png')))
    if not pngs:
        raise RuntimeError(f'No PNG files found in {args.images}')
    sample = cv2.imread(pngs[0])
    height, width = sample.shape[:2]
    print(f'Image size: {width}x{height}')
    print(f'Camera frames: {len(cam_timestamps)}')

    # Load all LiDAR scans up front
    print('Reading LiDAR scans from bag...')
    lidar_stamps, lidar_points = [], []
    with Reader(args.bag) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        if not conns:
            raise RuntimeError(f'{LIDAR_TOPIC} not found in bag')
        for _, _, rawdata in bag.messages(connections=conns):
            stamp, pts = parse_pointcloud2(rawdata)
            lidar_stamps.append(stamp)
            lidar_points.append(pts)

    lidar_stamps = np.array(lidar_stamps)
    print(f'Loaded {len(lidar_stamps)} LiDAR scans')
    print(f'LiDAR time range:  {lidar_stamps[0]:.3f} – {lidar_stamps[-1]:.3f}')
    print(f'Camera time range: {cam_timestamps[0]:.3f} – {cam_timestamps[-1]:.3f}')

    n_saved = n_skipped = 0
    for i, cam_ts in enumerate(cam_timestamps):
        idx = np.argmin(np.abs(lidar_stamps - cam_ts))
        gap = abs(lidar_stamps[idx] - cam_ts)

        if gap > MAX_TIME_GAP:
            n_skipped += 1
            continue

        depth = project_to_depth(lidar_points[idx], T_cam_lidar, K_new, width, height)
        np.save(os.path.join(args.out, f'{i:06d}.npy'), depth)
        n_saved += 1

        if i % 100 == 0:
            hits = np.count_nonzero(depth)
            coverage = 100 * hits / (width * height)
            print(f'  frame {i:04d}  gap={gap*1000:.1f}ms  '
                  f'LiDAR hits={hits} ({coverage:.1f}% coverage)')

    print(f'\nDone. {n_saved} depth maps saved, {n_skipped} frames skipped.')


if __name__ == '__main__':
    main()
