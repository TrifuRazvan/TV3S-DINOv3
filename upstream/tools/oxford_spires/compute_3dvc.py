"""
3D Video Consistency (3DVC) for Oxford Spires sequences.

For each source frame i, valid-depth pixels are backprojected to 3D using
undistorted camera intrinsics, transformed to the target frame j = i + gap
using ground-truth TUM poses, and reprojected.  The fraction of pixel pairs
whose semantic labels agree is 3DVC_gap.

Pose convention (TUM format): T_world_cam — transforms points from camera to
world frame.  One pose per line: "ts tx ty tz qx qy qz qw".

Usage (inside Docker from /workspace/TV3S):
    python3 tools/oxford_spires/compute_3dvc.py \
        --segs       results/oxford_spires/christ-church-02/<model>/segs_480x480 \
        --depth      data/oxford_spires/christ-church-02/depth_480x480 \
        --poses      /path/to/gt-tum.txt \
        --timestamps data/oxford_spires/christ-church-02/cam0_480x480/timestamps.txt \
        --K          data/oxford_spires/christ-church-02/cam0_480x480/K_new.txt \
        --gaps       1 2 4 8 16 \
        --out        results/oxford_spires/christ-church-02/<model>/3dvc.json
"""

import argparse
import json
import os
import glob
import numpy as np
import cv2


def _quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)    ],
        [    2*(qx*qy + qz*qw),  1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)    ],
        [    2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),  1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def load_poses(tum_path):
    """Return (timestamps float64 array, list of 4x4 float64 T_world_cam)."""
    timestamps, matrices = [], []
    with open(tum_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
            T[:3,  3] = [tx, ty, tz]
            timestamps.append(ts)
            matrices.append(T)
    return np.array(timestamps, dtype=np.float64), matrices


def _nearest_pose(pose_ts, pose_mats, query_ts):
    idx = int(np.argmin(np.abs(pose_ts - query_ts)))
    return pose_mats[idx]


def compute_pair(depth_i, seg_i, seg_j, K, T_world_i, T_world_j):
    """Return (n_consistent, n_total) for one frame pair (i → j)."""
    H, W = depth_i.shape
    mask = depth_i > 0.0
    if not mask.any():
        return 0, 0

    v_src, u_src = np.where(mask)
    z = depth_i[v_src, u_src].astype(np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Backproject to 3D in cam_i frame
    x = (u_src - cx) * z / fx
    y = (v_src - cy) * z / fy
    pts_h = np.stack([x, y, z, np.ones_like(z)])  # (4, N)

    # Transform cam_i → world → cam_j
    T_i_to_j = np.linalg.inv(T_world_j) @ T_world_i
    pts_j = T_i_to_j @ pts_h  # (4, N)

    Xj, Yj, Zj = pts_j[0], pts_j[1], pts_j[2]
    front = Zj > 0.1
    if not front.any():
        return 0, 0

    uj = np.round(fx * Xj[front] / Zj[front] + cx).astype(np.int32)
    vj = np.round(fy * Yj[front] / Zj[front] + cy).astype(np.int32)
    in_bounds = (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)

    uj = uj[in_bounds]
    vj = vj[in_bounds]
    ui = u_src[front][in_bounds]
    vi = v_src[front][in_bounds]

    if len(ui) == 0:
        return 0, 0

    labels_i = seg_i[vi, ui]
    labels_j = seg_j[vj, uj]
    return int((labels_i == labels_j).sum()), len(ui)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs',       required=True, help='Dir of segmentation PNGs')
    ap.add_argument('--depth',      required=True, help='Dir of depth NPY files (float32, metres)')
    ap.add_argument('--poses',      required=True, help='gt-tum.txt (T_world_cam per line)')
    ap.add_argument('--timestamps', required=True, help='Camera frame timestamps (one per line)')
    ap.add_argument('--K',          required=True, help='3x3 intrinsics matrix (K_new.txt)')
    ap.add_argument('--gaps',       type=int, nargs='+', default=[1, 2, 4, 8, 16])
    ap.add_argument('--stride',     type=int, default=1,
                    help='Use every N-th source frame (default=1 → all)')
    ap.add_argument('--max_dt',     type=float, default=0.5,
                    help='Max allowed seconds between camera frame timestamp and '
                         'nearest pose; frames exceeding this are skipped (default=0.5)')
    ap.add_argument('--out',        required=True, help='Output JSON path')
    args = ap.parse_args()

    K      = np.loadtxt(args.K, dtype=np.float64)
    cam_ts = np.loadtxt(args.timestamps, dtype=np.float64)
    N      = len(cam_ts)
    print(f'{N} camera frames')

    print('Loading poses...')
    pose_ts, pose_mats = load_poses(args.poses)
    print(f'  {len(pose_ts)} poses  (dt_first={cam_ts[0]-pose_ts[0]:.3f}s)')

    print('Matching frames to nearest pose...')
    frame_T  = []
    frame_dt = []
    for ts in cam_ts:
        idx = int(np.argmin(np.abs(pose_ts - ts)))
        frame_T.append(pose_mats[idx])
        frame_dt.append(abs(ts - pose_ts[idx]))
    frame_dt = np.array(frame_dt)
    n_valid = (frame_dt <= args.max_dt).sum()
    print(f'  max pose dt: {frame_dt.max()*1000:.1f} ms  |  '
          f'frames with pose dt ≤ {args.max_dt}s: {n_valid}/{N}')
    if n_valid < N:
        print(f'  WARNING: {N-n_valid} frames lack a close pose and will be excluded.')

    # Only allow frame indices whose nearest pose is within max_dt
    pose_valid_idx = set(int(i) for i in range(N) if frame_dt[i] <= args.max_dt)

    depth_idx = {
        int(os.path.splitext(os.path.basename(p))[0])
        for p in glob.glob(os.path.join(args.depth, '*.npy'))
    }
    seg_idx = {
        int(os.path.splitext(os.path.basename(p))[0])
        for p in glob.glob(os.path.join(args.segs, '*.png'))
    }
    print(f'Depth frames: {len(depth_idx)},  seg frames: {len(seg_idx)}')

    results = {'meta': {'n_frames': int(N), 'n_pose_valid': int(n_valid), 'max_dt': args.max_dt}}
    for gap in sorted(args.gaps):
        sources = sorted(i for i in depth_idx
                         if i in seg_idx
                         and (i + gap) in seg_idx
                         and i in pose_valid_idx
                         and (i + gap) in pose_valid_idx)
        sources = sources[::args.stride]
        print(f'\ngap={gap}: {len(sources)} source frames')

        consistent_sum = total_sum = 0
        for k, i in enumerate(sources):
            j = i + gap
            depth_i = np.load(os.path.join(args.depth, f'{i:06d}.npy'))
            seg_i   = cv2.imread(os.path.join(args.segs, f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
            seg_j   = cv2.imread(os.path.join(args.segs, f'{j:06d}.png'), cv2.IMREAD_GRAYSCALE)
            if seg_i is None or seg_j is None:
                continue

            c, t = compute_pair(depth_i, seg_i, seg_j, K, frame_T[i], frame_T[j])
            consistent_sum += c
            total_sum      += t

            if k % 200 == 0 and k > 0:
                print(f'  {k}/{len(sources)}  running 3DVC_{gap}: '
                      f'{consistent_sum/max(total_sum,1):.4f}  ({total_sum} pairs)')

        score = consistent_sum / total_sum if total_sum > 0 else 0.0
        results[f'3DVC_{gap}'] = {
            'score':      round(score, 6),
            'consistent': consistent_sum,
            'total':      total_sum,
            'n_source':   len(sources),
        }
        print(f'  3DVC_{gap} = {score:.4f}  ({consistent_sum}/{total_sum})')

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved → {args.out}')
    summary = {k: f"{v['score']:.4f}" for k, v in results.items() if k != 'meta'}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
