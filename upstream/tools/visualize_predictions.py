"""
Visualize segmentation predictions as side-by-side video.

Layout per frame:  [Original] | [GT mask] | [Prediction]

Usage (inside container at /workspace/TV3S/upstream/):
    python3 tools/visualize_predictions.py \
        --pred-dir  results/dinov3_vits16_bnhead_finetune_2sample_iter160k_lr3e-5/images/result_submission \
        --data-dir  data/vspw/VSPW_480p \
        --out-dir   results/dinov3_vits16_bnhead_finetune_2sample_iter160k_lr3e-5/videos \
        [--video-id 1002_3nW_Y_u1S08]   # omit to process all videos
        [--num-videos 5]                  # limit how many videos to render (omit = all)
        [--fps 8]
"""

import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# 124-class VSPW palette (HSV-spread, visually distinct, class 0 = background black)
# ---------------------------------------------------------------------------
def build_palette(num_classes=124):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(1, num_classes):
        hue = int((i * 360 / (num_classes - 1)) % 180)  # OpenCV hue is 0-179
        hsv = np.uint8([[[hue, 220, 200]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette[i] = rgb
    return palette  # shape (124, 3), BGR


def colorize_mask(mask_path, palette):
    """Load a class-index PNG and convert to BGR image."""
    mask = np.array(Image.open(str(mask_path)))
    return palette[mask.clip(0, len(palette) - 1)]


def make_side_by_side(origin_path, gt_path, pred_path, palette, size):
    """Return a single (H, W*3, 3) BGR frame."""
    origin = cv2.imread(str(origin_path))
    origin = cv2.resize(origin, size)

    gt = colorize_mask(gt_path, palette)
    gt = cv2.resize(gt, size)

    pred = colorize_mask(pred_path, palette)
    pred = cv2.resize(pred, size)

    # Add labels
    for img, label in [(origin, "Original"), (gt, "Ground Truth"), (pred, "Prediction")]:
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 1, cv2.LINE_AA)

    return np.concatenate([origin, gt, pred], axis=1)


def process_video(video_id, pred_dir, data_dir, out_dir, palette, fps, size):
    origin_dir = data_dir / "data" / video_id / "origin"
    gt_dir     = data_dir / "data" / video_id / "mask"
    pred_vdir  = pred_dir / video_id

    if not pred_vdir.exists():
        print(f"  [skip] no predictions for {video_id}")
        return

    frames = sorted(pred_vdir.glob("*.png"))
    if not frames:
        return

    out_path = out_dir / f"{video_id}.mp4"
    H, W = size[1], size[0]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W * 3, H),
    )

    written = 0
    for pred_frame in frames:
        stem = pred_frame.stem  # e.g. "00000543"
        origin_path = origin_dir / f"{stem}.jpg"
        gt_path     = gt_dir     / f"{stem}.png"

        if not origin_path.exists() or not gt_path.exists():
            continue

        combined = make_side_by_side(origin_path, gt_path, pred_frame, palette, size)
        writer.write(combined)
        written += 1

    writer.release()
    print(f"  {video_id}: {written} frames → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir",   required=True, help="result_submission/ directory")
    parser.add_argument("--data-dir",   required=True, help="VSPW_480p/ root directory")
    parser.add_argument("--out-dir",    required=True, help="Output directory for .mp4 files")
    parser.add_argument("--video-id",   default=None,  help="Single video ID to render")
    parser.add_argument("--num-videos", type=int, default=None, help="Max videos to render")
    parser.add_argument("--fps",        type=int, default=8)
    parser.add_argument("--width",      type=int, default=480, help="Per-panel width (height auto)")
    parser.add_argument("--height",     type=int, default=270, help="Per-panel height")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = build_palette(124)
    size = (args.width, args.height)  # (W, H) for cv2

    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = sorted([d.name for d in pred_dir.iterdir() if d.is_dir()])
        if args.num_videos:
            video_ids = video_ids[:args.num_videos]

    print(f"Rendering {len(video_ids)} video(s) → {out_dir}")
    for vid in video_ids:
        process_video(vid, pred_dir, data_dir, out_dir, palette, args.fps, size)


if __name__ == "__main__":
    main()
