"""
Visualize multiple model predictions side-by-side in the same video.

Default layout: [Original] | [GT] | [Pred1] | [Pred2] | ...  (single row)
With --cols N:  panels are arranged in a grid of N columns, multiple rows.
                Blank (black) panels pad the last row if needed.

Usage (inside container at /workspace/TV3S/upstream/):
    # Single row (original behaviour)
    python3 tools/visualize_compare.py \
        --data-dir data/vspw/VSPW_480p \
        --out-dir  results/compare/videos \
        --pred frozen=results/dinov3_vits16_tv3s_frozen_2sample_iter160k_lr6e-5/images/result_submission \
        --pred lpft=results/dinov3_vits16_tv3s_lpft_1sample_iter160k_lr3e-5/images/result_submission \
        --num-videos 5 --fps 8

    # 2-row grid with 4 columns (7 panels → row1: 4, row2: 3 + 1 blank)
    python3 tools/visualize_compare.py \
        --data-dir data/vspw/VSPW_480p \
        --out-dir  results/compare_all/videos \
        --pred b1=results/b1_mode0/images/result_submission \
        --pred bnhead_ft=results/dinov3_vits16_bnhead_finetune_2sample_iter160k_lr3e-5/images/result_submission \
        --pred tv3s_frozen=results/dinov3_vits16_tv3s_frozen_2sample_iter160k_lr6e-5/images/result_submission \
        --pred tv3s_ft=results/dinov3_vits16_tv3s_finetune_1sample_iter160k_lr3e-5/images/result_submission \
        --pred tv3s_lpft=results/dinov3_vits16_tv3s_lpft_1sample_iter160k_lr3e-5/images/result_submission \
        --cols 4 --num-videos 5 --fps 8
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def build_palette(num_classes=124):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(1, num_classes):
        hue = int((i * 360 / (num_classes - 1)) % 180)
        hsv = np.uint8([[[hue, 220, 200]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette[i] = rgb
    return palette


def colorize_mask(mask_path, palette):
    # PIL correctly returns raw class indices from both paletted PNGs (GT)
    # and true grayscale PNGs (predictions). cv2 IMREAD_GRAYSCALE converts
    # paletted PNG palette colors to luminance, giving wrong class indices.
    mask = np.array(Image.open(str(mask_path)))
    return palette[mask.clip(0, len(palette) - 1)]


def label_panel(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)


def panels_to_grid(panels, cols, panel_h, panel_w):
    """Arrange a flat list of panels into a grid of `cols` columns."""
    # Pad to a multiple of cols with black panels
    remainder = len(panels) % cols
    if remainder:
        blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panels = panels + [blank] * (cols - remainder)

    rows = []
    for i in range(0, len(panels), cols):
        rows.append(np.concatenate(panels[i:i + cols], axis=1))
    return np.concatenate(rows, axis=0)


def parse_pred_arg(raw):
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--pred value '{raw}' must be 'name=path'"
        )
    name, path = raw.split("=", 1)
    return name.strip(), Path(path.strip())


def process_video(video_id, preds, data_dir, out_dir, palette, fps, size, cols):
    origin_dir = data_dir / "data" / video_id / "origin"
    gt_dir     = data_dir / "data" / video_id / "mask"

    per_pred_frames = {
        name: {p.stem for p in (pdir / video_id).glob("*.png")}
        for name, pdir in preds
    }
    if any(len(f) == 0 for f in per_pred_frames.values()):
        print(f"  [skip] {video_id}: missing predictions in at least one model")
        return

    common_stems = sorted(set.intersection(*per_pred_frames.values()))
    if not common_stems:
        print(f"  [skip] {video_id}: no common frames across models")
        return

    n_panels = 2 + len(preds)
    W, H = size

    if cols is None:
        # Single row (original behaviour)
        frame_w, frame_h = W * n_panels, H
    else:
        import math
        n_rows = math.ceil(n_panels / cols)
        frame_w, frame_h = W * cols, H * n_rows

    out_path = out_dir / f"{video_id}.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    written = 0
    for stem in common_stems:
        origin_path = origin_dir / f"{stem}.jpg"
        gt_path     = gt_dir     / f"{stem}.png"
        if not origin_path.exists() or not gt_path.exists():
            continue

        panels = []

        origin = cv2.resize(cv2.imread(str(origin_path)), size)
        label_panel(origin, "Original")
        panels.append(origin)

        gt = cv2.resize(colorize_mask(gt_path, palette), size)
        label_panel(gt, "Ground Truth")
        panels.append(gt)

        for name, pdir in preds:
            pred_img = cv2.resize(
                colorize_mask(pdir / video_id / f"{stem}.png", palette),
                size,
            )
            label_panel(pred_img, name)
            panels.append(pred_img)

        if cols is None:
            frame = np.concatenate(panels, axis=1)
        else:
            frame = panels_to_grid(panels, cols, H, W)

        writer.write(frame)
        written += 1

    writer.release()
    layout = f"{frame_w}x{frame_h}"
    print(f"  {video_id}: {written} frames × {n_panels} panels ({layout}) → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True, help="VSPW_480p/ root")
    parser.add_argument("--out-dir",    required=True, help="Output dir for .mp4")
    parser.add_argument("--pred",       required=True, action="append",
                        type=parse_pred_arg,
                        help="name=path (repeat for each model). path = result_submission/")
    parser.add_argument("--cols",       type=int, default=None,
                        help="Panels per row. Omit for a single horizontal strip.")
    parser.add_argument("--video-id",   default=None)
    parser.add_argument("--num-videos", type=int, default=None)
    parser.add_argument("--fps",        type=int, default=8)
    parser.add_argument("--width",      type=int, default=480)
    parser.add_argument("--height",     type=int, default=270)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = args.pred
    for name, pdir in preds:
        if not pdir.exists():
            raise FileNotFoundError(f"Pred dir for '{name}' not found: {pdir}")

    palette = build_palette(124)
    size = (args.width, args.height)

    if args.video_id:
        video_ids = [args.video_id]
    else:
        common = set.intersection(*[
            {d.name for d in pdir.iterdir() if d.is_dir()}
            for _, pdir in preds
        ])
        video_ids = sorted(common)
        if args.num_videos:
            video_ids = video_ids[:args.num_videos]

    n_panels = 2 + len(preds)
    print(f"Rendering {len(video_ids)} video(s), {n_panels} panels"
          + (f" in {args.cols}-col grid" if args.cols else " single row")
          + f" → {out_dir}")
    for vid in video_ids:
        process_video(vid, preds, data_dir, out_dir, palette, args.fps, size, args.cols)


if __name__ == "__main__":
    main()
