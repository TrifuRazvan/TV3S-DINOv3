"""
Flickering rate: fraction of pixels that change predicted class between
consecutive frames, averaged over all frame pairs and all videos.

Unlike VC8/16 (which conditions on GT stability), this is unconditional —
it measures raw prediction jitter. Lower is more temporally stable.

NOTE: VC8/16 is the standard VSPW metric. Use flickering rate as a
supplementary diagnostic — it can be misleading if correct predictions
change (e.g. a moving object).

Usage:
    python3 tools/flicker_rate.py <pred_dir> [vspw_root]

    pred_dir  : path to result_submission/ folder (contains per-video subdirs)
    vspw_root : VSPW dataset root (default: data/vspw/VSPW_480p)
"""

import sys
import os
import numpy as np
from PIL import Image


def flicker_rate_for_video(pred_frames):
    """Mean fraction of pixels that change class between consecutive frames."""
    rates = []
    for i in range(len(pred_frames) - 1):
        changed = (pred_frames[i] != pred_frames[i + 1])
        rates.append(changed.mean())
    return rates


Pred = sys.argv[1]
DIR = sys.argv[2] if len(sys.argv) > 2 else 'data/vspw/VSPW_480p'

split = 'val.txt'
with open(os.path.join(DIR, split), 'r') as f:
    videolist = [line.strip() for line in f if line.strip() and line[0] != '.']

all_rates = []
skipped = 0

for video in videolist:
    pred_dir = os.path.join(Pred, video)
    if not os.path.isdir(pred_dir):
        skipped += 1
        continue

    frames = sorted(f for f in os.listdir(pred_dir) if f[0] != '.')
    if len(frames) < 2:
        continue

    pred_frames = [np.array(Image.open(os.path.join(pred_dir, f))) for f in frames]
    rates = flicker_rate_for_video(pred_frames)
    video_rate = np.mean(rates)
    all_rates.extend(rates)
    print(f'{video}  flicker={video_rate:.4f}')

overall = np.mean(all_rates)
print()
print(f'Predictions: {Pred}')
print('*' * 10)
print(f'Flickering rate: {overall:.4f}  ({overall*100:.2f}% of pixels change per frame)')
print('*' * 10)
if skipped:
    print(f'({skipped} videos not found in pred_dir)')
