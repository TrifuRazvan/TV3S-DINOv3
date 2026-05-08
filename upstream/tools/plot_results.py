"""
Plot mIoU vs backbone params and VC8 vs mIoU for all TV3S-DINOv3 experiments.
Run from anywhere — outputs two PNG files next to this script.

Usage:
    python3 tools/plot_results.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────
# (label, params_M, miou, mvc8, mvc16, variant)
# variant: 0=frozen, 1=ft, 2=lpft, 3=baseline

RESULTS = [
    # Paper baselines
    ("MiT-B1",    15,   39.99, 88.91, 84.52, 3),
    ("MiT-B2",    28,   46.30, 91.50, 88.35, 3),
    # ViT-S/16
    ("ViT-S/16",  22,   50.95, 90.60, 87.60, 0),
    ("ViT-S/16",  22,   53.32, 91.36, 88.63, 1),
    ("ViT-S/16",  22,   53.03, 92.02, 89.48, 2),
    # ViT-S/16+
    ("ViT-S/16+", 28.7, 53.87, 91.04, 88.20, 0),
    ("ViT-S/16+", 28.7, 56.08, 91.86, 89.24, 1),
    ("ViT-S/16+", 28.7, 55.10, 92.40, 89.84, 2),
    # ViT-B/16
    ("ViT-B/16",  86,   59.31, 91.82, 89.18, 0),
    ("ViT-B/16",  86,   59.81, 92.37, 89.76, 1),
    ("ViT-B/16",  86,   60.90, 93.18, 90.88, 2),
    # ConvNeXt-B
    ("CnX-B",     89,   58.62, 91.38, 88.57, 0),
    ("CnX-B",     89,   50.14, 91.22, 87.86, 1),
]

VARIANT_STYLE = {
    0: dict(color="#4C72B0", marker="o", label="Frozen"),
    1: dict(color="#DD8452", marker="s", label="Finetune"),
    2: dict(color="#55A868", marker="^", label="LP-FT"),
    3: dict(color="#888888", marker="D", label="TV3S paper"),
}

labels  = [r[0] for r in RESULTS]
params  = [r[1] for r in RESULTS]
mious   = [r[2] for r in RESULTS]
mvc8s   = [r[3] for r in RESULTS]
variants= [r[5] for r in RESULTS]


# ── Plot 1: mIoU vs Backbone Params ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    s = VARIANT_STYLE[v]
    ax.scatter(p, m, color=s["color"], marker=s["marker"],
               s=90, zorder=3, edgecolors="white", linewidths=0.6)

# Label each backbone group (offset so text doesn't overlap)
label_offsets = {
    "MiT-B1":   (-1,  -1.2),
    "MiT-B2":   (-1,  -1.2),
    "ViT-S/16": (-1,   0.6),
    "ViT-S/16+":(-1,   0.6),
    "ViT-B/16": ( 1.5, 0.3),
    "CnX-B":    ( 1.5,-1.2),
}
seen = set()
for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    if lbl not in seen:
        dx, dy = label_offsets.get(lbl, (1.5, 0.3))
        ax.annotate(lbl, (p, m), xytext=(p + dx, m + dy),
                    fontsize=8.5, color="#333333")
        seen.add(lbl)

legend_handles = [
    mpatches.Patch(color=s["color"], label=s["label"])
    for s in VARIANT_STYLE.values()
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
ax.set_xlabel("Backbone parameters (M)", fontsize=11)
ax.set_ylabel("mIoU (%)", fontsize=11)
ax.set_title("TV3S-DINOv3: mIoU vs Backbone Size", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(0, 105)
ax.set_ylim(36, 65)

plt.tight_layout()
out1 = OUT_DIR / "plot_miou_vs_params.png"
plt.savefig(out1, dpi=150)
print(f"Saved: {out1}")
plt.close()


# ── Plot 2: mVC8 vs mIoU (Pareto front) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    s = VARIANT_STYLE[v]
    ax.scatter(m, v8, color=s["color"], marker=s["marker"],
               s=90, zorder=3, edgecolors="white", linewidths=0.6)
    ax.annotate(lbl, (m, v8), xytext=(m + 0.2, v8 + 0.05),
                fontsize=7.5, color="#444444")

ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
ax.set_xlabel("mIoU (%)", fontsize=11)
ax.set_ylabel("mVC8 (%)", fontsize=11)
ax.set_title("TV3S-DINOv3: Temporal Consistency vs Spatial Accuracy", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
out2 = OUT_DIR / "plot_vc8_vs_miou.png"
plt.savefig(out2, dpi=150)
print(f"Saved: {out2}")
plt.close()


# ── Plot 3: mVC8 vs Backbone Params ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    s = VARIANT_STYLE[v]
    ax.scatter(p, v8, color=s["color"], marker=s["marker"],
               s=90, zorder=3, edgecolors="white", linewidths=0.6)

seen = set()
for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    if lbl not in seen:
        dx, dy = label_offsets.get(lbl, (1.5, 0.3))
        ax.annotate(lbl, (p, v8), xytext=(p + dx, v8 + 0.05),
                    fontsize=8.5, color="#333333")
        seen.add(lbl)

ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
ax.set_xlabel("Backbone parameters (M)", fontsize=11)
ax.set_ylabel("mVC8 (%)", fontsize=11)
ax.set_title("TV3S-DINOv3: Temporal Consistency vs Backbone Size", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(0, 105)

plt.tight_layout()
out3 = OUT_DIR / "plot_vc8_vs_params.png"
plt.savefig(out3, dpi=150)
print(f"Saved: {out3}")
plt.close()


# ── Plot 4: mVC16 vs Backbone Params ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    s = VARIANT_STYLE[v]
    ax.scatter(p, v16, color=s["color"], marker=s["marker"],
               s=90, zorder=3, edgecolors="white", linewidths=0.6)

seen = set()
for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    if lbl not in seen:
        dx, dy = label_offsets.get(lbl, (1.5, 0.3))
        ax.annotate(lbl, (p, v16), xytext=(p + dx, v16 + 0.05),
                    fontsize=8.5, color="#333333")
        seen.add(lbl)

ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
ax.set_xlabel("Backbone parameters (M)", fontsize=11)
ax.set_ylabel("mVC16 (%)", fontsize=11)
ax.set_title("TV3S-DINOv3: Long-range Temporal Consistency vs Backbone Size", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(0, 105)

plt.tight_layout()
out4 = OUT_DIR / "plot_vc16_vs_params.png"
plt.savefig(out4, dpi=150)
print(f"Saved: {out4}")
plt.close()


# ── Plot 5: mVC16 vs mIoU ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for i, (lbl, p, m, v8, v16, v) in enumerate(RESULTS):
    s = VARIANT_STYLE[v]
    ax.scatter(m, v16, color=s["color"], marker=s["marker"],
               s=90, zorder=3, edgecolors="white", linewidths=0.6)
    ax.annotate(lbl, (m, v16), xytext=(m + 0.2, v16 + 0.05),
                fontsize=7.5, color="#444444")

ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
ax.set_xlabel("mIoU (%)", fontsize=11)
ax.set_ylabel("mVC16 (%)", fontsize=11)
ax.set_title("TV3S-DINOv3: Long-range Temporal Consistency vs Spatial Accuracy", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
out5 = OUT_DIR / "plot_vc16_vs_miou.png"
plt.savefig(out5, dpi=150)
print(f"Saved: {out5}")
plt.close()
