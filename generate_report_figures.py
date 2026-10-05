"""
Generates publication-quality figures for the IEEE Technical Report:
1. pope_benchmark_chart.png: Performance comparison on POPE Random, Popular, Adversarial splits.
2. ablation_chart.png: Ablation analysis reproducing Table 5.
3. qualitative_comparison_figure.png: Multi-panel visual attention and hallucination mitigation demonstration.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.makedirs("outputs", exist_ok=True)

# Styling for academic paper figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12
})

# -------------------------------------------------------------
# Figure 1: POPE Benchmark Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)

splits = ["Random", "Popular", "Adversarial"]
x = np.arange(len(splits))
width = 0.35

# Metrics from POPE Evaluation
baseline_f1 = [100.0, 66.67, 66.67]
agla_f1 = [100.0, 85.71, 85.71]
baseline_acc = [100.0, 50.00, 62.50]
agla_acc = [100.0, 87.50, 87.50]

rects1 = ax.bar(x - width/2, baseline_f1, width, label="Baseline (LLaVA-1.5)", color="#94A3B8", edgecolor="#475569", hatch="//")
rects2 = ax.bar(x + width/2, agla_f1, width, label="AGLA (Ours)", color="#2563EB", edgecolor="#1E40AF")

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

autolabel(rects1)
autolabel(rects2)

ax.set_ylabel("F1 Score (%)")
ax.set_title("POPE Evaluation Benchmark Across Dataset Splits (Higher is Better)")
ax.set_xticks(x)
ax.set_xticklabels(splits, fontweight="bold")
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="lower left", framealpha=0.95)

plt.tight_layout()
fig.savefig("outputs/pope_benchmark_chart.png", dpi=300)
plt.close(fig)
print("Saved outputs/pope_benchmark_chart.png")

# -------------------------------------------------------------
# Figure 2: Ablation Study (Table 5)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=300)

variants = [
    "Full AGLA",
    "w/o Truncation\n(beta=0.0)",
    "w/o Adapt. Ratio\n(Fixed=0.3)",
    "w/o Assembly\n(Local only)",
    "Regular\n(Baseline)"
]

acc = [87.50, 62.50, 87.50, 62.50, 62.50]
prec = [100.00, 60.00, 100.00, 60.00, 60.00]
f1 = [85.71, 66.67, 85.71, 66.67, 66.67]

x = np.arange(len(variants))
width = 0.26

r1 = ax.bar(x - width, acc, width, label="Accuracy", color="#3B82F6", edgecolor="#1D4ED8")
r2 = ax.bar(x, prec, width, label="Precision", color="#10B981", edgecolor="#047857")
r3 = ax.bar(x + width, f1, width, label="F1 Score", color="#F59E0B", edgecolor="#B45309")

ax.set_ylabel("Metric Value (%)")
ax.set_title("Ablation Study on POPE Adversarial Split (Validation of Components)")
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=8)
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="lower left", framealpha=0.95, ncol=3)

plt.tight_layout()
fig.savefig("outputs/ablation_chart.png", dpi=300)
plt.close(fig)
print("Saved outputs/ablation_chart.png")

# -------------------------------------------------------------
# Figure 3: Qualitative Triple-Panel Visualization
# -------------------------------------------------------------
orig_img = Image.open("data/sample_images/skateboarder.jpg").convert("RGB")
heat_img = Image.open("outputs/gradcam_heatmap.jpg").convert("RGB")
aug_img = Image.open("outputs/augmented_masked.jpg").convert("RGB")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)

axes[0].imshow(orig_img)
axes[0].set_title("(a) Original Image (v)\n[Global Scene Context]", fontsize=8, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(heat_img)
axes[1].set_title("(b) GradCAM Heatmap\n[IPM Cross-Attention]", fontsize=8, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(aug_img)
axes[2].set_title("(c) Augmented View (v_aug)\n[Prompt-Relevant Masking]", fontsize=8, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
fig.savefig("outputs/qualitative_comparison_figure.png", dpi=300)
plt.close(fig)
print("Saved outputs/qualitative_comparison_figure.png")
