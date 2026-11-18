#!/usr/bin/env python3
"""Plot precision-recall curves (IoU 0.5) for each WIDER setting, comparing several models.

Reads the PR curves stored in results/<split>/<name>/metrics.json by evaluate.py, and
saves results/<split>/pr_curves.png with one panel each for Easy, Medium and Hard. Each
model's curve is marked with a circle at its best-F1 threshold and a square at
confidence 0.5. Up to 4 models are supported.

Run from face_detection/:
    python tools/plot_pr.py test resnet50 mobilenet0.25
Arguments: the split (default "test"), then the model names (default resnet50 and mobilenet0.25).
"""

import json
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # draw to a file, no display window needed
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
split = sys.argv[1] if len(sys.argv) > 1 else "test"
names = sys.argv[2:] or ["resnet50", "mobilenet0.25"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True, facecolor="#fcfcfb")  # Easy, Medium, Hard panels
for ax, s in zip(axes, ["easy", "medium", "hard"]):
    ax.set_facecolor("#fcfcfb")
    for c, n in zip(COLORS, names):  # one curve per model
        m = json.loads(Path(f"results/{split}/{n}/metrics.json").read_text())
        r = m["settings"][s]
        ax.plot(
            r["pr_curve"]["recall"],
            r["pr_curve"]["precision"],
            color=c,
            lw=2,
            label=f"{n}  AP {r['official_ap']*100:.1f}",
        )
        b = r["best_f1"]
        ax.plot(b["recall"], b["precision"], "o", ms=8, color=c, mec="#fcfcfb", mew=2)  # circle: best-F1 point
        o = r["at_conf_op"]
        ax.plot(o["recall"], o["precision"], "s", ms=8, color=c, mec="#fcfcfb", mew=2)  # square: conf 0.5 point
    # Panel styling.
    ax.set_title(s.capitalize(), loc="left", fontsize=12, color="#1f1f1e")
    ax.set_xlabel("Recall", color="#55554f")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.tick_params(colors="#55554f", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
axes[0].set_ylabel("Precision", color="#55554f")
fig.suptitle(
    f"WIDER FACE {split} — precision vs recall at IoU 0.5   (● best-F1 threshold, ■ conf 0.5)",
    x=0.01,
    ha="left",
    fontsize=12,
    color="#1f1f1e",
)
fig.tight_layout()
out = f"results/{split}/pr_curves.png"
fig.savefig(out, dpi=130)
print(out)
