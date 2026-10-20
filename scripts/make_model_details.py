"""Draw detailed module diagrams for the STAGC model.

Outputs:
  assets/detail_stag.png
  assets/detail_convlstm.png
  assets/detail_attention.png
  assets/detail_output.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "assets"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})


def box(ax, x, y, w, h, title, shape="", color="#e0f2fe", title_size=11,
        shape_size=9, depth=0):
    back = tuple(c * 0.86 for c in to_rgb(color))
    for i in range(depth, 0, -1):
        ax.add_patch(Rectangle(
            (x + 0.07 * i, y + 0.07 * i), w, h,
            facecolor=back, edgecolor="#334155", linewidth=1.0, zorder=1,
        ))
    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor=color, edgecolor="#334155",
        linewidth=1.2, zorder=2,
    ))
    if shape:
        ax.text(x + w / 2, y + h * 0.66, title,
                ha="center", va="center", fontsize=title_size,
                fontweight="bold", color="#0f172a")
        ax.text(x + w / 2, y + h * 0.25, shape,
                ha="center", va="center", fontsize=shape_size, color="#334155")
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center", fontsize=title_size,
                fontweight="bold", color="#0f172a")


def op(ax, x, y, symbol, radius=0.24):
    ax.add_patch(Circle((x, y), radius, facecolor="#f8fafc",
                        edgecolor="#334155", linewidth=1.2))
    ax.text(x, y, symbol, ha="center", va="center", fontsize=13, color="#0f172a")


def arrow(ax, p1, p2, dashed=False, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=lw,
        color="#475569", linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    ))


def poly_arrow(ax, pts, dashed=True, lw=1.3):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle="--" if dashed else "-",
                color="#475569", linewidth=lw)
    arrow(ax, pts[-2], pts[-1], dashed=dashed, lw=lw)


def label(ax, x, y, text, size=10):
    ax.text(x, y, text, fontsize=size, color="#475569", ha="center")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def stag():
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.8)
    ax.set_aspect("equal")
    ax.axis("off")

    box(ax, 0.3, 3.6, 2.0, 0.9, "$x_t$", "(B,5,16,16)", "#dbeafe", depth=2)
    box(ax, 0.3, 1.4, 2.0, 0.9, "$h_{t-1}$", "(B,64,16,16)", "#e2e8f0", depth=2)
    box(ax, 2.8, 2.5, 1.8, 0.9, "Concat", "69 channels", "#f1f5f9", depth=2)
    box(ax, 5.0, 2.5, 2.0, 0.9, "Conv2d 7x7", "69 -> 1, pad=3", "#bfdbfe")
    box(ax, 7.3, 2.5, 1.5, 0.9, "Sigmoid", "$\\sigma$", "#fed7aa")
    box(ax, 9.1, 2.5, 1.4, 0.9, "$g_t$", "(B,1,16,16)", "#fed7aa", depth=1)
    box(ax, 9.1, 4.0, 1.4, 0.7, "$1+g_t$", "", "#f8fafc")
    op(ax, 11.0, 4.35, r"$\odot$")
    box(ax, 11.55, 3.9, 1.3, 0.9, "$x'_t$", "(B,5,16,16)", "#bfdbfe", depth=2)

    arrow(ax, (2.3, 4.05), (2.8, 3.05))
    arrow(ax, (2.3, 1.85), (2.8, 2.85))
    arrow(ax, (4.6, 2.95), (5.0, 2.95))
    arrow(ax, (7.0, 2.95), (7.3, 2.95))
    arrow(ax, (8.8, 2.95), (9.1, 2.95))
    arrow(ax, (9.8, 3.4), (9.8, 4.0))
    arrow(ax, (10.5, 4.35), (10.76, 4.35))
    arrow(ax, (11.24, 4.35), (11.55, 4.35))
    poly_arrow(ax, [(1.3, 4.5), (1.3, 5.3), (11.0, 5.3), (11.0, 4.59)])

    label(ax, 6.6, 1.05, "$x'_t = x_t \\odot (1 + g_t)$", 14)
    label(ax, 6.6, 0.45, "Sigmoid gate; dashed line is the residual path", 10)

    fig.suptitle("STAG: spatial residual gate", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "detail_stag.png")


def convlstm():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    box(ax, 0.3, 4.0, 2.0, 0.9, "Concat", "(B,69,16,16)", "#f1f5f9", depth=2)
    box(ax, 2.8, 4.0, 2.0, 0.9, "Conv2d 3x3", "69 -> 256, pad=1", "#bfdbfe")
    box(ax, 5.3, 4.0, 1.5, 0.9, "Split", "4 x 64", "#e2e8f0")

    gates = [
        ("$i_t$", "Sigmoid, 64 ch", 5.6, "#dcfce7"),
        ("$f_t$", "Sigmoid, 64 ch", 4.5, "#dcfce7"),
        ("$o_t$", "Sigmoid, 64 ch", 3.4, "#dcfce7"),
        ("$g_t$", "Tanh, 64 ch", 2.3, "#fed7aa"),
    ]
    for title, shape, y, color in gates:
        box(ax, 8.0, y, 2.0, 0.8, title, shape, color, depth=1)

    arrow(ax, (2.3, 4.45), (2.8, 4.45))
    arrow(ax, (4.8, 4.45), (5.3, 4.45))
    for _, _, y, _ in gates:
        arrow(ax, (6.8, 4.45), (8.0, y + 0.4), rad=0.0)

    box(ax, 1.8, 0.9, 5.0, 0.8,
        r"$c_t = f_t \odot c_{t-1} + i_t \odot g_t$", "", "#f8fafc", 12)
    box(ax, 8.2, 0.9, 4.2, 0.8,
        r"$h_t = o_t \odot \tanh(c_t)$", "", "#f8fafc", 12)

    arrow(ax, (9.0, 2.3), (7.0, 1.7), rad=0.05)
    poly_arrow(ax, [(10.0, 3.4), (11.2, 3.4), (11.2, 1.7), (10.4, 1.7)], dashed=False)

    label(ax, 5.5, 1.95, "i, f, o: sigmoid    g: tanh", 10)
    label(ax, 5.5, 0.35, "Cell state update", 9)
    label(ax, 10.3, 0.35, "Hidden state", 9)
    label(ax, 7.0, 6.1, "4 gates x 64 channels = 256 channels", 10)

    fig.suptitle("ConvLSTM cell: gates, cell state, and hidden state", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "detail_convlstm.png")


def attention():
    fig, ax = plt.subplots(figsize=(15, 7), dpi=180)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    box(ax, 0.3, 4.8, 2.2, 0.9, "$h_t^{dec}$", "(B,64,16,16)", "#bbf7d0", depth=2)
    box(ax, 3.0, 4.8, 1.6, 0.9, "GAP", "H x W -> 1", "#dcfce7")
    box(ax, 5.0, 4.8, 1.8, 0.9, "Q: Linear", "64 -> 64", "#dcfce7")

    box(ax, 0.3, 2.0, 2.4, 0.9, "$H^{enc}$", "(B,12,64,16,16)", "#e2e8f0", depth=2)
    box(ax, 3.0, 2.0, 1.6, 0.9, "GAP", "H x W -> 1", "#e2e8f0")
    box(ax, 5.0, 2.0, 1.8, 0.9, "K: Linear", "64 -> 64", "#e2e8f0")

    box(ax, 7.3, 3.4, 2.2, 0.9, "Scores", "$QK^T / \\sqrt{64}$", "#fed7aa")
    box(ax, 10.0, 3.4, 2.0, 0.9, "Softmax", "weights (B,1,12)", "#fed7aa")
    box(ax, 10.0, 1.8, 2.2, 0.9, "Context", "(B,64,16,16)", "#fed7aa", depth=2)
    box(ax, 12.7, 3.0, 2.0, 0.9, "Concat + 1x1 Conv", "128 -> 64", "#bbf7d0")
    box(ax, 12.7, 4.8, 2.0, 0.9, "$h_t^{fused}$", "(B,64,16,16)", "#bbf7d0", depth=2)

    arrow(ax, (2.5, 5.25), (3.0, 5.25))
    arrow(ax, (4.6, 5.25), (5.0, 5.25))
    arrow(ax, (2.7, 2.45), (3.0, 2.45))
    arrow(ax, (4.6, 2.45), (5.0, 2.45))
    arrow(ax, (6.8, 5.25), (7.3, 4.0))
    arrow(ax, (6.8, 2.45), (7.3, 3.7))
    arrow(ax, (9.5, 3.85), (10.0, 3.85))
    arrow(ax, (11.0, 3.4), (11.1, 2.7))
    arrow(ax, (12.2, 2.25), (12.7, 3.15))
    arrow(ax, (13.7, 3.9), (13.7, 4.8))

    poly_arrow(ax, [(1.4, 4.8), (1.4, 1.1), (12.7, 1.1), (12.7, 2.9)])
    poly_arrow(ax, [(1.5, 2.0), (1.5, 1.5), (10.0, 1.5), (10.0, 1.8)], lw=1.1)

    label(ax, 4.2, 6.0, "Q from decoder state", 9)
    label(ax, 4.2, 2.95, "K from encoder states", 9)
    label(ax, 5.6, 1.35, "V path", 9)
    label(ax, 8.4, 2.9, "V = $H^{enc}$ (no projection)", 9)
    label(ax, 6.0, 0.65, "concat also includes $h_t^{dec}$", 9)

    fig.suptitle("Temporal attention and fusion", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "detail_attention.png")


def output():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    box(ax, 0.3, 4.0, 2.2, 0.9, "$h_t^{fused}$", "(B,64,16,16)", "#bbf7d0", depth=2)
    box(ax, 3.0, 4.0, 2.2, 0.9, "Output Conv2d 1x1", "64 -> 1, linear", "#bfdbfe")
    box(ax, 5.8, 4.0, 2.0, 0.9, "Predicted frame", "(B,1,16,16)", "#e2e8f0", depth=2)
    box(ax, 8.3, 4.0, 2.2, 0.9, "Denormalize", "x (vmax-vmin) + vmin", "#f8fafc")

    box(ax, 3.0, 1.8, 2.2, 0.9, "Target frame", "(B,1,16,16)", "#e2e8f0", depth=2)
    box(ax, 5.8, 1.8, 2.2, 0.9, "Masked Huber", "strict & observed", "#fed7aa")

    arrow(ax, (2.5, 4.45), (3.0, 4.45))
    arrow(ax, (5.2, 4.45), (5.8, 4.45))
    arrow(ax, (7.8, 4.45), (8.3, 4.45))
    arrow(ax, (5.2, 2.25), (5.8, 2.25))
    arrow(ax, (6.8, 4.0), (6.8, 2.7))

    poly_arrow(ax, [(6.8, 4.9), (6.8, 5.5), (1.4, 5.5), (1.4, 4.9)])
    label(ax, 4.1, 5.65, "predicted frame is fed back as the next decoder input", 9)
    label(ax, 6.9, 1.35, "loss only on observed cells", 9)
    label(ax, 9.4, 3.55, "no sigmoid/tanh at the output", 9)

    fig.suptitle("Output layer and masked loss", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "detail_output.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stag()
    convlstm()
    attention()
    output()


if __name__ == "__main__":
    main()
