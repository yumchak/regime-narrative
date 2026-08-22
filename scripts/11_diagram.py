"""Stage 11 -- the "how it works" strip for the one-page write-up.

A one-page PDF has room for exactly one picture, so it has to do real work. The
regime chart is the obvious candidate and the wrong one: shrunk to a column it
becomes unreadable, and it shows a *result* rather than a *mechanism*. The
write-up already states the results in a table.

What the page cannot say in words, and a reader most needs, is the shape of the
pipeline -- in particular that the boundary sits between the price data and the
news, and that the language model never touches a number. That is one strip of
five boxes, and it reads at a glance.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from regime_narrative.config import output_dir

INK = "#14202a"
MUT = "#5a6773"
BLUE = "#37567f"
RED = "#9a3b41"
WARN = "#9a5c17"
PALE = "#eef3f8"
PALE_R = "#fbf0f0"
LINE = "#c9d4de"


def box(ax, x, y, w, h, title, lines, face, edge):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="top",
            fontsize=8.4, fontweight="bold", color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.115 - i * 0.052, ln, ha="center", va="top",
                fontsize=6.9, color=MUT, zorder=3)


def arrow(ax, x0, x1, y, label=""):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color=MUT, zorder=1))
    if label:
        ax.text((x0 + x1) / 2, y + 0.022, label, ha="center", va="bottom",
                fontsize=6.4, color=MUT, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    w, h, y = 0.163, 0.40, 0.36
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    box(ax, xs[0], y, w, h, "Prices", ["SPY · TLT · GLD", "VIX · HYG", "2010–2026"],
        PALE, LINE)
    box(ax, xs[1], y, w, h, "HMM", ["refit every fold", "forward filter", "no lookahead"],
        PALE, BLUE)
    box(ax, xs[2], y, w, h, "Transition date", ["20-day dwell", "seed-stable", "20 dates"],
        PALE, BLUE)
    box(ax, xs[3], y, w, h, "News window", ["closes ON the date", "revision-pinned",
                                            "dates stripped"], PALE_R, WARN)
    box(ax, xs[4], y, w, h, "Explanation", ["every claim cites", "a supplied item",
                                            "+ confidence"], PALE_R, RED)

    for i in range(4):
        arrow(ax, xs[i] + w + 0.006, xs[i + 1] - 0.006, y + h / 2)

    # the boundary between the two halves -- the point of the whole design
    bx = (xs[2] + w + xs[3]) / 2
    ax.plot([bx, bx], [0.30, 0.88], color=WARN, linewidth=1.3,
            linestyle=(0, (4, 3)), zorder=4)
    ax.text(bx, 0.90, "the boundary", ha="center", va="bottom", fontsize=7.2,
            fontweight="bold", color=WARN)

    ax.annotate("", xy=(xs[0], 0.275), xytext=(xs[2] + w, 0.275),
                arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=1.0))
    ax.text((xs[0] + xs[2] + w) / 2, 0.225,
            "decides WHEN the regime changed", ha="center", va="top",
            fontsize=7.4, fontweight="bold", color=BLUE)
    ax.text((xs[0] + xs[2] + w) / 2, 0.155,
            "every number in the results comes from here", ha="center",
            va="top", fontsize=6.6, color=MUT)

    ax.annotate("", xy=(xs[3], 0.275), xytext=(xs[4] + w, 0.275),
                arrowprops=dict(arrowstyle="-", color=RED, linewidth=1.0))
    ax.text((xs[3] + xs[4] + w) / 2, 0.225,
            "describes WHAT WAS IN THE NEWS", ha="center", va="top",
            fontsize=7.4, fontweight="bold", color=RED)
    ax.text((xs[3] + xs[4] + w) / 2, 0.155,
            "sees no prices, no ticker, no dates", ha="center", va="top",
            fontsize=6.6, color=MUT)

    ax.text(0.5, 0.015,
            "Controls on both halves:  blind matching vs held-out articles  ·  "
            "citation grounding vs a random floor  ·  era-matched placebo dates",
            ha="center", va="bottom", fontsize=6.8, color=MUT)

    out = output_dir() / "pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
