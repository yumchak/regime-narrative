"""Stage 12 -- the tools slide for the video.

The submission requires the video to name the AI coding tools used and list the
models, APIs and platforms integrated. That is a checklist item, and the easiest
way to lose it is to say the list aloud and hope the judge catches every name.

So it goes on screen: 1920x1080, readable at a glance, held for the twenty
seconds it takes to say. Generated rather than hand-made in slide software so it
cannot drift out of step with settings.yaml.
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

from regime_narrative.config import load_settings, output_dir

INK = "#14202a"
MUT = "#5d6b78"
BLUE = "#37567f"
RED = "#9a3b41"
BG = "#ffffff"
RULE = "#d6dee6"


def main() -> None:
    cfg = load_settings()
    model = cfg["llm"]["model"]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.08, 0.86, "How AI was used", fontsize=46, fontweight="bold", color=INK)
    ax.plot([0.08, 0.92], [0.825, 0.825], color=INK, lw=2.4)

    # --- built with -------------------------------------------------------
    ax.text(0.08, 0.745, " ".join("BUILT WITH"), fontsize=17, fontweight="bold", color=MUT)
    ax.text(0.08, 0.665, "Claude Code", fontsize=40, fontweight="bold", color=BLUE)
    ax.text(0.08, 0.605, "Anthropic's command-line agent — pipeline, tests, dashboard",
            fontsize=22, color=MUT)

    # --- prompts ----------------------------------------------------------
    ax.text(0.08, 0.515, " ".join("PROMPTS"), fontsize=17, fontweight="bold", color=MUT)
    ax.text(0.08, 0.445, "Version-controlled files, never inline strings",
            fontsize=28, color=INK)
    ax.text(0.08, 0.392, "Every call logged: model id · prompt hash · input hash",
            fontsize=22, color=MUT)

    # --- integrated -------------------------------------------------------
    ax.text(0.08, 0.318, "MODELS,  APIs  AND  PLATFORMS  INTEGRATED", fontsize=17, fontweight="bold", color=MUT)

    rows = [
        (f"{model}", "Anthropic Messages API — one call per date, schema enforced"),
        ("Wikipedia MediaWiki API", "news, pinned to the revision that existed then"),
        ("yfinance", "prices"),
        ("hmmlearn · scikit-learn", "the regime model and the statistics"),
        ("Streamlit", "the dashboard"),
    ]
    y = 0.252
    for name, detail in rows:
        ax.text(0.08, y, name, fontsize=25, fontweight="bold", color=INK)
        ax.text(0.40, y, detail, fontsize=23, color=MUT)
        y -= 0.043

    ax.plot([0.08, 0.92], [0.058, 0.058], color=RULE, lw=1.2)
    ax.text(0.08, 0.022,
            "The statistical model decides WHEN.   The language model only describes "
            "WHAT WAS IN THE NEWS.",
            fontsize=21, color=RED)

    out = output_dir() / "slide_tools.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, 1920x1080)")


if __name__ == "__main__":
    main()
