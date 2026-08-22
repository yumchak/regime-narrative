"""The chart.

One image that should carry the whole project without narration: the price
series, the out-of-sample regime shading, the detected transitions, and the
explanation generated for each from news that closes on the transition date.

Design decisions worth defending:

*   **Out-of-sample states only.** The shading covers the pooled walk-forward
    test days, not an in-sample fit over the whole history. The blank stretch at
    the start is the first training window, and leaving it visibly blank is
    honest about what the model never had to predict.
*   **Filtered probability shown underneath.** The binary shading hides how
    confident the model was. The probability panel shows the transitions that
    were marginal, which matters when reading the narratives attached to them.
*   **Numbered callouts, not inline text.** Twenty annotations on a price series
    is unreadable. Numbers on the chart, text in the table below.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CALM = "#dfe9f2"
STRESSED = "#f6cfd0"
PRICE = "#1d2b36"
ACCENT = "#b4343a"
GRID = "#d9dee3"


def _shade_regimes(ax, states: pd.Series) -> None:
    """Shade contiguous runs, drawing one span per run rather than per day."""
    values = states.values.astype(int)
    idx = states.index
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            ax.axvspan(
                idx[start],
                idx[min(i, len(idx) - 1)],
                color=STRESSED if values[start] == 1 else CALM,
                linewidth=0,
                zorder=0,
            )
            start = i


def regime_chart(
    prices: pd.Series,
    states: pd.Series,
    stressed_prob: pd.Series,
    transitions: list[dict],
    *,
    excluded_transitions: list[dict] | None = None,
    narratives: dict[str, dict] | None = None,
    out_path: Path | None = None,
    title: str = "Out-of-sample volatility regimes in SPY, with generated explanations",
    subtitle: str = "",
) -> Path:
    """Render the headline chart. ``transitions`` are dicts with a 'date' key."""
    narratives = narratives or {}
    idx = states.index
    px = prices.reindex(idx).ffill()

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.1, 1.0], hspace=0.13, top=0.90, bottom=0.30,
        left=0.065, right=0.985,
    )
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1], sharex=ax)

    _shade_regimes(ax, states)
    ax.plot(idx, px.values, color=PRICE, linewidth=1.1, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("SPY close (log scale)", fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Transitions the persistence filter kept but the seed-stability filter
    # rejected. Drawn faintly and unnumbered so the shaded episodes they open
    # are accounted for rather than looking like unexplained bands.
    for t in (excluded_transitions or []):
        d = pd.Timestamp(t["date"])
        if idx[0] <= d <= idx[-1]:
            ax.axvline(d, color="#8d99a4", linewidth=0.8, linestyle=(0, (2, 2)),
                       alpha=0.75, zorder=3.5)

    # Transition markers, numbered.
    ymin, ymax = ax.get_ylim()
    for n, t in enumerate(transitions, 1):
        d = pd.Timestamp(t["date"])
        if d < idx[0] or d > idx[-1]:
            continue
        to_stressed = t.get("direction", "").startswith("calm")
        ax.axvline(d, color=ACCENT if to_stressed else "#4a6fa5",
                   linewidth=0.9, alpha=0.75, zorder=4)
        ax.annotate(
            str(n),
            xy=(d, ymax),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="circle,pad=0.22",
                facecolor=ACCENT if to_stressed else "#4a6fa5",
                edgecolor="none",
            ),
            zorder=6,
            clip_on=False,
        )

    _shade_regimes(axp, states)
    axp.plot(idx, stressed_prob.reindex(idx).values, color="#2f4858", linewidth=0.9)
    axp.axhline(0.5, color=ACCENT, linewidth=0.8, linestyle="--", alpha=0.8)
    axp.set_ylim(-0.03, 1.03)
    axp.set_ylabel("P(stressed)\nfiltered", fontsize=9)
    axp.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    axp.set_axisbelow(True)
    axp.xaxis.set_major_locator(mdates.YearLocator())
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), visible=False)

    handles = [
        Patch(facecolor=CALM, label="Calm regime (out-of-sample)"),
        Patch(facecolor=STRESSED, label="Stressed regime (out-of-sample)"),
        plt.Line2D([], [], color=ACCENT, lw=1.2, label="Transition into stress"),
        plt.Line2D([], [], color="#4a6fa5", lw=1.2, label="Transition out of stress"),
        plt.Line2D([], [], color="#8d99a4", lw=1.0, linestyle=(0, (2, 2)),
                   label="Not stable across seeds (excluded)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.94,
              ncol=2, borderpad=0.6)

    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.065, ha="left", y=0.965)
    if subtitle:
        fig.text(0.065, 0.928, subtitle, fontsize=9.5, color="#4a5560", ha="left")

    _draw_explanation_table(fig, transitions, narratives)

    out_path = out_path or Path("outputs/regime_chart.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_explanation_table(fig, transitions: list[dict], narratives: dict) -> None:
    """The numbered key beneath the chart: date, direction, generated explanation."""
    n = len(transitions)
    if not n:
        return
    n_left = (n + 1) // 2
    columns = [transitions[:n_left], transitions[n_left:]]

    fig.text(0.065, 0.262,
             "Explanation generated for each transition, from news published "
             "before that date and nothing else",
             fontsize=9.5, fontweight="bold", color="#1d2b36")

    for col_i, col in enumerate(columns):
        x = 0.065 + col_i * 0.48
        y = 0.238
        for j, t in enumerate(col):
            number = col_i * n_left + j + 1
            nar = narratives.get(f"transition_{t['date']}", {})
            to_stressed = t.get("direction", "").startswith("calm")

            if not nar:
                body = "— not yet generated —"
                colour = "#98a2ab"
            elif not nar.get("driver_identified", False):
                body = "No identifiable driver (model declined)"
                colour = "#7a848d"
            else:
                conf = str(nar.get("confidence", "")).upper()
                body = f"[{conf}] {nar.get('summary', '')}"
                colour = "#26313a"

            body = _wrap(body, 96)
            fig.text(x, y, f"{number}", fontsize=7.5, fontweight="bold",
                     color="white", ha="center", va="top",
                     bbox=dict(boxstyle="circle,pad=0.2",
                               facecolor=ACCENT if to_stressed else "#4a6fa5",
                               edgecolor="none"))
            fig.text(x + 0.016, y, f"{t['date']}", fontsize=8, fontweight="bold",
                     color="#1d2b36", va="top")
            fig.text(x + 0.062, y, body, fontsize=7.4, color=colour, va="top")
            y -= 0.021 * (1 + body.count("\n"))


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)[:2])


def dwell_histogram(dwell: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Episode-length distribution -- the count-discipline evidence."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for state, colour, label in ((0, "#7fa6cc", "Calm"), (1, "#d1656b", "Stressed")):
        sub = dwell[dwell["state"] == state]["length_days"]
        if len(sub):
            ax.hist(sub, bins=np.logspace(np.log10(max(1, sub.min())),
                                          np.log10(max(2, sub.max())), 18),
                    alpha=0.72, color=colour, label=f"{label} (n={len(sub)})")
    ax.set_xscale("log")
    ax.set_xlabel("Episode length (trading days, log scale)", fontsize=9)
    ax.set_ylabel("Episodes", fontsize=9)
    ax.set_title("Regime episode lengths after persistence filtering",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path = out_path or Path("outputs/dwell_histogram.png")
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path


def generalisation_chart(rows: list[dict], out_path: Path | None = None) -> Path:
    """Volatility ratio per index, fitted universe against never-seen indices."""
    rows = [r for r in rows if "ratio_oos" in r]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = [r["index"] for r in rows]
    vals = [r["ratio_oos"] for r in rows]
    fwd = [r.get("ratio_forward", np.nan) for r in rows]
    x = np.arange(len(labels))

    ax.bar(x - 0.19, vals, 0.36, label="Same-day volatility ratio", color="#4a6fa5")
    ax.bar(x + 0.19, fwd, 0.36, label="Forward 20-day volatility ratio", color="#d1656b")
    ax.axhline(1.0, color="#5a646d", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Stressed / calm volatility", fontsize=9)
    ax.set_title("Regime separation, out-of-sample", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path = out_path or Path("outputs/generalisation.png")
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path
