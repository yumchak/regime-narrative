"""Stage 5 -- render figures and the self-contained HTML report.

Runs on whatever exists. Sections whose inputs are missing say so rather than
vanishing, so an incomplete run is visibly incomplete.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import pandas as pd

from regime_narrative.config import output_dir
from regime_narrative.report import build_report
from regime_narrative.viz import dwell_histogram, generalisation_chart, regime_chart


def _load(name: str) -> dict | None:
    p = output_dir() / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    regimes = _load("regimes.json")
    if regimes is None:
        print("outputs/regimes.json missing -- run scripts/01_regimes.py first")
        sys.exit(1)
    news = _load("news_manifest.json") or {}
    ceiling = _load("blind_match_ceiling.json")
    narratives = _load("narrative_results.json")

    states = pd.read_csv(output_dir() / "oos_states.csv", index_col=0, parse_dates=True)
    transitions = regimes["seed_stability"]["transitions"]
    # Primary-seed transitions that the seed-stability filter dropped. Shown on
    # the chart, unnumbered, so no shaded episode is left unaccounted for.
    stable_dates = {t["date"] for t in transitions}
    excluded = [t for t in regimes["transitions_primary_seed"]
                if t["date"] not in stable_dates]

    nar_map = {}
    if narratives:
        nar_map = narratives.get("narratives", {})

    print("rendering figures")
    chart = regime_chart(
        states["spy"], states["state"], states["stressed_prob"], transitions,
        excluded_transitions=excluded,
        narratives=nar_map,
        subtitle=(
            f"Two-state Gaussian HMM refit inside each of "
            f"{regimes['walk_forward']['n_folds']} walk-forward folds; scaler fitted "
            f"per fold; states assigned by forward filter only. Pooled "
            f"out-of-sample volatility ratio "
            f"{regimes['out_of_sample']['ratio']:.2f}x "
            f"(in-sample was {regimes['in_sample']['ratio']:.2f}x)."
        ),
    )
    dwell = dwell_histogram(pd.DataFrame(regimes["dwell_distribution"]))
    gen = generalisation_chart(
        [{"index": "SPY (fitted)",
          "ratio_oos": regimes["out_of_sample"]["ratio"],
          "ratio_forward": regimes["out_of_sample_forward"]["ratio"]}]
        + regimes["generalisation"]
    )

    print("building report")
    out = build_report(
        regimes, news, ceiling, narratives,
        figures={"chart": chart, "dwell": dwell, "generalisation": gen},
        out_path=output_dir() / "report.html",
    )
    size_kb = out.stat().st_size // 1024
    print(f"wrote {out} ({size_kb} KB, self-contained)")
    if not narratives:
        print("\nnote: narrative sections are marked as not run "
              "(ANTHROPIC_API_KEY not set)")


if __name__ == "__main__":
    main()
