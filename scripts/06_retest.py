"""Stage 6 -- test-retest stability of the model's own labels.

Why this is necessary rather than nice to have
-----------------------------------------------
The headline placebo statistic is ``is_confident``: driver identified, at high
or medium confidence. That is a label the model emits about itself. The brief
assumed temperature 0 would make it deterministic, but sampling parameters have
been removed from the API and there is no determinism knob left.

So the same window, sent twice, can come back with different labels. If that
happens often, the Fisher p-value on the placebo comparison is measuring noise
as much as signal, and the honest thing is to know by how much before quoting
it.

This runs every window a second time under a separate cache namespace and
reports:

    * agreement on ``driver_identified`` and on ``is_confident``
    * Cohen's kappa, which corrects for agreement expected by chance
    * how far the headline placebo rates move between the two runs

A high kappa means the label is stable and the placebo statistic can be read at
face value. A low one is a finding in its own right and belongs in the report
next to the p-value, not instead of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from regime_narrative.config import has_env, load_settings, output_dir
from regime_narrative.narrative import generate_narrative

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_stage4 = import_module("04_narratives")


def _kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters over the same items."""
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="retest", help="cache namespace for the replicate")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not has_env("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set")
        sys.exit(2)

    cfg = load_settings()
    primary = json.loads(
        (output_dir() / "narrative_results.json").read_text(encoding="utf-8")
    )["narratives"]

    windows = _stage4._load_all_windows(cfg)
    keys = [k for k in windows if k in primary]
    if args.limit:
        keys = keys[: args.limit]
    print(f"re-running {len(keys)} windows under label {args.label!r}")

    rows = []
    for i, k in enumerate(keys, 1):
        n2 = generate_narrative(
            windows[k]["gen"], window_id=k, kind=windows[k]["kind"],
            run_label=args.label,
        )
        p = primary[k]
        conf1 = p["driver_identified"] and p["confidence"] in {"high", "medium"}
        rows.append({
            "window_id": k,
            "kind": windows[k]["kind"],
            "driver_1": p["driver_identified"], "driver_2": n2.driver_identified,
            "conf_label_1": p["confidence"], "conf_label_2": n2.confidence,
            "is_confident_1": conf1, "is_confident_2": n2.is_confident,
            "agree_driver": p["driver_identified"] == n2.driver_identified,
            "agree_confident": conf1 == n2.is_confident,
            "agree_exact_confidence": p["confidence"] == n2.confidence,
        })
        flag = "" if rows[-1]["agree_confident"] else "   <-- FLIPPED"
        print(f"  [{i:>2}/{len(keys)}] {k:<34} "
              f"{p['confidence']:>6} -> {n2.confidence:<6}{flag}")

    n = len(rows)
    d1 = [r["driver_1"] for r in rows]
    d2 = [r["driver_2"] for r in rows]
    c1 = [r["is_confident_1"] for r in rows]
    c2 = [r["is_confident_2"] for r in rows]

    def _rate(keyname, sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return None
        return round(sum(r[keyname] for r in sub) / len(sub), 3)

    out = {
        "n_windows": n,
        "model": cfg["llm"]["model"],
        "effort": cfg["llm"]["effort"],
        "agreement_driver_identified": round(sum(r["agree_driver"] for r in rows) / n, 3),
        "agreement_is_confident": round(sum(r["agree_confident"] for r in rows) / n, 3),
        "agreement_exact_confidence_label": round(
            sum(r["agree_exact_confidence"] for r in rows) / n, 3
        ),
        "kappa_driver_identified": round(_kappa(d1, d2), 3),
        "kappa_is_confident": round(_kappa(c1, c2), 3),
        "confident_rate_run1_transitions": _rate("is_confident_1", lambda r: r["kind"] == "transition"),
        "confident_rate_run2_transitions": _rate("is_confident_2", lambda r: r["kind"] == "transition"),
        "confident_rate_run1_placebos": _rate("is_confident_1", lambda r: r["kind"] == "placebo"),
        "confident_rate_run2_placebos": _rate("is_confident_2", lambda r: r["kind"] == "placebo"),
        "flipped": [r["window_id"] for r in rows if not r["agree_confident"]],
        "per_window": rows,
        "note": (
            "Sampling parameters were removed from the API, so identical inputs "
            "do not guarantee identical outputs. This measures how much of the "
            "placebo comparison is stable signal rather than run-to-run noise."
        ),
    }

    print(f"\nagreement on driver_identified : {out['agreement_driver_identified']} "
          f"(kappa {out['kappa_driver_identified']})")
    print(f"agreement on is_confident      : {out['agreement_is_confident']} "
          f"(kappa {out['kappa_is_confident']})")
    print(f"exact confidence label         : {out['agreement_exact_confidence_label']}")
    print(f"transition confident rate      : "
          f"{out['confident_rate_run1_transitions']} -> {out['confident_rate_run2_transitions']}")
    print(f"placebo confident rate         : "
          f"{out['confident_rate_run1_placebos']} -> {out['confident_rate_run2_placebos']}")

    path = output_dir() / "retest_stability.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
