"""Stage 10 -- replicate the whole narrative layer on a second model.

A result that only holds on one model is a result about that model. Running the
identical pipeline -- identical prompt, identical windows, identical scoring --
on a second model separates "this is a property of the method" from "this is a
property of claude-opus-5".

Three things are worth knowing and only this run can tell you:

  * Does the placebo gap survive? If the second model also produces confident
    explanations for ordinary dates at a similar rate, the limitation is a
    property of the task, not of one model's calibration.
  * Does blind matching survive? If it does, window-specificity is a property of
    having the news, not of a particular model's writing style.
  * How much do two models agree with each other, window by window? This is a
    ceiling on how much signal the confidence label can carry: two models that
    disagree constantly cannot both be measuring something real.

Cache is namespaced by run_label, so the primary results are untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np

from regime_narrative.config import has_env, load_settings, output_dir
from regime_narrative.narrative import generate_narrative

_stage4 = import_module("04_narratives")


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if exp >= 1 else (obs - exp) / (1 - exp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--label", default="sonnet5")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not has_env("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set")
        sys.exit(2)

    cfg = load_settings()
    primary_path = output_dir() / "narrative_results.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary_model = primary.get("model")
    prim_nar = primary["narratives"]

    print(f"replicating on {args.model} (primary was {primary_model})")
    windows = _stage4._load_all_windows(cfg)
    keys = [k for k in windows if k in prim_nar]
    if args.limit:
        keys = keys[: args.limit]
    windows = {k: windows[k] for k in keys}
    print(f"  {len(keys)} windows, identical prompt "
          f"({cfg['llm']['prompt_version']}) and identical inputs")

    narratives = {}
    for i, k in enumerate(keys, 1):
        w = windows[k]
        n = generate_narrative(
            w["gen"], window_id=k, kind=w["kind"],
            model=args.model, run_label=args.label,
        )
        narratives[k] = n
        p = prim_nar[k]
        pc = p["driver_identified"] and p["confidence"] in {"high", "medium"}
        flag = "" if pc == n.is_confident else "   <-- differs"
        print(f"  [{i:>2}/{len(keys)}] {k:<34} "
              f"{p['confidence']:>6} -> {n.confidence:<6}{flag}")

    print("\nscoring with the identical control suite")
    results = _stage4._evaluate(cfg, windows, narratives)
    results["model"] = args.model
    results["replicate_of"] = primary_model

    # ---- cross-model agreement ------------------------------------------
    a = [prim_nar[k]["driver_identified"] and
         prim_nar[k]["confidence"] in {"high", "medium"} for k in keys]
    b = [narratives[k].is_confident for k in keys]
    da = [prim_nar[k]["driver_identified"] for k in keys]
    db = [narratives[k].driver_identified for k in keys]

    def rate(sel, arr):
        idx = [i for i, k in enumerate(keys) if windows[k]["kind"] == sel]
        return round(sum(arr[i] for i in idx) / len(idx), 3) if idx else None

    results["cross_model"] = {
        "primary_model": primary_model,
        "replicate_model": args.model,
        "n_windows": len(keys),
        "agreement_is_confident": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
        "kappa_is_confident": round(kappa(a, b), 3),
        "agreement_driver_identified": round(
            sum(x == y for x, y in zip(da, db)) / len(da), 3),
        "kappa_driver_identified": round(kappa(da, db), 3),
        "confident_rate_transitions": {
            primary_model: rate("transition", a),
            args.model: rate("transition", b),
        },
        "confident_rate_controls": {
            primary_model: rate("placebo", a),
            args.model: rate("placebo", b),
        },
        "disagreements": [
            {"window_id": k,
             primary_model: prim_nar[k]["confidence"],
             args.model: narratives[k].confidence}
            for i, k in enumerate(keys) if a[i] != b[i]
        ],
    }

    cm = results["cross_model"]
    print(f"\ncross-model agreement on 'confident': "
          f"{cm['agreement_is_confident']} (kappa {cm['kappa_is_confident']})")
    print(f"transition confident rate: {cm['confident_rate_transitions']}")
    print(f"control confident rate:    {cm['confident_rate_controls']}")

    pb, bm, fa = results["placebo"], results["blind_match"], results["faithfulness"]
    # An arm can be empty -- on a --limit run, or if no control lands in the
    # clean stratum -- so read every rate defensively rather than indexing.
    print(f"\nplacebo    : transitions "
          f"{pb['transitions'].get('confident_rate', 'n/a')} vs "
          f"clean controls "
          f"{pb['placebos_clean_stratum'].get('confident_rate', 'n/a')} "
          f"(Fisher p={pb.get('fisher_p_clean')})")
    b1 = bm.get("transitions_all_sections", {})
    if "error" not in b1:
        print(f"blind match: {b1['n_correct']}/{b1['n_scored']} = "
              f"{100 * b1['accuracy']:.1f}% vs {100 * b1['chance']:.1f}% chance, "
              f"p={b1['p_value']:.4g}")
    print(f"grounding  : {fa['all']['grounding_rate']} "
          f"(fabricated {fa['all']['fabricated_citation_rate']})")

    out = output_dir() / f"narrative_results_{args.label}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
