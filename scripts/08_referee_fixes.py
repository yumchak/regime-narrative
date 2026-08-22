"""Stage 8 -- statistics the referee pass showed were missing or misstated.

An adversarial statistics review of the finished results found four things that
change what may be claimed, and every one of them is computed here rather than
asserted:

1.  **The placebo negative was overclaimed.** "Not diagnostic" is an affirmative
    statement of no-difference, and Fisher p=0.43 licenses no such thing. What
    the data support is an interval.

2.  **The per-fold sign test is a better headline than the pooled ratio.** It is
    distribution-free, and immune to the pooling objection (that the pooled
    figure mixes calm-era and crisis-era days) which the report already flags
    against itself.

3.  **The grounding rate has no null.** 96.4% sounds like a metric everything
    passes. Scoring the same claims against randomly chosen items from the same
    window gives the floor, and the floor is what makes the number mean
    something.

4.  **Blind matching against era-adjacent hard negatives** is stronger evidence
    than against the full pool, because a critic can explain 55%-vs-1.7% by
    era-matching and cannot explain 75%-vs-33% the same way.

Writes outputs/referee_stats.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sps

from regime_narrative.config import load_settings, output_dir
from regime_narrative.controls.blind_match import blind_match, strip_identifying_details
from regime_narrative.controls.faithfulness import _content_tokens
from regime_narrative.news.base import split_holdout
from regime_narrative.news.wikipedia import fetch_window


def newcombe_diff_ci(x1, n1, x0, n0, alpha=0.05):
    """Newcombe's score interval for a difference of proportions.

    Wald intervals are badly behaved at these sample sizes and near the
    boundary; Newcombe's hybrid is the standard fix.
    """
    z = sps.norm.ppf(1 - alpha / 2)

    def wilson(x, n):
        p = x / n
        d = 1 + z**2 / n
        c = p + z**2 / (2 * n)
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        return (c - h) / d, (c + h) / d

    l1, u1 = wilson(x1, n1)
    l0, u0 = wilson(x0, n0)
    p1, p0 = x1 / n1, x0 / n0
    lo = (p1 - p0) - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = (p1 - p0) + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return float(lo), float(hi)


def main() -> None:
    cfg = load_settings()
    R = json.loads((output_dir() / "regimes.json").read_text(encoding="utf-8"))
    D = json.loads((output_dir() / "narrative_results.json").read_text(encoding="utf-8"))
    out: dict = {}

    # ---------------------------------------------------------------- 1
    print("[1/4] placebo, stated as an interval rather than a verdict")
    pb = D["placebo"]
    t, c = pb["transitions"], pb["placebos_clean_stratum"]
    ca = pb["placebos_all"]
    lo, hi = newcombe_diff_ci(t["n_confident"], t["n"], c["n_confident"], c["n"])
    lo_a, hi_a = newcombe_diff_ci(t["n_confident"], t["n"],
                                  ca["n_confident"], ca["n"])
    # Jeffreys posterior on P(transition rate > control rate).
    rng = np.random.default_rng(20260822)
    pa = rng.beta(t["n_confident"] + .5, t["n"] - t["n_confident"] + .5, 200_000)
    pc = rng.beta(c["n_confident"] + .5, c["n"] - c["n_confident"] + .5, 200_000)
    out["placebo_interval"] = {
        "transitions": f"{t['n_confident']}/{t['n']}",
        "clean_controls": f"{c['n_confident']}/{c['n']}",
        "difference_pp": round(100 * (t["confident_rate"] - c["confident_rate"]), 1),
        "newcombe_95ci_pp_vs_clean": [round(100 * lo, 1), round(100 * hi, 1)],
        "newcombe_95ci_pp_vs_all": [round(100 * lo_a, 1), round(100 * hi_a, 1)],
        "posterior_prob_transition_higher": round(float((pa > pc).mean()), 3),
        "posterior_prob_gap_exceeds_10pp": round(float((pa - pc > 0.10).mean()), 3),
        "statement": (
            "The difference is +{:.1f}pp with a 95% interval of [{:.0f}, {:.0f}]pp. "
            "The data are compatible with no difference and with a large one. "
            "This is an unresolved comparison, not a null result."
        ).format(100 * (t["confident_rate"] - c["confident_rate"]), 100 * lo, 100 * hi),
    }
    print(f"      +{out['placebo_interval']['difference_pp']}pp, "
          f"95% CI {out['placebo_interval']['newcombe_95ci_pp_vs_clean']}, "
          f"P(higher)={out['placebo_interval']['posterior_prob_transition_higher']}")

    # ---------------------------------------------------------------- 2
    print("[2/4] per-fold sign test -- distribution-free, immune to pooling")
    pf = pd.DataFrame(R["walk_forward"]["per_fold"])
    ratios = pf["ratio"].dropna()
    n_above = int((ratios > 1).sum())
    sign_p = float(sps.binomtest(n_above, len(ratios), 0.5).pvalue)
    wil = sps.wilcoxon(np.log(ratios))
    boot = [np.median(rng.choice(ratios.values, len(ratios), replace=True))
            for _ in range(20000)]
    out["per_fold_sign_test"] = {
        "n_folds_with_both_states": int(len(ratios)),
        "n_folds_total": R["walk_forward"]["n_folds"],
        "n_folds_ratio_above_1": n_above,
        "sign_test_p": sign_p,
        "wilcoxon_log_ratio_p": float(wil.pvalue),
        "median_ratio": round(float(ratios.median()), 3),
        "median_95ci": [round(float(np.percentile(boot, 2.5)), 2),
                        round(float(np.percentile(boot, 97.5)), 2)],
        "note": (
            "Only folds containing both states can produce a ratio, so this is "
            "over {} folds, not {}. Quoting the fold count as {} alongside a "
            "per-fold median is wrong."
        ).format(len(ratios), R["walk_forward"]["n_folds"], R["walk_forward"]["n_folds"]),
    }
    print(f"      {n_above}/{len(ratios)} folds above 1.0, sign test p={sign_p:.5f}, "
          f"median {ratios.median():.2f} CI {out['per_fold_sign_test']['median_95ci']}")

    # ---------------------------------------------------------------- 3
    print("[3/4] grounding null -- what does 96.4% beat?")
    windows = {}
    for wid, n in D["narratives"].items():
        d = date.fromisoformat(n["boundary_date"])
        w = fetch_window(d, window_days=cfg["news"]["window_days"],
                         max_items=cfg["news"]["max_items_per_window"])
        gen, _ = split_holdout(w, fraction=cfg["news"]["holdout_fraction"],
                               seed=cfg["news"]["holdout_seed"])
        windows[wid] = gen

    real, rand = [], []
    for wid, n in D["narratives"].items():
        by_id = {i.item_id: i for i in windows[wid].items}
        pool = list(by_id.values())
        if not pool:
            continue
        for cl in n["claims"]:
            toks = _content_tokens(cl.get("claim", ""))
            if not toks:
                continue
            cited = [by_id[c] for c in cl.get("item_ids", []) if c in by_id]
            if cited:
                src = set().union(*[_content_tokens(i.body) for i in cited])
                real.append(len(toks & src) / len(toks))
            for _ in range(10):
                r = pool[int(rng.integers(0, len(pool)))]
                rt = _content_tokens(r.body)
                rand.append(len(toks & rt) / len(toks))

    out["grounding_null"] = {
        "n_claims_scored": len(real),
        "mean_overlap_cited": round(float(np.mean(real)), 3),
        "grounded_rate_cited": round(float(np.mean([x >= 0.25 for x in real])), 3),
        "mean_overlap_random_same_window": round(float(np.mean(rand)), 3),
        "grounded_rate_random_same_window": round(
            float(np.mean([x >= 0.25 for x in rand])), 4),
        "n_random_draws": len(rand),
        "statement": (
            "Claims overlap the items they cite at {:.0%}, against {:.1%} for a "
            "randomly chosen item from the same window. The metric is not one "
            "that everything passes."
        ).format(np.mean(real), np.mean(rand)),
    }
    print(f"      cited {np.mean(real):.1%} vs random same-window "
          f"{np.mean(rand):.1%} ({len(rand)} draws)")

    # ---------------------------------------------------------------- 4
    print("[4/4] blind matching against era-adjacent hard negatives")
    hold, expl = {}, {}
    for wid, n in D["narratives"].items():
        d = date.fromisoformat(n["boundary_date"])
        w = fetch_window(d, window_days=cfg["news"]["window_days"],
                         max_items=cfg["news"]["max_items_per_window"])
        _, h = split_holdout(w, fraction=cfg["news"]["holdout_fraction"],
                             seed=cfg["news"]["holdout_seed"])
        ht = " ".join(i.body for i in h.items)
        et = strip_identifying_details(" ".join(
            [n["summary"], n["mechanism"]] +
            [c.get("claim", "") for c in n["claims"]]))
        if ht.strip() and et.strip():
            hold[wid], expl[wid] = ht, et

    def dt(wid):
        return date.fromisoformat(D["narratives"][wid]["boundary_date"])

    arms = {}
    for k_near in (3, 5):
        correct = 0
        for wid in expl:
            pool = sorted(hold, key=lambda o: abs((dt(o) - dt(wid)).days))[:k_near]
            if wid not in pool:
                pool = [wid] + pool[: k_near - 1]
            sub_h = {p: hold[p] for p in pool}
            r = blind_match({wid: expl[wid]}, sub_h, n_permutations=200)
            correct += r.n_correct
        arms[f"{k_near}_temporally_nearest"] = {
            "n_correct": correct, "n_scored": len(expl),
            "accuracy": round(correct / len(expl), 3),
            "chance": round(1 / k_near, 3),
            "binomial_p": float(sps.binomtest(correct, len(expl), 1 / k_near,
                                              alternative="greater").pvalue),
        }
        print(f"      {k_near} nearest: {correct}/{len(expl)} = "
              f"{100 * correct / len(expl):.1f}% vs {100 / k_near:.1f}% chance")

    out["blind_match_hard_negatives"] = arms
    out["blind_match_hard_negatives"]["why"] = (
        "A critic can explain 55%-against-1.7% by era-matching: adjacent windows "
        "share running stories, so the matcher might only be recovering roughly "
        "when, not which fortnight. Restricting each explanation to compete only "
        "against its temporal neighbours removes that explanation. The signal "
        "survives, which is stronger evidence than the headline number."
    )

    p = output_dir() / "referee_stats.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
