"""Stage 4 -- generate one explanation per window, then run the controls.

This is the only stage that needs ANTHROPIC_API_KEY. Everything upstream runs
offline from cache, so the statistical spine and the retrieval controls can be
rebuilt and audited without spending anything.

Order matters here. Narratives are generated for transitions and placebo dates
through the identical code path and the identical prompt; nothing about the call
differs except the news. Then:

    placebo comparison   -- confident-explanation rate, transitions vs controls
    blind matching       -- explanations vs held-out articles, against 1/N
    faithfulness         -- does each claim's citation actually support it

Run with --dry-run to render the exact prompts to disk and print token estimates
without making a single API call.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from scipy import stats as sps

from regime_narrative.config import cache_dir, has_env, load_settings, output_dir
from regime_narrative.controls.blind_match import blind_match, strip_identifying_details
from regime_narrative.controls.faithfulness import (
    aggregate,
    audit_narrative,
    threshold_sensitivity,
)
from regime_narrative.narrative import generate_narrative, load_prompt, render_window
from regime_narrative.news.base import split_holdout
from regime_narrative.news.wikipedia import fetch_window


def _load_all_windows(cfg) -> dict:
    regimes = json.loads((output_dir() / "regimes.json").read_text(encoding="utf-8"))
    t_dates = [t["date"] for t in regimes["seed_stability"]["transitions"]]
    placebos = regimes["placebo"]["dates"]

    windows = {}
    for dstr in t_dates:
        windows[f"transition_{dstr}"] = ("transition", dstr, "n/a")
    for p in placebos:
        windows[f"placebo_{p['date']}"] = ("placebo", p["date"], p["stratum"])

    out = {}
    for wid, (kind, dstr, stratum) in windows.items():
        w = fetch_window(
            date.fromisoformat(dstr),
            window_days=cfg["news"]["window_days"],
            max_items=cfg["news"]["max_items_per_window"],
        )
        gen, hold = split_holdout(
            w, fraction=cfg["news"]["holdout_fraction"], seed=cfg["news"]["holdout_seed"]
        )
        out[wid] = {"kind": kind, "date": dstr, "stratum": stratum,
                    "full": w, "gen": gen, "hold": hold}
    return out


def dry_run(cfg, windows: dict) -> None:
    system_prompt, prompt_hash = load_prompt()
    outdir = output_dir("dry_run")
    total_chars = 0
    for wid, w in windows.items():
        text, _ = render_window(w["gen"])
        total_chars += len(text)
        (outdir / f"{wid}.txt").write_text(
            f"=== SYSTEM (prompt {cfg['llm']['prompt_version']} "
            f"sha {prompt_hash}) ===\n{system_prompt}\n\n=== USER ===\n{text}",
            encoding="utf-8",
        )
    est_in = total_chars / 3.7
    print(f"rendered {len(windows)} prompts to {outdir}")
    print(f"total input ~{total_chars:,} chars ~= {est_in:,.0f} tokens")
    print(f"model: {cfg['llm']['model']}  effort: {cfg['llm']['effort']}")
    print("no API calls made")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="render prompts and estimate cost without calling the API")
    ap.add_argument("--limit", type=int, default=None, help="only the first N windows")
    args = ap.parse_args()

    cfg = load_settings()
    print("loading cached news windows")
    windows = _load_all_windows(cfg)
    if args.limit:
        windows = dict(list(windows.items())[: args.limit])
    print(f"  {len(windows)} windows "
          f"({sum(1 for w in windows.values() if w['kind'] == 'transition')} transitions, "
          f"{sum(1 for w in windows.values() if w['kind'] == 'placebo')} placebos)")

    if args.dry_run:
        dry_run(cfg, windows)
        return

    if not has_env("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY is not set.")
        print("Everything except this stage runs without it. To see exactly what")
        print("would be sent, and what it would cost:")
        print("    python scripts/04_narratives.py --dry-run")
        sys.exit(2)

    print(f"generating with {cfg['llm']['model']} at effort "
          f"{cfg['llm']['effort']}, schema-enforced output, one call per window")
    narratives = {}
    for i, (wid, w) in enumerate(windows.items(), 1):
        n = generate_narrative(w["gen"], window_id=wid, kind=w["kind"])
        narratives[wid] = n
        flag = "" if n.parse_ok else "  [PARSE FAILED]"
        print(f"  [{i:>2}/{len(windows)}] {wid:<34} "
              f"driver={str(n.driver_identified):<5} conf={n.confidence:<7} "
              f"claims={len(n.claims)}{flag}")

    results = _evaluate(cfg, windows, narratives)
    path = output_dir() / "narrative_results.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")


def _evaluate(cfg, windows: dict, narratives: dict) -> dict:
    results: dict = {"model": cfg["llm"]["model"],
                     "prompt_version": cfg["llm"]["prompt_version"]}

    # --- Control 3: placebo -------------------------------------------------
    def _rate(keys):
        if not keys:
            return {"n": 0}
        conf = sum(1 for k in keys if narratives[k].is_confident)
        ident = sum(1 for k in keys if narratives[k].driver_identified)
        return {"n": len(keys), "n_confident": conf,
                "confident_rate": round(conf / len(keys), 3),
                "n_driver_identified": ident,
                "driver_rate": round(ident / len(keys), 3)}

    t_keys = [k for k, w in windows.items() if w["kind"] == "transition"]
    p_keys = [k for k, w in windows.items() if w["kind"] == "placebo"]
    clean_keys = [k for k in p_keys if windows[k]["stratum"] == "clean"]

    t_rate, p_rate, c_rate = _rate(t_keys), _rate(p_keys), _rate(clean_keys)
    table = [
        [t_rate.get("n_confident", 0), t_rate["n"] - t_rate.get("n_confident", 0)],
        [p_rate.get("n_confident", 0), p_rate["n"] - p_rate.get("n_confident", 0)],
    ]
    _, fisher_p = sps.fisher_exact(table) if all(sum(r) for r in table) else (None, None)
    table_clean = [
        [t_rate.get("n_confident", 0), t_rate["n"] - t_rate.get("n_confident", 0)],
        [c_rate.get("n_confident", 0), c_rate["n"] - c_rate.get("n_confident", 0)],
    ]
    _, fisher_clean = (
        sps.fisher_exact(table_clean) if all(sum(r) for r in table_clean) else (None, None)
    )

    results["placebo"] = {
        "transitions": t_rate,
        "placebos_all": p_rate,
        "placebos_clean_stratum": c_rate,
        "fisher_p_all": fisher_p,
        "fisher_p_clean": fisher_clean,
        "note": (
            "Identical prompt and identical pipeline for both arms; only the "
            "news differs. The clean stratum is the pre-declared primary "
            "comparison."
        ),
    }
    print(f"\nplacebo: transitions {t_rate.get('confident_rate')} "
          f"vs all controls {p_rate.get('confident_rate')} "
          f"vs clean controls {c_rate.get('confident_rate')} "
          f"(Fisher p={fisher_p})")

    # --- Blind matching -----------------------------------------------------
    BIZ = {"business and economy", "economics and business", "business"}

    def _match_arm(label, keys, item_filter):
        expl, hold = {}, {}
        for k in keys:
            h = " ".join(
                i.body for i in windows[k]["hold"].items if item_filter(i)
            )
            e = narratives[k].summary + " " + narratives[k].mechanism + " " + " ".join(
                c.get("claim", "") for c in narratives[k].claims if isinstance(c, dict)
            )
            e = strip_identifying_details(e)
            if h.strip() and e.strip():
                hold[k], expl[k] = h, e
        if len(hold) < 2:
            return {"error": f"only {len(hold)} usable windows"}
        r = blind_match(expl, hold, n_permutations=10000)
        print(f"  blind match [{label}]: {r.summary()}")
        return {
            "n_candidates": r.n_candidates, "n_scored": r.n_scored,
            "n_correct": r.n_correct, "accuracy": round(r.accuracy, 4),
            "chance": round(r.chance, 4), "top3": round(r.top3_accuracy, 4),
            "mrr": round(r.mean_reciprocal_rank, 4), "p_value": r.p_value,
            "per_item": r.per_item,
        }

    print("\nblind matching")
    results["blind_match"] = {
        "transitions_all_sections": _match_arm(
            "transitions, all sections", t_keys, lambda i: True),
        "transitions_business_only": _match_arm(
            "transitions, business only", t_keys,
            lambda i: str(i.extra.get("section", "")).lower() in BIZ),
        "transitions_non_business": _match_arm(
            "transitions, non-business", t_keys,
            lambda i: str(i.extra.get("section", "")).lower() not in BIZ),
        "all_windows": _match_arm(
            "all windows", list(windows), lambda i: True),
        "caveat": (
            "The all-sections arm is inflated by generic world news: in the "
            "ceiling test, non-business sections matched at 90% while "
            "business-only matched at 16.7% against 8.3% chance (p=0.27). The "
            "business arm also has far less text (median 183 words vs 2772), so "
            "that gap is partly power and not only content. Both arms are "
            "reported; neither alone is the answer."
        ),
    }

    # --- Control 2: faithfulness -------------------------------------------
    print("\nfaithfulness")
    reports = [audit_narrative(narratives[k], windows[k]["gen"]) for k in windows]
    results["faithfulness"] = aggregate(reports)
    results["faithfulness"]["threshold_sensitivity"] = threshold_sensitivity(
        [(narratives[k], windows[k]["gen"]) for k in windows]
    )
    results["faithfulness"]["per_window"] = [r.as_dict() for r in reports]
    print(f"  grounding rate: {results['faithfulness']['all'].get('grounding_rate')}")
    print(f"  fabricated citations: "
          f"{results['faithfulness']['all'].get('fabricated_citation_rate')}")

    results["narratives"] = {k: n.as_dict() for k, n in narratives.items()}
    return results


if __name__ == "__main__":
    main()
