"""Stage 2 -- retrieve and cache a boundary-enforced news window for every date.

Covers both real transitions and era-matched placebo dates, using the identical
code path, so the placebo comparison cannot be an artefact of different
retrieval behaviour.

Also measures the hindsight leak on a sample of pages: how much of each page was
written after the boundary. That number is the justification for revision
pinning and it belongs in the report.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from regime_narrative.config import load_settings, output_dir
from regime_narrative.news.base import split_holdout
from regime_narrative.news.wikipedia import fetch_window, measure_hindsight_leak


def main() -> None:
    cfg = load_settings()
    window_days = cfg["news"]["window_days"]
    max_items = cfg["news"]["max_items_per_window"]

    regimes = json.loads((output_dir() / "regimes.json").read_text(encoding="utf-8"))
    transitions = [r["date"] for r in regimes["seed_stability"]["transitions"]]
    placebos = [p["date"] for p in regimes["placebo"]["dates"]]

    targets = [("transition", d) for d in transitions] + [("placebo", d) for d in placebos]
    print(f"fetching {len(targets)} windows of {window_days} days "
          f"({len(transitions)} transitions, {len(placebos)} placebos)")

    manifest: dict = {"window_days": window_days, "windows": {}}
    failures = []

    for i, (kind, dstr) in enumerate(targets, 1):
        d = date.fromisoformat(dstr)
        try:
            w = fetch_window(d, window_days=window_days, max_items=max_items)
            gen, hold = split_holdout(
                w,
                fraction=cfg["news"]["holdout_fraction"],
                seed=cfg["news"]["holdout_seed"],
            )
            wid = f"{kind}_{dstr}"
            manifest["windows"][wid] = {
                "kind": kind,
                "date": dstr,
                **w.manifest(),
                "n_generation": len(gen),
                "n_holdout": len(hold),
            }
            print(f"  [{i:>2}/{len(targets)}] {wid:<32} items={len(w):>4} "
                  f"gen={len(gen):>3} hold={len(hold):>3} "
                  f"pages_ok={w.provenance['n_pages_ok']}/{window_days}")
        except Exception as exc:
            failures.append({"kind": kind, "date": dstr, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{i:>2}/{len(targets)}] {kind}_{dstr} FAILED: {exc}")

    counts = [m["n_items"] for m in manifest["windows"].values()]
    t_counts = [m["n_items"] for m in manifest["windows"].values() if m["kind"] == "transition"]
    p_counts = [m["n_items"] for m in manifest["windows"].values() if m["kind"] == "placebo"]
    manifest["summary"] = {
        "n_windows": len(manifest["windows"]),
        "n_failures": len(failures),
        "failures": failures,
        "total_items": sum(counts),
        "mean_items_transition": round(sum(t_counts) / len(t_counts), 1) if t_counts else None,
        "mean_items_placebo": round(sum(p_counts) / len(p_counts), 1) if p_counts else None,
        "min_items": min(counts) if counts else None,
        "max_items": max(counts) if counts else None,
        "any_post_boundary_items": any(
            m["dropped_post_boundary"] > 0 for m in manifest["windows"].values()
        ),
    }

    print("\nmeasuring hindsight leak on a sample of pages")
    leak = []
    for dstr in transitions[:8]:
        d = date.fromisoformat(dstr)
        try:
            leak.append(measure_hindsight_leak(d, d))
        except Exception as exc:
            leak.append({"page": dstr, "status": f"error: {exc}"})
    ok = [x for x in leak if x.get("status") == "ok"]
    manifest["hindsight_leak"] = {
        "samples": leak,
        "mean_pct_written_after_boundary": (
            round(sum(x["pct_written_after_boundary"] for x in ok) / len(ok), 1) if ok else None
        ),
        "max_pct_written_after_boundary": (
            max(x["pct_written_after_boundary"] for x in ok) if ok else None
        ),
    }

    path = output_dir() / "news_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nmean items/window: transitions={manifest['summary']['mean_items_transition']} "
          f"placebos={manifest['summary']['mean_items_placebo']}")
    print(f"hindsight leak: mean "
          f"{manifest['hindsight_leak']['mean_pct_written_after_boundary']}% of page "
          f"written after the boundary")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
