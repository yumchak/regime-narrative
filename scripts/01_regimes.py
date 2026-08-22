"""Stage 1 -- regime detection, generalisation, transitions.

Produces outputs/regimes.json, which every later stage reads. Runs offline once
prices are cached.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import pandas as pd

from regime_narrative.config import load_settings, output_dir
from regime_narrative.controls.placebo import placebo_balance_report, sample_placebo_dates
from regime_narrative.data import load_universe, load_unseen_index
from regime_narrative.features import build_features, daily_returns, forward_realised_vol
from regime_narrative.hmm_model import fit_full_sample, walk_forward_states
from regime_narrative.regime_stats import (
    forward_vol_separation,
    per_fold_ratios,
    vol_separation,
)
from regime_narrative.transitions import (
    dwell_distribution,
    dwell_sensitivity,
    extract_transitions,
    stable_transitions,
    transitions_to_frame,
)


def main() -> None:
    cfg = load_settings()
    results: dict = {"specification": {
        "min_dwell_days": cfg["transitions"]["min_dwell_days"],
        "news_window_days": cfg["news"]["window_days"],
        "placebo_offset_days": [
            cfg["placebo"]["offset_days_min"], cfg["placebo"]["offset_days_max"]
        ],
        "train_days": cfg["walk_forward"]["train_days"],
        "test_days": cfg["walk_forward"]["test_days"],
        "primary_seed": cfg["hmm"]["primary_seed"],
    }}

    print("[1/5] loading prices and building features")
    px = load_universe()
    feat = build_features(px["spy"], px["vix"], px["hyg"])
    rets = daily_returns(px["spy"])
    fwd = forward_realised_vol(px["spy"], 20)
    results["data"] = {
        "n_days": int(len(feat)),
        "start": str(feat.index[0].date()),
        "end": str(feat.index[-1].date()),
    }

    print("[2/5] in-sample fit (reproducing the original notebook)")
    ins, _ = fit_full_sample(feat)
    v_ins = vol_separation(rets, ins, label="in-sample (original procedure)")
    results["in_sample"] = v_ins.as_dict()
    print(f"      in-sample ratio = {v_ins.ratio:.3f}")

    print("[3/5] walk-forward, causal (filtered) states")
    wf = walk_forward_states(feat)
    v_oos = vol_separation(rets, wf.states, label="out-of-sample (pooled test days)")
    fv = forward_vol_separation(fwd, wf.states)
    pf = per_fold_ratios(rets, wf.states, wf.fold_id)

    results["out_of_sample"] = v_ins and v_oos.as_dict()
    results["out_of_sample_forward"] = fv
    results["walk_forward"] = {
        "n_folds": wf.n_folds,
        "n_oos_days": int(len(wf.states)),
        "failures": wf.failures,
        "folds_with_both_states": int(pf["ratio"].notna().sum()),
        "per_fold_ratio_median": float(pf["ratio"].median()),
        "per_fold_ratio_mean": float(pf["ratio"].mean()),
        "per_fold_ratio_above_1": int((pf["ratio"] > 1).sum()),
        "per_fold": pf.to_dict("records"),
    }
    print(f"      OOS pooled ratio  = {v_oos.ratio:.3f} "
          f"[{v_oos.ratio_ci_low:.2f}, {v_oos.ratio_ci_high:.2f}]")
    print(f"      OOS forward-20d   = {fv['ratio']:.3f}")
    print(f"      per-fold median   = {pf['ratio'].median():.3f}")

    print("[4/5] generalisation to unseen indices")
    gen_rows = []
    for name in cfg["unseen_indices"]:
        try:
            idx_px = load_unseen_index(name)
            series = idx_px[name]
            f2 = build_features(series, px["vix"], px["hyg"])
            wf2 = walk_forward_states(f2)
            r2 = daily_returns(series)
            v2 = vol_separation(r2, wf2.states, label=name)
            fwd2 = forward_realised_vol(series, 20)
            fv2 = forward_vol_separation(fwd2, wf2.states)
            pf2 = per_fold_ratios(r2, wf2.states, wf2.fold_id)
            gen_rows.append({
                "index": name,
                "n_days": int(len(f2)),
                "n_folds": wf2.n_folds,
                "vol_calm": v2.vol_calm,
                "vol_stressed": v2.vol_stressed,
                "ratio_oos": v2.ratio,
                "ratio_ci": [v2.ratio_ci_low, v2.ratio_ci_high],
                "ratio_forward": fv2["ratio"],
                "per_fold_median": float(pf2["ratio"].median()),
                "levene_p": v2.levene_p,
                "n_calm": v2.n_calm,
                "n_stressed": v2.n_stressed,
            })
            print(f"      {name:10s} ratio={v2.ratio:.3f} fwd={fv2['ratio']:.3f} "
                  f"folds={wf2.n_folds}")
        except Exception as exc:
            gen_rows.append({"index": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"      {name:10s} FAILED: {exc}")
    results["generalisation"] = gen_rows

    print("[5/5] transitions, dwell sensitivity, seed stability, placebos")
    sens = dwell_sensitivity(
        wf.states, cfg["transitions"]["dwell_sensitivity"] + [1, 15, 45, 60],
        fold_id=wf.fold_id,
    ).sort_values("min_dwell_days")
    results["dwell_sensitivity"] = sens.to_dict("records")

    primary = extract_transitions(
        wf.states, fold_id=wf.fold_id, stressed_prob=wf.stressed_prob
    )
    results["transitions_primary_seed"] = transitions_to_frame(primary).to_dict("records")

    by_seed = {}
    for seed in cfg["hmm"]["stability_seeds"]:
        try:
            w = walk_forward_states(feat, seed=seed)
            by_seed[seed] = extract_transitions(
                w.states, fold_id=w.fold_id, stressed_prob=w.stressed_prob
            )
        except Exception as exc:
            print(f"      seed {seed} failed: {exc}")
    stable = stable_transitions(by_seed)
    results["seed_stability"] = {
        "n_seeds": len(by_seed),
        "counts_per_seed": {str(k): len(v) for k, v in by_seed.items()},
        "n_stable_transitions": len(stable),
        "n_primary_seed_transitions": len(primary),
        "transitions": transitions_to_frame(stable).to_dict("records"),
    }
    print(f"      primary seed: {len(primary)} transitions")
    print(f"      counts across {len(by_seed)} seeds: "
          f"{sorted(len(v) for v in by_seed.values())}")
    print(f"      stable across seeds: {len(stable)}")

    placebos = sample_placebo_dates(stable or primary, wf.states)
    results["placebo"] = {
        "dates": [p.as_dict() for p in placebos],
        "balance": placebo_balance_report(placebos, stable or primary),
    }
    print(f"      placebo dates: {len(placebos)}")

    dd = dwell_distribution(wf.states)
    results["dwell_distribution"] = dd.assign(
        start=dd["start"].astype(str), end=dd["end"].astype(str)
    ).to_dict("records")

    # Persist the state series for the chart and later stages.
    #
    # Both the raw and the persistence-filtered sequences are saved. The chart
    # must shade the *filtered* series, because that is what the transitions are
    # read off; shading the raw series would show regime bands with no
    # corresponding transition marker, which is exactly the noise the filter
    # exists to remove. The raw filtered probability is kept for the lower panel,
    # where showing that noise is the point.
    from regime_narrative.transitions import apply_persistence_filter

    states_out = pd.DataFrame({
        "state_raw": wf.states,
        "state": apply_persistence_filter(
            wf.states, cfg["transitions"]["min_dwell_days"]
        ),
        "stressed_prob": wf.stressed_prob,
        "fold": wf.fold_id,
        "spy": px["spy"].reindex(wf.states.index),
    })
    states_out.to_csv(output_dir() / "oos_states.csv")
    n_flips_raw = int((wf.states.diff().abs() > 0).sum())
    n_flips_filtered = int((states_out["state"].diff().abs() > 0).sum())
    results["persistence_filter"] = {
        "min_dwell_days": cfg["transitions"]["min_dwell_days"],
        "raw_state_flips": n_flips_raw,
        "flips_after_filter": n_flips_filtered,
        "flips_removed_as_noise": n_flips_raw - n_flips_filtered,
    }
    print(f"      raw flips {n_flips_raw} -> {n_flips_filtered} after persistence filter")

    path = output_dir() / "regimes.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
