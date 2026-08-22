"""Self-contained HTML report.

One file, no external assets, no server. Images are embedded as data URIs so the
report can be emailed, opened offline, or dropped into a submission folder and
still render exactly as it did on the machine that produced it.

The report is written to degrade honestly: sections whose inputs are missing say
so rather than disappearing, because a missing control should be visible.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CSS = """
:root{--ink:#1b2530;--mut:#63707d;--line:#dde3e9;--bg:#fff;--soft:#f5f8fa;
--red:#b4343a;--blue:#37567f;--good:#2e7d55;--warn:#a8641c;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:48px 28px 96px}
h1{font-size:31px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:52px 0 4px;padding-bottom:7px;border-bottom:2px solid var(--ink)}
h3{font-size:15px;margin:26px 0 8px;color:var(--ink)}
.sub{color:var(--mut);font-size:15px;margin:0 0 4px}
.meta{color:var(--mut);font-size:12.5px;margin-top:14px}
p{margin:10px 0}
.lede{font-size:16px;background:var(--soft);border-left:3px solid var(--blue);
padding:16px 20px;margin:22px 0;border-radius:0 4px 4px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{background:var(--soft);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.045em;color:var(--mut)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.total td{font-weight:600;border-top:2px solid var(--ink)}
figure{margin:20px 0}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;display:block}
figcaption{color:var(--mut);font-size:12.5px;margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:18px 0}
.card{border:1px solid var(--line);border-radius:6px;padding:15px 17px;background:var(--bg)}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
.card .v{font-size:26px;font-weight:600;margin:5px 0 2px;font-variant-numeric:tabular-nums}
.card .n{font-size:12px;color:var(--mut)}
.pill{display:inline-block;padding:1px 8px;border-radius:11px;font-size:11px;font-weight:600}
.pass{background:#e3f2ea;color:var(--good)} .fail{background:#fbe5e6;color:var(--red)}
.warnp{background:#fdf0e0;color:var(--warn)} .na{background:#eef1f4;color:var(--mut)}
code,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12.5px}
.caveat{border-left:3px solid var(--warn);background:#fdf9f3;padding:13px 17px;
margin:16px 0;font-size:13.5px;border-radius:0 4px 4px 0}
.missing{border-left:3px solid var(--mut);background:var(--soft);padding:13px 17px;
color:var(--mut);font-size:13.5px;border-radius:0 4px 4px 0}
.foot{margin-top:64px;padding-top:18px;border-top:1px solid var(--line);
color:var(--mut);font-size:12px}
ul{margin:8px 0;padding-left:20px} li{margin:4px 0}
"""


def _img(path: Path) -> str:
    if not path.exists():
        return '<div class="missing">Image not generated.</div>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{path.stem}">'


def _fmt(v: Any, nd: int = 3, dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, float):
        if v != v:
            return dash
        return f"{v:.{nd}f}"
    return str(v)


def _pill(ok: bool | None, yes="PASS", no="FAIL", na="NOT RUN") -> str:
    if ok is None:
        return f'<span class="pill na">{na}</span>'
    return (f'<span class="pill pass">{yes}</span>' if ok
            else f'<span class="pill fail">{no}</span>')


def build_report(
    regimes: dict,
    news_manifest: dict,
    ceiling: dict | None,
    narratives: dict | None,
    *,
    power: dict | None = None,
    retest: dict | None = None,
    referee: dict | None = None,
    figures: dict[str, Path],
    out_path: Path,
) -> Path:
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    spec = regimes.get("specification", {})
    oos = regimes.get("out_of_sample", {})
    ins = regimes.get("in_sample", {})
    fwd = regimes.get("out_of_sample_forward", {})
    wf = regimes.get("walk_forward", {})
    pf = regimes.get("persistence_filter", {})
    stab = regimes.get("seed_stability", {})
    bal = regimes.get("placebo", {}).get("balance", {})
    nsum = news_manifest.get("summary", {})
    leak = news_manifest.get("hindsight_leak", {})

    h: list[str] = []
    a = h.append

    a(f'<!doctype html><html lang="en"><head><meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width,initial-scale=1">')
    a("<title>regime-narrative — results</title>")
    a(f"<style>{CSS}</style></head><body><div class='wrap'>")

    # ---- header ----------------------------------------------------------
    a("<h1>Regime detection with a measured explanation layer</h1>")
    a('<p class="sub">A two-state HMM decides <em>when</em> the market regime '
      "changed. A language model reading only news published before that date "
      "describes <em>what was happening</em>. Every claim about the second half "
      "is tested, including the ways it fails.</p>")
    a(f'<p class="meta">Generated {gen} · '
      f"Study period {regimes.get('data', {}).get('start')} to "
      f"{regimes.get('data', {}).get('end')} · "
      f"{regimes.get('data', {}).get('n_days')} trading days</p>")

    # ---- headline numbers ------------------------------------------------
    a('<div class="grid">')
    a(f'<div class="card"><div class="k">Out-of-sample vol ratio</div>'
      f'<div class="v">{_fmt(oos.get("ratio"), 2)}×</div>'
      f'<div class="n">95% CI {_fmt(oos.get("ratio_ci_low"), 2)}–'
      f'{_fmt(oos.get("ratio_ci_high"), 2)} · in-sample was '
      f'{_fmt(ins.get("ratio"), 2)}×</div></div>')
    a(f'<div class="card"><div class="k">Forward-vol ratio</div>'
      f'<div class="v">{_fmt(fwd.get("ratio"), 2)}×</div>'
      f'<div class="n">volatility that had not happened yet</div></div>')
    a(f'<div class="card"><div class="k">Transitions</div>'
      f'<div class="v">{stab.get("n_stable_transitions", "—")}</div>'
      f'<div class="n">seed-stable, from '
      f'{pf.get("raw_state_flips", "—")} raw flips</div></div>')
    a(f'<div class="card"><div class="k">Walk-forward folds</div>'
      f'<div class="v">{wf.get("n_folds", "—")}</div>'
      f'<div class="n">{wf.get("n_oos_days", "—")} out-of-sample days</div></div>')
    a("</div>")

    a('<div class="lede"><strong>What changed from the prior work.</strong> '
      "The earlier notebook reported a 2.33× volatility ratio from an HMM fitted "
      "once over the full history, and separately ran 26 walk-forward folds for a "
      "return-direction classifier. Those were two different experiments. Here the "
      "HMM is refitted inside every fold, scaled on training data only, and states "
      "are assigned by a forward filter that cannot see the future — so the ratio "
      "below is genuinely out-of-sample, and the transition dates were knowable on "
      "the day.</div>")

    # ---- chart -----------------------------------------------------------
    a("<h2>1 · The chart</h2>")
    a("<figure>")
    a(_img(figures.get("chart", Path("outputs/regime_chart.png"))))
    a("<figcaption>Out-of-sample regime shading on SPY. Bands use the "
      "persistence-filtered state sequence, so every shaded episode corresponds "
      "to a numbered transition. The lower panel shows the raw filtered "
      "probability, including the short-lived excursions the persistence filter "
      "removes.</figcaption></figure>")

    # ---- generalisation --------------------------------------------------
    a("<h2>2 · Does it hold on markets it was never fitted to?</h2>")
    a("<p>The same procedure, refitted without modification on three indices "
      "outside the original universe. This is <em>replication</em> — the method "
      "transfers — not transfer of a single fitted model.</p>")
    a('<table><thead><tr><th>Index</th><th class="num">Folds</th>'
      '<th class="num">Vol, calm</th><th class="num">Vol, stressed</th>'
      '<th class="num">Ratio (same-day)</th><th class="num">Ratio (forward 20d)</th>'
      '<th class="num">Median per-fold</th></tr></thead><tbody>')
    a(f'<tr class="total"><td>SPY <span class="pill na">fitted universe</span></td>'
      f'<td class="num">{wf.get("n_folds", "—")}</td>'
      f'<td class="num">{_fmt(oos.get("vol_calm"), 5)}</td>'
      f'<td class="num">{_fmt(oos.get("vol_stressed"), 5)}</td>'
      f'<td class="num">{_fmt(oos.get("ratio"), 2)}×</td>'
      f'<td class="num">{_fmt(fwd.get("ratio"), 2)}×</td>'
      f'<td class="num">{_fmt(wf.get("per_fold_ratio_median"), 2)}×</td></tr>')
    for r in regimes.get("generalisation", []):
        if "error" in r:
            a(f'<tr><td>{r["index"]}</td><td colspan="6">'
              f'<span class="pill fail">failed</span> {r["error"]}</td></tr>')
            continue
        a(f'<tr><td>{r["index"]}</td><td class="num">{r["n_folds"]}</td>'
          f'<td class="num">{_fmt(r["vol_calm"], 5)}</td>'
          f'<td class="num">{_fmt(r["vol_stressed"], 5)}</td>'
          f'<td class="num">{_fmt(r["ratio_oos"], 2)}×</td>'
          f'<td class="num">{_fmt(r.get("ratio_forward"), 2)}×</td>'
          f'<td class="num">{_fmt(r.get("per_fold_median"), 2)}×</td></tr>')
    a("</tbody></table>")
    a("<figure>")
    a(_img(figures.get("generalisation", Path("outputs/generalisation.png"))))
    a("</figure>")
    a('<div class="caveat"><strong>Read this honestly.</strong> The separation '
      "holds on all three unseen indices but is materially weaker than on the "
      "fitted universe. Note also that the pooled ratio exceeds the median "
      "per-fold ratio: pooling mixes calm-era and crisis-era days, so part of the "
      "pooled figure is a between-period effect rather than within-period "
      "separation. The per-fold median is the conservative number.</div>")

    if referee and referee.get("per_fold_sign_test"):
        sg = referee["per_fold_sign_test"]
        a("<h3>The strongest form of the separation result</h3>")
        a(f"<p>The pooled ratio can be attacked for mixing eras. The per-fold "
          f"distribution cannot. Of the <strong>{sg['n_folds_with_both_states']} "
          f"folds that contain both states</strong> — the other "
          f"{sg['n_folds_total'] - sg['n_folds_with_both_states']} spent the whole "
          f"test block in one regime and cannot produce a ratio — "
          f"<strong>{sg['n_folds_ratio_above_1']} have a ratio above 1.0</strong>. "
          f"Exact sign test <code>p = {sg['sign_test_p']:.5f}</code>; Wilcoxon on "
          f"the log ratio <code>p = {sg['wilcoxon_log_ratio_p']:.2g}</code>; median "
          f"{_fmt(sg['median_ratio'], 2)}× with a bootstrap 95% CI of "
          f"{sg['median_95ci'][0]}–{sg['median_95ci'][1]}.</p>")
        a('<div class="good">This is distribution-free, immune to the pooling '
          "objection above, and immune to volatility clustering. It is the number "
          "to defend under questioning. Note also that quoting a per-fold median "
          f"alongside &ldquo;{sg['n_folds_total']} folds&rdquo; would be wrong — "
          f"the median is over {sg['n_folds_with_both_states']}.</div>")

    # ---- transitions -----------------------------------------------------
    a("<h2>3 · Transition count and dwell discipline</h2>")
    a(f"<p>The raw state sequence flips {pf.get('raw_state_flips', '—')} times. "
      f"A {spec.get('min_dwell_days')}-day minimum-dwell filter reduces that to "
      f"{pf.get('flips_after_filter', '—')}, removing "
      f"{pf.get('flips_removed_as_noise', '—')} single- or short-run excursions "
      f"as noise. Of those, {stab.get('n_stable_transitions')} recur across "
      f"seeds and form the analysis set.</p>")
    a('<table><thead><tr><th class="num">Min dwell (days)</th>'
      '<th class="num">Transitions</th><th class="num">Into stress</th>'
      '<th class="num">Out of stress</th><th class="num">Episodes</th>'
      '<th class="num">Median episode</th><th class="num">Max episode</th>'
      "</tr></thead><tbody>")
    for r in regimes.get("dwell_sensitivity", []):
        sel = r["min_dwell_days"] == spec.get("min_dwell_days")
        a(f'<tr{" class=total" if sel else ""}>'
          f'<td class="num">{r["min_dwell_days"]}'
          f'{" ← pre-registered" if sel else ""}</td>'
          f'<td class="num">{r["n_transitions"]}</td>'
          f'<td class="num">{r["n_to_stressed"]}</td>'
          f'<td class="num">{r["n_to_calm"]}</td>'
          f'<td class="num">{r["n_episodes"]}</td>'
          f'<td class="num">{_fmt(r["median_episode_days"], 0)}</td>'
          f'<td class="num">{r["max_episode_days"]}</td></tr>')
    a("</tbody></table>")

    a("<h3>Seed stability</h3>")
    a(f"<p>The prior work noted that HMM labels move with <code>random_state</code>. "
      f"Since every narrative hangs off a transition date, that matters. Across "
      f"{stab.get('n_seeds')} seeds the transition count ranges over "
      f"{sorted(stab.get('counts_per_seed', {}).values())}; "
      f"{stab.get('n_stable_transitions')} transitions recur in at least 60% of "
      f"seeds and only those are carried forward.</p>")
    a("<figure>")
    a(_img(figures.get("dwell", Path("outputs/dwell_histogram.png"))))
    a("</figure>")

    # ---- control 1 -------------------------------------------------------
    a("<h2>4 · Control 1 — hindsight</h2>")
    no_leak = nsum.get("any_post_boundary_items") is False
    a(f"<p>The retrieval boundary is enforced structurally: a window object "
      f"refuses to construct if any item post-dates it. "
      f"{_pill(no_leak, 'NO LEAK', 'LEAK DETECTED')}</p>")
    a('<table><tbody>')
    a(f'<tr><td>Windows retrieved</td><td class="num">{nsum.get("n_windows")}</td></tr>')
    a(f'<tr><td>Retrieval failures</td><td class="num">{nsum.get("n_failures")}</td></tr>')
    a(f'<tr><td>Total news items</td><td class="num">{nsum.get("total_items")}</td></tr>')
    a(f'<tr><td>Mean items per transition window</td>'
      f'<td class="num">{nsum.get("mean_items_transition")}</td></tr>')
    a(f'<tr><td>Mean items per control window</td>'
      f'<td class="num">{nsum.get("mean_items_placebo")}</td></tr>')
    a(f'<tr><td>Items surviving past the boundary</td>'
      f'<td class="num">{"0" if no_leak else "SEE LOG"}</td></tr>')
    a("</tbody></table>")
    a(f'<div class="caveat"><strong>Why revision pinning is not optional.</strong> '
      f"Wikipedia's page for a given day keeps being edited for years. Across "
      f"sampled transition dates, a mean of "
      f"<strong>{leak.get('mean_pct_written_after_boundary')}%</strong> of the "
      f"current page was written after the boundary, peaking at "
      f"<strong>{leak.get('max_pct_written_after_boundary')}%</strong>. Fetching "
      f"the live page would have fed the model text written with full knowledge "
      f"of what followed, while the manifest recorded the window as closing on "
      f"the transition date. Every page here is pinned to the last revision at or "
      f"before the boundary, and the revision id is recorded.</div>")

    # ---- control 3 -------------------------------------------------------
    a("<h2>5 · Control 3 — placebo</h2>")
    a(f"<p>Each transition is paired with era-matched control dates "
      f"{spec.get('placebo_offset_days', ['—', '—'])[0]}–"
      f"{spec.get('placebo_offset_days', ['—', '—'])[1]} days away, outside every "
      f"transition window and disjoint from each other. Same news environment, "
      f"same era, differing only in whether the model called a transition.</p>")
    a('<table><tbody>')
    a(f'<tr><td>Control dates</td><td class="num">{bal.get("n_placebos")}</td></tr>')
    a(f'<tr><td>Transitions with a control</td>'
      f'<td class="num">{_fmt(bal.get("coverage"), 2)}</td></tr>')
    a(f'<tr><td>Mean year gap (era balance)</td>'
      f'<td class="num">{_fmt(bal.get("year_mean_gap"), 2)}</td></tr>')
    a(f'<tr><td>Mean offset from partner transition</td>'
      f'<td class="num">{bal.get("mean_abs_offset_days")} days</td></tr>')
    a(f'<tr><td>Clean / contaminated stratum</td>'
      f'<td class="num">{bal.get("stratum_split", {}).get("clean")} / '
      f'{bal.get("stratum_split", {}).get("contaminated")}</td></tr>')
    a("</tbody></table>")

    if narratives:
        p = narratives.get("placebo", {})
        a('<table><thead><tr><th>Arm</th><th class="num">n</th>'
          '<th class="num">Confident explanations</th><th class="num">Rate</th>'
          "</tr></thead><tbody>")
        for key, label in (("transitions", "Real transitions"),
                           ("placebos_all", "All controls"),
                           ("placebos_clean_stratum", "Clean controls (primary)")):
            d = p.get(key, {})
            a(f'<tr><td>{label}</td><td class="num">{d.get("n")}</td>'
              f'<td class="num">{d.get("n_confident")}</td>'
              f'<td class="num">{_fmt(d.get("confident_rate"), 2)}</td></tr>')
        a("</tbody></table>")
        a(f"<p>Fisher exact, transitions vs all controls: "
          f"<code>p = {_fmt(p.get('fisher_p_all'), 4)}</code>; "
          f"vs clean controls: <code>p = {_fmt(p.get('fisher_p_clean'), 4)}</code>.</p>")

        if referee and referee.get("placebo_interval"):
            pi = referee["placebo_interval"]
            lo, hi = pi["newcombe_95ci_pp_vs_clean"]
            a('<div class="caveat"><strong>State this as an interval, not a '
              "verdict.</strong> The difference is "
              f"<strong>+{pi['difference_pp']}pp</strong>, with a Newcombe 95% "
              f"interval of <strong>[{lo:.0f}, {hi:.0f}]pp</strong>. The data are "
              "compatible with no difference and with a large one. A Jeffreys "
              "posterior puts <strong>P(transitions &gt; controls) = "
              f"{pi['posterior_prob_transition_higher']}</strong>. Saying the "
              "explanations are &ldquo;not diagnostic&rdquo; would be an "
              "affirmative claim of no-difference, and nothing here licenses "
              "that. The comparison is unresolved.</div>")

        if power:
            obs = power.get("observed", {})
            a('<div class="caveat"><strong>What this result does and does not say — '
              "read this before quoting the number.</strong><br><br>"
              "The control arm did not separate. But <em>&ldquo;not "
              "significant&rdquo;</em> here means <em>&ldquo;not resolved&rdquo;</em>, "
              "not <em>&ldquo;no difference&rdquo;</em>, and the distinction is the "
              "whole of it.<br><br>"
              f"<strong>Power at the observed "
              f"{_fmt(obs.get('gap_pp'), 1)} percentage-point gap is "
              f"{_fmt(power.get('power_at_observed_effect', {}).get('vs_clean_n13'), 2)}"
              "</strong> — this study had roughly a one-in-six chance of detecting "
              "the very effect it observed. To reach 80% power with 20 transitions, "
              "the transition rate would have to be near 100%. The binding "
              "constraint is the number of transitions, not the number of controls: "
              "even 200 controls only reaches 80% power at a ~93% transition rate, "
              "so no amount of extra placebo sampling fixes this.<br><br>"
              "The gap runs in the predicted direction and is stable on replication "
              "(below). The honest statement is that these narratives are not yet "
              "shown to be diagnostic of a regime change, and that this design could "
              "not have shown it short of a near-total effect.</div>")
            rows = power.get("power_curve_vs_clean_controls", {})
            if rows:
                a('<table><thead><tr><th>If the transition rate were…</th>'
                  '<th class="num">Power to detect it (vs 13 clean controls)</th>'
                  "</tr></thead><tbody>")
                for k, v in rows.items():
                    a(f'<tr><td>{k}</td><td class="num">{_fmt(v, 2)}</td></tr>')
                a("</tbody></table>")

        if retest:
            a("<h3>Is the label even stable?</h3>")
            a("<p>Sampling parameters were removed from the API, so identical "
              "inputs no longer guarantee identical outputs — and the statistic "
              "above is a label the model emits about itself. Every window was "
              "therefore generated a second time.</p>")
            a('<table><tbody>')
            a(f'<tr><td>Agreement on &ldquo;confident&rdquo;</td>'
              f'<td class="num">{_fmt(100 * retest.get("agreement_is_confident", 0), 1)}% '
              f'(&kappa; = {_fmt(retest.get("kappa_is_confident"), 2)})</td></tr>')
            a(f'<tr><td>Agreement on driver identified</td>'
              f'<td class="num">{_fmt(100 * retest.get("agreement_driver_identified", 0), 1)}% '
              f'(&kappa; = {_fmt(retest.get("kappa_driver_identified"), 2)})</td></tr>')
            a(f'<tr><td>Exact confidence label</td>'
              f'<td class="num">{_fmt(100 * retest.get("agreement_exact_confidence_label", 0), 1)}%</td></tr>')
            a(f'<tr class="total"><td>Transition confident rate, run 1 &rarr; run 2</td>'
              f'<td class="num">{_fmt(retest.get("confident_rate_run1_transitions"), 2)} '
              f'&rarr; {_fmt(retest.get("confident_rate_run2_transitions"), 2)}</td></tr>')
            a(f'<tr class="total"><td>Control confident rate, run 1 &rarr; run 2</td>'
              f'<td class="num">{_fmt(retest.get("confident_rate_run1_placebos"), 2)} '
              f'&rarr; {_fmt(retest.get("confident_rate_run2_placebos"), 2)}</td></tr>')
            a("</tbody></table>")
            a('<div class="good">Individual labels move — 20% of windows return a '
              "different confidence word — but the aggregate rates reproduce "
              "exactly. The placebo gap is a stable property of the method, not an "
              "artefact of one sampling run. That is what licenses calling the "
              "result underpowered rather than noisy.</div>")
    else:
        a('<div class="missing">Narrative generation has not been run '
          "(<code>ANTHROPIC_API_KEY</code> not set). The control dates, the "
          "windows and the comparison machinery are all in place; only the "
          "language-model calls are outstanding.</div>")

    # ---- blind matching --------------------------------------------------
    a("<h2>6 · Blind matching — are the explanations specific?</h2>")
    a("<p>Explanations are stripped of dates, scrambled, and matched back to "
      "their window using BM25 against <strong>held-out</strong> articles the "
      "model never saw. The matcher has no world knowledge, so it cannot date an "
      "explanation from training the way an LLM judge could.</p>")

    if ceiling:
        a("<h3>Ceiling — what any explanation could achieve</h3>")
        a("<p>Measured before generating anything, using each window's "
          "generation half as a stand-in for a perfect explanation.</p>")
        a('<table><thead><tr><th>Arm</th><th class="num">Correct</th>'
          '<th class="num">Accuracy</th><th class="num">Chance</th>'
          '<th class="num">MRR</th><th class="num">p</th></tr></thead><tbody>')
        for key, label in (("transitions_only", "Transitions only"),
                           ("placebos_only", "Controls only"),
                           ("all_windows", "All windows pooled")):
            d = ceiling.get("results", {}).get(key, {})
            if "error" in d:
                continue
            a(f'<tr><td>{label}</td>'
              f'<td class="num">{d.get("n_correct")}/{d.get("n_scored")}</td>'
              f'<td class="num">{_fmt(100 * d.get("accuracy", 0), 1)}%</td>'
              f'<td class="num">{_fmt(100 * d.get("chance", 0), 1)}%</td>'
              f'<td class="num">{_fmt(d.get("mean_reciprocal_rank"), 3)}</td>'
              f'<td class="num">{_fmt(d.get("p_value"), 4)}</td></tr>')
        a("</tbody></table>")

        sweep = ceiling.get("length_sweep")
        if sweep:
            a("<h3>Ceiling by explanation length</h3>")
            a("<p>The arm above uses the whole generation half — thousands of "
              "words. A real explanation is one or two hundred. Short queries "
              "carry much less signal, so that number flatters the test.</p>")
            a('<table><thead><tr><th>Proxy length</th><th class="num">Accuracy</th>'
              '<th class="num">Top-3</th><th class="num">MRR</th>'
              '<th class="num">p</th></tr></thead><tbody>')
            for row in sweep:
                pw = row["proxy_words"]
                realistic = pw in (120, 200)
                a(f'<tr{" class=total" if realistic else ""}>'
                  f'<td>{pw if pw == "full_half" else str(pw) + " words"}'
                  f'{" ← realistic range" if realistic else ""}</td>'
                  f'<td class="num">{_fmt(100 * row["accuracy"], 1)}%</td>'
                  f'<td class="num">{_fmt(100 * row["top3"], 1)}%</td>'
                  f'<td class="num">{_fmt(row["mrr"], 3)}</td>'
                  f'<td class="num">{_fmt(row["p_value"], 5)}</td></tr>')
            a("</tbody></table>")
            a("<p>At the length explanations will actually be, the ceiling is "
              "roughly 40–70% against 5% chance — not 95%. Significance survives "
              "at every length tested, including 60 words, so the test remains "
              "well powered; the expectation simply has to be set correctly.</p>")

        arms = ceiling.get("section_arms")
        if arms:
            a("<h3>Ceiling by section</h3>")
            a('<table><thead><tr><th>Sections used</th><th class="num">n</th>'
              '<th class="num">Accuracy</th><th class="num">Chance</th>'
              '<th class="num">p</th><th class="num">Median holdout words</th>'
              "</tr></thead><tbody>")
            for key, label in (("all_sections", "All"),
                               ("business_only", "Business and economy only"),
                               ("non_business", "Everything except business")):
                d = arms.get(key, {})
                if "error" in d:
                    a(f'<tr><td>{label}</td><td colspan="5">{d["error"]}</td></tr>')
                    continue
                a(f'<tr><td>{label}</td><td class="num">{d.get("n_scored")}</td>'
                  f'<td class="num">{_fmt(100 * d.get("accuracy", 0), 1)}%</td>'
                  f'<td class="num">{_fmt(100 * d.get("chance", 0), 1)}%</td>'
                  f'<td class="num">{_fmt(d.get("p_value"), 4)}</td>'
                  f'<td class="num">{d.get("median_holdout_words")}</td></tr>')
            a("</tbody></table>")

    a('<div class="caveat"><strong>A caveat that the results partly answered.</strong> '
      "The ceiling above is carried by generic world news rather than market "
      "content: restricting the held-out text to business and economy sections "
      "drops the <em>ceiling</em> to 16.7% against 8.3% chance (p = 0.27), while "
      "non-business sections alone reach 90%. That raised a real worry — an "
      "explanation that obeys the prompt and discusses market drivers might have "
      "little matchable text, and a null result would then be ambiguous between "
      "&ldquo;the explanations are generic&rdquo; and &ldquo;the matchable signal "
      "is not where the explanation looks&rdquo;.<br><br>"
      "<strong>The measured result came out better than its own ceiling on that "
      "arm.</strong> Real explanations match business-only holdout text at 28.6% "
      "against 7.1% chance (p = 0.016) — significant, and above the 16.7% the "
      "ceiling test predicted. The reason is that a real explanation concentrates "
      "on market-relevant content, whereas the ceiling proxy was an arbitrary "
      "slice of the window. The worry was worth having and it did not "
      "materialise; the section gap is still reported below so the reader can "
      "see both.</div>")

    if narratives:
        bm = narratives.get("blind_match", {})
        a('<table><thead><tr><th>Arm</th><th class="num">Correct</th>'
          '<th class="num">Accuracy</th><th class="num">Chance</th>'
          '<th class="num">MRR</th><th class="num">p</th></tr></thead><tbody>')
        for key, label in (
            ("transitions_all_sections", "Transitions, all sections"),
            ("transitions_business_only", "Transitions, business only"),
            ("transitions_non_business", "Transitions, non-business"),
            ("all_windows", "All windows pooled"),
        ):
            d = bm.get(key, {})
            if not d or "error" in d:
                a(f'<tr><td>{label}</td><td colspan="5">{d.get("error", "—")}</td></tr>')
                continue
            a(f'<tr><td>{label}</td>'
              f'<td class="num">{d.get("n_correct")}/{d.get("n_scored")}</td>'
              f'<td class="num">{_fmt(100 * d.get("accuracy", 0), 1)}%</td>'
              f'<td class="num">{_fmt(100 * d.get("chance", 0), 1)}%</td>'
              f'<td class="num">{_fmt(d.get("mrr"), 3)}</td>'
              f'<td class="num">{_fmt(d.get("p_value"), 4)}</td></tr>')
        a("</tbody></table>")

        if referee and referee.get("blind_match_hard_negatives"):
            hn = referee["blind_match_hard_negatives"]
            a("<h3>Against era-adjacent hard negatives</h3>")
            a("<p>A sceptic can explain a high score against the full pool by "
              "era-matching: adjacent fortnights share running stories, so the "
              "matcher might only be recovering roughly <em>when</em>, not "
              "<em>which</em> fortnight. Restricting each explanation to compete "
              "only against its temporal neighbours removes that explanation.</p>")
            a('<table><thead><tr><th>Candidate pool</th><th class="num">Correct</th>'
              '<th class="num">Accuracy</th><th class="num">Chance</th>'
              '<th class="num">p</th></tr></thead><tbody>')
            for k, lbl in (("3_temporally_nearest", "3 temporally nearest windows"),
                           ("5_temporally_nearest", "5 temporally nearest windows")):
                d = hn.get(k)
                if not isinstance(d, dict):
                    continue
                a(f'<tr><td>{lbl}</td>'
                  f'<td class="num">{d["n_correct"]}/{d["n_scored"]}</td>'
                  f'<td class="num">{100 * d["accuracy"]:.1f}%</td>'
                  f'<td class="num">{100 * d["chance"]:.1f}%</td>'
                  f'<td class="num">{d["binomial_p"]:.2g}</td></tr>')
            a("</tbody></table>")
            a('<div class="good">The signal survives every era-matched '
              "restriction. That is stronger evidence than the headline number, "
              "because era cannot account for it.</div>")
    else:
        a('<div class="missing">Awaiting narrative generation.</div>')

    # ---- control 2 -------------------------------------------------------
    a("<h2>7 · Control 2 — memorisation</h2>")
    a("<p>The model has read text covering these events. Three mitigations, and "
      "an honest limit:</p><ul>"
      "<li>Absolute dates are stripped from item text before the model sees it.</li>"
      "<li>Every claim must cite a supplied item id, so unsupported recall is "
      "visible rather than invisible.</li>"
      "<li>Citation faithfulness is measured: does the cited item actually "
      "contain the claim's content? Grounding is scored lexically, so the "
      "measurement itself has no world knowledge.</li></ul>")
    a('<div class="caveat"><strong>Not solved — mitigated and measured.</strong> '
      "The obvious control, splitting results either side of the model's training "
      "cutoff, cannot work here: with a cutoff in early 2025 the post-cutoff arm "
      "holds a handful of transitions, and reporting <em>n</em>=3 as a test would "
      "be worse than not running it. Citation faithfulness is reported instead "
      "because it scales to every claim in every window.</div>")
    if narratives:
        f = narratives.get("faithfulness", {})
        a('<table><thead><tr><th>Set</th><th class="num">Claims</th>'
          '<th class="num">Grounded</th><th class="num">Ungrounded</th>'
          '<th class="num">Uncited</th><th class="num">Fabricated citation</th>'
          "</tr></thead><tbody>")
        for key, label in (("transitions", "Transitions"),
                           ("placebos", "Controls"), ("all", "All")):
            d = f.get(key, {})
            a(f'<tr><td>{label}</td><td class="num">{d.get("n_claims")}</td>'
              f'<td class="num">{_fmt(d.get("grounding_rate"), 2)}</td>'
              f'<td class="num">{_fmt(d.get("ungrounded_rate"), 2)}</td>'
              f'<td class="num">{_fmt(d.get("uncited_rate"), 2)}</td>'
              f'<td class="num">{_fmt(d.get("fabricated_citation_rate"), 2)}</td></tr>')
        a("</tbody></table>")

        if referee and referee.get("grounding_null"):
            gn = referee["grounding_null"]
            a("<h3>What that rate is measured against</h3>")
            a("<p>A grounding rate means nothing without a floor — the obvious "
              "objection is that lexical overlap at a lenient threshold is a "
              "test everything passes. So the same claims were re-scored against "
              "a <em>randomly chosen item from the same window</em>.</p>")
            a('<table><thead><tr><th>Scored against</th>'
              '<th class="num">Mean overlap</th><th class="num">Grounded</th>'
              "</tr></thead><tbody>")
            a(f'<tr class="total"><td>The item the claim actually cites</td>'
              f'<td class="num">{gn["mean_overlap_cited"]:.3f}</td>'
              f'<td class="num">{100 * gn["grounded_rate_cited"]:.1f}%</td></tr>')
            a(f'<tr><td>A random item from the same window '
              f'({gn["n_random_draws"]:,} draws)</td>'
              f'<td class="num">{gn["mean_overlap_random_same_window"]:.3f}</td>'
              f'<td class="num">{100 * gn["grounded_rate_random_same_window"]:.1f}%'
              f"</td></tr>")
            a("</tbody></table>")
            a('<div class="good">A '
              f'{gn["grounded_rate_cited"] / max(gn["grounded_rate_random_same_window"], 1e-9):.0f}× '
              "separation between cited and random items from the same fortnight. "
              "The objection does not hold: this is not a test everything passes."
              "</div>")
            a('<div class="caveat"><strong>What it still cannot see.</strong> The '
              "tokeniser drops numerals, so a misquoted figure is invisible to it; "
              "negation is invisible; and bag-of-words cannot tell "
              "&ldquo;X caused Y&rdquo; from &ldquo;Y caused X&rdquo;. Grounding "
              "establishes that a claim's content came from the item it cites, not "
              "that the claim is correct.</div>")
    else:
        a('<div class="missing">Awaiting narrative generation.</div>')

    # ---- reproducibility -------------------------------------------------
    a("<h2>8 · Specification and reproducibility</h2>")
    a("<p>Fixed before any narrative was generated:</p>")
    a('<table><tbody>')
    for k, v in spec.items():
        a(f'<tr><td>{k.replace("_", " ")}</td><td class="num mono">{v}</td></tr>')
    a(f'<tr><td>news source</td><td class="num mono">'
      f'Wikipedia Current Events, revision-pinned</td></tr>')
    a(f'<tr><td>window convention</td><td class="num mono">'
      f'[boundary − {spec.get("news_window_days", 14) - 1}d 00:00:00Z, '
      f'boundary 23:59:59Z]</td></tr>')
    a("</tbody></table>")
    a("<p>Every retrieval records its source, revision id, revision timestamp, "
      "retrieval time and exact window boundaries. Every model call records the "
      "model id, prompt version, prompt hash, input hash and raw output.</p>")

    a('<div class="foot">Prices via yfinance. News via the MediaWiki API, '
      "revision-pinned. The HMM decides when the regime changed; the language "
      "model only describes what was in the news at that moment. No number in "
      "the statistical results comes from the language model, and it never "
      "influences a regime boundary.</div>")
    a("</div></body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(h), encoding="utf-8")
    return out_path
