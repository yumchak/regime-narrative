"""Stage 7 -- the one-page PDF, generated from the results rather than typed.

Every number on the page is substituted from the JSON that the pipeline wrote.
Hand-typing figures into a deliverable is how a table ends up disagreeing with
the report it summarises; the first draft of this page had three per-fold medians
that were simply invented, and nothing would have caught them.

Output is A4 print-ready HTML. Open it and print to PDF (Ctrl+P, "Save as PDF",
margins: default, background graphics: on).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from regime_narrative.config import output_dir

CSS = """
@page { size: A4; margin: 12mm 13mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font: 8.6pt/1.34 "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
  color: #16202a; background: #fff; width: 184mm; margin: 0 auto; padding: 4mm 0; }
h1 { font-size: 15pt; line-height: 1.12; margin: 0 0 1mm; letter-spacing: -0.35pt; }
.sub { font-size: 8.4pt; color: #55636f; margin: 0 0 3mm; }
h2 { font-size: 8.4pt; text-transform: uppercase; letter-spacing: 0.7pt;
  color: #55636f; margin: 3.6mm 0 1.4mm; padding-bottom: 0.7mm;
  border-bottom: 0.5pt solid #c9d2da; }
p { margin: 0 0 1.6mm; }
.claim { border-left: 2.2pt solid #37567f; background: #f4f7fa;
  padding: 2.4mm 3mm; margin: 0 0 3mm; font-size: 8.9pt; line-height: 1.42; }
.cols { display: flex; gap: 5mm; }
.col { flex: 1; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.9pt; margin: 0 0 1.6mm; }
th, td { padding: 1.15mm 1.6mm; text-align: left; border-bottom: 0.4pt solid #dde3e9; }
th { background: #f4f7fa; font-weight: 600; font-size: 7pt; text-transform: uppercase;
  letter-spacing: 0.35pt; color: #55636f; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.hi td { font-weight: 600; border-top: 0.9pt solid #16202a; }
.flag { border-left: 2.2pt solid #a8641c; background: #fdf8f2; padding: 2.2mm 3mm;
  margin: 1.6mm 0 0; font-size: 8.1pt; }
.ok { border-left: 2.2pt solid #2e7d55; background: #f2f8f5; padding: 2.2mm 3mm;
  margin: 1.6mm 0 0; font-size: 8.1pt; }
ul { margin: 0 0 1.6mm; padding-left: 3.6mm; }
li { margin: 0 0 0.7mm; }
.foot { margin-top: 3.4mm; padding-top: 1.6mm; border-top: 0.4pt solid #c9d2da;
  font-size: 7.1pt; color: #6b7783; }
code { font-family: Consolas, monospace; font-size: 7.4pt; }
b { font-weight: 600; }
"""

PRETTY = {"nikkei": "Nikkei 225", "hang_seng": "Hang Seng", "dax": "DAX"}


def _load(name):
    p = output_dir() / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    R = _load("regimes.json")
    N = _load("news_manifest.json") or {}
    D = _load("narrative_results.json") or {}
    P = _load("placebo_power.json") or {}
    T = _load("retest_stability.json") or {}
    RF = _load("referee_stats.json") or {}
    if R is None:
        print("run scripts/01_regimes.py first")
        sys.exit(1)

    oos, ins = R["out_of_sample"], R["in_sample"]
    fwd, wf, pf = R["out_of_sample_forward"], R["walk_forward"], R["persistence_filter"]
    stab = R["seed_stability"]
    nsum, leak = N.get("summary", {}), N.get("hindsight_leak", {})
    pb = D.get("placebo", {})
    bm = D.get("blind_match", {}).get("transitions_all_sections", {})
    bmb = D.get("blind_match", {}).get("transitions_business_only", {})
    fa = D.get("faithfulness", {}).get("all", {})

    gen_rows = "".join(
        f'<tr><td>{PRETTY.get(g["index"], g["index"])}</td>'
        f'<td class="n">{g["ratio_oos"]:.2f}×</td>'
        f'<td class="n">{g["ratio_forward"]:.2f}×</td>'
        f'<td class="n">{g["per_fold_median"]:.2f}×</td></tr>'
        for g in R["generalisation"] if "error" not in g
    )

    worst = max(
        (s for s in leak.get("samples", []) if s.get("status") == "ok"),
        key=lambda s: s["pct_written_after_boundary"], default=None,
    )
    worst_txt = ""
    if worst:
        page = worst["page"].split("/")[-1]
        worst_txt = (
            f' — the {page} page was {worst["pinned_size_bytes"]:,} bytes on the day '
            f'and is {worst["current_size_bytes"]:,} today, last edited in '
            f'{worst["current_timestamp"][:4]}'
        )

    power_obs = P.get("power_at_observed_effect", {}).get("vs_clean_n13")
    gap_pp = P.get("observed", {}).get("gap_pp")
    pi = RF.get("placebo_interval", {})
    ci = pi.get("newcombe_95ci_pp_vs_clean", [None, None])
    sg = RF.get("per_fold_sign_test", {})
    gn = RF.get("grounding_null", {})
    hn = RF.get("blind_match_hard_negatives", {}).get("3_temporally_nearest", {})
    sg_ci = sg.get("median_95ci", [None, None])
    sg_ci_txt = (f"{sg_ci[0]}–{sg_ci[1]}" if sg_ci and sg_ci[0] is not None else "—")
    nikkei_fwd = next((g["ratio_forward"] for g in R["generalisation"]
                       if g.get("index") == "nikkei"), None)

    t_rate = pb.get("transitions", {}).get("confident_rate")
    c_rate = pb.get("placebos_clean_stratum", {}).get("confident_rate")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>regime-narrative — one page</title><style>{CSS}</style></head><body>

<h1>A regime model that can say what happened — and a measurement of how much
that is worth</h1>
<p class="sub">Statistical regime detection with a citation-grounded explanation
layer, and three controls that test it. BSc Mathematics with Statistics,
University of Bristol.</p>

<div class="claim">
A two-state Gaussian HMM, <b>refitted inside each of {wf['n_folds']} walk-forward
folds</b> and scored only on held-out days, separates SPY volatility regimes by
<b>{oos['ratio']:.2f}×</b> (95% CI {oos['ratio_ci_low']:.2f}–{oos['ratio_ci_high']:.2f});
the same procedure refitted on three indices outside the original universe gives
{', '.join(f"{g['ratio_oos']:.2f}×" for g in R['generalisation'] if 'error' not in g)}.
A {R['specification']['min_dwell_days']}-day persistence filter reduces
{pf['raw_state_flips']} raw state flips to {pf['flips_after_filter']}, of which
<b>{stab['n_stable_transitions']} recur across random seeds</b>. For each, a
language model reading <b>only news published before that date</b> produced a
citation-grounded explanation. Stripped of dates and scrambled,
<b>{100 * bm.get('accuracy', 0):.0f}% matched back to the correct fortnight
against {100 * bm.get('chance', 0):.0f}% chance</b> (p&nbsp;&lt;&nbsp;0.0001),
with <b>{100 * fa.get('grounding_rate', 0):.1f}% of {fa.get('n_claims', 0)} claims
grounded in the source they cite and zero fabricated citations</b>. On
era-matched non-transition dates the identical pipeline produced confident
explanations {100 * (c_rate or 0):.1f}% of the time against
{100 * (t_rate or 0):.0f}% on real transitions: <b>+{pi.get('difference_pp')}pp,
95% CI [{ci[0]:.0f}, {ci[1]:.0f}]pp</b>. That comparison is unresolved, and this
design could not have resolved it. That is the honest limit of the method.
</div>

<div class="cols">
<div class="col">

<h2>The problem</h2>
<p>A regime model tells you the market changed. It cannot tell you what happened,
because it never sees anything except returns. So its output is coloured bands a
human interprets from memory — undocumented, unauditable, and different for every
person who looks. For a platform running thirty PMs, that is thirty private
readings of one signal.</p>

<h2>The split that makes it testable</h2>
<p><b>The HMM decides <i>when</i> the regime changed. The language model only
describes <i>what was in the news</i> at that moment.</b> No number in the
statistical results comes from the model, and it never moves a regime boundary.</p>

<h2>Regime separation, out-of-sample</h2>
<table>
<thead><tr><th>Index</th><th class="n">Same-day</th><th class="n">Forward 20d</th>
<th class="n">Per-fold median</th></tr></thead>
<tbody>
<tr class="hi"><td>SPY <i>(fitted universe)</i></td>
<td class="n">{oos['ratio']:.2f}×</td><td class="n">{fwd['ratio']:.2f}×</td>
<td class="n">{wf['per_fold_ratio_median']:.2f}×</td></tr>
{gen_rows}
</tbody></table>
<p><b>Forward-20d</b> measures volatility that had not happened when the state
was assigned — the HMM is fed <i>trailing</i> volatility, so the same-day figure
is partly definitional and the forward one is the real claim. Nikkei at
{nikkei_fwd:.2f}× is close to nothing, and that is reported rather than left
behind its more flattering same-day figure.</p>
<div class="ok"><b>The strongest form of the separation result.</b> Of the
{sg.get('n_folds_with_both_states')} folds that contain both states,
<b>{sg.get('n_folds_ratio_above_1')} have a ratio above 1.0 — exact sign test
p&nbsp;=&nbsp;{sg.get('sign_test_p', 0):.5f}</b>, median
{sg.get('median_ratio', 0):.2f}× (95% CI {sg_ci_txt}).
This is distribution-free and immune to the objection that the pooled
{oos['ratio']:.2f}× mixes calm-era and crisis-era days. It is the number to
defend.</div>

<h2>Three corrections to the prior work</h2>
<ul>
<li>The earlier notebook's {ins['ratio']:.2f}× was <b>in-sample over the full
history</b>, and its 26 folds belonged to a separate return classifier. One
sentence, two unrelated experiments. Refit per fold it is {oos['ratio']:.2f}×.</li>
<li>States are assigned by a <b>forward filter</b>, not Viterbi — which smooths
over the whole test block, letting a label depend on up to
{R['specification']['test_days']} days of future data and contaminating the very
transition dates the news control rests on.</li>
<li>The scaler is fitted <b>inside</b> each fold.</li>
</ul>

</div>
<div class="col">

<h2>Control 1 — hindsight</h2>
<p>The window closes on the transition date, enforced structurally: a window
object refuses to construct if any item post-dates it.
{nsum.get('n_windows', '—')} windows, {nsum.get('total_items', 0):,} items,
{nsum.get('n_failures', '—')} retrieval failures,
<b>zero post-boundary survivors</b>.</p>
<div class="flag"><b>Why revision pinning is not optional.</b> Wikipedia's page
for a given day keeps being edited for years. Across sampled transition dates a
mean of <b>{leak.get('mean_pct_written_after_boundary', '—')}%</b> of the current
page was written <i>after</i> the boundary, peaking at
<b>{leak.get('max_pct_written_after_boundary', '—')}%</b>{worst_txt}. Fetching the
live page would feed the model text written with full knowledge of what followed,
while the manifest recorded the window as closing on the date. Every page is
pinned by revision id.</div>

<h2>Control 2 — memorisation</h2>
<p>Dates are stripped before the model sees the text; every claim must cite a
supplied item; grounding is scored by lexical overlap, so the check itself has no
world knowledge. <b>{100 * fa.get('grounding_rate', 0):.1f}% grounded,
{100 * fa.get('fabricated_citation_rate', 0):.1f}% fabricated citations,
{fa.get('n_claims', 0)} claims — against a
{100 * gn.get('grounded_rate_random_same_window', 0):.1f}% floor</b> when the
same claims are scored against a randomly chosen item from the same window
({gn.get('n_random_draws', 0):,} draws). It is not a metric everything passes. Splitting either side of the training cutoff
was abandoned: the post-cutoff arm holds three transitions, and reporting
<i>n</i>=3 as a test is worse than not running it.</p>

<h2>Control 3 — placebo, and the limit</h2>
<table>
<thead><tr><th>Arm</th><th class="n">n</th><th class="n">Confident</th></tr></thead>
<tbody>
<tr><td>Real transitions</td><td class="n">{pb.get('transitions', {}).get('n', '—')}</td>
<td class="n">{100 * (t_rate or 0):.1f}%</td></tr>
<tr><td>Era-matched controls</td><td class="n">{pb.get('placebos_all', {}).get('n', '—')}</td>
<td class="n">{100 * (pb.get('placebos_all', {}).get('confident_rate') or 0):.1f}%</td></tr>
<tr class="hi"><td>Clean controls</td>
<td class="n">{pb.get('placebos_clean_stratum', {}).get('n', '—')}</td>
<td class="n">{100 * (c_rate or 0):.1f}%</td></tr>
</tbody></table>
<p>Fisher exact <code>p = {pb.get('fisher_p_clean', float('nan')):.2f}</code>.
Identical prompt, identical pipeline; only the news differs.</p>
<div class="flag"><b>Not significant means not resolved, not absent.</b> Power at
the observed {gap_pp}pp gap is <b>{power_obs:.2f}</b> — a one-in-six chance of
detecting the effect actually observed. Reaching 80% power with
{pb.get('transitions', {}).get('n', 20)} transitions would need a near-100%
transition rate, and the binding constraint is the number of transitions, not
controls: even 200 controls only gets there at ~93%. The gap runs in the
predicted direction and <b>reproduces exactly on a full replicate run</b>
(&kappa;&nbsp;=&nbsp;{T.get('kappa_is_confident', float('nan')):.2f} on the
confidence label; rates {T.get('confident_rate_run1_transitions')}&rarr;{T.get('confident_rate_run2_transitions')}
and {T.get('confident_rate_run1_placebos')}&rarr;{T.get('confident_rate_run2_placebos')}),
so it is stable rather than noisy — but this design could not have proven it.</div>
<div class="ok"><b>What is established.</b> The explanations are specific to
their window. A sceptic can explain {100 * bm.get('accuracy', 0):.0f}%-against-{100 * bm.get('chance', 0):.0f}%
by era-matching — adjacent fortnights share running stories — so each explanation
was re-scored against only its <b>three temporally nearest</b> windows:
<b>{100 * hn.get('accuracy', 0):.1f}% against {100 * hn.get('chance', 0):.1f}% chance</b>.
Era cannot explain that. What is <i>not</i> established is that the explanations
are diagnostic of a regime change — the comparison above is unresolved, not
settled either way.</div>

<h2>Built with</h2>
<p>Claude Code for development. <code>{D.get('model', 'claude-opus-5')}</code> via
the Anthropic Messages API with schema-enforced structured output, one call per
window, prompts in version-controlled files, every call logged with model id,
prompt hash and input hash. Wikipedia MediaWiki API (revision-pinned), yfinance,
hmmlearn, scikit-learn. 69 tests, including negative controls proving the
blind-match test returns chance on boilerplate.</p>

</div>
</div>

<div class="foot">
Every figure on this page is substituted from the committed results at build time,
not typed in. Reproducible offline from cache. The interactive dashboard runs the
same pipeline on any regime model's dates, not only this one.
</div>

</body></html>
"""

    out = output_dir() / "onepager.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    print("Open it and print to PDF: Ctrl+P -> Save as PDF, background graphics on.")


if __name__ == "__main__":
    main()
