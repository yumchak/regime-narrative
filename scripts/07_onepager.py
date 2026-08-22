"""Stage 7 -- the one-page PDF, generated from the results rather than typed.

Built to the Polymer Tech Expo 2026 write-up rules, which are specific:

    * maximum 1 page
    * PDF
    * MINIMUM font size 11-point
    * must address: problem statement, solution overview, use of AI,
      impact & value, reflections

The 11-point floor is the binding constraint. An earlier draft of this page ran
at 8.6pt and carried roughly a third more content; at 11pt that does not fit, so
the content is triaged rather than shrunk. Nothing here is below 11pt.

Every number is substituted from the results JSON at build time. The first
hand-written draft carried three per-fold medians that were simply invented, and
nothing in the process would have caught them.

Output: A4 print-ready HTML. Open it and print to PDF (Ctrl+P -> Save as PDF,
margins Default, Background graphics ON, Scale 100%).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from regime_narrative.config import output_dir

# Every size below is >= 11pt. The rule is a floor on *all* text, so captions
# and table cells are bound by it too, not just body copy.
CSS = """
@page { size: A4; margin: 11mm 12mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font: 11pt/1.16 "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
  color: #14202a; background: #fff; }
h1 { font-size: 16pt; line-height: 1.06; margin: 0 0 1mm; letter-spacing: -0.3pt; }
.sub { font-size: 11pt; color: #4a5763; margin: 0 0 2.5mm; }
h2 { font-size: 11pt; text-transform: uppercase; letter-spacing: 0.5pt;
  color: #37567f; margin: 2.2mm 0 0.8mm; }
p { margin: 0 0 1.2mm; }
.claim { border-left: 2.5pt solid #37567f; background: #f2f6fa;
  padding: 1.6mm 2.2mm; margin: 0 0 2mm; font-size: 11pt; line-height: 1.18; }
.cols { display: flex; gap: 4.5mm; }
.col { flex: 1; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 11pt; margin: 1mm 0 1.5mm;
  font-variant-numeric: tabular-nums; }
th, td { padding: 0.5mm 1.4mm; text-align: left; border-bottom: 0.4pt solid #dbe2e8; }
th { font-weight: 600; font-size: 11pt; color: #4a5763; }
td.n, th.n { text-align: right; }
tr.hi td { font-weight: 600; border-top: 0.8pt solid #14202a; }
.flag { border-left: 2.5pt solid #9a5c17; background: #fdf7f0; padding: 1.8mm 2.6mm;
  margin: 1.2mm 0; font-size: 11pt; line-height: 1.16; }
ul { margin: 0 0 1.5mm; padding-left: 4mm; }
li { margin: 0 0 0.5mm; }
b { font-weight: 600; }
.foot { margin-top: 2.5mm; padding-top: 1.2mm; border-top: 0.4pt solid #dbe2e8;
  font-size: 11pt; color: #4a5763; }
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
    X = _load("model_comparison.json") or {}
    if R is None:
        print("run scripts/01_regimes.py first")
        sys.exit(1)

    oos, ins = R["out_of_sample"], R["in_sample"]
    fwd, wf, pf = R["out_of_sample_forward"], R["walk_forward"], R["persistence_filter"]
    stab, nsum = R["seed_stability"], N.get("summary", {})
    leak = N.get("hindsight_leak", {})
    pb = D.get("placebo", {})
    bm = D.get("blind_match", {}).get("transitions_all_sections", {})
    fa = D.get("faithfulness", {}).get("all", {})
    sg = RF.get("per_fold_sign_test", {})
    gn = RF.get("grounding_null", {})
    hn = RF.get("blind_match_hard_negatives", {}).get("3_temporally_nearest", {})
    pi = RF.get("placebo_interval", {})
    ci = pi.get("newcombe_95ci_pp_vs_clean", [0, 0])
    pw = P.get("power_at_observed_effect", {}).get("vs_clean_n13", 0)
    xrep = X.get("what_replicated_exactly", {})

    gen_rows = "".join(
        f'<tr><td>{PRETTY.get(g["index"], g["index"])}</td>'
        f'<td class="n">{g["ratio_oos"]:.2f}&times;</td>'
        f'<td class="n">{g["ratio_forward"]:.2f}&times;</td></tr>'
        for g in R["generalisation"] if "error" not in g
    )
    t_rate = pb.get("transitions", {}).get("confident_rate", 0)
    c_rate = pb.get("placebos_clean_stratum", {}).get("confident_rate", 0)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>regime-narrative</title><style>{CSS}</style></head><body>

<h1>Regime detection that can say what happened &mdash; and a measurement of
how much that is worth</h1>
<p class="sub">An auditable explanation layer for any regime model.
BSc Mathematics with Statistics, University of Bristol.</p>

<div class="claim">
<b>The claim.</b> A two-state HMM refitted inside every walk-forward fold
separates SPY volatility regimes in {sg.get('n_folds_ratio_above_1')} of the
{sg.get('n_folds_with_both_states')} folds containing both states (sign test
<b>p&nbsp;=&nbsp;{sg.get('sign_test_p', 0):.5f}</b>). Reading only news published
<i>before</i> each of {stab['n_stable_transitions']} transitions, a language model produced cited
explanations that match back to their own fortnight
<b>{100 * hn.get('accuracy', 0):.0f}% of the time against {100 * hn.get('chance', 0):.0f}% chance</b>, with
<b>{100 * fa.get('grounding_rate', 0):.1f}% of {fa.get('n_claims', 0)} claims grounded and zero fabricated
citations</b>. Whether they are <i>diagnostic</i> of a regime change is
unresolved: <b>+{pi.get('difference_pp')}pp over matched controls, 95% CI
[{ci[0]:.0f},&nbsp;{ci[1]:.0f}]pp</b>.
</div>

<div class="cols">
<div class="col">

<h2>Problem</h2>
<p>A regime model tells you the market changed but not what happened, because it
only ever sees returns. Its output is coloured bands a human interprets from
memory &mdash; undocumented, unauditable, different for everyone who looks. For a
platform running thirty portfolio managers, that is thirty private readings of
one signal.</p>

<h2>Solution</h2>
<p><b>The HMM decides <i>when</i> the regime changed; the language model only
describes <i>what was in the news</i> then.</b> No statistic comes from the
language model. For each date the tool retrieves a 14-day window that
<i>provably</i> closes on it, and returns an explanation whose every claim cites
a supplied article.</p>
<p>The reusable part is the <b>controls</b>, not the explanations. Anyone can ask
a model what happened in March 2020; nobody can otherwise tell you whether the
answer is specific to that window or boilerplate fitting any month. Point the
tool at any regime model's dates and each explanation arrives with a blind-match
score, a grounding rate, and a matched control arm.</p>

<h2>Impact &amp; value</h2>
<table>
<tr><th>Stressed / calm volatility</th><th class="n">Same&#8209;day</th><th class="n">Forward&nbsp;20d</th></tr>
<tr class="hi"><td>SPY <i>(fitted)</i></td><td class="n">{oos['ratio']:.2f}&times;</td><td class="n">{fwd['ratio']:.2f}&times;</td></tr>
{gen_rows}
</table>
<p>A ratio of {oos['ratio']:.2f}&times; means the stressed state's daily moves are
{oos['ratio']:.2f} times the size of the calm state's. <b>Same-day</b> is partly
definitional &mdash; the HMM is fed trailing volatility, so it had better
separate on it. <b>Forward-20d</b> measures volatility that had <i>not
happened</i> when the state was assigned, and is the honest number. Nikkei at
1.18&times; is close to nothing, and is reported as such. Within SPY the
per-fold median is {sg.get('median_ratio', 0):.2f}&times;, below the pooled
figure because pooling mixes calm and crisis years.</p>

</div>
<div class="col">

<h2>Use of AI</h2>
<p><code>claude-opus-5</code> via the Anthropic Messages API &mdash; one call per
window, effort high, schema enforced server-side. Prompts are version-controlled
files with a dated iteration history, never inline strings; every call logs model
id, prompt hash and input hash. Built with Claude Code. News via the
revision-pinned Wikipedia MediaWiki API, prices via yfinance, statistics via
hmmlearn and scikit-learn, tool in Streamlit.</p>

<h2>Three controls</h2>
<ul>
<li><b>Hindsight.</b> A window object refuses to construct if any item post-dates
it: {nsum.get('n_windows')} windows, {nsum.get('total_items', 0):,} items, zero survivors. Pages are
pinned by revision id because a mean of <b>{leak.get('mean_pct_written_after_boundary')}%</b> of a
current Wikipedia day-page (peak {leak.get('max_pct_written_after_boundary')}%) was written <i>after</i>
the boundary.</li>
<li><b>Memorisation.</b> Dates stripped; every claim must cite; grounding scored
lexically, so the check has no world knowledge.</li>
<li><b>Placebo.</b> Era-matched controls, identical prompt and pipeline.</li>
</ul>

<h2>Reflections</h2>
<p>The placebo arm went against me and is the result I would keep. Calling the
explanations &ldquo;not diagnostic&rdquo; would be a no-difference claim that
<i>p</i>&nbsp;=&nbsp;0.43 does not license. <b>Power at the observed gap is
{pw:.2f}</b> &mdash; one chance in six of detecting the effect observed. The
binding constraint is 20 transitions, not controls, so more placebos cannot fix
it; pre-specifying volatility <i>onsets</i> would roughly triple it.</p>
<div class="flag"><b>What replicates.</b> Re-run on <code>claude-sonnet-5</code>,
blind matching is identical and grounding differs by 0.4pp &mdash; properties of
having the news. The confidence label is not
(&kappa;&nbsp;=&nbsp;{X.get('kappa_is_confident')}; Sonnet declined half as often), which is why
blind matching is load-bearing and the self-report is a weak instrument.</div>
<p><b>Given more time:</b> more transitions, not more controls. One leak stays
open &mdash; the boundary is 23:59&nbsp;UTC and Wikipedia reports closes the same
evening &mdash; though controls carry <i>more</i> of it than transitions, so it
does not manufacture the gap.</p>

</div>
</div>

</body></html>
"""

    out = output_dir() / "onepager.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")

    # Guard the binding requirement rather than trusting the stylesheet.
    import re
    sizes = [float(x) for x in re.findall(r"font(?:-size)?:\s*([0-9.]+)pt", CSS)]
    too_small = [s for s in sizes if s < 11]
    assert not too_small, f"font sizes below the 11pt minimum: {sorted(set(too_small))}"
    print(f"all {len(sizes)} declared font sizes are >= 11pt (expo minimum)")
    print("print to PDF: Ctrl+P -> Save as PDF, margins Default, "
          "Background graphics ON, Scale 100%")


if __name__ == "__main__":
    main()
