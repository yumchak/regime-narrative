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
    nikkei_fwd = next((g["ratio_forward"] for g in R["generalisation"]
                       if g.get("index") == "nikkei"), float("nan"))
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
separates SPY volatility regimes in <b>{sg.get('n_folds_ratio_above_1')} of {sg.get('n_folds_with_both_states')} folds</b>
containing both states (sign test p&nbsp;=&nbsp;{sg.get('sign_test_p', 0):.5f}). Reading only news
published <i>before</i> each of {stab['n_stable_transitions']} transitions, a language model produced
cited explanations matching back to their own fortnight
<b>{100 * hn.get('accuracy', 0):.0f}% of the time against {100 * hn.get('chance', 0):.0f}% chance</b>, with
<b>{100 * fa.get('grounding_rate', 0):.1f}% of {fa.get('n_claims', 0)} claims grounded, zero fabricated</b>. Whether they
are <i>diagnostic</i> is unresolved: <b>+{pi.get('difference_pp')}pp, 95% CI
[{ci[0]:.0f},&nbsp;{ci[1]:.0f}]pp</b>.
</div>

<div class="cols">
<div class="col">

<h2>Problem</h2>
<p>A regime model tells you the market changed but not what happened, because it
only ever sees returns. Its output is coloured bands a human interprets from
memory &mdash; undocumented, unauditable, different for everyone who looks.</p>

<h2>Solution</h2>
<p><b>The HMM decides <i>when</i> the regime changed; the language model only
describes <i>what was in the news</i> then.</b> No statistic comes from the
language model. For each date the tool retrieves a 14-day window that
<i>provably</i> closes on it, and returns an explanation whose every claim cites
a supplied article.</p>

<h2>Evidence the detector works</h2>
<table>
<tr><th>Stressed / calm volatility</th><th class="n">Same&#8209;day</th><th class="n">Forward&nbsp;20d</th></tr>
<tr class="hi"><td>SPY <i>(fitted)</i></td><td class="n">{oos['ratio']:.2f}&times;</td><td class="n">{fwd['ratio']:.2f}&times;</td></tr>
{gen_rows}
</table>
<p>{oos['ratio']:.2f}&times; means stressed days move {oos['ratio']:.2f} times as much as calm
days. Same-day is partly definitional &mdash; the HMM is fed trailing volatility;
<b>forward-20d</b> is volatility that had <i>not happened</i> when the state was
assigned, and is the honest number. Nikkei at {nikkei_fwd:.2f}&times; is close to nothing
and is reported as such.</p>

<h2>Impact &amp; value</h2>
<p>A desk with thirty portfolio managers gets thirty private readings of one
regime chart. None is written down, none is auditable, and the reasoning leaves
with whoever remembers that week. This replaces &ldquo;trust me, that band is the
Greek referendum&rdquo; with a written explanation, citing dated sources,
carrying a number that says how specific it is.</p>
<p><b>The reusable asset is the controls, not the explanations.</b> Anyone can
ask a model what happened in a given month; nobody can otherwise tell you
whether the answer is specific to that fortnight or commentary fitting any
month. Point this at any regime model's dates &mdash; any asset, any method
&mdash; and each explanation returns with a blind-match score against chance, a
grounding rate against a random-citation floor, and a matched control arm.</p>

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
<li><b>Placebo.</b> Era-matched controls, identical prompt.</li>
</ul>

<div class="flag"><b>The honest bound on that value.</b> The placebo arm says
these explanations are not yet shown to be <i>diagnostic</i> of a regime change.
So what this buys today is auditability, not signal generation: an undocumented
interpretation becomes a documented one, with a figure attached saying how far
to trust it. A smaller claim than the obvious one, and unlike the obvious one it
is defensible.</div>

<h2>Reflections</h2>
<p><b>What worked:</b> writing the controls before generating anything,
including the negative controls that prove the blind-match test can fail on
boilerplate. Making the retrieval boundary structural rather than a convention.</p>
<p><b>What I would change:</b> the placebo test was doomed by design &mdash;
power at the observed gap is {pw:.2f}, and the binding constraint is
{pb.get('transitions', {}).get('n', 20)} transitions, not the number of controls, so more placebos
could never have helped. Pre-specifying volatility <i>onsets</i> at a shorter
dwell threshold would roughly triple it. One leak also stays open: the boundary
is 23:59&nbsp;UTC and Wikipedia reports the US close the same evening, though
controls carry <i>more</i> of it than transitions, so it does not manufacture
the result.</p>
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
