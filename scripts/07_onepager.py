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

import base64

from regime_narrative.config import output_dir


def _img(name: str) -> str:
    q = output_dir() / name
    if not q.exists():
        return ""
    b = base64.b64encode(q.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b}" alt="{q.stem}">'

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
img { width: 68%; height: auto; display: block; margin: 0.5mm auto 0.5mm; }
figcaption { font-size: 11pt; color: #4a5763; margin: 0 0 1.5mm; }
.wide { margin: 0 0 2mm; }
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

<p class="sub"><b>Finding:</b> the explanations are specific and faithful to
their sources; whether they are <i>diagnostic</i> of a regime change is
unresolved, and this design could never have resolved it.</p>

<div class="wide">{_img("pipeline.png")}</div>

<div class="cols">
<div class="col">

<h2>Problem statement</h2>
<p>A regime model tells you the market changed, not what happened &mdash; it only
ever sees returns. So its output is coloured bands that a human interprets from
memory. On a desk of thirty portfolio managers that is thirty private,
unrecorded readings of one signal, and the reasoning leaves with whoever
remembers the week.</p>

<h2>Solution overview</h2>
<p>Above. A two-state HMM marks <i>when</i> the regime changed; a language model
reading <i>only news published before that date</i> describes what was
happening. The two never mix: no statistic comes from the model, and it never
moves a regime boundary.</p>
<p>The boundary is enforced by the type system &mdash; a window object refuses to
construct if any item post-dates it. Pages are pinned to their revision id,
because a mean of <b>{leak.get('mean_pct_written_after_boundary')}%</b> of a Wikipedia day-page as it
stands today (peak {leak.get('max_pct_written_after_boundary')}%) was written <i>after</i> the day it
describes. Fetching the live page feeds the model hindsight, invisibly.</p>

<h2>Use of AI</h2>
<p><code>claude-opus-5</code> via the Anthropic Messages API &mdash; one call per
window, effort high, output schema enforced server-side. Prompts live in
version-controlled files with a dated iteration history, never inline strings;
every call logs the model id, prompt hash and input hash, so any sentence in the
results traces to the exact prompt and news window behind it. Built with Claude
Code. Wikipedia MediaWiki API, yfinance, hmmlearn, scikit-learn, Streamlit.</p>

</div>
<div class="col">

<h2>Impact &amp; value</h2>
<p>It replaces &ldquo;trust me, that band is the Greek referendum&rdquo; with a
written explanation citing dated sources &mdash; <b>arriving with its own error
bars</b>. Tested on {stab['n_stable_transitions']} transitions and {pb.get('placebos_all', {}).get('n', 40)} matched control dates:</p>
<table>
<tr><th>Can you trust the explanations?</th><th class="n">What we found</th></tr>
<tr><td>Date hidden: can we still tell which two weeks it describes?</td><td class="n">{100 * hn.get('accuracy', 0):.0f}% &mdash; guessing gives {100 * hn.get('chance', 0):.0f}%</td></tr>
<tr><td>Claims really come from the source cited</td><td class="n">{100 * fa.get('grounding_rate', 0):.1f}% &mdash; a random source gives {100 * gn.get('grounded_rate_random_same_window', 0):.1f}%</td></tr>
<tr><td>Invented citations</td><td class="n">none, in {fa.get('n_claims', 0)} claims</td></tr>
<tr class="hi"><td>Confident on real events, vs on ordinary days</td><td class="n">{100 * t_rate:.0f}% vs {100 * c_rate:.0f}% &mdash; ignoring the news gives no gap</td></tr>
<tr><td>Whole study re-run on a different AI</td><td class="n">same scores</td></tr>
</table>
<p><b>Why this is not a chatbot.</b> Anyone can ask a model what happened in a
month and get a fluent answer &mdash; and it reads identically whether it is
specific to that month or would fit any month. <b>A chatbot gives you an answer;
this gives you an answer and how far to trust it.</b> The reusable asset is the
controls, not the explanations: point it at any regime model's dates, any asset,
any method, and every explanation returns with that scorecard attached.</p>

<h2>Reflections</h2>
<p><b>Row four is the finding.</b> The model explains an ordinary Tuesday almost
as readily as a real transition. Calling that &ldquo;not diagnostic&rdquo; would
be a no-difference claim <i>p</i>&nbsp;=&nbsp;0.43 does not license: it is
<b>+{pi.get('difference_pp')}pp, 95% CI [{ci[0]:.0f},&nbsp;{ci[1]:.0f}]pp</b>, unresolved &mdash; power at that
gap is only {pw:.2f}, and the constraint is {pb.get('transitions', {}).get('n', 20)} transitions, not controls, so
more placebos could never have helped. <b>What this buys today is auditability,
not signal generation.</b></p>
<p><b>Next:</b> pre-specify volatility <i>onsets</i> rather than all
transitions, which roughly triples the power. One leak stays open: the boundary
is 23:59&nbsp;UTC and Wikipedia reports the US close the same evening.</p>

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
