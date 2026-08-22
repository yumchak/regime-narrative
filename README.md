# regime-narrative

A statistical regime detector with a language explanation layer, and a measured
answer to whether the explanations mean anything.

A regime model tells you that the market changed. It cannot tell you what
happened, because it never sees anything except returns. So every regime model's
output is a chart of coloured bands that a human interprets from memory — and
that interpretation is undocumented, unauditable, and different for every person
who looks at it.

This project attaches an explanation to each detected transition, generated from
news that closes on the transition date, and then measures how much that
explanation is worth.

---

## The claim

> A two-state Gaussian HMM, refitted inside each of 27 walk-forward folds and
> scored only on held-out days, separates SPY volatility regimes by **2.18×**
> (95% CI 1.76–2.69); the same procedure refitted on three indices outside the
> original universe gives 1.54×, 1.46× and 1.87×. A 20-day persistence filter
> reduces 84 raw state flips to 24 transitions, of which **20 recur across
> random seeds**. For each, a language model reading only news published before
> that date produced a citation-grounded explanation. Stripped of dates and
> scrambled, *X*% matched back to the correct window against 1/20 chance. On
> era-matched non-transition dates the same pipeline produced confident
> explanations *Y*% of the time, and that is the honest limit of the method.

*X* and *Y* require `ANTHROPIC_API_KEY`. Everything before them is computed and
reproducible offline.

## The architectural split

The HMM decides **when** the regime changed. The language model only describes
**what was in the news** at that moment. No number in the statistical results
comes from the language model, and it never influences a regime boundary.

---

## What makes the result defensible

**The volatility ratio is genuinely out-of-sample.** The HMM is refitted in every
fold on training data only, the scaler is fitted inside the fold, and states are
assigned by a **forward filter** — `P(state_t | observations up to t)`. The usual
`predict()` runs Viterbi over the whole sequence, so a day's label can depend on
days that came *after* it. That would contaminate the transition dates, and the
entire news control rests on those dates having been knowable at the time.

**The retrieval boundary is structural, not conventional.** A `NewsWindow`
refuses to construct if any item post-dates its boundary. 26 tests enforce it,
including timezone edge cases and a property sweep across window lengths.

**Wikipedia pages are pinned to revisions.** The page for a given day keeps being
edited for years. Across sampled transition dates, a mean of **38%** of the
current page was written after the boundary — peaking at **71%** for 26 January
2018, whose latest edit is from 2025. Fetching the live page would feed the model
text written with full knowledge of what followed, invisibly.

**The blind-match test can fail.** Its negative controls are in the test suite:
handed boilerplate it returns chance, and a single lucky hit is not significant.

**Controls are era-matched and mutually disjoint.** Each transition is paired
with control dates 30–90 days away — same news environment, same era (year-mean
gap 0.00) — and controls are disjoint from each other, not only from transitions.

---

## Layout

```
settings.yaml                 every tunable, including the model name
prompts/                      version-controlled prompts, never inline strings
src/regime_narrative/
  config.py                   settings loader
  data.py                     yfinance with caching and a manifest
  features.py                 the four features; forward realised vol
  hmm_model.py                walk-forward refit, forward filter
  regime_stats.py             vol separation, block-bootstrap CI, per-fold
  transitions.py              persistence filter, dwell, seed stability
  narrative.py                the language layer, fully logged
  news/
    base.py                   NewsItem, NewsWindow, the boundary invariant
    wikipedia.py              revision-pinned Current Events retrieval
  controls/
    placebo.py                era-matched control dates
    blind_match.py            BM25, permutation test, date scrubbing
    faithfulness.py           citation grounding
  viz.py, report.py           the chart and the HTML report
scripts/01..05                the pipeline, in order
tests/                        47 tests; the boundary and the negative controls
```

## Running it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

```bash
python scripts/01_regimes.py && python scripts/02_fetch_news.py && python scripts/03_ceiling_test.py && python scripts/05_report.py
```

Stage 4 needs a key. To see exactly what would be sent, and what it would cost,
without spending anything:

```bash
python scripts/04_narratives.py --dry-run
```

## Deliberately out of scope

Not a trading strategy, backtest, allocator, sentiment model or forecaster. The
prior work's XGBoost direction classifier was dropped: it is a forecaster, it
lost to the naive baseline (56.9% against 61.4%), and its regime feature came
from an HMM fitted over the same days it was tested on. Removing it costs the
claim nothing and removes three separate objections.
