# Status — end of overnight session, 2026-08-22

Day 1 of the three-day plan is complete, and part of Day 2. One thing is
blocked on you.

## Blocked on you

**`ANTHROPIC_API_KEY` is not set.** It is the only thing standing between the
current state and a finished set of results. Everything else runs offline.

```bash
export ANTHROPIC_API_KEY=sk-...
python scripts/04_narratives.py && python scripts/05_report.py
```

60 calls, ~320k input tokens total, `claude-opus-5` at effort `high`. To see the
exact prompts without spending anything: `python scripts/04_narratives.py --dry-run`
(they are already rendered to `outputs/dry_run/`).

No other credential is needed. Wikipedia and yfinance are keyless; GDELT was
evaluated and rejected.

## Done

| | |
|---|---|
| Environment | venv on Python 3.12 (3.14 has no `hmmlearn`/`xgboost` wheels) |
| Regime spine | HMM refit per fold, scaler per fold, forward-filtered states |
| Generalisation | Nikkei, Hang Seng, DAX |
| Transitions | 84 raw flips → 24 filtered → 20 seed-stable |
| News | 60 windows, 10,888 items, 0 failures, revision-pinned |
| Controls | Boundary enforced structurally; 40 era-matched controls; blind-match machinery with negative controls |
| Tests | 69 passing |
| Outputs | `report.html` (self-contained), 3 figures, 4 JSON result files |

## Headline numbers

- In-sample ratio **2.32×** (reproduces the notebook's 2.33×)
- Out-of-sample ratio **2.18×**, 95% CI [1.76, 2.69]
- Forward-20d ratio **1.69×** — the non-circular version
- Per-fold median **1.52×** — the conservative version
- Unseen indices: Nikkei **1.54×**, Hang Seng **1.46×**, DAX **1.87×**
- Hindsight leak in unpinned Wikipedia: mean **38%**, max **71%** of page written after the boundary

## Three findings that change what you should claim

**1. The original claim described an experiment that was never run.** The 2.33×
was in-sample over the full history; the 26 folds belonged to the XGBoost
forecaster. Refit per fold it becomes 2.18×. Never restate 2.33× as
out-of-sample.

**2. Pooled ratio > per-fold median (2.18× vs 1.52×).** Pooling mixes calm-era
and crisis-era days, so part of the pooled figure is a between-period effect
rather than within-period separation. Lead with 2.18× but have 1.52× ready; a
sharp judge will ask, and having the answer first is worth more than the larger
number.

**3. The blind-matching ceiling is carried by generic world news.** All sections
match at 95%; business-only at 16.7% against 8.3% chance (p=0.27). Partly a
power effect — business holdouts have 183 words against 2,772 — but either way, a
null result on the real test would be ambiguous between "explanations are
generic" and "the matchable signal is not where the explanation looks". Both arms
are reported. **This is the most important caveat in the project and it should be
in the video, not buried.**

Also: at realistic explanation lengths (120–200 words) the ceiling is 40–70%, not
95%. Expect that range. Significance survives even at 60 words, so the test is
well powered.

## Two bugs found and fixed

- **Placebo overlap.** Two controls were sampled seven days apart, sharing
  thirteen of fourteen days of news. The ceiling test caught it by predicting one
  from the other. Then a second, subtler version: the two picks for a single
  transition were filtered before either was recorded, so the second never saw
  the first — that produced 2019-05-30 and 2019-05-31. Both fixed, regression
  tests added, minimum gap now 21 days.
- **Month names leaked into prompts.** The date scrubber caught "March 3" and
  "3 March" but not a bare "in March". Nine leaked into the first rendered
  prompt. Fixed, with `audit_date_leaks` now reporting zero across all windows.

## Decisions I made that you should overrule if you disagree

1. **Dropped the XGBoost layer.** It is a forecaster (out of scope by your own
   rules), it lost to the naive baseline, and its regime feature came from an HMM
   fitted over the same days it was tested on.
2. **Transitions come from the out-of-sample filtered sequence, not a
   full-sample fit.** Otherwise the transition date itself is contaminated by
   hindsight and the news control is undermined at its root.
3. **Forward filter instead of Viterbi.** `predict()` smooths over the whole
   test block, so a day's label could depend on up to 126 days of future data.
4. **All Wikipedia sections, not just business.** Restricting to business would
   pre-select for market relevance and make the placebo test easier to pass.
5. **Two controls per transition** rather than one, for power and to build a
   usable clean stratum.

## Not done

- Stage 4 and everything downstream of it (needs the key)
- The one-page PDF (Day 3; depends on final numbers)
- Video script and prompt-iteration reflections
