# Status — 2026-08-22

The study is complete. Every number in the report, the one-pager and the
dashboard is computed from committed code and reproducible offline from cache.

## Results

**Regime separation.** 18 of the 20 folds that contain both states have a
volatility ratio above 1.0 — exact sign test *p* = 0.00040, median 1.52×
(95% CI 1.37–1.90). The pooled figure is 2.18× [1.76, 2.69] but mixes calm-era
and crisis-era days, so the per-fold result is the one to defend. Forward-20d
(non-circular) 1.69×. In-sample, reproducing the prior notebook, 2.32×.

**Generalisation.** Refitted unchanged on three indices outside the original
universe: forward-vol ratios 1.18× (Nikkei), 1.32× (Hang Seng), 1.58× (DAX).
Nikkei is close to nothing and is reported as such.

**Transitions.** 84 raw state flips → 24 after a 20-day dwell filter → 20 that
recur in ≥60% of ten random seeds.

**News.** 60 windows, 10,888 items, zero retrieval failures, zero post-boundary
survivors. Wikipedia Current Events pinned by revision id; a mean of 38% of the
current page was written after the boundary, peaking at 71%.

**Blind matching.** 55% against 5% chance (*p* < 0.0001); 71.7% against 33.3%
when each explanation competes only with its three temporally nearest windows,
so era cannot account for it; 28.6% against 7.1% on business-only text
(*p* = 0.016).

**Faithfulness.** 96.4% of 474 claims grounded in the item they cite, against a
1.8% random-citation floor — a 54× separation. Zero fabricated citations.

**Placebo — the honest limit.** Transitions 80.0%, clean controls 61.5%:
+18.5pp, 95% CI [−12, +47]pp, Fisher *p* = 0.43. Power at the observed gap is
0.17. Stable on replication (κ = 0.84; rates reproduce exactly). Unresolved, not
null.

## Deliverables

| | |
|---|---|
| Shareable report | https://claude.ai/code/artifact/60222501-c141-4976-ac6a-6db30a74bd96 |
| Local report | `outputs/report.html` (self-contained, 949 KB) |
| One-page PDF | `outputs/onepager.html` → Ctrl+P → Save as PDF |
| Dashboard | `streamlit run app.py` — six views |
| Video script | `docs/VIDEO_SCRIPT.md` |
| Referee report | `docs/REFEREE_REPORT.md` |
| Tests | 69 passing |

## Running it

```bash
python scripts/01_regimes.py && python scripts/02_fetch_news.py && python scripts/03_ceiling_test.py && python scripts/05_report.py
```

Stages 04, 06, 08 and 10 need `ANTHROPIC_API_KEY`; all four are cached, so a
rerun is free and instant. `python scripts/04_narratives.py --dry-run` renders
every prompt without calling anything.

## Known limitations, stated rather than discovered

1. **The placebo comparison is unresolved.** The design could only have detected
   a gap of ~50pp. Fixing it needs more transitions, not more controls —
   pre-specifying calm→stressed onsets at a 5-day dwell threshold would give
   ~24 onsets and raise power from 0.25 to ~0.73.

2. **A same-day outcome path survives revision pinning.** The boundary is 23:59
   UTC on the transition day; US equities close at 20:00–21:00 UTC, and
   Wikipedia reports market moves the same evening. So a window's final day can
   describe the very move that defined the label. Mitigating: transitions carry
   *less* of this than controls (30% vs 47.5%), so it does not manufacture the
   placebo gap.

3. **Grounding is lexical.** It establishes that a claim's content came from the
   item it cites, not that the claim is true. Numerals, negation and causal
   direction are invisible to it.

4. **Three global equity indices are not three independent replications.** They
   share one global volatility factor over the same 2015/2018/2020/2022 window.

5. **Outputs are not bit-reproducible.** Sampling parameters were removed from
   the API. Reproducibility comes from the archive — every call cached by input
   hash and logged with model id, prompt hash and token counts — not from
   re-execution.

## Not done

- Recording the video
- Printing `outputs/onepager.html` to PDF
- A `narrative_v2` prompt (the header leaks item count; see `PROMPT_HISTORY.md`)
