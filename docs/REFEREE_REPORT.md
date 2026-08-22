## Referee report — ranked by how much damage each finding does to the claims as stated

---

### 1. The headline negative is stated as a finding about the world. It is a finding about the sample size. (Most damaging, and it is a wording problem, not a data problem.)

**The attack.** The text says the explanations "are NOT DIAGNOSTIC of a regime transition" and "the model finds a confident story for an ordinary date almost as often as for a real one." That is an affirmative claim of no-difference. Fisher p=0.43 licenses no such claim. I computed the exact Fisher power surface (`scratchpad/pwr.py`, enumeration over all (x₁,x₀), α=0.05 two-sided):

- Power at the observed effect, 20 vs 13: **0.162**. 20 vs 40: **0.172**. (Your `outputs/placebo_power.json` says 0.166/0.163 — confirmed.)
- **Minimum detectable difference at 80% power, holding transitions at 80%: the control rate must fall to ≤29.5%, i.e. a gap of ≥50.5pp.** At 20 vs 40 it is ≥39.5pp. Even with *400* controls it is ≥31.5pp, because the binding constraint is n_transitions=20.
- Newcombe 95% CI on the risk difference: **[−11.5, +47.1] pp**. The data are compatible with the transition arm being 47 points better. Odds ratio 2.50.
- Jeffreys posterior: **P(transition rate > control rate) = 0.878**; P(true gap > 10pp) = 0.693; P(gap < 0) = 0.122.

**Damage: severe to the interpretation, zero to the numbers.** A judge who knows statistics will read "not diagnostic" as an unlicensed acceptance of the null and will discount everything else you say about rigour, because it is the one place you claimed more than you measured — in the humble direction, which is still overclaiming.

**Strongest honest defence.** You already computed this and wrote the correct sentence in `placebo_power.json`: *"underpowered, difference not resolved, NOT no difference exists."* The defence is that the correct reading exists in your own output; only the summary prose contradicts it.

**Fix.** Replace "not diagnostic" with the interval. One line: *"80% vs 61.5%, difference +18.5pp, 95% CI [−12, +47]pp. This study could only have resolved a gap of 50pp or more. It did not resolve this one."* Then state the design consequence: at the observed rates a 1:2 design needs **~80 transitions and 160 controls for 80% power** (I get power 0.836 at n=80/160, 0.248 at the current 20/40).

---

### 2. The confidence outcome is driven by a covariate that is not the regime label, and adjusting for it moves the estimate the *other* way.

**The attack.** I coded each window for whether any item in its **final two days** uses market vocabulary (stock/index/futures/sell-off/plunge/rally/volatility/…). Result (`scratchpad/adj2.py`, n=60):

- Market language present → **21/25 = 84% confident**; absent → **21/35 = 60%**. Fisher **p=0.052**.
- Logistic `confident ~ transition + market_language`: market_language **OR 4.29 [1.15, 15.99], p=0.030**; transition OR 2.90 [0.76, 11.15], p=0.121.
- Transitions carry *less* of this language than controls (6/20 = 30% vs 19/40 = 47.5%), so it is a **negative confounder**. Mantel–Haenszel adjusted OR **3.04** vs crude **2.15**.

So the single strongest predictor of the model saying "high/medium confidence" is not whether the HMM flagged a change — it is whether the fortnight happened to contain a story phrased in market vocabulary. And because that vocabulary is *anti*-correlated with your transitions, your unadjusted 18.5pp is biased **toward** the null.

**Damage: high, and double-edged.** It undermines "is_confident" as a measure of transition-detection (it partly measures news vocabulary), while simultaneously showing the reported gap understates the adjusted one.

**Defence.** This covariate was chosen by me after seeing the data; p=0.030 is not corrected for my own search, and n=60 splits into strata of 25 and 35. It is exploratory. Also, the direction of the bias is not something you engineered — it argues in your favour and you did not claim it.

**Fix.** Pre-specify boundary-window market-vocabulary presence as a stratification variable and report the CMH-adjusted OR alongside the crude one. It costs nothing and it is defensible because the covariate is measurable from the news alone, with no knowledge of the label. And check the deeper problem it exposes: an item saying *"U.S. futures markets fall by 650 points as the UK votes to leave"* is sitting inside `placebo_2016-06-28` — your news source reports market outcomes, so the outcome is partly in the input.

---

### 3. Half your transitions are the wrong direction for the hypothesis, which halves the effect you were trying to detect.

**The attack.** Of the 20 seed-stable transitions, exactly **10 are stressed→calm** and 10 calm→stressed. Volatility *onset* should have a news driver; volatility *decay* is the absence of one. Pooling them tests a hypothesis nobody holds. Split:

- calm→stressed: **9/10 = 90%** confident. vs 8/13 clean controls, Fisher p=0.179 (one-sided ≈0.11).
- stressed→calm: **7/10 = 70%**. vs clean controls, p=1.00.

The pooled 80% is a 90% subgroup diluted by a 70% subgroup, and the theoretically-motivated half is the one that moves.

**Damage: high on design, moderate on interpretation.** With n=10 vs 13 the power is 0.247 even at a true 90/61.5 split, so this is not a rescue — it is an explanation of why the pooled test was doomed.

**Defence.** Direction is knowable before any narrative is generated, from the HMM alone, so pre-specifying it is not p-hacking. And you did not run this split and report the favourable half — I did.

**Fix, quantified.** Your `dwell_sensitivity` table shows min_dwell=10 yields 36 transitions and min_dwell=5 yields 47, against 24 at your chosen 20. Pre-specify **calm→stressed onsets at min_dwell=5** (~24 onsets, 48 controls): power at a true 0.90 vs 0.615 is **0.725**, against 0.247 today. That is the single change that turns this from an unpowered study into a nearly powered one, and the cost — noisier transition dates — is measurable via your existing seed-stability filter.

---

### 4. "p < 0.0001" on the blind match is defensible but understated, and the permutation null does not do what its docstring claims.

**The attack, part one.** `_permutation_p_value` returns (n_at_least+1)/(B+1) = 9.999e-05, i.e. **zero exceedances in 10,000**. Quoting "p < 0.0001" is the correct convention for a Monte Carlo floor. But the docstring in `blind_match.py` asserts the permutation is used because *"the candidate windows are not exchangeable… a binomial null would be too generous."* That is false as implemented: the scheme holds `predicted` fixed and permutes the true labels, which is the classical matching problem. I simulated it: **null mean = 0.994 matches, identical to Binomial(20, 1/20)**, with max 6 in 20,000 draws. It is marginally *more* conservative than binomial in the tail (Poisson(1) P(≥11)=1.0e-8 vs Binomial P(≥11)=5.4e-10) but it does not model window-to-window dependence at all.

**The attack, part two — the real one.** Adjacent windows share running stories, and your errors prove it: for the 60-window arm the median |true − predicted| gap is 420 days, and **12/27 errors land within 365 days** of the truth against a 21.0% random-pair baseline (6/27 within 120 days vs 8.1% baseline). Some of the 55% could be era-matching, not fortnight-matching. That is a materially weaker claim.

**Damage: moderate — and I could not make it stick.** I re-ran the test with hard negatives (`scratchpad/hard2.py`, reproducing your 33/60 exactly before modifying anything):

| Candidate pool | Result | Chance | p |
|---|---|---|---|
| all 60 (published) | 33/60 = 55.0% | 1.7% | <5e-5 |
| **3 temporally nearest windows** | **43/60 = 71.7%** | 33.3% | <5e-5 |
| 5 nearest | 37/60 = 61.7% | 20.0% | <5e-5 |
| **each transition vs its own 2 era-matched controls** | **15/20 = 75.0%** | 33.3% | 3e-4 |

The signal survives every era-matched restriction. **Report the 3-nearest and own-controls rows** — they are stronger evidence than 55%-vs-1.7% because they cannot be explained by era.

**Fix.** Correct the docstring (the permutation is a matching null, not a dependence-preserving one), quote the exact statement "0/10,000 permutations reached 11; p ≤ 1/10001" rather than a bare inequality, and add the hard-negative rows.

---

### 5. The business-only arm is the one that fails multiplicity, and it exceeds its own ceiling.

**The attack.** Family: 4 blind-match arms + 2 placebo arms + 2 driver-identified arms = 8 tests. Holm and BH:

| test | raw p | Holm | BH |
|---|---|---|---|
| blind all-sections / non-business / all-60 | 1.0e-4 | 8.0e-4 | 2.7e-4 |
| **blind business-only** | **0.0163** | **0.0815** | **0.0326** |
| placebo & driver arms (4) | 0.36–0.47 | 1.00 | 0.47 |

Business-only **does not survive Holm**. It survives BH. The three near-zero arms survive anything. The placebo arms are unaffected — multiplicity is not their problem.

Worse: business-only real accuracy is **28.6% (4/14)** while your ceiling test for business-only was **16.7% (2/12), p=0.27**. A real, lossy 342-word explanation beat the "best any explanation could do." A ceiling that the result exceeds is not a ceiling; it means the proxy (a several-thousand-word generation half as a BM25 query, against 183-word business holdouts) is a bad query, not an upper bound.

**Damage: moderate, and it lands on the arm your own STATUS.md calls "the most important caveat in the project."**

**Defence.** The four arms are nested, not independent — all-sections and non-business share 10 of their 11 correct windows, so the family is really ~2 questions, and Holm over 8 is punitive. The primary arms were pre-declared and are unaffected.

**Fix.** Declare all-sections as primary and the other three as sensitivity; report Holm on the two-test family (primary blind match, primary placebo). Rename "ceiling" to "proxy reference" for the business arm, and say plainly that the reference is invalid there because the proxy exceeds nothing.

---

### 6. Grounding at 96.4%: what it rules out is more than you claim, and what it misses is different from what a critic will guess.

**What a critic will say:** lexical overlap at a 0.25 threshold is a metric everything passes. **That is wrong and I disproved it.** Re-scoring all 473 cited claims against a *random* item from the same window (`scratchpad/ground.py`, 30 draws per claim):

- real cited items: mean overlap **0.614**, grounded **96.6%**
- random same-window items: mean overlap **0.037**, grounded **2.3%** (0.0% at every threshold ≥0.25 across draws)
- random other-window items: mean overlap 0.020
- whole-window union (all ~180 items): mean overlap 0.748, grounded 99.4%

So 1.93 cited items recover 82% of what all 180 items recover, against a 2.3% random baseline. **This is the strongest single control in the project and you are not quoting the null.** Quote it.

**What it genuinely cannot see.** The tokeniser is `[a-z][a-z0-9'-]+` with a >3-character filter, so **every numeral is invisible**: `_content_tokens("75bp rate cut to 2.5% on 14 April")` returns `{'rate','april'}`. Negation is invisible (26.6% of claims contain a negation/failure word). Direction, causation, and entity-role binding are invisible — bag-of-words cannot distinguish "X caused Y" from "Y caused X". And overlap is against the *union* of cited items, so grounding gets easier as citations increase (mean 1.93, max 9).

**But the leak isn't there either.** I ran the numeric audit the metric never does: **150/473 claims (31.7%) contain a non-year numeral; in 136 (90.7%) every numeral appears verbatim in the cited item.** Of the 14 misses I inspected, they are formatting and artefact, not confabulation: "5,600" vs the source's "5600"; "650 points" *is* present verbatim; most of the rest are "Day 13"/"Day 14" references, which are your own relative-day labels and by construction absent from source text.

**Damage: low. The claim survives; the framing needs tightening.**

**Fix.** Say "96.4% of claims are lexically grounded in the item they cite, against a 2.3% random-citation floor; a separate numeric audit finds 136/150 numerals verbatim in source." Add a comma-normalising numeric check to the audit — it is ten lines and it closes the one structural blind spot that matters for a finance audience.

---

### 7. The statistical spine quotes the wrong summary statistic in two places.

**7a. "27 walk-forward folds" — only 20 folds contain both states.** Folds 2, 3 and four others have zero stressed days, so the per-fold median of 1.52 is over 20, not 27. Say 20.

**7b. You are burying your best number.** Per-fold ratios: **18/20 exceed 1.0. Exact sign test p=0.00040**; Wilcoxon on log-ratio p=1e-5; fold-cluster bootstrap 95% CI on the median **[1.37, 1.90]**, on the geometric mean **[1.41, 1.98]**. This is distribution-free, immune to the pooling objection you already flagged, and immune to the volatility-clustering objection. It is a better headline than 2.18×.

**7c. The published CI is ~15–20% too narrow.** `_block_bootstrap_ratio_ci` uses a fixed 20-day block, but your own dwell filter guarantees regimes last ≥20 days and the median observed run is **74 days** (mean 136, max 503). Re-running at longer blocks: 20d → [1.77, 2.75]; 63d → [1.71, 2.85]; 126d → [1.69, 2.87]. And a **fold-cluster bootstrap gives [1.64, 2.81]** against your published [1.76, 2.69]. Neither changes the conclusion, but the block length should be justified against the dwell distribution, not left at 20.

**7d. The generalisation claim quotes the circular metric.** You report Nikkei 1.54× / Hang Seng 1.46× / DAX 1.87× — those are *contemporaneous* ratios, which your own `regime_stats.py` docstring calls "partly definitional." The **forward-20d** ratios in `regimes.json` are **1.18 / 1.32 / 1.58**. Nikkei at 1.18 is close to nothing. By your own honesty standard the forward numbers are the ones to quote, and quoting the contemporaneous ones for the generalisation test while quoting the forward one (1.69×) for the primary is inconsistent. Also: three global equity indices over the same 2015/2018/2020/2022 window are not three independent replications — they share one global volatility factor.

**Damage: moderate. 7d is the one a sharp judge will catch on the one-pager.**

---

### 8. Remaining hindsight and memorisation paths, given pinning and date-stripping.

Three survive, in decreasing order of concern.

**(a) The boundary is the end of the transition day, not the moment of the move.** Manifests show `window_end_utc = <boundary>T23:59:59Z` and pinned revisions at e.g. 20:58Z on the boundary day. US equities close at 20:00–21:00 UTC. Wikipedia Current Events has a Business and economy section that reports market moves the same evening. So a transition window's final day can legitimately contain a description of the very move that defined the label. I confirmed the mechanism exists: `placebo_2016-06-28` contains *"The Dow Jones ends the day down 611.21 points"* and *"U.S. futures markets fall by 650 points."* **This is a same-day outcome-into-input path that revision pinning does not close.** Mitigating evidence: transitions do not carry more of it than controls (30% vs 47.5%, Fisher p=0.27; mean market items in the final two days 0.65 vs 0.65, p=0.59), so it does not manufacture the placebo gap — it does the opposite.

**(b) Blind matching does not separate reading from recalling.** BM25 is knowledge-free, but the *query* is model-generated. If the model recognises March 2020 and writes about it from memory, BM25 will still match that text to the March 2020 holdout. The blind match is therefore not a memorisation control — the faithfulness audit is (see §6), and it is a strong one.

**(c) The post-cutoff evidence you have and are not using.** Only two windows post-date the claude-opus-5 knowledge cutoff (`placebo_2026-06-23`, `placebo_2026-07-28`). Both were **matched correctly, 2/2** (binomial vs 1/60 chance, p=0.0003), and both grounded at **100%** across 21 claims, against 96.2% pre-cutoff. n=2 is an anecdote as you said — but "the only two windows the model provably cannot have memorised behaved identically" is worth one sentence.

---

### 9. Effective sample size is smaller than the Fisher tests assume.

Controls are strictly paired, 2 per transition (`Counter({2: 20})`), each drawn 30–90 days from its partner (mean |offset| 54 days). Fisher's exact test treats all 60 windows as independent draws; they are 20 clusters. The 13 clean controls come from **12 distinct transitions**. And the minimum `days_to_nearest_transition` across all controls is **14** — the bare disjointness bound — so some control windows are the fortnight immediately adjacent to a transition window and share running stories with it.

**Damage: low-moderate, and it cuts toward the null.** Clustering inflates the variance of the control rate, so the true p is *larger* than 0.43 and the true power lower than 0.16. It does not create the gap.

**Fix.** Report the placebo comparison as a paired/clustered analysis — conditional logistic or an exact permutation that shuffles the transition/control label *within* each matched set of three. With 20 clusters that permutation has 3²⁰ arrangements, so it is exact and cheap.

---

### 10. Where the pre-registration claim holds, and where it does not.

I checked git rather than taking it on trust. Commit `e255e00` (2026-08-22 12:23:49) contains `placebo.py` with the `stratum` property and the words *"the headline comparison is pre-declared against the clean stratum"*, and `narrative.py` with `is_confident = driver_identified and confidence in {high, medium}`. `outputs/narrative_results.json` first appears in `f463d22` at 13:53:00. **The primary outcome, its threshold, and the primary comparison are all timestamped 90 minutes before the results existed.** That is real, verifiable pre-declaration and you should show the `git log` at the walkthrough.

Two caveats. First, `settings.yaml`'s "PRE-REGISTERED PRIMARY SPECIFICATION" banner sits only over the `transitions:` block; the placebo primary and the confidence threshold live in Python docstrings, which is a weaker place for them. Second, 90 minutes and a single author is not an external timestamp — a hostile reader can note that nothing prevented reordering. Move both declarations under the settings banner.

**One thing the pre-declaration got right, and I tried to break it.** The clean stratum is defined by `state==0 and days_to_nearest ≥30`, which is a filter on the model's own output and skews the year composition badly (clean controls include 2017, 2023, 2024 — years with zero transitions — while dropping most of 2019, which has five). I expected this to destroy the era matching that the whole placebo design rests on. It does not. Trailing-20d annualised SPY volatility at each date: transitions median **0.110**, clean controls **0.106** (Mann–Whitney p=0.62); *all* controls **0.152** (p=0.042); contaminated controls 0.179. **The clean stratum is the better-matched comparison on the variable that matters, and the "all controls" arm is the confounded one** — its controls sit in significantly more turbulent periods, which inflates the 65% and biases that arm toward the null. Pre-declaring the clean stratum as primary was the correct call, and now you can prove it with a number instead of an argument.

---

### The two sentences I would put on the one-page PDF

> Explanations are window-specific: 43/60 matched back to the correct fortnight when the only distractors were the three temporally nearest windows (33% chance, p<1e-4), and 15/20 transitions beat their own era-matched controls (p=3e-4). Claims are grounded in their cited item at 96.4% against a 2.3% random-citation floor, with zero fabricated citations.

> The placebo test did not separate — 80% vs 61.5%, +18.5pp, 95% CI [−12, +47]pp — but at n=20 this design could only ever have resolved a gap of 50pp or more. That is an unpowered test, not a null result, and the fix is 80 transitions, not more controls.

Working scripts are in `C:/Users/SZEMIN~1/AppData/Local/Temp/claude/C--Users-sze-ming-chak-Documents-GitHub-regime-narrative/d82d60d7-99ff-4487-a304-6ddc69fb64e3/scratchpad/` (`pwr.py`, `ci.py`, `bm.py`, `adj.py`, `hard2.py`, `ground.py`, `num.py`, `leak.py`, `adj2.py`, `spine.py`). Nothing in the repository was modified.