# regime-narrative — verified fact sheet

Every number here is read from a file in `outputs/`, with the source named. Built
because this project has twice had a claim drift away from what was actually
computed; if you are about to write one of these numbers into a CV, an
application or a conversation, take it from here rather than from memory.

**One line:** a statistical regime detector for equity markets, plus a language
layer that says what was in the news on each date it flags — and a measurement
of whether that explanation is worth anything.

---

## 1 · The competition

**Polymer Tech Expo 2026**, Hong Kong. Open to university students (2nd year and
above) in technology, AI, quantitative finance or data analytics.

| | |
|---|---|
| Submission deadline | 23 August 2026 |
| Exposition day | 10 September 2026 |
| Deliverables | Video (max 7:00) + one-page PDF (min 11-point font) |
| Write-up must cover | problem statement · solution overview · use of AI · impact & value · reflections |
| Video must cover | self-introduction · what you built · how it works · how AI was used · iterations and reflections |
| Prize pool | HKD 50,000, shared by the top 4 video submissions |
| Internships | up to 20 fast-tracked summer positions |
| File naming | `FirstName_LastName_University` |

**Polymer Capital Management** — Hong Kong, founded 2019 by Angus Wai in
partnership with PAG. ~$4bn AUM. A **multi-manager platform**: capital allocated
across internally managed PM teams, Asia-Pacific public equities, market-neutral
and equity long/short, with centralised risk management. Describes itself as
"a unique fusion of human insights and science." Flagship market-neutral Asia
fund returned 11.6% in 2024. Licensed by the HK SFC for Type 1, 4 and 9.
*(Preqin, Hedgeweek, Reuters, polymercapital.com — not from the repo.)*

---

## 2 · Part A — the regime detector

### Method
- **Model:** 2-state Gaussian HMM (`hmmlearn`), refit independently inside every
  walk-forward fold. `StandardScaler` fitted on train data only, inside the fold.
- **Data:** SPY, with TLT / GLD / VIX / HYG as features. 2010-02-02 → 2026-08-21.
- **Forward filter, not Viterbi.** `predict()` smooths over the whole test block,
  which lets a label depend on up to 126 days of future data. A custom
  forward-only recursion is used instead, so no transition date is contaminated
  by information from after it. This is the single most important design choice
  in the statistical half.
- **Which state is "stressed" is decided on training data only**, never on test.
- **Persistence filter:** minimum dwell 20 days.
- **Seed stability:** 10 random seeds; only transitions surviving across seeds kept.

### Results — `outputs/regimes.json`

| Measure | Value |
|---|---|
| Trading days | **4,164** |
| Walk-forward folds | **27** (3,402 out-of-sample days, 0 failures) |
| Pooled OOS volatility ratio, stressed ÷ calm | **2.18×** (95% CI 1.76 – 2.69) |
| Levene p | 3.6 × 10⁻⁷⁴ |
| In-sample ratio (the prior, weaker procedure) | 2.32× |
| Forward-20d ratio | 1.69× (median 1.60×) |
| Per-fold median ratio | **1.52×** (95% CI 1.37 – 1.90) |
| Folds with ratio > 1 | **18 of 20** that contain both states |
| Sign test p | **0.0004** |
| Wilcoxon on log ratios p | 9.5 × 10⁻⁶ |
| Raw state flips → after persistence filter | 84 → 24 (60 removed as noise) |
| Seed-stable transitions | **20** (primary seed alone gave 24) |

**Watch the fold count.** 27 folds exist, but only **20** contain both states, so
only 20 can produce a ratio. `referee_stats.json` records this explicitly:
quoting 27 alongside a per-fold median is wrong.

### Generalisation — markets never refitted on

| Index | OOS ratio | 95% CI | Forward-20d | Per-fold median |
|---|---|---|---|---|
| DAX | **1.87×** | 1.63 – 2.18 | 1.58× | 1.56× |
| Nikkei 225 | **1.54×** | 1.35 – 1.77 | 1.18× | 1.38× |
| Hang Seng | **1.46×** | 1.28 – 1.67 | 1.32× | 1.37× |

All three Levene p < 10⁻²³.

---

## 3 · Part B — the news layer

### Method
- Source: **Wikipedia Current Events portal**, via the MediaWiki API.
- **Revision-pinned** using `rvstart` + `rvdir=older` — each day-page is fetched
  as it existed on the boundary date, not as it reads today.
- 14-day window ending *on* the flagged date.
- **Hard boundary invariant:** `NewsWindow.__post_init__` raises
  `BoundaryViolation` if any item post-dates the boundary. The window cannot be
  constructed with contaminated data — it is not a filter, it is a refusal.
- Absolute dates stripped from item text before the model sees it.
- Deterministic held-out split for the blind-matching control.
- Rate-limited client: 1.1s minimum interval, 6 retries, exponential backoff,
  honours `Retry-After`.

### Results — `outputs/news_manifest.json`

| Measure | Value |
|---|---|
| Windows retrieved | **60** (0 failures) |
| Total news items | **10,888** |
| Mean items per window | 180.6 transitions / 181.9 controls |
| Range | 91 – 250 |
| Post-boundary items surviving | **0** |

### The hindsight measurement — why pinning matters

Comparing each pinned revision against the live page today:

- **38.2% mean** of a current Wikipedia day-page was written *after* that day
- **71.2% worst case** (2018 January 26: 1,266 bytes then, 4,401 bytes now)

Without pinning, more than a third of the "news" fed to the model would have
been written with full knowledge of what happened next. This is the number that
justifies the whole retrieval design.

---

## 4 · Part C — the narrator

- **Model:** `claude-opus-5`, Anthropic Messages API, effort `high`.
- **Output schema enforced server-side** via `output_config.format` — the model
  structurally cannot return off-schema output. Not requested in prose.
- **One call per window**, 60 windows (20 transitions + 40 era-matched controls).
- **Prompt lives in a version-controlled file** (`prompts/narrative_v1.md`) with
  a dated iteration history — never a string literal in code.
- **Identical prompt for both arms.** If the framing differed between transitions
  and placebos, the placebo test would measure the framing, not the news.
- Every call logs: model id, prompt version, prompt hash, input hash, effort,
  token usage and raw response. **187 call logs committed.**
- The model never sees a price, a ticker, or the date.

### The five prompt rules
1. Use only the supplied items — no outside recall
2. **Every factual claim must cite at least one item id**
3. No speculation about what happened next
4. **You may decline** — declining honestly is a correct answer
5. Do not infer or name the date

Rule 4 is load-bearing. Without it the model produces a confident story for
every window, and the placebo test collapses.

---

## 5 · Part D — validation

### 1. Blind matching — are the explanations specific, or boilerplate?

Dates stripped, explanations scrambled, matched back to their window by **BM25
against held-out articles the model never saw**. BM25 has no world knowledge, so
it cannot date an explanation from training the way an LLM judge could.

| Arm | Result | Chance |
|---|---|---|
| All 20 windows as candidates | 55% | 5% |
| **Against its 3 nearest neighbours** | **43/60 = 71.7%** | **33.3%** |
| Against its 5 nearest neighbours | 37/60 = 61.7% | 20% |

The hard-negative arm is the better claim. 55%-against-5% invites the objection
that adjacent windows share running stories, so the matcher may only be
recovering roughly *when*. Forcing competition against nearest neighbours kills
that reading — `referee_stats.json` calls it "stronger evidence than the
headline number."

### 2. Citation grounding — do the sources say what is claimed?

| Measure | Value |
|---|---|
| Claims audited | **474** (carrying **913** citation links) |
| Grounded in the article cited | **96.4%** |
| Mean lexical overlap with cited article | 61.4% |
| **Random article from the same window** | **1.8%** (overlap 2.4%) |
| Random draws | 4,740 |
| **Fabricated citations** | **0** |

The floor is what makes the 96% mean anything: this is not a metric everything
passes.

### 3. Placebo — are the explanations diagnostic of a real regime change?

Era-matched control dates, drawn from each transition's own neighbourhood,
mutually disjoint. Identical prompt, identical pipeline; only the news differs.

| Arm | Confident | Rate |
|---|---|---|
| Real transitions | 16/20 | **80.0%** |
| All controls | 26/40 | 65.0% |
| **Clean controls** (pre-declared primary) | 8/13 | **61.5%** |

- Difference **+18.5pp**, in the predicted direction
- Fisher exact **p = 0.425**
- Newcombe 95% CI **[−11.5, +47.1] pp**
- **Statistical power at the observed effect: 0.166** — about one in six

**This did not resolve.** The honest statement is *"underpowered, difference not
resolved"*, not *"no difference exists."* The binding constraint is the 20
transitions, not the 13 controls: even 200 controls would need a ~93% transition
rate to reach 80% power.

### 4. Retest stability — same model, same inputs, run twice

Sampling parameters were removed from current models, so identical inputs no
longer guarantee identical outputs.

| Measure | Value |
|---|---|
| Agreement on "confident" | 93.3% (κ = **0.841**) |
| Agreement on driver identified | 93.3% (κ = 0.76) |
| Exact confidence label | 80.0% |
| Headline rates | unchanged (0.80 / 0.65 both runs) |

### 5. Cross-model replication — the whole study re-run on `claude-sonnet-5`

| Measure | opus-5 | sonnet-5 |
|---|---|---|
| Blind match accuracy | 0.55 | **0.55** |
| Grounding rate | 0.964 | **0.968** |
| Fabricated citations | 0.0 | **0.0** |
| Transitions confident | 16/20 | 20/20 |
| Controls confident | 8/13 | 10/13 |

**The structural measurements are model-independent** — they are properties of
having the news and citing it. **The self-reported confidence label is not:**
window-by-window agreement between the two models is **κ = 0.25**. That is the
argument for treating blind matching as the load-bearing result and the
confidence label as a weak instrument.

Mantel-Haenszel OR 4.375, 95% CI [1.29, 14.88], p = 0.067 — **stratified rather
than pooled**, and the file states its own caveat: MH assumes independent strata,
and these are two readings of one sample, so that interval is optimistic.

### 6. Ceiling test — run *before* spending API budget

Measured the maximum any explanation could score, to avoid interpreting a null
result that was never achievable:
- All sections: 95% · business-only text: 16.7%
- Length sweep: at realistic explanation length (100–200 words) the ceiling is
  **40–70%**, not 95%. Significance survives even at 60 words.

---

## 6 · Honest limits (state these; they are the strongest part)

- **The placebo test did not resolve.** Power 0.166. Not a null result — an
  unresolved one.
- **Business-only text matches near chance.** A null here would be ambiguous
  between "explanations are generic" and "the matchable signal isn't where the
  explanation looks."
- **The confidence label is a weak instrument** — κ = 0.25 across models.
- **Memorisation is mitigated, not solved.** Splitting either side of the
  training cutoff leaves n≈3 — worse than not running it. Citation faithfulness
  replaces it because it scales to every claim.
- **Open gap:** the boundary is midnight UTC, and Wikipedia reports the US close
  that same evening.

### Two findings computed in analysis, not in the repo — quote as direction only

- **Onsets vs exits:** calm→stressed transitions score 9/10 confident; the
  stressed→calm transitions score 7/10, against 8/13 for controls. The exits
  carry no news driver and were diluting the test. **10 cases per side** — the
  95% CI on 9/10 is 0.60–0.98, so this is a direction, never a measurement.
- **The "triples" claim is wrong and is still in `scripts/07_onepager.py:203`
  and `docs/VIDEO_SCRIPT.md:65`.** Testing only the days volatility starts gives
  **10 cases, not 60** — it halves the sample. Three times is the extra
  *history* that would be needed, not the cases. Do not repeat it.

---

## 7 · Engineering

| | |
|---|---|
| Python | **9,036 lines** across 38 files |
| Tests | **101** |
| Commits | 37 |
| Pipeline stages | 13 numbered scripts, `01_regimes` → `13_check_script` |
| Dashboard | Streamlit, deployed to Streamlit Community Cloud |
| Stack | hmmlearn · scikit-learn · scipy · statsmodels · pandas · yfinance · plotly · Streamlit · Anthropic SDK |
| Built with | Claude Code |

Things worth naming as engineering rather than analysis:
- **A secret-scanning test** over all git-tracked files, so an API key cannot be
  committed.
- **A test asserting every imported package is in `requirements.txt`** — verified
  by deleting one and watching it fail. Caught a missing dependency that would
  have broken the deploy.
- **A font-size guard** in the PDF generator asserting the competition's 11pt
  minimum, rather than trusting the stylesheet.
- **A script that measures the video script** (`13_check_script.py`) — fails on
  overrun, on any gap or overlap between cells, and on any cell above 120 wpm.
- **Every number in the PDF substituted from JSON at build time.** An early
  hand-written draft carried three per-fold medians that were simply invented,
  and nothing in the process would have caught them.
- **Bring-your-own-data mode:** the controls run against any regime model's
  dates, any asset, any method — the reusable part is the controls, not the
  explanations.

---

## 8 · Stack and methods inventory

What was actually used, for applications and interviews. Read off the code, not
recalled.

**Data sources / APIs**
- `yfinance 1.6.0` — prices. Fitted: SPY, TLT, GLD, `^VIX`, HYG.
  Never fitted on: `^N225`, `^HSI`, `^GDAXI`.
- **Wikipedia MediaWiki API** — Current Events portal, revision-pinned with
  `rvstart` + `rvdir=older`.
- **Anthropic Messages API** — `claude-opus-5`, plus `claude-sonnet-5` for the
  replication arm.
- **GDELT** — evaluated and rejected on day-0 reconnaissance; it refused queries
  before ~2015. Wikipedia was the fallback that became primary. The config still
  records the original intent.

**Features** — four, 20-day lookback: trailing realised volatility, momentum,
VIX level, HYG 20-day return (credit proxy). Plus *forward* realised volatility
(t+1 … t+20) as the anti-circularity statistic — the model is fed trailing vol,
so separating states by trailing vol is near-definitional; separating them by
volatility that has not happened yet is a real claim.

**Model** — `GaussianHMM` (hmmlearn), 2 states, full covariance, 1000 iters.
Walk-forward 756 train / 126 test. `StandardScaler` fitted inside each fold.
**Custom forward-filter recursion with its own log-sum-exp**, replacing
`predict()`. Persistence filter (20d, swept at 5/10/20/30). Seed stability over
10 seeds at a 60% threshold, ±5-day match tolerance. Primary specification
pre-registered before any narrative was generated.

**Statistics** — Levene · Welch t-test · Fisher exact · binomial test ·
Wilcoxon signed-rank · sign test · **block bootstrap** · **permutation test**
(10,000 perms) · **Newcombe score interval** · **Mantel-Haenszel stratified OR**
· **Cohen's κ** · Monte Carlo power simulation.
The bootstrap, Newcombe interval and κ are hand-written, not imported.

**Retrieval / NLP** — **Okapi BM25 implemented from scratch** (no BM25 library
in the project at all). This is methodological, not stylistic: the blind-match
control is only valid because the matcher has no world knowledge, and a reader
can verify that by reading the scoring. Lexical-overlap scoring for citation
grounding. Deterministic 50% held-out split.

**LLM engineering** — Anthropic SDK `1.0.0`; `output_config` carrying
`effort: high` and a server-enforced JSON schema; **no temperature** (sampling
params were removed on current models and now 400). `max_tokens` 4000, ~320k
input tokens over 60 calls. Prompts as versioned files. Per-call logging of
model id, prompt version and hash, input hash, token usage and raw response.
Disk cache keyed by input hash. Rate-limited client: 1.1s interval, 6 retries,
exponential backoff, honours `Retry-After`.

**Tooling** — pytest · Streamlit `1.62` · plotly `6.9` · matplotlib · pandas
`3.0` · numpy `2.5` · scipy `1.18` · scikit-learn `1.9` · Jinja2 · PyYAML ·
Chrome headless `--print-to-pdf` · Streamlit `AppTest` for headless UI checks.

---

## 9 · CV-ready lines

**One-liner**
> Built a walk-forward regime detector for equity markets (2.18× out-of-sample
> volatility separation, sign test p = 0.0004) and an LLM explanation layer with
> a measured audit trail — 96% citation grounding against a 2% random-article
> floor, and 72% blind matching against 33% chance.

**Three bullets**
> - Walk-forward 2-state Gaussian HMM across 27 folds and 4,164 trading days;
>   2.18× out-of-sample volatility separation (95% CI 1.76–2.69), 18 of 20
>   eligible folds independently positive (sign test p = 0.0004). Held up
>   untouched on the DAX, Nikkei and Hang Seng.
> - Built a leakage-proof news retrieval layer (Wikipedia revision-pinning;
>   measured that 38% of a current day-page is written after the fact, peaking at
>   71%) and an LLM narrator whose every claim cites a supplied source.
> - Validated the explanations rather than assuming them: 96.4% citation
>   grounding against a 1.8% random-article floor over 474 claims, 72% blind
>   matching against 33% chance, zero fabricated citations — and reported the
>   placebo control as unresolved at power 0.166 rather than claiming a result.

**The interview answer, if asked what you are proudest of**
> The placebo test went against me. The honest thing was to compute what the
> experiment could ever have detected — power 0.166 — and report it as
> unresolved rather than as a null. Anyone can generate an explanation; the
> measurement is the part that transfers.

**Keywords** — Hidden Markov Models · walk-forward validation · look-ahead bias ·
data leakage · statistical power · Fisher exact · Newcombe interval ·
Mantel-Haenszel · Cohen's κ · sign test · block bootstrap · BM25 retrieval ·
LLM evaluation · citation grounding · placebo controls · schema-enforced
structured output · Streamlit · reproducible pipelines
