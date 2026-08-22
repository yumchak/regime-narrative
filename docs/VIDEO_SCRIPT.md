# 7-minute video script — regime-narrative

**Polymer Tech Expo 2026.** Maximum 7 minutes, two required parts.

Every narration cell is written to a budget of **~150 words per minute**, a
normal presenting pace. An earlier draft of this script ran at 219 wpm overall
with segments at 336 — physically unspeakable. If you add a sentence, cut one.

Name, university, branding, AI journey and career aspiration are filled in and
grounded in your own repos (`purgedcv`, `pypbo`, `quantguard`), so every word is
checkable. Two brackets remain because they are genuinely yours. Read it aloud
before recording — if a sentence doesn't sound like you, change it.

Figures are the measured values in `outputs/`; none is rounded in your favour.

---

## PART 1 — Self-introduction · 0:00–1:05

*Required: one-line branding · AI journey · fun fact · internship aspirations.*

| Time | On screen | Narration |
|---|---|---|
| **0:00–0:12**<br>(12s · 28w) | To camera. Lower third: *Victor Sze · BSc Mathematics with Statistics · University of Bristol*. | Hi, I'm Victor, a penultimate-year Maths and Statistics student at Bristol. I build tools that stop quantitative research fooling itself. |
| **0:12–0:45**<br>(33s · 80w) | Montage of your repos — `purgedcv`, `pypbo`, `quantguard`. Hold two seconds on quantguard's line *"The LLM proposes; deterministic code verifies."* Cut to this repo's tests scrolling past. | Look at what I build and it's all one thing. Purged cross-validation. Probability of backtest overfitting. An auditor for leakage in quant research. Tools for not fooling yourself. My AI journey went the same way — it started as autocomplete, and what changed is that I stopped asking models for answers and started asking what would prove them wrong. **[OPTIONAL: one sentence on the moment that changed.]** So: the model proposes, code verifies. |
| **0:45–0:55**<br>(10s · 22w) | Something visual and personal. | A fun fact — **[FUN FACT]**. |
| **0:55–1:05**<br>(10s · 26w) | To camera. | **[PICK ONE OF THE THREE BELOW.]** |

### Career aspiration — three directions, pick one

All three fit the 10-second slot (~26 words) and all three are consistent with
what you actually built. They point at different things, so choose the one you
would still believe in an interview.

**A — The researcher.** Leans on the through-line already established in the
previous beat. Safest, and the most coherent with the rest of the video.

> I want a quant research internship, where you build a signal and then spend
> most of your time trying to break it. **[WHY POLYMER.]**

**B — The one who ships.** Positions you as someone whose research reaches other
people, which is what the dashboard actually demonstrates. Use this if you want
the tool, not the statistics, to be the thing they remember.

> I want to do research that other people can actually use — signals that arrive
> with their error bars attached, not just a number in a notebook. **[WHY
> POLYMER.]**

**C — The one who wants to be wrong faster.** The boldest, and the most
memorable if you deliver it plainly rather than as a slogan. It also sets up
your placebo result, so the video pays it off later.

> I want to work somewhere that treats a negative result as a finding. This
> project's best number is the one that went against me. **[WHY POLYMER.]**

**On the Polymer sentence.** One line, specific, and true. Something you can
name — a strategy, a team, something they published, the fact that the expo is
judged on a code walkthrough rather than a pitch. A generic "I admire your
culture" is worse than saying nothing; if you have nothing specific, cut the
sentence and use the extra two seconds on the line above.

---

## PART 2 — Project showcase · 1:05–7:00

*Required: what you built · how it works · how AI was used (coding tools, system
prompts with examples, models/APIs/platforms) · iterations and reflections.*

| Time | On screen | Narration |
|---|---|---|
| **1:05–1:35**<br>(30s · 72w) | Overview regime chart, full width. SPY log axis, calm bands pale blue, stressed pale red, twenty transition lines. At 1:28 one word fades in: **"Why?"** | Every regime model produces a picture like this. Calm here, stressed here. A real result — but the model only ever saw returns. It has never read a newspaper. It can tell you the market changed today, and nothing about why. A human fills that gap from memory: undocumented, unauditable, different for everyone. This tool attaches an explanation to each line, then measures what that explanation is worth. |
| **1:35–2:10**<br>(35s · 86w) | Metric tiles: per-fold median **1.52×**, sign test **p = 0.00040**. Then `hmm_model.py`, forward filter highlighted, with the line where Viterbi is deliberately *not* used. | First, the part with no language model in it. A two-state hidden Markov model, refitted inside every walk-forward fold, scored only on held-out days. Of the twenty folds containing both states, eighteen show the stressed state more volatile. Sign test, p of four in ten thousand. One detail decides whether that's real: states come from a forward filter — observations up to that day only. The standard call runs Viterbi over the whole sequence, so a label can depend on the future. That would contaminate every date this rests on. |
| **2:10–2:45**<br>(35s · 87w) | Controls → Hindsight. **60 windows, 10,888 items, 0 post-boundary survivors.** Leak table: mean **38.2%**, max **71.2%**. Cut to `news/base.py` — the constructor raising — then green pytest output. | Now give a model the news, and only the news that existed then. Two things make that defensible. The boundary is structural — a window object refuses to construct if one item post-dates it. Enforced by the type, not by a comment. And every page is pinned to its revision. That matters: Wikipedia keeps editing a day's page for years. On average thirty-eight percent of the current page was written after the boundary. Worst case, seventy-one. Fetch the live page and you feed the model hindsight, invisibly. |
| **2:45–3:20**<br>(35s · 88w) | **REQUIREMENT: system prompts.** Provenance → Prompt tab, `prompts/narrative_v1.md` with its sha256. Highlight in turn: the frontmatter `notes:`; Rule 2 — *every claim must cite an item id*; Rule 4 — *you may decline*. | This is the system prompt. One file, version controlled, hashed, never an inline string. Three lines are load-bearing. The frontmatter: the same prompt runs for real transitions and for control dates, because an earlier draft framed them differently and would have measured my framing instead of the news. Second: every claim must cite an item — that's what makes memorisation countable. Third: the model is told it may decline. Without that line it confabulates on controls and the comparison collapses. |
| **3:20–3:50**<br>(30s · 74w) | **Live view, real recording, no cuts.** Date **2015-06-29**, tile reads **TRANSITION**. Run. Status log types out: `0` post-boundary, date scrub, "Asking claude-opus-5…". Output: **high confidence**, Greek bailout summary, claims ticked with `cites …` tags. | This is running now. I give it a date. It fetches the window, checks the boundary, strips the dates out of the text, and sends it. The model sees items labelled Day one to Day fourteen. No prices, no ticker, no date. Here's what comes back — ministers refusing to extend negotiations, a referendum called, deposit flight from Greek banks. Every sentence carries the id of the item it came from. |
| **3:50–4:15**<br>(25s · 60w) | **The ordinary Tuesday.** Same view. Type **2019-12-03**. Tile flips to **no transition**. Run. Output: **medium confidence**, a fluent, fully-cited US–China story. Hold three seconds in silence. | Now watch. Third of December, twenty-nineteen. An ordinary Tuesday. Same prompt, same permission to decline. And it finds a story anyway. Medium confidence, well sourced, every claim grounded. It isn't wrong — that news is real. But it's not evidence of anything, because there's always something to point at. |
| **4:15–4:55**<br>(40s · 98w) | Blind matching: **11/20 = 55% vs 5%**, then hard negatives **71.7% vs 33.3%**. Cut to Memorisation: **474 claims, 96.4% grounded, 1.8% random floor, 0.0% fabricated**. | So how do I separate a real explanation from macro boilerplate? Two measurements. Blind matching — strip the dates, scramble them, and match each explanation back to its own two-week window using held-out articles the model never saw. Scored with BM25, which has no world knowledge, so it can't date a paragraph from training. Fifty-five percent against one in twenty. And against only its three nearest neighbours in time, seventy-two against thirty-three — era can't explain that. Second, faithfulness: ninety-six percent of claims grounded in what they cite, against a two percent random floor. Zero fabricated citations. |
| **4:55–5:45**<br>(50s · 123w) | Placebo tab, full screen. Table held throughout: transitions **20 / 16 / 80.0%**; clean controls **13 / 8 / 61.5%**. Then **+18.5pp, 95% CI [−12, +47]pp** and **power 0.17**. | And here's the result that goes against me. Twenty transitions, forty controls drawn from each transition's own neighbourhood, so era and news environment match. Identical prompt. Only the news differs. On real transitions the model is confident eighty percent of the time. On clean controls, sixty-one. Now — the tempting thing is to say the explanations aren't diagnostic. I can't say that. A non-significant result is not evidence of no difference. What I can say is the interval: plus eighteen points, from minus twelve to plus forty-seven. It's that wide because power at the observed gap is zero point one seven. One chance in six of detecting the effect I observed. Unresolved — and this design could never have resolved it. |
| **5:45–6:10**<br>(25s · 62w) | Cross-model table: blind match **55% / 55%**, grounding **96.4% / 96.8%**, fabricated **0 / 0**; then confidence **80% / 100%**, **κ = 0.25**. | I ran everything again on a second model. Blind matching came back identical. Grounding differed by four tenths of a point. Zero fabricated citations in either. Those belong to the method. But the confidence label didn't replicate — kappa of zero point two five, and the second model declined half as often. So that number belongs to the model, not to the data. |
| **6:10–6:40**<br>(30s · 74w) | **REQUIREMENT: tools, models, APIs.** Full-screen card. **Built with:** Claude Code. **Integrated:** `claude-opus-5`, Anthropic Messages API, effort high, schema-enforced output · Wikipedia MediaWiki API, revision-pinned · yfinance · hmmlearn · scikit-learn · Streamlit. Highlight `model: "claude-opus-5"` in `settings.yaml`. | The build. Code, tests and refactors written with Claude Code in this repository. The prompts were not — those are hand-written, with the reasoning for every rule dated in a prompt history. The narrative layer is claude-opus-five through the Messages API, one call per window, schema enforced server-side. News from the MediaWiki API, pinned by revision. Prices from yfinance. The model name is one line in one settings file. |
| **6:40–7:00**<br>(20s · 50w) | **REQUIREMENT: reflections.** Regime chart. Three lines fade in: *The HMM decides **when**.* / *The model describes **what was in the news**.* / *No statistic comes from the language model.* | What worked: writing the controls before generating anything, including the ones that prove the test can fail. What I'd change: pre-specify volatility onsets, which triples the power on the test that came out unresolved. A method whose limit has been measured beats one that's only been claimed. |

---

## Before you record

- [ ] Fill `[FUN FACT]` and `[WHY POLYMER]`
- [ ] Read Part 2 aloud against a timer — if a cell runs long, cut a sentence
      rather than speaking faster
- [ ] Key in `.env` or the dashboard sidebar, so the Live view works on camera
- [ ] **Pre-warm both Live dates** (2015-06-29, 2019-12-03) once before
      recording — cached windows return instantly instead of a 15-second wait
- [ ] Total **≤ 7:00**
- [ ] Both files named `FirstName_LastName_University`
