# 6-minute video script — regime-narrative

**Polymer Tech Expo 2026.** Limit is 7:00; this runs **6:00**, leaving a minute
of slack for pauses and a slower delivery.

Written at **~120 words per minute** — deliberate and unhurried. You should be
able to breathe between sentences without running over. If you naturally speak
faster you will finish early, which is fine.

**SHOW** is what's on screen. Record the app segments as screen capture with
your voice over; use your face for the intro and the closing line only.

Two brackets are yours. Everything else is written.

---

## PART 1 — Self-introduction · 0:00–0:55

| Time | SHOW | Say |
|---|---|---|
| **0:00–0:15**<br>*30w* | **Your face.** Lower third: *Victor Sze · BSc Mathematics with Statistics · University of Bristol* | Hi, I'm Victor, a penultimate-year Maths and Statistics student at Bristol. I build tools that stop quantitative research fooling itself. |
| **0:15–0:35**<br>*45w* | **Your GitHub**, scrolling slowly past `purgedcv`, `pypbo`, `quantguard`. Pause two seconds on quantguard's tagline. | Everything I build is the same idea. Purged cross-validation. Probability of backtest overfitting. An auditor for leakage. Tools for not fooling yourself. My AI journey went the same way — I stopped asking models for answers, and started asking what would prove them wrong. |
| **0:35–0:45**<br>*20w* | Something personal — a photo, an object. | A fun fact: **[FUN FACT]**. |
| **0:45–0:55**<br>*22w* | **Your face.** | I want a quant research internship, where you build a signal and then try hard to break it. **[ONE SENTENCE: WHY POLYMER.]** |

---

## PART 2 — The project · 0:55–6:00

| Time | SHOW | Say |
|---|---|---|
| **0:55–1:25**<br>*60w* | **App → Overview.** The SPY chart, full screen. Let it sit for two seconds. Move the cursor slowly along a few red transition lines. | Every regime model gives you a picture like this. Calm here. Stressed here. It's a real statistical result. But the model only ever saw price data — it has never read a newspaper. So it can tell you the market changed on this day, and nothing at all about why. |
| **1:25–1:55**<br>*60w* | Same page, **scroll to the metric tiles.** Hover the first tile so the confidence-interval tooltip appears. | Underneath is a hidden Markov model, refitted inside every walk-forward window and scored only on data it never saw. Stressed days move about twice as much as calm days, and eighteen of the twenty windows show that independently. So the detector works. It just can't say a single word about what happened. |
| **1:55–2:25**<br>*62w* | **Controls → Hindsight tab.** Point at *0 post-boundary survivors*. Then the leak table — rest the cursor on the 71% row. | So I give a language model the news, and only news that existed at the time. Two things make that safe. Nothing published after the date can enter the window — the code refuses to build one. And every Wikipedia page is pinned to the version that existed then, because on average thirty-eight percent of one of those pages gets written afterwards. Worst case, seventy-one. |
| **2:25–2:55**<br>*58w* | **Provenance → Prompt tab.** Scroll slowly. Pause on Rule 2, then on Rule 4. | This is the prompt. One file, version controlled, never buried inside code. Two lines matter most. Every claim must cite one of the articles I supplied — that's what makes it checkable. And it's explicitly told it may refuse to answer. Without that line it invents a story every time, and the whole test collapses. |
| **2:55–3:30**<br>*72w* | **Live view. Real recording, no cuts.** Type `2015-06-29`. Tile reads **TRANSITION**. Click Run and let the log type out. | Here it is running. I give it a date. It fetches the news, checks the boundary, strips every date out of the text, and sends it. The model sees fourteen days of headlines and nothing else. No prices. No ticker. No date. And back it comes — Greek bailout talks collapsing, a referendum called, money leaving Greek banks. Every sentence carries the ID of the article it came from. |
| **3:30–4:05**<br>*72w* | **Same view, do not reload.** Clear the box, type `2019-12-03`. Tile flips to **no transition**. Click Run. **Hold three seconds in silence when the answer appears.** | Now watch this. Third of December, twenty-nineteen. An ordinary Tuesday. Nothing happened. Same prompt, same pipeline, and it's allowed to say *I don't know*. … And it gives me a confident story anyway. US–China trade tension. Well sourced. Every claim ticked green. It isn't wrong — that news is real. But it's not evidence of anything, because there is always something to point at. |
| **4:05–4:45**<br>*80w* | **Controls → Blind matching**, then the **Memorisation** tab. Point at each number as you say it. | So how do I tell a real explanation from one that would fit any month? I hide the dates, shuffle them, and try to match each explanation back to its own two weeks. Seventy-two percent right — guessing gets thirty-three. Then I check every claim against the article it cited. Ninety-six percent are genuinely supported, against a two percent floor if I'd used a random article instead. Nine hundred citations, not one invented. |
| **4:45–5:20**<br>*76w* | **Controls → Placebo tab.** Full screen. Leave the arm table up for the whole segment. | And here's the result that goes against me. On real events the model is confident eighty percent of the time. On ordinary days, sixty-two. Those should be far apart. They aren't. But I can't call the explanations useless either, because with only twenty real transitions this test had a seventeen percent chance of finding a difference even if one was there. It was never going to work. That's the honest limit. |
| **5:20–5:45**<br>*56w* | **Full-screen card.** *Built with:* Claude Code. *Integrated:* `claude-opus-5`, Anthropic Messages API · Wikipedia MediaWiki API · yfinance · hmmlearn · scikit-learn · Streamlit. | The build. Code and tests written with Claude Code. The narrative layer is Claude Opus 5 through the Messages API, one call per date, with the output format enforced by the API rather than requested in the prompt. News from Wikipedia, pinned by revision. And I ran the whole study twice, on two models, and got the same scores. |
| **5:45–6:00**<br>*36w* | **Your face**, or back to the chart. | The statistical model decides *when*. The language model only describes *what was in the news*. What I've built is an explanation you can check — and a measurement of exactly how far to trust it. |

---

## Before you record

- [ ] Fill `[FUN FACT]` and `[WHY POLYMER]`
- [ ] **Pre-warm both Live dates** — run `2015-06-29` and `2019-12-03` once
      beforehand. They then return instantly instead of pausing 15 seconds.
- [ ] Key in `.env`, or pasted into the app sidebar
- [ ] Zoom the browser to ~125% so text stays readable once compressed
- [ ] Record in short takes, one row at a time — far easier than one long run
- [ ] Stay **under 7:00**
- [ ] Both files named `FirstName_LastName_University`

## If you overrun

Cut **1:25–1:55** — the metric tiles — first. It is the least essential: the
chart already showed the detector works, and the write-up carries the numbers.
That buys thirty seconds without touching the argument.

## The two moments that matter

**3:30–4:05** — the ordinary Tuesday. This is the whole project in thirty
seconds. Do not rush it, and do not fill the silence.

**4:45–5:20** — the result that goes against you. Say it evenly. Anyone can
present a win; presenting a measured limit is what separates this from a demo.
