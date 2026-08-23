# 6-minute video script — regime-narrative

**Polymer Tech Expo 2026.** Limit 7:00; this runs **6:35**, leaving 25 seconds
of slack.

Every cell is timed to its own word count at **~118 words per minute** — an
unhurried pace. No cell is faster than 120, so you can breathe between
sentences without running over.

Part 2 follows the brief's four points in order, so a judge can tick them off:
**what you built · how it works · how AI was used · iterations and reflections.**

**SHOW** is what's on screen. In the demo, *use* the tool — click, type, wait —
rather than narrating every part of it. The point is that it works, not that you
can describe it.

Two brackets are yours. Everything else is written.

---

## PART 1 — Self-introduction · 0:00–1:09

| Time | SHOW | Say |
|---|---|---|
| **0:00–0:11** | **Your face.** Lower third: *Victor Sze · BSc Maths with Statistics · University of Bristol* | Hi, I'm Victor, a penultimate-year Maths and Statistics student at Bristol. I learn by implementing things and seeing whether they actually work. |
| **0:11–0:47** | **Your GitHub profile**, scrolling slowly past the pinned repos. Don't name them. | It started with data science on a live database of about five million records — at that scale you stop writing code and start designing workflows. Then a competition on predicting how markets react to earnings. The lesson was knowing what to hand a model and what to keep deterministic. You still label the data and run the checks yourself, and that's what tells you whether an answer is real. |
| **0:47–0:57** | Something personal. | A fun fact: **[FUN FACT]**. |
| **0:57–1:09** | **Your face.** | What I'd want from a summer at Polymer is this same problem at real scale. **[WHY POLYMER.]** It's what I'd be doing anyway. |

---

## PART 2 — Project showcase · 1:09–6:35

### 1 · What I built · 1:09–1:50

| Time | SHOW | Say |
|---|---|---|
| **1:09–1:27** | **App → Overview.** SPY chart full screen. Let it sit. Move the cursor along two or three red lines. | Every regime model gives you a picture like this. Calm here, stressed here. It's a real statistical result — but the model only ever saw price data. It knows something changed. It has no idea what. |
| **1:27–1:50** | Scroll to the metric tiles. | So I built the missing half. For every date this thing flags, a language model reads the news from before that date and tells me what was happening. Then three checks measure how far that explanation can be trusted. |

### 2 · How it works · 1:50–3:45

| Time | SHOW | Say |
|---|---|---|
| **1:50–2:12** | **The pipeline diagram** (from the write-up), full screen. Trace it left to right with the cursor. | Two halves with a hard line between them. On the left the statistical model picks the dates. On the right a language model reads fourteen days of news ending on that date. It never sees a price, a ticker, or even the date itself. |
| **2:12–2:39** | **Controls → Hindsight.** Point at *0 post-boundary survivors*, then the 71% row. | The line between them is the whole design. Nothing published after the date can get in — the code refuses to build the window. And each Wikipedia page is pinned to the version that existed back then, because on average thirty-eight percent of one of those pages gets written later. Worst case, seventy-one. |
| **2:39–3:01** | **Live view — just use it.** Type `2015-06-29`. Click Run. Let the log run. Read the summary off the screen. | So — pick a date. Run it. June 2015. Talks between Greece and its lenders break down, a referendum is called, and money starts leaving Greek banks. A negotiation became a deadline with a default at the end of it. |
| **3:01–3:20** | **Same answer — scroll down** to *Competing explanation*, then to any claim's citation chip. | And it argues with itself. Same fortnight, three terror attacks — it names them, then says why the Greek deadline is the bigger driver. Every line cites its source, so I can open it. |
| **3:20–3:45** | **Same view.** Clear it, type `2019-12-03`. Run. **Three seconds of silence when the answer lands.** Point at the confidence label. | Now an ordinary Tuesday — one the model never flagged. It's allowed to say *I don't know*. … It gives me a story anyway. A little less sure, but only a little. That gap is what this project is about. |

### 3 · How AI was used · 3:45–5:09

| Time | SHOW | Say |
|---|---|---|
| **3:45–4:07** | **Your terminal / editor with Claude Code**, and a git log scrolling. | The code was written with Claude Code, Anthropic's command-line agent — the pipeline, the tests, the dashboard. A hundred and one tests. The ones I care about are the ones that can fail, because a test that can't fail isn't a test. |
| **4:07–4:35** | **Provenance → Prompt tab.** Scroll slowly. Stop on Rule 2, then Rule 4. | This is the system prompt. One file, version controlled, never buried in code. Two rules carry it. Every claim has to cite one of the articles I gave it — that's what makes the answer checkable. And it's told it may refuse. Without that line it invents something every single time, and the test collapses. |
| **4:35–5:09** | **Open `outputs/slide_tools.png` full screen** (double-click it, then maximise the image viewer). Hold it for the whole segment. | The narrative layer is Claude Opus 5 through the Messages API — one call per date, and the output format is enforced by the API rather than politely asked for in the prompt. News comes from Wikipedia's API, pinned by revision. Prices from yfinance, the model from hmmlearn, and the dashboard is Streamlit. Every call is logged with the model, the prompt hash and the input hash. |

### 4 · Iterations and reflections · 5:09–6:35

| Time | SHOW | Say |
|---|---|---|
| **5:09–5:40** | **Controls → Blind matching**, then **Memorisation**. Point at each number. | Here's what worked. I hide the dates, shuffle the explanations, and try to match each one back to its own two weeks. Seventy-two percent right, where guessing gets thirty-three. Then I check every claim against the article it cited — ninety-six percent hold up, against a two percent floor if I'd picked a random article. Nine hundred citations, not one invented. |
| **5:40–6:12** | **Controls → Placebo tab.** Hold it. | And here's what didn't. On real events it's confident eighty percent of the time. On ordinary days, sixty-two. Those should be far apart, and they aren't. But I can't call it a failure either — with only twenty real transitions, this test had a seventeen percent chance of finding a difference even if one was there. It was never going to work. |
| **6:12–6:35** | **Your face**, or back to the chart. | Given more time I'd test only the days volatility starts, which roughly triples the cases that count. What I've got is an explanation you can check, and an honest measurement of how far to trust it. That second part is the bit I'd keep. |

---

## Before you record

- [ ] Fill `[FUN FACT]` and `[WHY POLYMER]`
- [ ] **Pre-warm both Live dates** — run `2015-06-29` and `2019-12-03` once
      first, so they return instantly instead of pausing 15 seconds
- [ ] Key in `.env` or pasted into the sidebar
- [ ] Browser at ~125% zoom so text survives compression
- [ ] Record one row at a time
- [ ] Stay under **7:00**
- [ ] `outputs/slide_tools.png` open in a window you can bring up at 4:35
- [ ] Files named `FirstName_LastName_University`
- [ ] Re-check the timings after any edit: `python scripts/13_check_script.py`

## Two moments that carry it

**3:20–3:45** — the ordinary Tuesday. The whole project in one answer.
Don't rush, and don't fill the silence.

**5:40–6:12** — the result that went against you. Say it evenly. Anyone can
present a win.

## If you overrun

Cut **1:27–1:50** first — the metric tiles. The chart already made the point.
Buys twenty-three seconds without touching the argument.
