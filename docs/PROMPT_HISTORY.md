# Prompt iteration history

Kept as it happens. Reconstructing this afterwards looks fake because it is.

Each entry records what changed, why, and what it cost. Prompts themselves live
in `prompts/` under version control; this file is the reasoning behind them.

---

## `narrative_v1` — 2026-08-22

First version. Not yet run against the API, so nothing here is a post-hoc
rationalisation of an observed failure — these are the decisions taken up front,
and later entries will record which ones survived contact with real output.

### Decisions and why

**Identical prompt for transitions and controls.**
The placebo test only means something if the two arms differ in exactly one
respect: the news. An earlier draft framed control dates neutrally ("was
anything happening?") and transitions assertively ("a regime shift was
detected"). That would have measured the framing, not the news, and the placebo
rate would have been an artefact. The prompt now tells the model a possible
change was flagged in both cases. This is written into the prompt's frontmatter
so nobody later "improves" one arm.

**Relative day labels, not absolute dates.**
Sequence is genuinely informative — an escalating series of items reads
differently from an isolated shock — but absolute dates hand the model the
answer. Items are therefore labelled `Day 1 … Day 14`, with Day 14 the flagged
day. Absolute dates inside item text are separately scrubbed by
`narrative.strip_dates`.

**Mandatory citation on every claim.**
Not for presentation. It is what makes the memorisation control measurable: a
claim whose cited item does not contain its content is recall leaking in, and
`controls/faithfulness.py` counts exactly that. Without per-claim citations
there is no way to distinguish inference from recollection at scale.

**An explicit licence to decline.**
Rule 4 states that declining is a correct answer and that most windows contain
nothing market-moving. Without this the model will confabulate on control dates
and the placebo comparison collapses to "the model always says something". The
risk in the other direction — that an over-permissive licence to decline
suppresses real drivers too — is what the transition arm measures.

**Confidence judged on evidence, not on trusting the model.**
The prompt says explicitly that the flag may be noise. Otherwise the model
reasons backwards from "a shift was detected, so something must explain it",
which is precisely the failure the placebo test exists to catch.

**Banned forward-looking phrasing.**
"This would go on to", "marked the beginning of" — these are hindsight tells.
Naming them explicitly is cheap and gives the audit something concrete to grep
for.

**A `competing_explanation` field.**
A single confident story is easy to produce and hard to trust. Forcing the
strongest alternative reading makes over-confidence visible in the output rather
than only in the aggregate statistics.

### Known risks in this version, to check on first run

1. **Over-declining.** The decline licence is emphatic. If the transition arm
   declines often, the confidence bar is too high and the wording needs
   softening — but that change must be applied to both arms simultaneously.
2. **Citation padding.** The model may cite many items per claim to appear
   grounded, inflating lexical overlap. Watch mean citations per claim; if it
   drifts high, cap it in the prompt.
3. **Section bias.** Wikipedia's Current Events sections are mostly non-business.
   The model may reach for geopolitics because that is what is abundant. Worth
   checking against the blind-match section-stratified arms, where non-business
   text matches at 90% and business text at chance.
4. **JSON compliance.** `max_tokens` is 2000. A window of 250 items with many
   claims could truncate mid-object. The parser degrades to a recorded failure
   rather than raising, so truncation will be visible in `parse_ok`.

### Not yet decided

Whether to include the item count in the prompt header. It currently says how
many items there are, which is a weak signal about how eventful the window was
and could bias confidence. Transition and control windows are closely matched on
item count (180.6 vs 186.1 mean), so the leak is small — but it is a leak, and
removing it costs nothing if the first run shows confidence correlating with
item count.


---

## Determinism note — 2026-08-22, before the first run

The brief specified "temperature zero, one transition per call, no batching".
Two of those three survive; the first does not, and it is worth stating plainly
rather than quietly leaving `temperature: 0.0` in a settings file where a reader
would assume it took effect.

**Sampling parameters have been removed on current Claude models.** Passing
`temperature` (or `top_p`/`top_k`) to `claude-opus-5` returns a 400. The code
originally passed `temperature=0.0` on every call, so all 60 would have failed
before a single narrative was produced. Caught before the first run, not after.

What replaces it, and what it means for reproducibility:

* `output_config.effort` controls reasoning depth. Set to `high`.
* `output_config.format` now enforces the JSON schema server-side rather than
  asking the prompt nicely for valid JSON. This removes the parse-failure path
  entirely instead of handling it, and it is strictly better than the previous
  arrangement -- the `parse_ok` flag stays in the dataclass as a belt-and-braces
  check but should never fire.
* **Outputs are no longer bit-reproducible by re-running.** Nothing in the API
  now offers a determinism knob. Reproducibility therefore comes from the
  archive rather than from re-execution: every call is cached by input hash, and
  every call is logged with the model id, prompt hash, input hash, token counts
  and raw response. A reader can verify exactly what was sent and what came
  back; they cannot regenerate it byte-for-byte, and the report should say so.

The prompt text itself is unchanged. It still describes every output field,
because the model fills a schema more coherently when it knows what the fields
mean -- the schema guarantees shape, the prompt supplies intent.

---

## `narrative_v1` — outcome, 2026-08-22, after the first full run

The four risks logged above before generating anything, checked against what
actually came back. No prompt change is being made on the strength of this: the
prompt is frozen for the reported results, and any revision becomes `v2` with
its own run.

**1. Over-declining — did not materialise, and the direction is informative.**
The model declined on 2 of 20 transitions and 8 of 40 controls. It declines, but
not often, and — importantly for the placebo test — it declines *more* on
controls than on transitions, which is the direction the design hoped for. The
confidence bar is not set too high. Transition mix: 6 high, 10 medium, 2 low,
2 none. Control mix: 8 high, 18 medium, 6 low, 8 none.

**2. Citation padding — did not materialise.** Mean 1.93 cited items per claim,
median 1, maximum 9. The model is not spraying citations to look grounded. This
matters because the grounding metric scores a claim against the *union* of its
cited items, so padding would have inflated it for free. It did not need a cap
in the prompt after all.

**3. Section bias — real, and it cuts the opposite way to the worry.** The
concern was that the model would reach for geopolitics because Wikipedia's
Current Events is mostly non-business. What the blind-match arms show is that
the explanations carry *more* market-relevant content than an arbitrary slice of
the same window: real explanations match business-only held-out text at 28.6%
against 7.1% chance (p = 0.016), while the ceiling proxy managed 16.7% (p =
0.27). A real explanation beat the "best any explanation could do" on that arm,
which means the proxy was a bad query rather than an upper bound. The section
labelled "ceiling" for the business arm should be read as a reference point, not
a ceiling.

**4. JSON compliance — solved by construction, not by the prompt.** Zero parse
failures in 60 calls. This risk was written when the output contract lived in the
prompt and `max_tokens` was 2000. Both changed: the schema is now enforced
server-side via `output_config.format`, and `max_tokens` is 4000. The largest
single output carried 13 claims and did not truncate. The `parse_ok` flag stays
in the dataclass as a belt-and-braces check but no longer has a failure mode to
catch.

**The undecided question from before is now decided.** The prompt header states
how many items are in the window, which is a weak signal about how eventful the
period was. Transition and control windows are closely matched on item count
(180.6 vs 181.9 mean), so the leak is small — and since the placebo comparison
did not separate, there is no favourable result for it to have manufactured.
Left as-is for the reported run; a `v2` would remove it, because a leak that
happens not to have mattered is still a leak.
