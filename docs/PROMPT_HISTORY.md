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
