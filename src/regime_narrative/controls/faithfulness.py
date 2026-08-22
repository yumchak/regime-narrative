"""Control 2: memorisation, measured through citation faithfulness.

The problem
-----------
The model has read text covering these events. Handed news from a window it
recognises, it may describe what it remembers rather than what it was given --
and the output would look identical either way.

Why the pre/post training-cutoff split is not enough
----------------------------------------------------
The obvious control is to compare quality before and after the model's training
cutoff. With a cutoff in early 2025 and a study period running to 2026, the
post-cutoff arm holds a handful of transitions. That is an anecdote, not a test,
and reporting it as one would be worse than not running it.

What scales instead
-------------------
Every claim must cite a supplied item. So for every claim we can ask: does the
cited item actually contain the claim's content? Three failure modes fall out,
each countable across all windows:

    fabricated citation   -- cites an id that is not in the window at all
    uncited claim         -- cites nothing
    ungrounded claim      -- cites a real item whose text does not support it

The last is the interesting one. It is where recall leaks in: the model attaches
a real citation to a statement the citation does not license. Grounding is
measured lexically, by content-word overlap, so the measurement itself has no
world knowledge and cannot be fooled the way an LLM judge could be.

This is mitigation and measurement, not a solution. The honest framing in the
report is: memorisation is mitigated by date-stripping, constrained by mandatory
citation, and quantified here -- not eliminated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from ..narrative import Narrative
from ..news.base import NewsWindow

_RE_TOKEN = re.compile(r"[a-z][a-z0-9\-']+")

# Words that carry no evidential weight; overlap on these does not ground a claim.
_GENERIC = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
    "was", "were", "are", "been", "will", "would", "could", "may", "might",
    "its", "their", "they", "which", "when", "while", "after", "before",
    "market", "markets", "investor", "investors", "volatility", "risk",
    "financial", "economic", "economy", "price", "prices", "increase",
    "decrease", "significant", "major", "amid", "following", "reported",
    "announced", "according", "also", "more", "than", "into", "over", "under",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _RE_TOKEN.findall(text.lower()) if t not in _GENERIC and len(t) > 3}


@dataclass
class ClaimAudit:
    claim: str
    cited_ids: list[str]
    valid_ids: list[str]
    fabricated_ids: list[str]
    overlap: float
    grounded: bool
    status: str          # "grounded" | "ungrounded" | "uncited" | "fabricated_citation"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaithfulnessReport:
    window_id: str
    kind: str
    n_claims: int
    n_grounded: int
    n_ungrounded: int
    n_uncited: int
    n_fabricated: int
    grounding_rate: float
    fabricated_citation_rate: float
    mean_overlap: float
    claims: list[ClaimAudit]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = [c.as_dict() if hasattr(c, "as_dict") else c for c in self.claims]
        return d


def audit_narrative(
    narrative: Narrative,
    window: NewsWindow,
    *,
    overlap_threshold: float = 0.25,
) -> FaithfulnessReport:
    """Check every claim against the items it cites.

    ``overlap_threshold`` is the fraction of a claim's content words that must
    appear in its cited items for the claim to count as grounded. It is
    deliberately lenient: paraphrase is expected and should not be punished. The
    threshold is reported alongside the results and swept in the sensitivity
    table, because a grounding rate is meaningless without it.
    """
    by_id = {item.item_id: item for item in window.items}
    audits: list[ClaimAudit] = []

    for entry in narrative.claims:
        if isinstance(entry, dict):
            claim_text = str(entry.get("claim", ""))
            cited = [str(x) for x in (entry.get("item_ids") or [])]
        else:
            claim_text, cited = str(entry), []

        valid = [c for c in cited if c in by_id]
        fabricated = [c for c in cited if c not in by_id]

        claim_tokens = _content_tokens(claim_text)
        if valid:
            source_tokens: set[str] = set()
            for cid in valid:
                source_tokens |= _content_tokens(by_id[cid].body)
            overlap = (
                len(claim_tokens & source_tokens) / len(claim_tokens)
                if claim_tokens
                else 0.0
            )
        else:
            overlap = 0.0

        if not cited:
            status = "uncited"
        elif fabricated and not valid:
            status = "fabricated_citation"
        elif overlap >= overlap_threshold:
            status = "grounded"
        else:
            status = "ungrounded"

        audits.append(
            ClaimAudit(
                claim=claim_text,
                cited_ids=cited,
                valid_ids=valid,
                fabricated_ids=fabricated,
                overlap=round(overlap, 3),
                grounded=status == "grounded",
                status=status,
            )
        )

    n = len(audits)
    return FaithfulnessReport(
        window_id=narrative.window_id,
        kind=narrative.kind,
        n_claims=n,
        n_grounded=sum(1 for a in audits if a.status == "grounded"),
        n_ungrounded=sum(1 for a in audits if a.status == "ungrounded"),
        n_uncited=sum(1 for a in audits if a.status == "uncited"),
        n_fabricated=sum(1 for a in audits if a.fabricated_ids),
        grounding_rate=(sum(1 for a in audits if a.grounded) / n) if n else 0.0,
        fabricated_citation_rate=(sum(1 for a in audits if a.fabricated_ids) / n) if n else 0.0,
        mean_overlap=(sum(a.overlap for a in audits) / n) if n else 0.0,
        claims=audits,
    )


def aggregate(reports: list[FaithfulnessReport]) -> dict:
    """Pooled faithfulness, split by transition versus placebo.

    A placebo window that produces confident, well-cited claims is a different
    problem from one that produces confident, ungrounded claims. Splitting the
    rate says which.
    """
    def _pool(subset: list[FaithfulnessReport]) -> dict:
        total = sum(r.n_claims for r in subset)
        if not total:
            return {"n_windows": len(subset), "n_claims": 0}
        return {
            "n_windows": len(subset),
            "n_claims": total,
            "grounding_rate": round(sum(r.n_grounded for r in subset) / total, 3),
            "ungrounded_rate": round(sum(r.n_ungrounded for r in subset) / total, 3),
            "uncited_rate": round(sum(r.n_uncited for r in subset) / total, 3),
            "fabricated_citation_rate": round(sum(r.n_fabricated for r in subset) / total, 3),
            "mean_overlap": round(
                sum(r.mean_overlap * r.n_claims for r in subset) / total, 3
            ),
        }

    return {
        "all": _pool(reports),
        "transitions": _pool([r for r in reports if r.kind == "transition"]),
        "placebos": _pool([r for r in reports if r.kind == "placebo"]),
    }


def threshold_sensitivity(
    narratives_and_windows: list[tuple[Narrative, NewsWindow]],
    thresholds: tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
) -> list[dict]:
    """Grounding rate as a function of the overlap threshold.

    Reported so the headline grounding number cannot be read as a single
    conveniently chosen cut-off.
    """
    rows = []
    for th in thresholds:
        reports = [audit_narrative(n, w, overlap_threshold=th)
                   for n, w in narratives_and_windows]
        agg = aggregate(reports)
        rows.append({"overlap_threshold": th, **agg["all"]})
    return rows
