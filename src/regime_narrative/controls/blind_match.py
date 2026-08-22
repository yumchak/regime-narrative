"""Blind matching: are the explanations specific, or is this macro boilerplate?

The test
--------
Strip each explanation of dates and identifying details, scramble the order, and
try to match each one back to the window it was generated from. Score against
the rate expected by chance, ``1/N``.

Two design decisions make this test mean something.

**Matched against held-out articles, never the generation set.**
The naive version scores an explanation against the same articles it was written
from. Lexical overlap is then guaranteed -- you would be measuring whether the
summariser copies words from its input, which it does, which proves nothing. Each
window is split in half: the model sees one half, the matcher scores against the
other. Overlap then reflects what the *window* was about, not what the model was
handed.

**The matcher has no world knowledge.**
An LLM matcher could date "pandemic lockdowns" to early 2020 from training alone,
without reading a single supplied article -- turning a specificity test into a
memorisation test with the sign flipped. BM25 over token counts cannot do that.
It knows nothing except which words co-occur in the documents in front of it.

Significance is assessed by permutation rather than a closed-form binomial. The
candidate windows are not exchangeable -- adjacent windows share vocabulary and
crisis-era windows resemble each other -- so a binomial null would be too
generous. Shuffling the assignment preserves that structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "had", "he", "her", "his", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "they", "this", "to", "was",
    "were", "will", "with", "which", "after", "over", "into", "than", "then",
    "there", "these", "those", "when", "while", "who", "would", "could",
    "also", "more", "most", "other", "some", "such", "only", "own", "same",
    "so", "not", "no", "nor", "s", "t", "d", "ll", "m", "o", "re", "ve", "y",
    "said", "says", "say", "new", "first", "two", "one", "three", "about",
    "up", "down", "out", "off", "all", "any", "both", "each", "few", "he",
    "market", "markets", "regime", "volatility", "shift", "state", "period",
    "window", "news", "article", "articles", "report", "reports", "reported",
}

_RE_TOKEN = re.compile(r"[a-z][a-z0-9\-']+")


def tokenise(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens, optionally without stopwords.

    Generic market vocabulary ("volatility", "regime", "market") is treated as a
    stopword. Every explanation contains it, so it carries no information about
    *which* window an explanation came from, and leaving it in would inflate
    every similarity score uniformly.
    """
    toks = _RE_TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    return toks


# ---------------------------------------------------------------------------
# BM25 -- implemented directly so the scoring is inspectable in a walkthrough
# ---------------------------------------------------------------------------


class BM25:
    """Okapi BM25 over a fixed corpus of documents."""

    def __init__(self, corpus: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [list(doc) for doc in corpus]
        self.n_docs = len(self.corpus)
        self.doc_lens = np.array([len(d) for d in self.corpus], dtype=float)
        self.avg_len = float(self.doc_lens.mean()) if self.n_docs else 0.0

        self.term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for doc in self.corpus:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # Robertson/Sparck-Jones idf with the +1 smoothing that keeps it positive.
        self.idf = {
            term: np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n_docs)
        if not self.n_docs or self.avg_len == 0:
            return out
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avg_len)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    n_candidates: int
    n_scored: int
    n_correct: int
    accuracy: float
    chance: float
    mean_reciprocal_rank: float
    top3_accuracy: float
    p_value: float
    n_permutations: int
    per_item: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.n_correct}/{self.n_scored} matched "
            f"({100 * self.accuracy:.1f}%) against {100 * self.chance:.1f}% chance; "
            f"MRR {self.mean_reciprocal_rank:.3f}; permutation p={self.p_value:.4g}"
        )


def blind_match(
    explanations: dict[str, str],
    holdout_docs: dict[str, str],
    *,
    n_permutations: int = 10000,
    seed: int = 20260822,
) -> MatchResult:
    """Match each explanation to a candidate window using held-out text only.

    ``explanations`` and ``holdout_docs`` are both keyed by window id. Only ids
    present in both are scored; the full set of ``holdout_docs`` forms the
    candidate pool, so a window with no explanation still acts as a distractor.
    """
    candidate_ids = sorted(holdout_docs)
    if len(candidate_ids) < 2:
        raise ValueError("blind matching needs at least two candidate windows")

    corpus = [tokenise(holdout_docs[cid]) for cid in candidate_ids]
    bm25 = BM25(corpus)

    scored_ids = [eid for eid in sorted(explanations) if eid in holdout_docs]
    if not scored_ids:
        raise ValueError("no explanation ids overlap the candidate windows")

    per_item: list[dict] = []
    correct = 0
    recip_ranks: list[float] = []
    top3 = 0

    for eid in scored_ids:
        query = tokenise(explanations[eid])
        scores = bm25.scores(query)
        order = np.argsort(-scores)
        ranked = [candidate_ids[i] for i in order]
        rank = ranked.index(eid) + 1

        hit = rank == 1
        correct += int(hit)
        top3 += int(rank <= 3)
        recip_ranks.append(1.0 / rank)
        per_item.append(
            {
                "window_id": eid,
                "predicted": ranked[0],
                "correct": hit,
                "rank": rank,
                "score_correct": float(scores[candidate_ids.index(eid)]),
                "score_best": float(scores[order[0]]),
                "top3": ranked[:3],
            }
        )

    n = len(scored_ids)
    accuracy = correct / n
    chance = 1.0 / len(candidate_ids)

    p_value = _permutation_p_value(
        explanations={eid: explanations[eid] for eid in scored_ids},
        candidate_ids=candidate_ids,
        bm25=bm25,
        observed_correct=correct,
        n_permutations=n_permutations,
        seed=seed,
    )

    return MatchResult(
        n_candidates=len(candidate_ids),
        n_scored=n,
        n_correct=correct,
        accuracy=accuracy,
        chance=chance,
        mean_reciprocal_rank=float(np.mean(recip_ranks)),
        top3_accuracy=top3 / n,
        p_value=p_value,
        n_permutations=n_permutations,
        per_item=per_item,
    )


def _permutation_p_value(
    *,
    explanations: dict[str, str],
    candidate_ids: list[str],
    bm25: BM25,
    observed_correct: int,
    n_permutations: int,
    seed: int,
) -> float:
    """How often does a random relabelling do this well?

    The score matrix is computed once; permutation then only reshuffles which
    window each explanation is *supposed* to belong to. This preserves the
    similarity structure between windows, which a binomial null would ignore.
    """
    rng = np.random.default_rng(seed)
    eids = sorted(explanations)
    score_matrix = np.vstack([bm25.scores(tokenise(explanations[e])) for e in eids])
    predicted = np.argmax(score_matrix, axis=1)

    true_positions = np.array([candidate_ids.index(e) for e in eids])
    n_at_least = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(true_positions)
        if int(np.sum(predicted == shuffled)) >= observed_correct:
            n_at_least += 1

    return (n_at_least + 1) / (n_permutations + 1)


def strip_identifying_details(text: str) -> str:
    """Remove dates and explicit period references from an explanation.

    Matching must not succeed because the explanation literally said "March
    2020". Years, month names, ISO dates and quarter labels all go. This is
    applied to the explanation before matching, and the removal count is
    reported so the scrubbing itself is auditable.
    """
    out = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    out = re.sub(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        " ",
        out,
        flags=re.I,
    )
    out = re.sub(r"\b[qQ][1-4]\b", " ", out)
    out = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", out)
    out = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def count_stripped(text: str) -> int:
    """How many identifying tokens the scrubber removed, for the audit trail."""
    before = len(tokenise(text, drop_stopwords=False))
    after = len(tokenise(strip_identifying_details(text), drop_stopwords=False))
    return before - after
