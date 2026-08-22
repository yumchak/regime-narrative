"""Core news types and the retrieval boundary.

The single most important invariant in this project lives here:

    no item in a NewsWindow may have been published after the window's
    boundary date.

It is enforced structurally -- ``NewsWindow`` refuses to construct if the
invariant is broken -- rather than by convention in each fetcher. Every
fetcher routes its results through ``build_window``; there is no other
supported way to make a window.

Timezone convention (recorded in every manifest):
    boundary  = <boundary_date> 23:59:59 UTC   inclusive
    start     = <boundary_date - (window_days - 1)> 00:00:00 UTC inclusive

All timestamps are normalised to tz-aware UTC on ingest. A naive timestamp
from a source is treated as UTC and flagged in the manifest, because a source
whose timezone we cannot pin is a source whose boundary we cannot fully trust.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Sequence


class BoundaryViolation(AssertionError):
    """Raised when an item published after the boundary reaches a window."""


@dataclass(frozen=True)
class NewsItem:
    """One dated piece of text. ``published`` is always tz-aware UTC."""

    source: str
    published: datetime
    title: str
    text: str = ""
    url: str = ""
    # True when the source gave us no timezone and we assumed UTC.
    tz_assumed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.published.tzinfo is None:
            raise ValueError(
                f"NewsItem.published must be tz-aware; got naive {self.published!r}. "
                "Use normalise_timestamp() on ingest."
            )

    @property
    def item_id(self) -> str:
        """Stable content hash -- used for citation checking and dedup."""
        payload = f"{self.source}|{self.published.isoformat()}|{self.url}|{self.title}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def body(self) -> str:
        """Title plus text, which is what a matcher or an LLM actually reads."""
        return f"{self.title}\n{self.text}".strip()


@dataclass(frozen=True)
class NewsWindow:
    """A boundary-enforced collection of items.

    Construct via ``build_window``; the constructor re-checks the invariant so
    that even a hand-built window cannot violate it.
    """

    boundary_date: date
    window_days: int
    source: str
    items: tuple[NewsItem, ...]
    retrieved_at: datetime
    # Items the source returned that post-dated the boundary and were dropped.
    # Non-zero is not fatal, but it is recorded: it tells us the source does
    # not enforce the boundary server-side and we are filtering client-side.
    dropped_post_boundary: int = 0
    dropped_pre_window: int = 0
    query: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cutoff = boundary_instant(self.boundary_date)
        offenders = [i for i in self.items if i.published > cutoff]
        if offenders:
            worst = max(o.published for o in offenders)
            raise BoundaryViolation(
                f"{len(offenders)} item(s) post-date the boundary "
                f"{cutoff.isoformat()} (latest {worst.isoformat()}) "
                f"in {self.source} window ending {self.boundary_date}."
            )

    @property
    def start_instant(self) -> datetime:
        return window_start_instant(self.boundary_date, self.window_days)

    @property
    def end_instant(self) -> datetime:
        return boundary_instant(self.boundary_date)

    def __len__(self) -> int:
        return len(self.items)

    def manifest(self) -> dict[str, Any]:
        """Everything needed to reproduce and audit this retrieval."""
        return {
            "source": self.source,
            "boundary_date": self.boundary_date.isoformat(),
            "window_days": self.window_days,
            "window_start_utc": self.start_instant.isoformat(),
            "window_end_utc": self.end_instant.isoformat(),
            "retrieved_at_utc": self.retrieved_at.isoformat(),
            "n_items": len(self.items),
            "dropped_post_boundary": self.dropped_post_boundary,
            "dropped_pre_window": self.dropped_pre_window,
            "n_tz_assumed": sum(1 for i in self.items if i.tz_assumed),
            "query": self.query,
            "earliest_item_utc": (
                min(i.published for i in self.items).isoformat() if self.items else None
            ),
            "latest_item_utc": (
                max(i.published for i in self.items).isoformat() if self.items else None
            ),
            "item_ids": [i.item_id for i in self.items],
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------
# boundary arithmetic -- one definition, used everywhere
# --------------------------------------------------------------------------


def boundary_instant(boundary_date: date) -> datetime:
    """Last instant that is inside the window. Inclusive of the whole day."""
    return datetime.combine(boundary_date, time(23, 59, 59), tzinfo=timezone.utc)


def window_start_instant(boundary_date: date, window_days: int) -> datetime:
    """First instant inside the window. ``window_days`` counts the boundary day."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    start_day = boundary_date - timedelta(days=window_days - 1)
    return datetime.combine(start_day, time(0, 0, 0), tzinfo=timezone.utc)


def normalise_timestamp(
    value: datetime, *, assume_utc: bool = True
) -> tuple[datetime, bool]:
    """Return (tz-aware UTC datetime, whether the timezone was assumed)."""
    if value.tzinfo is None:
        if not assume_utc:
            raise ValueError(f"naive timestamp {value!r} and assume_utc=False")
        return value.replace(tzinfo=timezone.utc), True
    return value.astimezone(timezone.utc), False


def build_window(
    *,
    boundary_date: date,
    window_days: int,
    source: str,
    candidate_items: Iterable[NewsItem],
    retrieved_at: datetime | None = None,
    query: str = "",
    provenance: dict[str, Any] | None = None,
) -> NewsWindow:
    """The only supported way to make a NewsWindow.

    Filters candidates to the window, counts what it dropped on each side, and
    hands the survivors to a constructor that re-checks the invariant.
    """
    start = window_start_instant(boundary_date, window_days)
    end = boundary_instant(boundary_date)

    kept: list[NewsItem] = []
    dropped_after = 0
    dropped_before = 0
    for item in candidate_items:
        if item.published > end:
            dropped_after += 1
        elif item.published < start:
            dropped_before += 1
        else:
            kept.append(item)

    kept.sort(key=lambda i: (i.published, i.item_id))

    # Dedup on content hash; sources overlap and repeat wire copy.
    seen: set[str] = set()
    deduped: list[NewsItem] = []
    for item in kept:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        deduped.append(item)

    return NewsWindow(
        boundary_date=boundary_date,
        window_days=window_days,
        source=source,
        items=tuple(deduped),
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
        dropped_post_boundary=dropped_after,
        dropped_pre_window=dropped_before,
        query=query,
        provenance=provenance or {},
    )


def split_holdout(
    window: NewsWindow, *, fraction: float, seed: int
) -> tuple[NewsWindow, NewsWindow]:
    """Split a window into (generation set, held-out matching set).

    The blind-matching test scores an explanation against articles the model
    never saw. Without this split the test measures whether the summariser
    copies words from its input, which it does, which proves nothing.

    The split is deterministic in the item ids, so it does not depend on
    retrieval order and survives a re-fetch.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")

    ordered = sorted(window.items, key=lambda i: _stable_hash(i.item_id, seed))
    n_holdout = max(1, round(len(ordered) * fraction)) if ordered else 0
    holdout = ordered[:n_holdout]
    generation = ordered[n_holdout:]

    def _rebuild(items: Sequence[NewsItem], role: str) -> NewsWindow:
        prov = dict(window.provenance)
        prov.update({"split_role": role, "split_fraction": fraction, "split_seed": seed})
        return NewsWindow(
            boundary_date=window.boundary_date,
            window_days=window.window_days,
            source=window.source,
            items=tuple(sorted(items, key=lambda i: (i.published, i.item_id))),
            retrieved_at=window.retrieved_at,
            dropped_post_boundary=window.dropped_post_boundary,
            dropped_pre_window=window.dropped_pre_window,
            query=window.query,
            provenance=prov,
        )

    return _rebuild(generation, "generation"), _rebuild(holdout, "holdout")


def _stable_hash(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
