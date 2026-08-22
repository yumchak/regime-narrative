"""The hindsight control, as executable tests.

Control 1 in the brief: the news window closes at the transition date, and
that is enforced in the retrieval function rather than by convention. These
tests are what makes that sentence checkable. If any of them fail, no
narrative result in the project means anything.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from regime_narrative.news.base import (
    BoundaryViolation,
    NewsItem,
    NewsWindow,
    boundary_instant,
    build_window,
    normalise_timestamp,
    split_holdout,
    window_start_instant,
)

BOUNDARY = date(2020, 3, 3)
WINDOW_DAYS = 14


def _item(published: datetime, title: str = "headline", source: str = "test") -> NewsItem:
    return NewsItem(source=source, published=published, title=title, text="body text")


def _utc(y: int, m: int, d: int, hh: int = 12, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# boundary arithmetic
# ---------------------------------------------------------------------------


def test_boundary_instant_includes_the_whole_boundary_day():
    assert boundary_instant(BOUNDARY) == _utc(2020, 3, 3, 23, 59, 59)


def test_window_start_counts_the_boundary_day():
    # A 14-day window ending 3 March starts on 19 February, not 18 February.
    assert window_start_instant(BOUNDARY, 14) == _utc(2020, 2, 19, 0, 0, 0)


def test_single_day_window_is_just_that_day():
    start = window_start_instant(BOUNDARY, 1)
    assert start == _utc(2020, 3, 3, 0, 0, 0)
    assert start < boundary_instant(BOUNDARY)


def test_zero_or_negative_window_is_rejected():
    with pytest.raises(ValueError):
        window_start_instant(BOUNDARY, 0)


# ---------------------------------------------------------------------------
# the invariant itself
# ---------------------------------------------------------------------------


def test_item_one_second_after_boundary_is_dropped():
    late = _item(_utc(2020, 3, 4, 0, 0, 0), "published just after midnight")
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=[late],
    )
    assert len(window) == 0
    assert window.dropped_post_boundary == 1


def test_item_at_the_last_second_of_the_boundary_day_is_kept():
    edge = _item(_utc(2020, 3, 3, 23, 59, 59), "23:59:59 on the day")
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=[edge],
    )
    assert len(window) == 1


def test_item_before_the_window_is_dropped_and_counted_separately():
    early = _item(_utc(2020, 2, 18, 23, 59, 59), "one second too early")
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=[early],
    )
    assert len(window) == 0
    assert window.dropped_pre_window == 1
    assert window.dropped_post_boundary == 0


def test_constructing_a_window_directly_with_a_late_item_raises():
    """The structural guarantee: you cannot hand-build a leaking window."""
    late = _item(_utc(2020, 3, 10))
    with pytest.raises(BoundaryViolation):
        NewsWindow(
            boundary_date=BOUNDARY,
            window_days=WINDOW_DAYS,
            source="test",
            items=(late,),
            retrieved_at=datetime.now(timezone.utc),
        )


def test_mixed_batch_keeps_only_the_in_window_items():
    items = [
        _item(_utc(2020, 2, 10), "way before"),
        _item(_utc(2020, 2, 25), "inside"),
        _item(_utc(2020, 3, 3, 9, 30), "morning of the transition"),
        _item(_utc(2020, 3, 4), "the day after"),
        _item(_utc(2020, 6, 1), "months of hindsight later"),
    ]
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=items,
    )
    assert len(window) == 2
    assert window.dropped_pre_window == 1
    assert window.dropped_post_boundary == 2
    assert all(i.published <= window.end_instant for i in window.items)


@pytest.mark.parametrize("window_days", [1, 3, 7, 14, 30, 60, 90])
def test_no_item_ever_survives_past_the_boundary_for_any_window_length(window_days):
    """Property test: sweep a year of daily items through every window length."""
    items = [_item(_utc(2020, 1, 1) + timedelta(days=n)) for n in range(365)]
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=window_days,
        source="test",
        candidate_items=items,
    )
    cutoff = boundary_instant(BOUNDARY)
    assert all(i.published <= cutoff for i in window.items)
    # One item per day, so the count is the window length -- capped by how far
    # back the synthetic series actually runs (1 Jan 2020 -> 3 Mar 2020 = 63 days).
    available = (BOUNDARY - date(2020, 1, 1)).days + 1
    assert len(window) == min(window_days, available)


# ---------------------------------------------------------------------------
# timezone handling -- a source in a later timezone must not leak
# ---------------------------------------------------------------------------


def test_naive_timestamps_are_rejected_at_the_item_level():
    with pytest.raises(ValueError):
        NewsItem(source="test", published=datetime(2020, 3, 1, 12, 0), title="naive")


def test_normalise_flags_an_assumed_timezone():
    value, assumed = normalise_timestamp(datetime(2020, 3, 1, 12, 0))
    assert assumed is True
    assert value.tzinfo is timezone.utc


def test_a_tokyo_morning_after_the_utc_boundary_is_excluded():
    """4 March 08:00 in Tokyo is 3 March 23:00 UTC -- inside.
    4 March 10:00 in Tokyo is 4 March 01:00 UTC -- outside.
    The boundary is defined in UTC and the conversion must respect it."""
    jst = timezone(timedelta(hours=9))
    inside, _ = normalise_timestamp(datetime(2020, 3, 4, 8, 0, tzinfo=jst))
    outside, _ = normalise_timestamp(datetime(2020, 3, 4, 10, 0, tzinfo=jst))

    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=[
            _item(inside, "tokyo, still 3 March UTC"),
            _item(outside, "tokyo, now 4 March UTC"),
        ],
    )
    assert len(window) == 1
    assert window.items[0].title == "tokyo, still 3 March UTC"
    assert window.dropped_post_boundary == 1


def test_manifest_records_assumed_timezones():
    naive_utc, assumed = normalise_timestamp(datetime(2020, 3, 1, 12, 0))
    item = NewsItem(
        source="test", published=naive_utc, title="no tz given", tz_assumed=assumed
    )
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=[item],
    )
    assert window.manifest()["n_tz_assumed"] == 1


# ---------------------------------------------------------------------------
# manifest completeness -- the audit trail the brief asks for
# ---------------------------------------------------------------------------


def test_manifest_carries_every_field_needed_to_reproduce():
    items = [_item(_utc(2020, 2, 25)), _item(_utc(2020, 3, 2))]
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="gdelt",
        candidate_items=items,
        query="stock market",
    )
    m = window.manifest()
    for key in (
        "source",
        "boundary_date",
        "window_days",
        "window_start_utc",
        "window_end_utc",
        "retrieved_at_utc",
        "n_items",
        "dropped_post_boundary",
        "query",
        "item_ids",
    ):
        assert key in m, f"manifest missing {key}"
    assert m["window_end_utc"] == "2020-03-03T23:59:59+00:00"
    assert m["n_items"] == 2


def test_items_are_deduplicated_on_content():
    dup = _item(_utc(2020, 3, 1), "same wire copy")
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=[dup, dup, dup],
    )
    assert len(window) == 1


# ---------------------------------------------------------------------------
# held-out split -- the blind-matching confound fix
# ---------------------------------------------------------------------------


def test_holdout_split_is_disjoint_and_complete():
    items = [_item(_utc(2020, 2, 20) + timedelta(hours=6 * n), f"item {n}") for n in range(20)]
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=items,
    )
    gen, hold = split_holdout(window, fraction=0.5, seed=7)

    gen_ids = {i.item_id for i in gen.items}
    hold_ids = {i.item_id for i in hold.items}
    assert gen_ids.isdisjoint(hold_ids), "generation and holdout share items"
    assert gen_ids | hold_ids == {i.item_id for i in window.items}
    assert len(hold) == 10


def test_holdout_split_is_deterministic_across_calls():
    items = [_item(_utc(2020, 2, 20) + timedelta(hours=6 * n), f"item {n}") for n in range(20)]
    window = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=items,
    )
    a_gen, a_hold = split_holdout(window, fraction=0.5, seed=7)
    b_gen, b_hold = split_holdout(window, fraction=0.5, seed=7)
    assert [i.item_id for i in a_hold.items] == [i.item_id for i in b_hold.items]
    assert [i.item_id for i in a_gen.items] == [i.item_id for i in b_gen.items]


def test_holdout_split_does_not_depend_on_retrieval_order():
    """A re-fetch that returns items in a different order must split the same."""
    items = [_item(_utc(2020, 2, 20) + timedelta(hours=6 * n), f"item {n}") for n in range(20)]
    w1 = build_window(
        boundary_date=BOUNDARY, window_days=WINDOW_DAYS, source="test", candidate_items=items
    )
    w2 = build_window(
        boundary_date=BOUNDARY,
        window_days=WINDOW_DAYS,
        source="test",
        candidate_items=list(reversed(items)),
    )
    _, h1 = split_holdout(w1, fraction=0.5, seed=7)
    _, h2 = split_holdout(w2, fraction=0.5, seed=7)
    assert [i.item_id for i in h1.items] == [i.item_id for i in h2.items]


def test_both_halves_of_a_split_still_respect_the_boundary():
    items = [_item(_utc(2020, 2, 20) + timedelta(hours=6 * n)) for n in range(20)]
    window = build_window(
        boundary_date=BOUNDARY, window_days=WINDOW_DAYS, source="test", candidate_items=items
    )
    gen, hold = split_holdout(window, fraction=0.5, seed=7)
    cutoff = boundary_instant(BOUNDARY)
    assert all(i.published <= cutoff for i in gen.items)
    assert all(i.published <= cutoff for i in hold.items)
