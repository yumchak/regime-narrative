"""Tests for what actually reaches the language model.

Control 2 (memorisation) is mitigated partly by removing dates from item text
before the model sees it. That mitigation is only worth what these tests say it
is worth: a single leaked month name narrows the window enough to trigger recall.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from regime_narrative.narrative import (
    audit_date_leaks,
    load_prompt,
    render_window,
    strip_dates,
)
from regime_narrative.news.base import NewsItem, build_window


def _item(day: int, text: str) -> NewsItem:
    return NewsItem(
        source="test",
        published=datetime(2020, 3, day, 12, 0, tzinfo=timezone.utc),
        title="headline",
        text=text,
        extra={"section": "Business and economy"},
    )


# ---------------------------------------------------------------------------
# date scrubbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "On 3 March the Fed cut rates",
        "On March 3 the Fed cut rates",
        "In 2020 the Fed cut rates",
        "On 2020-03-03 the Fed cut rates",
        "Talks resume in September",
        "The deal closes next quarter",
        "Rates were cut in Feb",
    ],
)
def test_no_date_signal_survives_scrubbing(raw):
    assert audit_date_leaks(strip_dates(raw)) == {
        "bare_years": 0, "iso_dates": 0, "month_day": 0,
        "day_month": 0, "bare_months": 0, "relative_periods": 0,
    }


def test_scrubbing_keeps_the_substance():
    out = strip_dates("On 3 March 2020 the Federal Reserve cut rates by 50 basis points")
    assert "Federal Reserve" in out
    assert "50 basis points" in out


def test_the_verb_may_is_not_mistaken_for_the_month():
    out = strip_dates("Officials said the ruling may be revised")
    assert "may be revised" in out


def test_the_month_May_is_still_removed():
    out = strip_dates("The summit is scheduled for May")
    assert "May" not in out
    assert audit_date_leaks(out)["bare_months"] == 0


def test_numbers_that_are_not_years_survive():
    out = strip_dates("The index fell 1200 points and 350 firms were affected")
    assert "1200" in out and "350" in out


# ---------------------------------------------------------------------------
# window rendering
# ---------------------------------------------------------------------------


def _window():
    items = [
        _item(1, "In 2020 the central bank met in March to discuss policy"),
        _item(2, "Manufacturing output fell sharply across the eurozone"),
        _item(3, "An emergency rate cut of 0.5% was announced on 3 March"),
    ]
    return build_window(
        boundary_date=items[-1].published.date(),
        window_days=14,
        source="test",
        candidate_items=items,
    )


def test_rendered_prompt_carries_no_date_signal():
    text, _ = render_window(_window())
    assert audit_date_leaks(text) == {
        "bare_years": 0, "iso_dates": 0, "month_day": 0,
        "day_month": 0, "bare_months": 0, "relative_periods": 0,
    }


def test_items_are_labelled_by_relative_day_not_by_date():
    text, _ = render_window(_window())
    assert "Day 12" in text and "Day 14" in text
    assert "2020" not in text


def test_every_item_is_addressable_by_a_stable_id():
    window = _window()
    text, id_map = render_window(window)
    for item in window.items:
        assert item.item_id in text
        assert item.item_id in id_map


def test_empty_window_renders_without_crashing():
    window = build_window(
        boundary_date=datetime(2020, 3, 3, tzinfo=timezone.utc).date(),
        window_days=14, source="test", candidate_items=[],
    )
    text, id_map = render_window(window)
    assert "no items" in text.lower()
    assert id_map == {}


# ---------------------------------------------------------------------------
# prompt file discipline
# ---------------------------------------------------------------------------


def test_prompt_loads_and_hashes():
    text, digest = load_prompt("narrative_v1")
    assert len(digest) == 16
    assert "driver_identified" in text


def test_prompt_frontmatter_is_stripped_before_sending():
    text, _ = load_prompt("narrative_v1")
    assert not text.startswith("---")
    assert "version: narrative_v1" not in text


def test_prompt_grants_permission_to_decline():
    """If the model cannot decline, the placebo test cannot mean anything."""
    text, _ = load_prompt("narrative_v1")
    assert "decline" in text.lower()


def test_prompt_requires_citations():
    text, _ = load_prompt("narrative_v1")
    assert "cite" in text.lower()


def test_missing_prompt_version_fails_loudly():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist_v9")
