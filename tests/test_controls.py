"""Tests for the placebo and blind-matching controls.

The important cases here are the negative ones. A blind-matching test that
scores well on distinguishable documents proves only that BM25 works. What
matters is that the same test returns *chance* when handed boilerplate -- if it
cannot fail, a high score on real data means nothing.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from regime_narrative.controls.blind_match import (
    BM25,
    blind_match,
    count_stripped,
    strip_identifying_details,
    tokenise,
)
from regime_narrative.controls.placebo import (
    PlaceboSamplingError,
    placebo_balance_report,
    sample_placebo_dates,
)
from regime_narrative.transitions import Transition


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_ranks_the_document_containing_the_query_terms_first():
    corpus = [
        tokenise("central bank raises interest rates inflation surges"),
        tokenise("hurricane makes landfall coastal evacuation flooding"),
        tokenise("football championship final penalty shootout stadium"),
    ]
    bm25 = BM25(corpus)
    scores = bm25.scores(tokenise("hurricane flooding evacuation"))
    assert scores.argmax() == 1


def test_bm25_returns_zeros_for_a_query_with_no_overlap():
    bm25 = BM25([tokenise("alpha beta gamma"), tokenise("delta epsilon zeta")])
    scores = bm25.scores(tokenise("xylophone quixotic"))
    assert scores.tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# blind matching -- positive and, crucially, negative cases
# ---------------------------------------------------------------------------


def _distinguishable_windows(n: int = 8) -> tuple[dict, dict]:
    """Windows about genuinely different topics, split into two halves."""
    topics = [
        ("banking", "bank lender deposit withdrawal solvency capital regulator"),
        ("hurricane", "hurricane storm landfall evacuation flooding coastal levee"),
        ("election", "election ballot candidate parliament coalition vote turnout"),
        ("pandemic", "outbreak quarantine hospital infection transmission virus"),
        ("energy", "pipeline refinery barrel crude opec embargo tanker"),
        ("conflict", "airstrike ceasefire troops border shelling casualties"),
        ("tech", "semiconductor chip fabrication export licence foundry"),
        ("shipping", "container port berth freight congestion dockworkers"),
    ][:n]
    gen = {f"w{i}": f"{words} {words}" for i, (_, words) in enumerate(topics)}
    hold = {f"w{i}": f"{words} {words} {words}" for i, (_, words) in enumerate(topics)}
    return gen, hold


def test_blind_match_recovers_distinguishable_windows():
    gen, hold = _distinguishable_windows()
    res = blind_match(gen, hold, n_permutations=2000)
    assert res.n_correct == res.n_scored
    assert res.accuracy == 1.0
    assert res.p_value < 0.01


def test_blind_match_returns_chance_on_boilerplate():
    """The negative control. If this passes at high accuracy the test is broken."""
    boilerplate = (
        "market conditions deteriorated amid heightened uncertainty as investors "
        "reassessed risk across asset classes and volatility increased"
    )
    gen = {f"w{i}": boilerplate for i in range(10)}
    hold = {f"w{i}": boilerplate for i in range(10)}
    res = blind_match(gen, hold, n_permutations=2000)
    # Every document is identical, so nothing is recoverable beyond chance.
    assert res.accuracy <= 0.2
    assert res.p_value > 0.05


def test_blind_match_p_value_is_not_significant_for_a_single_lucky_hit():
    gen, hold = _distinguishable_windows(n=8)
    # Scramble all but one explanation into boilerplate.
    noise = "generic commentary about conditions and sentiment and positioning"
    scrambled = {k: (v if k == "w0" else noise) for k, v in gen.items()}
    res = blind_match(scrambled, hold, n_permutations=4000)
    assert res.n_correct <= 2
    assert res.p_value > 0.05


def test_blind_match_reports_chance_as_one_over_candidate_count():
    gen, hold = _distinguishable_windows(n=8)
    res = blind_match(gen, hold, n_permutations=500)
    assert res.chance == pytest.approx(1 / 8)


def test_blind_match_uses_all_holdout_windows_as_distractors():
    """A window with no explanation must still compete as a candidate."""
    gen, hold = _distinguishable_windows(n=8)
    gen.pop("w7")
    res = blind_match(gen, hold, n_permutations=500)
    assert res.n_candidates == 8
    assert res.n_scored == 7
    assert res.chance == pytest.approx(1 / 8)


def test_blind_match_needs_at_least_two_candidates():
    with pytest.raises(ValueError):
        blind_match({"a": "text"}, {"a": "text"}, n_permutations=10)


# ---------------------------------------------------------------------------
# date scrubbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "In March 2020 the market fell",
        "On 2020-03-03 the Fed cut rates",
        "During Q1 the index declined",
        "The 3/3/2020 announcement",
        "Events of February 2018 drove the move",
    ],
)
def test_scrubber_removes_identifying_dates(text):
    cleaned = strip_identifying_details(text)
    assert not any(tok in cleaned.lower() for tok in
                   ["2020", "2018", "march", "february", "q1", "3/3/2020"])


def test_scrubber_keeps_the_substance():
    cleaned = strip_identifying_details(
        "In March 2020 the Federal Reserve cut rates by 50 basis points"
    )
    assert "Federal Reserve" in cleaned
    assert "basis points" in cleaned


def test_count_stripped_reports_removals():
    assert count_stripped("nothing to remove here") == 0
    assert count_stripped("March 2020 and December 2021") > 0


# ---------------------------------------------------------------------------
# placebo sampling
# ---------------------------------------------------------------------------


def _states_series(start: date = date(2013, 1, 1), n: int = 3000) -> pd.Series:
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([0] * n, index=idx)


def _transition(d: date) -> Transition:
    return Transition(
        date=d, from_state=0, to_state=1, dwell_before=30, dwell_after=30,
        at_fold_boundary=False,
    )


def test_offset_must_exceed_window_or_sampling_refuses():
    """The constraint that keeps control and transition windows disjoint."""
    states = _states_series()
    with pytest.raises(PlaceboSamplingError):
        sample_placebo_dates(
            [_transition(date(2018, 6, 1))], states, window_days=30, offset_min=14,
            offset_max=90,
        )


def test_placebo_windows_never_overlap_any_transition_window():
    states = _states_series()
    transitions = [
        _transition(d)
        for d in [date(2018, 1, 26), date(2018, 5, 8), date(2018, 10, 10),
                  date(2019, 3, 1), date(2019, 8, 5)]
    ]
    placebos = sample_placebo_dates(
        transitions, states, window_days=14, offset_min=30, offset_max=90,
        n_per_transition=1, seed=1,
    )
    assert placebos
    for p in placebos:
        for t in transitions:
            assert abs((p.date - t.date).days) >= 14, (
                f"placebo {p.date} overlaps the window of transition {t.date}"
            )


def test_placebo_offsets_stay_inside_the_requested_band():
    states = _states_series()
    transitions = [_transition(date(2018, 6, 1)), _transition(date(2019, 6, 1))]
    placebos = sample_placebo_dates(
        transitions, states, window_days=14, offset_min=30, offset_max=90, seed=3
    )
    for p in placebos:
        assert 30 <= abs(p.offset_days) <= 90


def test_placebo_dates_are_trading_days_present_in_the_state_series():
    states = _states_series()
    transitions = [_transition(date(2018, 6, 1))]
    placebos = sample_placebo_dates(transitions, states, window_days=14, seed=5)
    observed = {d.date() for d in states.index}
    for p in placebos:
        assert p.date in observed


def test_placebo_dates_are_not_reused_across_transitions():
    states = _states_series()
    transitions = [_transition(date(2018, 6, 1) + timedelta(days=120 * i)) for i in range(6)]
    placebos = sample_placebo_dates(transitions, states, window_days=14, seed=9)
    assert len({p.date for p in placebos}) == len(placebos)


def test_balance_report_detects_an_era_mismatch():
    """Controls drawn from a different era must show up as a year gap."""
    states = _states_series()
    transitions = [_transition(date(2020, 3, 2))]
    good = sample_placebo_dates(transitions, states, window_days=14, seed=11)
    report = placebo_balance_report(good, transitions)
    assert report["year_mean_gap"] <= 1.0

    from regime_narrative.controls.placebo import PlaceboDate

    bad = [PlaceboDate(date=date(2013, 5, 2), matched_to=date(2020, 3, 2),
                       offset_days=-2496, state_at_date=0,
                       days_to_nearest_transition=2496)]
    bad_report = placebo_balance_report(bad, transitions)
    assert bad_report["year_mean_gap"] > 5


def test_multiple_controls_per_transition_are_mutually_disjoint():
    """Regression: two picks for one transition must be checked against each other.

    An earlier version filtered all candidates against a stale ``used`` set and
    then sliced, so the second control for a transition never saw the first.
    That produced 2019-05-30 and 2019-05-31 -- one day apart, sharing thirteen
    of fourteen days of news.
    """
    states = _states_series()
    transitions = [_transition(date(2019, 6, 28)), _transition(date(2020, 3, 2))]
    placebos = sample_placebo_dates(
        transitions, states, window_days=14, offset_min=30, offset_max=90,
        n_per_transition=2, min_gap_days=21, seed=17,
    )
    assert len(placebos) >= 3
    for i, a in enumerate(placebos):
        for b in placebos[i + 1:]:
            assert abs((a.date - b.date).days) >= 21, (
                f"controls {a.date} and {b.date} are only "
                f"{abs((a.date - b.date).days)} days apart"
            )


def test_min_gap_is_never_below_the_window_length():
    """Even if a caller passes a small gap, windows must not overlap."""
    states = _states_series()
    transitions = [_transition(date(2019, 6, 28))]
    placebos = sample_placebo_dates(
        transitions, states, window_days=14, offset_min=30, offset_max=90,
        n_per_transition=3, min_gap_days=2, seed=23,
    )
    for i, a in enumerate(placebos):
        for b in placebos[i + 1:]:
            assert abs((a.date - b.date).days) >= 14
