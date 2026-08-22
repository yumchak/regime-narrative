"""Control 3: era-matched placebo dates.

The brief originally proposed sampling placebo dates from quiet periods. That
biases the test toward passing. Transitions cluster in 2015-16, 2018, 2020 and
2022; quiet periods are 2013, 2017 and 2023-24. Comparing crisis-era news
against calm-era news measures the era, not the regime label, and the model
declining to explain a slow Tuesday in 2017 would prove nothing.

So each transition gets a control drawn from its own neighbourhood: between
``offset_days_min`` and ``offset_days_max`` away, same news environment, same
volume of world events, differing only in whether the HMM called a transition.

Four constraints on a valid control date:

1.  **Window disjointness.** The control's news window must share no day with
    the transition's window, or the two would be partly the same articles.
    Guaranteed by requiring ``offset_days_min > window_days``, which is
    asserted rather than assumed.
2.  **Not near any other transition.** A control must be clear of *every*
    transition's window, not just its own partner's.
3.  **Disjoint from other controls.** Two controls sampled days apart would
    share most of their articles. This was not hypothetical: an early run drew
    2019-04-08 and 2019-04-15, seven days apart, and the blind-matching ceiling
    test then predicted one from the other.
4.  **Inside the observed sample.** The date must be a day the model actually
    labelled, so a state is defined there.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import load_settings
from ..transitions import Transition


@dataclass
class PlaceboDate:
    date: date
    matched_to: date          # the transition it is era-matched against
    offset_days: int          # signed: negative = before the transition
    state_at_date: int
    days_to_nearest_transition: int

    @property
    def stratum(self) -> str:
        """Controls are not all equally clean.

        A control that happens to sit inside a stressed regime, or close to
        another transition, is a weaker counterfactual than one drawn from a
        calm stretch well away from any detected change. Both are legitimate
        non-transition dates and both are kept, but the headline comparison is
        pre-declared against the clean stratum so the number cannot be read as
        either flattered or sandbagged by the mix.
        """
        if self.state_at_date == 0 and self.days_to_nearest_transition >= 30:
            return "clean"
        return "contaminated"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        d["matched_to"] = self.matched_to.isoformat()
        d["stratum"] = self.stratum
        return d


class PlaceboSamplingError(RuntimeError):
    pass


def sample_placebo_dates(
    transitions: list[Transition],
    states: pd.Series,
    *,
    window_days: int | None = None,
    offset_min: int | None = None,
    offset_max: int | None = None,
    n_per_transition: int | None = None,
    min_gap_days: int | None = None,
    seed: int | None = None,
) -> list[PlaceboDate]:
    """One or more era-matched control dates per transition."""
    cfg = load_settings()
    window_days = window_days or cfg["news"]["window_days"]
    offset_min = offset_min or cfg["placebo"]["offset_days_min"]
    offset_max = offset_max or cfg["placebo"]["offset_days_max"]
    n_per_transition = n_per_transition or cfg["placebo"]["n_per_transition"]
    min_gap_days = min_gap_days or cfg["placebo"].get("min_gap_days", window_days)
    seed = cfg["placebo"]["seed"] if seed is None else seed

    if offset_min <= window_days:
        raise PlaceboSamplingError(
            f"offset_days_min ({offset_min}) must exceed news.window_days "
            f"({window_days}), otherwise the control window overlaps the "
            f"transition window and they share articles."
        )

    rng = np.random.default_rng(seed)
    observed = pd.DatetimeIndex(states.index)
    observed_dates = {d.date() for d in observed}
    transition_dates = [t.date for t in transitions]

    # Controls must be disjoint from each other, not only from transitions.
    # Without this, two controls can land days apart and share most of their
    # articles -- which happened: 2019-04-08 and 2019-04-15 were sampled seven
    # days apart, and the blind-matching ceiling test then predicted one from
    # the other, because they were largely the same fortnight of news.
    min_gap = max(window_days, min_gap_days or window_days)

    used: set[date] = set()
    out: list[PlaceboDate] = []

    def _is_clear(candidate: date) -> bool:
        """Clear of every transition window and every control already chosen."""
        for td in transition_dates:
            # Two windows of length W ending at a and b overlap iff |a-b| < W.
            if abs((candidate - td).days) < window_days:
                return False
        for pd_ in used:
            if abs((candidate - pd_).days) < min_gap:
                return False
        return True

    for t in transitions:
        candidates: list[date] = []
        for delta in range(offset_min, offset_max + 1):
            for signed in (-delta, delta):
                c = t.date + timedelta(days=signed)
                if c in observed_dates and c not in used:
                    candidates.append(c)

        if not candidates:
            continue

        rng.shuffle(candidates)
        # Accept one at a time, registering each before testing the next.
        # Filtering the whole list first and slicing would compare every
        # candidate against a stale ``used`` set, so the second control for a
        # given transition would never be checked against the first -- which is
        # exactly how 2019-05-30 and 2019-05-31 were once both selected, one day
        # apart, sharing thirteen of their fourteen days of news.
        chosen: list[date] = []
        for c in candidates:
            if len(chosen) >= n_per_transition:
                break
            if _is_clear(c):
                chosen.append(c)
                used.add(c)

        for c in chosen:
            nearest = min(abs((c - td).days) for td in transition_dates)
            out.append(
                PlaceboDate(
                    date=c,
                    matched_to=t.date,
                    offset_days=(c - t.date).days,
                    state_at_date=int(states.loc[pd.Timestamp(c)]),
                    days_to_nearest_transition=nearest,
                )
            )

    out.sort(key=lambda p: p.date)
    return out


def placebo_balance_report(
    placebos: list[PlaceboDate], transitions: list[Transition]
) -> dict:
    """Evidence that the controls really are era-matched.

    If the two sets differ systematically in calendar year, the comparison is
    confounded and the placebo number cannot be interpreted. This is the check
    that says so.
    """
    t_years = pd.Series([t.date.year for t in transitions])
    p_years = pd.Series([p.date.year for p in placebos])

    return {
        "n_transitions": len(transitions),
        "n_placebos": len(placebos),
        "coverage": (
            round(len({p.matched_to for p in placebos}) / len(transitions), 3)
            if transitions
            else 0.0
        ),
        "transition_year_mean": round(float(t_years.mean()), 2) if len(t_years) else None,
        "placebo_year_mean": round(float(p_years.mean()), 2) if len(p_years) else None,
        "year_mean_gap": (
            round(abs(float(t_years.mean()) - float(p_years.mean())), 3)
            if len(t_years) and len(p_years)
            else None
        ),
        "mean_abs_offset_days": (
            round(float(np.mean([abs(p.offset_days) for p in placebos])), 1)
            if placebos
            else None
        ),
        "min_days_to_any_transition": (
            int(min(p.days_to_nearest_transition for p in placebos)) if placebos else None
        ),
        "state_split": (
            pd.Series([p.state_at_date for p in placebos]).value_counts().to_dict()
            if placebos
            else {}
        ),
        "stratum_split": {
            "clean": sum(1 for p in placebos if p.stratum == "clean"),
            "contaminated": sum(1 for p in placebos if p.stratum == "contaminated"),
        },
        "transitions_without_control": [
            t.date.isoformat()
            for t in transitions
            if t.date not in {p.matched_to for p in placebos}
        ],
    }
