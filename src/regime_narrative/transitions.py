"""Transition extraction with persistence filtering and seed stability.

A single-day flip is noise. This module turns a state sequence into a short
list of defensible transition dates, and records everything needed to defend
each one:

*   **Persistence filter.** Runs shorter than ``min_dwell_days`` are absorbed
    into their neighbours before transitions are read off. Absorption is
    iterative and shortest-run-first, so the result does not depend on scan
    direction.

*   **Fold-boundary flagging.** A state change on the first day of a new
    walk-forward fold may be an artefact of refitting the model rather than a
    change in the market. These are flagged, and the primary specification
    excludes them.

*   **Seed stability.** The original notebook's own limitations section admits
    HMM labels move with ``random_state``. Every narrative in this project
    hangs off a transition date, so a date that survives only one seed is not
    a finding. ``stable_transitions`` keeps dates that recur across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

import numpy as np
import pandas as pd

from .config import load_settings


@dataclass
class Transition:
    date: date
    from_state: int
    to_state: int
    dwell_before: int
    dwell_after: int
    at_fold_boundary: bool
    fold_id: int | None = None
    stressed_prob: float = float("nan")
    seed_support: float = float("nan")   # fraction of seeds producing this date
    n_seeds_supporting: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d

    @property
    def direction(self) -> str:
        return "calm->stressed" if self.to_state == 1 else "stressed->calm"


# ---------------------------------------------------------------------------
# run-length machinery
# ---------------------------------------------------------------------------


def _runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    """Return [(state, start_idx, length), ...] for a state array."""
    if len(states) == 0:
        return []
    out = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append((int(states[start]), start, i - start))
            start = i
    return out


def apply_persistence_filter(states: pd.Series, min_dwell: int) -> pd.Series:
    """Absorb runs shorter than ``min_dwell`` into their neighbours.

    Repeatedly takes the shortest sub-threshold run and merges it into the
    adjacent run it is most compatible with (the longer neighbour; the only
    neighbour at a boundary), until every surviving run is long enough or the
    whole series is one run.
    """
    if min_dwell <= 1:
        return states.copy()

    values = states.values.astype(int).copy()

    while True:
        runs = _runs(values)
        if len(runs) <= 1:
            break
        short = [r for r in runs if r[2] < min_dwell]
        if not short:
            break

        # Shortest first; ties broken by earliest, so the result is deterministic.
        state_val, start, length = min(short, key=lambda r: (r[2], r[1]))
        pos = runs.index((state_val, start, length))

        if pos == 0:
            target = runs[1][0]
        elif pos == len(runs) - 1:
            target = runs[pos - 1][0]
        else:
            before, after = runs[pos - 1], runs[pos + 1]
            target = before[0] if before[2] >= after[2] else after[0]

        values[start : start + length] = target

    return pd.Series(values, index=states.index, name=states.name)


def extract_transitions(
    states: pd.Series,
    *,
    min_dwell: int | None = None,
    fold_id: pd.Series | None = None,
    stressed_prob: pd.Series | None = None,
    exclude_fold_boundaries: bool = True,
) -> list[Transition]:
    """Persistence-filtered transitions from a state sequence."""
    cfg = load_settings()
    min_dwell = cfg["transitions"]["min_dwell_days"] if min_dwell is None else min_dwell

    filtered = apply_persistence_filter(states, min_dwell)
    runs = _runs(filtered.values.astype(int))
    idx = filtered.index

    fold_boundary_dates: set[pd.Timestamp] = set()
    if fold_id is not None:
        changes = fold_id.ne(fold_id.shift())
        changes.iloc[0] = False
        fold_boundary_dates = set(fold_id.index[changes])

    transitions: list[Transition] = []
    for i in range(1, len(runs)):
        prev_state, _, prev_len = runs[i - 1]
        new_state, start, new_len = runs[i]
        when = idx[start]
        at_boundary = when in fold_boundary_dates

        if exclude_fold_boundaries and at_boundary:
            continue

        transitions.append(
            Transition(
                date=when.date(),
                from_state=prev_state,
                to_state=new_state,
                dwell_before=prev_len,
                dwell_after=new_len,
                at_fold_boundary=at_boundary,
                fold_id=int(fold_id.loc[when]) if fold_id is not None else None,
                stressed_prob=(
                    float(stressed_prob.loc[when]) if stressed_prob is not None else float("nan")
                ),
            )
        )
    return transitions


def dwell_distribution(states: pd.Series, *, min_dwell: int | None = None) -> pd.DataFrame:
    """Length of every regime episode, for the dwell-time histogram."""
    cfg = load_settings()
    min_dwell = cfg["transitions"]["min_dwell_days"] if min_dwell is None else min_dwell
    filtered = apply_persistence_filter(states, min_dwell)
    runs = _runs(filtered.values.astype(int))
    idx = filtered.index
    return pd.DataFrame(
        [
            {
                "state": s,
                "regime": "stressed" if s == 1 else "calm",
                "start": idx[start].date(),
                "end": idx[start + length - 1].date(),
                "length_days": length,
            }
            for s, start, length in runs
        ]
    )


def dwell_sensitivity(
    states: pd.Series,
    thresholds: list[int],
    *,
    fold_id: pd.Series | None = None,
    exclude_fold_boundaries: bool = True,
) -> pd.DataFrame:
    """Transition count as a function of the persistence threshold."""
    rows = []
    for t in thresholds:
        trans = extract_transitions(
            states,
            min_dwell=t,
            fold_id=fold_id,
            exclude_fold_boundaries=exclude_fold_boundaries,
        )
        dwell = dwell_distribution(states, min_dwell=t)
        rows.append(
            {
                "min_dwell_days": t,
                "n_transitions": len(trans),
                "n_to_stressed": sum(1 for x in trans if x.to_state == 1),
                "n_to_calm": sum(1 for x in trans if x.to_state == 0),
                "n_episodes": len(dwell),
                "median_episode_days": float(dwell["length_days"].median()),
                "mean_episode_days": float(dwell["length_days"].mean()),
                "max_episode_days": int(dwell["length_days"].max()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# seed stability
# ---------------------------------------------------------------------------


def stable_transitions(
    transitions_by_seed: dict[int, list[Transition]],
    *,
    tolerance_days: int | None = None,
    support_threshold: float | None = None,
) -> list[Transition]:
    """Keep transitions that recur across seeds.

    Two transitions from different seeds are the same event if they point the
    same direction and fall within ``tolerance_days`` of each other. The
    representative date is the median of the supporting dates, which is more
    robust than picking any single seed's answer.
    """
    cfg = load_settings()
    tolerance_days = (
        cfg["transitions"]["date_tolerance_days"] if tolerance_days is None else tolerance_days
    )
    support_threshold = (
        cfg["transitions"]["seed_stability_threshold"]
        if support_threshold is None
        else support_threshold
    )

    n_seeds = len(transitions_by_seed)
    if n_seeds == 0:
        return []

    # Flatten, then greedily cluster by (direction, date proximity).
    flat: list[tuple[int, Transition]] = [
        (seed, t) for seed, ts in transitions_by_seed.items() for t in ts
    ]
    flat.sort(key=lambda st: st[1].date)

    clusters: list[list[tuple[int, Transition]]] = []
    for seed, t in flat:
        placed = False
        for cluster in clusters:
            ref = cluster[0][1]
            if ref.to_state == t.to_state and abs((ref.date - t.date).days) <= tolerance_days:
                cluster.append((seed, t))
                placed = True
                break
        if not placed:
            clusters.append([(seed, t)])

    out: list[Transition] = []
    for cluster in clusters:
        seeds = {s for s, _ in cluster}
        support = len(seeds) / n_seeds
        if support < support_threshold:
            continue
        dates = sorted(t.date for _, t in cluster)
        rep_date = dates[len(dates) // 2]
        rep = next(t for _, t in cluster if t.date == rep_date)
        out.append(
            Transition(
                date=rep_date,
                from_state=rep.from_state,
                to_state=rep.to_state,
                dwell_before=int(np.median([t.dwell_before for _, t in cluster])),
                dwell_after=int(np.median([t.dwell_after for _, t in cluster])),
                at_fold_boundary=rep.at_fold_boundary,
                fold_id=rep.fold_id,
                stressed_prob=float(
                    np.nanmean([t.stressed_prob for _, t in cluster])
                ),
                seed_support=support,
                n_seeds_supporting=len(seeds),
            )
        )

    out.sort(key=lambda t: t.date)
    return out


def transitions_to_frame(transitions: list[Transition]) -> pd.DataFrame:
    if not transitions:
        return pd.DataFrame(
            columns=[
                "date", "direction", "dwell_before", "dwell_after",
                "at_fold_boundary", "seed_support", "stressed_prob",
            ]
        )
    return pd.DataFrame(
        [
            {
                "date": t.date.isoformat(),
                "direction": t.direction,
                "dwell_before": t.dwell_before,
                "dwell_after": t.dwell_after,
                "at_fold_boundary": t.at_fold_boundary,
                "seed_support": round(t.seed_support, 3) if t.seed_support == t.seed_support else None,
                "n_seeds": t.n_seeds_supporting,
                "stressed_prob": round(t.stressed_prob, 3) if t.stressed_prob == t.stressed_prob else None,
            }
            for t in transitions
        ]
    )
