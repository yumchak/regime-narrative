"""The reusable core: an auditable explanation layer for any regime model.

The study in this repository is one application of this pipeline, not the whole
of it. Given *any* list of dates -- from any regime model, any asset, any method
-- this module will:

    1. retrieve a news window that provably closes on each date
    2. generate a citation-grounded explanation from that window and nothing else
    3. sample era-matched control dates and run the identical pipeline on them
    4. measure whether the explanations are specific (blind matching) and
       whether their claims are supported (citation faithfulness)

Step 4 is the part that does not exist elsewhere. Asking a language model what
happened on a date is trivial; knowing whether the answer is window-specific or
generic boilerplate is not, and that is what makes this usable in research
rather than merely impressive in a demo.

Nothing here knows anything about HMMs. The regime model is the caller's
problem; this layer only ever sees dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import load_settings
from .controls.blind_match import blind_match, strip_identifying_details
from .controls.faithfulness import aggregate, audit_narrative
from .narrative import Narrative, generate_narrative
from .news.base import NewsWindow, split_holdout
from .news.wikipedia import fetch_window

BUSINESS_SECTIONS = {"business and economy", "economics and business", "business"}


@dataclass
class PipelineConfig:
    window_days: int = 14
    max_items: int = 250
    holdout_fraction: float = 0.5
    holdout_seed: int = 20260822
    placebo_offset_min: int = 30
    placebo_offset_max: int = 90
    placebo_per_transition: int = 2
    placebo_min_gap: int = 21
    placebo_seed: int = 20260822
    run_label: str = ""

    @classmethod
    def from_settings(cls) -> "PipelineConfig":
        cfg = load_settings()
        return cls(
            window_days=cfg["news"]["window_days"],
            max_items=cfg["news"]["max_items_per_window"],
            holdout_fraction=cfg["news"]["holdout_fraction"],
            holdout_seed=cfg["news"]["holdout_seed"],
            placebo_offset_min=cfg["placebo"]["offset_days_min"],
            placebo_offset_max=cfg["placebo"]["offset_days_max"],
            placebo_per_transition=cfg["placebo"]["n_per_transition"],
            placebo_min_gap=cfg["placebo"].get("min_gap_days", 21),
            placebo_seed=cfg["placebo"]["seed"],
        )

    def validate(self) -> None:
        if self.placebo_offset_min <= self.window_days:
            raise ValueError(
                f"placebo_offset_min ({self.placebo_offset_min}) must exceed "
                f"window_days ({self.window_days}), or control and transition "
                f"windows share articles."
            )


@dataclass
class PipelineResult:
    transitions: list[date]
    controls: list[date]
    windows: dict[str, dict] = field(default_factory=dict)
    narratives: dict[str, Narrative] = field(default_factory=dict)
    placebo: dict = field(default_factory=dict)
    blind_match: dict = field(default_factory=dict)
    faithfulness: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# input parsing -- accept whatever a researcher actually has
# ---------------------------------------------------------------------------


def parse_input_frame(df: pd.DataFrame) -> tuple[list[date], pd.Series | None, list[str]]:
    """Work out what the user uploaded.

    Three shapes are accepted, because these are the three things people
    actually have lying around:

      A. a single column of transition dates
      B. a date column plus a state/regime column -- transitions are derived
      C. a date index plus a state column

    Returns (transition_dates, state_series_or_None, notes).
    """
    notes: list[str] = []
    work = df.copy()

    # A DatetimeIndex is unambiguous -- promote it before anything else, or a
    # numeric state column can win the date-column contest below (pandas parses
    # integers as nanosecond epochs, so [0, 1] "succeeds" as dates).
    if isinstance(work.index, pd.DatetimeIndex):
        work = work.reset_index()
        work.columns = ["date"] + [str(c).strip() for c in work.columns[1:]]
        notes.append("Using the DatetimeIndex as the date column.")

    work.columns = [str(c).strip() for c in work.columns]
    lower = {c.lower(): c for c in work.columns}

    date_col = next(
        (lower[k] for k in ("date", "dates", "timestamp", "time", "day", "dt")
         if k in lower),
        None,
    )
    if date_col is None:
        # Fall back to whichever column parses best as dates. Tolerate bad rows:
        # requiring every value to parse would reject a real-world column for a
        # single typo and silently fall through to the index instead.
        best, best_frac = None, 0.0
        for c in work.columns:
            # Skip plain numeric columns: pandas happily reads them as epochs,
            # so a regime label of 0/1 parses to 1970 and wins on coverage.
            if pd.api.types.is_numeric_dtype(work[c]):
                continue
            try:
                parsed = pd.to_datetime(work[c], errors="coerce")
            except Exception:
                continue
            ok = parsed.dropna()
            if ok.empty:
                continue
            # Reject anything outside a plausible range for market data.
            if ok.min().year < 1900 or ok.max().year > 2100:
                continue
            frac = float(parsed.notna().mean())
            if frac > best_frac:
                best, best_frac = c, frac
        if best is not None and best_frac >= 0.6:
            date_col = best
            notes.append(
                f"Using column {best!r} as the date column "
                f"({best_frac:.0%} of its values parse as dates)."
            )
    if date_col is None:
        # Maybe the index carries the dates.
        try:
            work = work.reset_index()
            work.columns = [str(c).strip() for c in work.columns]
            date_col = work.columns[0]
            notes.append(f"Using the index as the date column ({date_col!r}).")
        except Exception as exc:
            raise ValueError(f"could not find a date column: {exc}")

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    _ok = work[date_col].dropna()
    if _ok.empty or _ok.min().year < 1900 or _ok.max().year > 2100:
        raise ValueError(
            f"column {date_col!r} does not contain plausible dates. Supply a CSV "
            f"with a date column, or a date column plus a state/regime column."
        )
    dropped = int(work[date_col].isna().sum())
    if dropped:
        notes.append(f"Dropped {dropped} row(s) with unparseable dates.")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    state_col = next(
        (lower[k] for k in ("state", "regime", "label", "cluster", "state_id")
         if k in lower),
        None,
    )

    if state_col is not None and state_col in work.columns:
        states = pd.Series(
            pd.to_numeric(work[state_col], errors="coerce").values,
            index=pd.DatetimeIndex(work[date_col]),
        ).dropna().astype(int)
        notes.append(
            f"Found a state column ({state_col!r}) with "
            f"{states.nunique()} distinct values over {len(states)} rows. "
            "Transitions will be derived with the persistence filter."
        )
        return [], states, notes

    dates = sorted({d.date() for d in work[date_col]})
    notes.append(f"Read {len(dates)} distinct dates and treated them as transitions.")
    return dates, None, notes


def derive_transitions(states: pd.Series, min_dwell: int) -> list[date]:
    """Transition dates from a caller-supplied state series."""
    from .transitions import extract_transitions

    binary = states
    if states.nunique() > 2:
        # Collapse multi-state to "is it the most volatile-looking state" is not
        # knowable here, so treat every change of label as a candidate instead.
        changes = states.ne(states.shift())
        changes.iloc[0] = False
        raw = [d.date() for d in states.index[changes]]
        runs = pd.Series(1, index=states.index).groupby(
            (states != states.shift()).cumsum()
        ).transform("size")
        keep = [
            d for d in raw
            if runs.loc[pd.Timestamp(d)] >= min_dwell
        ]
        return sorted(set(keep))

    return [t.date for t in extract_transitions(binary, min_dwell=min_dwell)]


def sample_controls(
    transitions: Sequence[date],
    *,
    universe: Iterable[date] | None,
    cfg: PipelineConfig,
) -> list[date]:
    """Era-matched control dates, disjoint from transitions and from each other.

    Mirrors ``controls.placebo`` but works from a bare date list so it can serve
    an arbitrary uploaded dataset with no state series behind it.
    """
    rng = np.random.default_rng(cfg.placebo_seed)
    allowed = set(universe) if universe is not None else None
    used: list[date] = []
    out: list[date] = []
    min_gap = max(cfg.window_days, cfg.placebo_min_gap)

    def clear(c: date) -> bool:
        if allowed is not None and c not in allowed:
            return False
        if any(abs((c - t).days) < cfg.window_days for t in transitions):
            return False
        if any(abs((c - u).days) < min_gap for u in used):
            return False
        return True

    for t in transitions:
        cands = [
            t + timedelta(days=s * d)
            for d in range(cfg.placebo_offset_min, cfg.placebo_offset_max + 1)
            for s in (-1, 1)
        ]
        rng.shuffle(cands)
        picked = 0
        for c in cands:
            if picked >= cfg.placebo_per_transition:
                break
            if clear(c):
                used.append(c)
                out.append(c)
                picked += 1

    return sorted(out)


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def estimate_cost(n_windows: int, cfg: PipelineConfig) -> dict:
    """What a run will cost in requests, wall-clock and tokens, before starting.

    A researcher deciding whether to press the button deserves to know. Wikipedia
    is throttled to roughly one request per second by policy, and that dominates
    the wall-clock, not the model.
    """
    http = n_windows * cfg.window_days
    retrieval_s = http * 1.15
    llm_s = n_windows * 35
    in_tok = n_windows * 5300
    out_tok = n_windows * 1500
    return {
        "n_windows": n_windows,
        "wikipedia_requests": http,
        "estimated_retrieval_minutes": round(retrieval_s / 60, 1),
        "estimated_generation_minutes": round(llm_s / 60, 1),
        "estimated_total_minutes": round((retrieval_s + llm_s) / 60, 1),
        "estimated_input_tokens": in_tok,
        "estimated_output_tokens": out_tok,
        "estimated_usd_opus": round(in_tok / 1e6 * 5 + out_tok / 1e6 * 25, 2),
        "note": (
            "Cached windows and cached narratives are free and instant on a "
            "rerun. Wikipedia throttling dominates the wall-clock, not the model."
        ),
    }


def run(
    transitions: Sequence[date],
    *,
    controls: Sequence[date] | None = None,
    cfg: PipelineConfig | None = None,
    generate: bool = True,
    progress: Callable[[str, float], None] | None = None,
) -> PipelineResult:
    """Retrieve, explain and measure. Works on any list of dates.

    ``generate=False`` runs retrieval and the controls that do not need a model,
    which is useful for checking coverage before spending anything.
    """
    cfg = cfg or PipelineConfig.from_settings()
    cfg.validate()

    transitions = sorted(set(transitions))
    controls = sorted(set(controls or []))
    result = PipelineResult(transitions=list(transitions), controls=list(controls))

    targets = [("transition", d) for d in transitions] + [("control", d) for d in controls]
    if not targets:
        result.warnings.append("no dates supplied")
        return result

    def tick(msg: str, frac: float) -> None:
        if progress:
            progress(msg, min(max(frac, 0.0), 1.0))

    # --- retrieval -------------------------------------------------------
    for i, (kind, d) in enumerate(targets):
        tick(f"Retrieving news for {d} ({i + 1}/{len(targets)})", i / (2 * len(targets)))
        try:
            w = fetch_window(d, window_days=cfg.window_days, max_items=cfg.max_items)
        except Exception as exc:
            result.warnings.append(f"{d}: retrieval failed -- {type(exc).__name__}: {exc}")
            continue
        if len(w) == 0:
            result.warnings.append(f"{d}: window is empty; skipped")
            continue
        gen, hold = split_holdout(w, fraction=cfg.holdout_fraction, seed=cfg.holdout_seed)
        result.windows[f"{kind}_{d.isoformat()}"] = {
            "kind": kind, "date": d.isoformat(),
            "full": w, "gen": gen, "hold": hold,
        }

    result.retrieval = {
        "n_requested": len(targets),
        "n_retrieved": len(result.windows),
        "n_failed": len(targets) - len(result.windows),
        "total_items": sum(len(v["full"]) for v in result.windows.values()),
        "any_post_boundary_survivors": any(
            len([i for i in v["full"].items
                 if i.published > v["full"].end_instant])
            for v in result.windows.values()
        ),
    }

    if not generate:
        return result

    # --- generation ------------------------------------------------------
    for i, (wid, w) in enumerate(result.windows.items()):
        tick(f"Explaining {w['date']} ({i + 1}/{len(result.windows)})",
             0.5 + i / (2 * len(result.windows)))
        try:
            result.narratives[wid] = generate_narrative(
                w["gen"], window_id=wid,
                kind="transition" if w["kind"] == "transition" else "placebo",
                run_label=cfg.run_label,
            )
        except Exception as exc:
            result.warnings.append(f"{w['date']}: generation failed -- {exc}")

    tick("Measuring", 0.97)
    _measure(result)
    tick("Done", 1.0)
    return result


def _measure(result: PipelineResult) -> None:
    """Run the controls over whatever was generated."""
    from scipy import stats as sps

    nar, win = result.narratives, result.windows
    t_keys = [k for k in nar if win[k]["kind"] == "transition"]
    c_keys = [k for k in nar if win[k]["kind"] == "control"]

    def rate(keys):
        if not keys:
            return {"n": 0}
        conf = sum(1 for k in keys if nar[k].is_confident)
        ident = sum(1 for k in keys if nar[k].driver_identified)
        return {
            "n": len(keys), "n_confident": conf,
            "confident_rate": round(conf / len(keys), 3),
            "n_driver_identified": ident,
            "driver_rate": round(ident / len(keys), 3),
        }

    tr, cr = rate(t_keys), rate(c_keys)
    fisher_p = None
    if tr.get("n") and cr.get("n"):
        table = [[tr["n_confident"], tr["n"] - tr["n_confident"]],
                 [cr["n_confident"], cr["n"] - cr["n_confident"]]]
        if all(sum(r) for r in table):
            _, fisher_p = sps.fisher_exact(table)

    result.placebo = {
        "transitions": tr, "controls": cr, "fisher_p": fisher_p,
        "power_note": (
            "With small n this test is easily underpowered. A non-significant "
            "result means the difference was not resolved, not that none exists."
        ),
    }

    # blind matching, over every window that has both an explanation and holdout
    def arm(keys, keep):
        expl, hold = {}, {}
        for k in keys:
            h = " ".join(i.body for i in win[k]["hold"].items if keep(i))
            n = nar[k]
            e = " ".join([n.summary, n.mechanism] +
                         [c.get("claim", "") for c in n.claims if isinstance(c, dict)])
            e = strip_identifying_details(e)
            if h.strip() and e.strip():
                hold[k], expl[k] = h, e
        if len(hold) < 2:
            return {"error": f"needs >= 2 windows, have {len(hold)}"}
        r = blind_match(expl, hold, n_permutations=10000)
        return {
            "n_correct": r.n_correct, "n_scored": r.n_scored,
            "n_candidates": r.n_candidates, "accuracy": round(r.accuracy, 4),
            "chance": round(r.chance, 4), "mrr": round(r.mean_reciprocal_rank, 4),
            "p_value": r.p_value, "per_item": r.per_item,
        }

    all_keys = list(nar)
    result.blind_match = {
        "transitions_all_sections": arm(t_keys, lambda i: True),
        "transitions_business_only": arm(
            t_keys, lambda i: str(i.extra.get("section", "")).lower() in BUSINESS_SECTIONS),
        "all_windows": arm(all_keys, lambda i: True),
    }

    reports = [audit_narrative(nar[k], win[k]["gen"]) for k in nar]
    result.faithfulness = aggregate(reports)
    result.faithfulness["per_window"] = [r.as_dict() for r in reports]


def to_records(result: PipelineResult) -> pd.DataFrame:
    """Flat table of everything, for export."""
    rows = []
    for wid, n in result.narratives.items():
        w = result.windows[wid]
        # .get rather than indexing: a malformed or partial faithfulness entry
        # should cost this row its grounding columns, not crash the whole export
        # the user is trying to download.
        fa = next(
            (r for r in result.faithfulness.get("per_window", [])
             if isinstance(r, dict) and r.get("window_id") == wid), {}
        )
        rows.append({
            "window_id": wid,
            "kind": w["kind"],
            "date": w["date"],
            "window_start": w["full"].start_instant.date().isoformat(),
            "window_end": w["full"].end_instant.date().isoformat(),
            "n_items": len(w["full"]),
            "n_shown_to_model": len(w["gen"]),
            "n_held_out": len(w["hold"]),
            "driver_identified": n.driver_identified,
            "confidence": n.confidence,
            "is_confident": n.is_confident,
            "summary": n.summary,
            "mechanism": n.mechanism,
            "competing_explanation": n.competing_explanation,
            "n_claims": len(n.claims),
            "grounding_rate": fa.get("grounding_rate"),
            "fabricated_citations": fa.get("n_fabricated"),
        })
    return pd.DataFrame(rows).sort_values(["kind", "date"]) if rows else pd.DataFrame()
