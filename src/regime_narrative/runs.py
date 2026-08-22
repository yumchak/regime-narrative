"""Saved runs, so a browser refresh does not throw away fifty minutes of work.

A run over twenty dates takes the better part of an hour, most of it waiting on
a rate-limited news API. Holding the assembled result only in Streamlit session
state means a refresh, a laptop sleep or a stray Ctrl-R loses all of it.

What is actually at risk is narrower than it looks. Individual narratives are
already cached by input hash, and news windows are cached on disk, so a repeat
run costs nothing and returns almost instantly. What is lost is the *assembled*
result: the control statistics, the blind-match scores, and the table the user
was reading. That is what this module persists.

Runs are stored as one self-describing JSON file each. No database, no schema
migration, and a file a researcher can open, diff or email.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import cache_dir

RUNS_DIRNAME = "runs"
_SAFE = re.compile(r"[^a-z0-9_-]+")


def runs_dir() -> Path:
    return cache_dir(RUNS_DIRNAME)


def _slug(text: str, limit: int = 40) -> str:
    out = _SAFE.sub("-", str(text).strip().lower()).strip("-")
    return (out[:limit] or "run").strip("-")


@dataclass(frozen=True)
class RunSummary:
    """Enough to populate a picker without opening the whole file."""

    run_id: str
    label: str
    saved_at: str
    n_events: int
    n_controls: int
    n_narratives: int
    model: str
    window_days: int
    path: Path

    @property
    def display(self) -> str:
        when = self.saved_at[:16].replace("T", " ")
        return (f"{self.label}  ·  {self.n_events} events + {self.n_controls} "
                f"controls  ·  {when}")


def save_run(
    result: Any,
    *,
    cfg: Any,
    model: str,
    label: str = "",
    now: datetime | None = None,
) -> str:
    """Persist an assembled pipeline result. Returns the run id.

    ``now`` is injectable so tests do not depend on the clock.
    """
    from .pipeline import to_records

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    label = label.strip() or f"{len(result.transitions)} dates"
    run_id = f"{stamp}_{_slug(label)}"

    windows = {
        wid: {
            "kind": w["kind"],
            "date": w["date"],
            "n_items": len(w["full"]),
            "n_generation": len(w["gen"]),
            "n_holdout": len(w["hold"]),
            "window_start": w["full"].start_instant.isoformat(),
            "window_end": w["full"].end_instant.isoformat(),
            "dropped_post_boundary": w["full"].dropped_post_boundary,
        }
        for wid, w in result.windows.items()
    }

    payload = {
        "run_id": run_id,
        "label": label,
        "saved_at": (now or datetime.now(timezone.utc)).isoformat(),
        "model": model,
        "settings": {
            "window_days": cfg.window_days,
            "max_items": cfg.max_items,
            "holdout_fraction": cfg.holdout_fraction,
            "controls_per_event": cfg.placebo_per_transition,
            "control_offset_min": cfg.placebo_offset_min,
            "control_offset_max": cfg.placebo_offset_max,
            "control_min_gap": cfg.placebo_min_gap,
        },
        "event_dates": [d.isoformat() for d in result.transitions],
        "control_dates": [d.isoformat() for d in result.controls],
        "retrieval": result.retrieval,
        "placebo": result.placebo,
        # per_item is large and only used for debugging; drop it from the file.
        "blind_match": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_item"}
            for k, v in result.blind_match.items() if isinstance(v, dict)
        },
        "faithfulness": {
            k: v for k, v in result.faithfulness.items() if k != "per_window"
        },
        "windows": windows,
        "narratives": {k: n.as_dict() for k, n in result.narratives.items()},
        "records": to_records(result).to_dict("records"),
        "warnings": list(result.warnings),
    }

    path = runs_dir() / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return run_id


def list_runs() -> list[RunSummary]:
    """Newest first. A corrupt or half-written file is skipped, not fatal."""
    out: list[RunSummary] = []
    for path in sorted(runs_dir().glob("*.json"), reverse=True):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict) or "run_id" not in d:
            continue
        out.append(RunSummary(
            run_id=d["run_id"],
            label=d.get("label", d["run_id"]),
            saved_at=d.get("saved_at", ""),
            n_events=len(d.get("event_dates", [])),
            n_controls=len(d.get("control_dates", [])),
            n_narratives=len(d.get("narratives", {})),
            model=d.get("model", "?"),
            window_days=d.get("settings", {}).get("window_days", 0),
            path=path,
        ))
    out.sort(key=lambda r: r.saved_at, reverse=True)
    return out


def load_run(run_id: str) -> dict | None:
    path = runs_dir() / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def delete_run(run_id: str) -> bool:
    path = runs_dir() / f"{run_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True
