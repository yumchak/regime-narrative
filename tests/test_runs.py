"""Saved runs must survive the thing they exist to survive: losing the process.

The failure this guards against is a user losing fifty minutes of work to a
browser refresh, so the tests write a run, throw away every in-memory object,
and read it back from disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from regime_narrative import runs as runstore
from regime_narrative.narrative import Narrative
from regime_narrative.news.base import NewsItem, build_window, split_holdout
from regime_narrative.pipeline import PipelineConfig, PipelineResult


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    """Never touch the real cache from a test."""
    monkeypatch.setattr(runstore, "runs_dir", lambda: tmp_path)
    return tmp_path


def _window(day: date, n_items: int = 6):
    items = [
        NewsItem(
            source="test",
            published=datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc),
            title=f"headline {i}",
            text=f"body text number {i} about something specific",
            extra={"section": "Business and economy", "event_date": day.isoformat()},
        )
        for i in range(n_items)
    ]
    return build_window(boundary_date=day, window_days=14, source="test",
                        candidate_items=items)


def _narrative(wid: str, kind: str, when: date) -> Narrative:
    return Narrative(
        window_id=wid, kind=kind, boundary_date=when.isoformat(),
        driver_identified=True, confidence="high",
        summary="A specific thing happened.", mechanism="It transmits like so.",
        claims=[{"claim": "A specific thing happened.", "item_ids": ["abc"]}],
        primary_item_ids=["abc"], competing_explanation="",
        insufficient_evidence_reason="",
    )


def _result() -> PipelineResult:
    ev, ctl = date(2020, 3, 3), date(2020, 5, 12)
    res = PipelineResult(transitions=[ev], controls=[ctl])
    for kind, d in (("transition", ev), ("control", ctl)):
        w = _window(d)
        gen, hold = split_holdout(w, fraction=0.5, seed=1)
        wid = f"{kind}_{d.isoformat()}"
        res.windows[wid] = {"kind": kind, "date": d.isoformat(),
                            "full": w, "gen": gen, "hold": hold}
        res.narratives[wid] = _narrative(wid, kind, d)
    res.retrieval = {"n_requested": 2, "n_retrieved": 2, "n_failed": 0,
                     "total_items": 12, "any_post_boundary_survivors": False}
    res.placebo = {"transitions": {"n": 1, "n_confident": 1, "confident_rate": 1.0},
                   "controls": {"n": 1, "n_confident": 0, "confident_rate": 0.0},
                   "fisher_p": 1.0}
    res.blind_match = {"transitions_all_sections": {
        "n_correct": 1, "n_scored": 1, "accuracy": 1.0, "chance": 0.5,
        "p_value": 0.5, "per_item": [{"big": "payload"}]}}
    res.faithfulness = {"all": {"n_claims": 2, "grounding_rate": 1.0},
                        "per_window": [{"big": "payload"}]}
    res.warnings = ["a warning worth keeping"]
    return res


def _cfg() -> PipelineConfig:
    return PipelineConfig(window_days=14, placebo_per_transition=2)


# ---------------------------------------------------------------------------
# the point of the module
# ---------------------------------------------------------------------------


def test_a_saved_run_survives_losing_every_in_memory_object():
    rid = runstore.save_run(_result(), cfg=_cfg(), model="claude-opus-5",
                            label="my run")
    loaded = runstore.load_run(rid)          # read back from disk only
    assert loaded is not None
    assert loaded["label"] == "my run"
    assert loaded["model"] == "claude-opus-5"
    assert loaded["event_dates"] == ["2020-03-03"]
    assert loaded["control_dates"] == ["2020-05-12"]
    assert len(loaded["narratives"]) == 2
    assert loaded["retrieval"]["total_items"] == 12
    assert loaded["placebo"]["fisher_p"] == 1.0
    assert loaded["warnings"] == ["a warning worth keeping"]


def test_saved_run_carries_the_table_the_user_was_reading():
    rid = runstore.save_run(_result(), cfg=_cfg(), model="m")
    recs = runstore.load_run(rid)["records"]
    assert len(recs) == 2
    assert {"window_id", "kind", "date", "confidence"} <= set(recs[0])


def test_settings_are_recorded_so_a_reload_says_what_produced_it():
    rid = runstore.save_run(_result(), cfg=_cfg(), model="m")
    st = runstore.load_run(rid)["settings"]
    assert st["window_days"] == 14
    assert st["controls_per_event"] == 2


def test_bulky_debug_payloads_are_not_written():
    """per_item and per_window are large and only useful while debugging."""
    rid = runstore.save_run(_result(), cfg=_cfg(), model="m")
    raw = (runstore.runs_dir() / f"{rid}.json").read_text(encoding="utf-8")
    assert "big" not in raw
    loaded = runstore.load_run(rid)
    assert "per_item" not in loaded["blind_match"]["transitions_all_sections"]
    assert "per_window" not in loaded["faithfulness"]


# ---------------------------------------------------------------------------
# listing, deleting, and not falling over
# ---------------------------------------------------------------------------


def test_runs_list_newest_first():
    for i, lbl in enumerate(["oldest", "middle", "newest"]):
        runstore.save_run(_result(), cfg=_cfg(), model="m", label=lbl,
                          now=datetime(2026, 8, 20 + i, 12, tzinfo=timezone.utc))
    assert [r.label for r in runstore.list_runs()] == ["newest", "middle", "oldest"]


def test_summary_has_what_a_picker_needs():
    runstore.save_run(_result(), cfg=_cfg(), model="claude-opus-5", label="x")
    s = runstore.list_runs()[0]
    assert s.n_events == 1 and s.n_controls == 1 and s.n_narratives == 2
    assert s.model == "claude-opus-5" and s.window_days == 14
    assert "1 events" in s.display or "events" in s.display


def test_delete_removes_it():
    rid = runstore.save_run(_result(), cfg=_cfg(), model="m")
    assert runstore.delete_run(rid) is True
    assert runstore.load_run(rid) is None
    assert runstore.list_runs() == []


def test_deleting_something_that_is_not_there_is_not_an_error():
    assert runstore.delete_run("nope") is False


def test_loading_something_that_is_not_there_returns_none():
    assert runstore.load_run("nope") is None


def test_a_corrupt_file_is_skipped_rather_than_breaking_the_list(isolated_runs_dir):
    """A half-written file must not make every saved run unreachable."""
    runstore.save_run(_result(), cfg=_cfg(), model="m", label="good")
    (isolated_runs_dir / "20260101-000000_broken.json").write_text(
        "{not valid json", encoding="utf-8")
    (isolated_runs_dir / "20260101-000001_wrongshape.json").write_text(
        '["a list, not a run"]', encoding="utf-8")
    listed = runstore.list_runs()
    assert len(listed) == 1
    assert listed[0].label == "good"


def test_labels_with_awkward_characters_produce_a_safe_filename():
    rid = runstore.save_run(_result(), cfg=_cfg(), model="m",
                            label="../../etc/passwd  &  <script>")
    assert "/" not in rid and "\\" not in rid and "<" not in rid
    assert runstore.load_run(rid) is not None


def test_two_runs_saved_in_the_same_second_do_not_collide():
    when = datetime(2026, 8, 22, 10, 30, 0, tzinfo=timezone.utc)
    a = runstore.save_run(_result(), cfg=_cfg(), model="m", label="alpha", now=when)
    b = runstore.save_run(_result(), cfg=_cfg(), model="m", label="beta", now=when)
    assert a != b
    assert runstore.load_run(a)["label"] == "alpha"
    assert runstore.load_run(b)["label"] == "beta"
