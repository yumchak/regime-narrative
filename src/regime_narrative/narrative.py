"""The language layer.

Architectural boundary, stated here because it is the point of the project:
the HMM decides *when* the regime changed; this module only describes *what was
in the news* at that moment. No number in the results comes from the language
model, and nothing here can move a regime boundary. The only inputs are a window
of dated items that the retrieval layer has already proved closes at the
transition date.

Every call is logged with the model id, the prompt version, a hash of the exact
input, and the raw output, so any narrative in the report can be traced back to
the precise prompt and the precise news window that produced it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, cache_dir, load_settings, require_env
from .news.base import NewsWindow

PROMPT_DIR = PROJECT_ROOT / "prompts"

# Schema-enforced output shape. Kept in lockstep with the field list in
# prompts/narrative_v1.md -- the prompt explains what each field means, the
# schema guarantees the structure.
NARRATIVE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "driver_identified", "confidence", "summary", "mechanism", "claims",
        "primary_item_ids", "competing_explanation", "insufficient_evidence_reason",
    ],
    "properties": {
        "driver_identified": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "none"]},
        "summary": {"type": "string"},
        "mechanism": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "item_ids"],
                "properties": {
                    "claim": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "primary_item_ids": {"type": "array", "items": {"type": "string"}},
        "competing_explanation": {"type": "string"},
        "insufficient_evidence_reason": {"type": "string"},
    },
}

_RE_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)

# Absolute-date patterns stripped from item text before the model sees it.
_RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RE_MONTH_DAY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}(st|nd|rd|th)?\b",
    re.I,
)
_RE_DAY_MONTH = re.compile(
    r"\b\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\b",
    re.I,
)
_RE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Standalone month names ("elections in November") narrow the date almost as
# much as a full date does, and the Month-DD / DD-Month patterns above miss
# them. Matched case-sensitively so the verb "may" survives while the month
# "May" does not; a sentence-initial verbal "May" is rare enough in news copy
# to accept as over-scrubbing.
_RE_BARE_MONTH = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b"
)
# Relative-year phrasing that pins the window just as effectively.
# "May" is excluded here on purpose: "this may be revised" is a verb, and the
# case-insensitive match would mangle it. A genuine "this May" is still caught
# by _RE_BARE_MONTH, which is case-sensitive.
_RE_REL_YEAR = re.compile(
    r"\b(last|next|this|the following|the previous)\s+(year|month|week|quarter)\b",
    re.I,
)


@dataclass
class Narrative:
    window_id: str
    kind: str                     # "transition" or "placebo"
    boundary_date: str
    driver_identified: bool
    confidence: str
    summary: str
    mechanism: str
    claims: list[dict]
    primary_item_ids: list[str]
    competing_explanation: str
    insufficient_evidence_reason: str
    raw_response: str = ""
    parse_ok: bool = True
    parse_error: str = ""
    call_log_id: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_confident(self) -> bool:
        """The headline placebo statistic counts high and medium as confident."""
        return self.driver_identified and self.confidence in {"high", "medium"}


# ---------------------------------------------------------------------------
# prompt handling
# ---------------------------------------------------------------------------


def load_prompt(version: str | None = None) -> tuple[str, str]:
    """Return (prompt text without frontmatter, sha256 of the whole file).

    Prompts live in version-controlled files, never as inline strings, so the
    exact wording behind any result is recoverable from git.
    """
    cfg = load_settings()
    version = version or cfg["llm"]["prompt_version"]
    path = PROMPT_DIR / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt {version!r} not found at {path}")
    raw = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return _RE_FRONTMATTER.sub("", raw).strip(), digest


def strip_dates(text: str) -> str:
    """Remove absolute dates from item text.

    Mitigation, not a cure. The model may still recognise an event from its
    content -- that is precisely why the placebo and citation-faithfulness
    checks exist. What this removes is the cheapest possible shortcut: reading
    the date off the page and reciting what it knows happened.
    """
    out = _RE_ISO.sub("[date]", text)
    out = _RE_MONTH_DAY.sub("[date]", out)
    out = _RE_DAY_MONTH.sub("[date]", out)
    out = _RE_BARE_MONTH.sub("[month]", out)
    out = _RE_REL_YEAR.sub("[relative period]", out)
    out = _RE_YEAR.sub("[year]", out)
    return out


def audit_date_leaks(text: str) -> dict:
    """Count residual date signals in text that has already been scrubbed.

    Run over every rendered prompt before generation. A non-zero count is a hole
    in the memorisation control, and the control is only as good as this audit.
    """
    return {
        "bare_years": len(_RE_YEAR.findall(text)),
        "iso_dates": len(_RE_ISO.findall(text)),
        "month_day": len(_RE_MONTH_DAY.findall(text)),
        "day_month": len(_RE_DAY_MONTH.findall(text)),
        "bare_months": len(_RE_BARE_MONTH.findall(text)),
        "relative_periods": len(_RE_REL_YEAR.findall(text)),
    }


def render_window(window: NewsWindow) -> tuple[str, dict[str, str]]:
    """Format a window as prompt text, returning (text, id -> original text).

    Items are labelled by relative day position rather than by date. Sequence is
    genuinely informative -- an escalating series reads differently from an
    isolated shock -- but absolute dates would hand the model the answer.
    """
    if not window.items:
        return "(no items were published in this window)", {}

    start = window.start_instant.date()
    lines: list[str] = []
    id_map: dict[str, str] = {}

    for item in window.items:
        day_index = (item.published.date() - start).days + 1
        section = item.extra.get("section", "")
        text = strip_dates(item.text or item.title)
        id_map[item.item_id] = item.text or item.title
        prefix = f"[{item.item_id}] Day {day_index}"
        if section:
            prefix += f" | {section}"
        lines.append(f"{prefix}\n{text}")

    header = (
        f"There are {len(window.items)} items, covering a window of "
        f"{window.window_days} days. Day {window.window_days} is the day the "
        f"model flagged.\n"
    )
    return header + "\n\n".join(lines), id_map


# ---------------------------------------------------------------------------
# call logging
# ---------------------------------------------------------------------------


def _log_call(entry: dict) -> str:
    """Append one call record and return its id."""
    log_dir = cache_dir("llm_calls")
    call_id = hashlib.sha256(
        json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    entry = {"call_id": call_id, **entry}
    (log_dir / f"{call_id}.json").write_text(
        json.dumps(entry, indent=2, default=str), encoding="utf-8"
    )
    with open(log_dir / "calls.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: v for k, v in entry.items() if k != "response_text"},
                            default=str) + "\n")
    return call_id


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def generate_narrative(
    window: NewsWindow,
    *,
    window_id: str,
    kind: str,
    prompt_version: str | None = None,
    model: str | None = None,
    use_cache: bool = True,
) -> Narrative:
    """One window in, one structured explanation out. No batching."""
    cfg = load_settings()
    model = model or cfg["llm"]["model"]
    system_prompt, prompt_hash = load_prompt(prompt_version)
    user_text, _ = render_window(window)

    input_hash = hashlib.sha256(
        f"{model}|{prompt_hash}|{user_text}".encode("utf-8")
    ).hexdigest()[:16]

    cache_path = cache_dir("narratives") / f"{window_id}_{input_hash}.json"
    if use_cache and cache_path.exists():
        return Narrative(**json.loads(cache_path.read_text(encoding="utf-8")))

    import anthropic

    client = anthropic.Anthropic(
        api_key=require_env(
            "ANTHROPIC_API_KEY",
            hint="Set it before running stage 3; every other stage runs without it.",
        )
    )

    response = client.messages.create(
        model=model,
        max_tokens=cfg["llm"]["max_tokens"],
        system=system_prompt,
        messages=[{"role": "user", "content": user_text}],
        # Sampling parameters were removed on current models -- passing
        # temperature returns a 400. Depth is controlled by effort instead.
        output_config={
            "effort": cfg["llm"]["effort"],
            # Schema-constrained output. The prompt still describes the fields,
            # because the model reasons better when it knows what it is filling
            # in, but the schema is what guarantees the shape. This removes the
            # parse-failure path entirely rather than handling it.
            "format": {"type": "json_schema", "schema": NARRATIVE_SCHEMA},
        },
    )
    raw = "".join(block.text for block in response.content if block.type == "text")

    call_id = _log_call(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "window_id": window_id,
            "kind": kind,
            "boundary_date": window.boundary_date.isoformat(),
            "model": model,
            "prompt_version": prompt_version or cfg["llm"]["prompt_version"],
            "prompt_sha256_16": prompt_hash,
            "effort": cfg["llm"]["effort"],
            "output_schema_enforced": True,
            "input_hash": input_hash,
            "n_items_in_window": len(window),
            "input_chars": len(user_text),
            "usage": {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            },
            "response_text": raw,
        }
    )

    narrative = _parse(raw, window_id=window_id, kind=kind,
                       boundary_date=window.boundary_date.isoformat(), call_log_id=call_id)
    cache_path.write_text(json.dumps(narrative.as_dict(), indent=2), encoding="utf-8")
    return narrative


def _parse(raw: str, *, window_id: str, kind: str, boundary_date: str,
           call_log_id: str) -> Narrative:
    """Parse the model's JSON, degrading to a recorded failure rather than raising."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        text = brace.group(0)

    base = dict(
        window_id=window_id, kind=kind, boundary_date=boundary_date,
        raw_response=raw, call_log_id=call_log_id,
    )
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        return Narrative(
            **base, driver_identified=False, confidence="none", summary="",
            mechanism="", claims=[], primary_item_ids=[], competing_explanation="",
            insufficient_evidence_reason="", parse_ok=False,
            parse_error=f"JSONDecodeError: {exc}",
        )

    return Narrative(
        **base,
        driver_identified=bool(d.get("driver_identified", False)),
        confidence=str(d.get("confidence", "none")).lower(),
        summary=str(d.get("summary", "")),
        mechanism=str(d.get("mechanism", "")),
        claims=list(d.get("claims", []) or []),
        primary_item_ids=list(d.get("primary_item_ids", []) or []),
        competing_explanation=str(d.get("competing_explanation", "")),
        insufficient_evidence_reason=str(d.get("insufficient_evidence_reason", "")),
        parse_ok=True,
    )
