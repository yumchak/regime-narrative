"""Wikipedia Current Events Portal, pinned to revisions that predate the boundary.

Why revision pinning is not optional
------------------------------------
The portal page for a given day keeps being edited for years afterwards. Fetching
``Portal:Current events/2020 March 3`` today returns 9,041 bytes; the revision as
it stood at 23:59 on 3 March 2020 was 7,548 bytes. Roughly a fifth of the page
was written later -- the most recent edit measured was from January 2023, almost
three years after the fact.

Using the current page would therefore feed the model text written with full
knowledge of what happened next, while the manifest cheerfully recorded the
window as closing on the transition date. The leak would be invisible and it
would invalidate every narrative result in the project.

So every page is fetched with ``rvstart`` set to the boundary instant and
``rvdir=older``, which returns the last revision at or before that moment. The
revision id and its timestamp go into the manifest, making each retrieval exactly
reproducible and independently checkable.

Two distinct dates are tracked per item:
    event date      -- the day the portal page covers      (<= boundary)
    knowledge date  -- the timestamp of the pinned revision (<= boundary)
Both must precede the boundary, and both are asserted.
"""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import cache_dir
from .base import NewsItem, NewsWindow, boundary_instant, build_window, window_start_instant

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia's user-agent policy asks for a descriptive agent with a contact
# route. Anonymous clients that ignore it get throttled hard, which is exactly
# what happened on the first full run.
USER_AGENT = (
    "regime-narrative/0.1 (University of Bristol student research project; "
    "https://github.com/regime-narrative) python-urllib"
)
MIN_REQUEST_INTERVAL = 1.1    # seconds between requests
MAX_RETRIES = 6
BACKOFF_BASE = 5.0            # seconds; doubled each attempt
MAX_BACKOFF = 120.0

SOURCE_NAME = "wikipedia_current_events"

_last_request_at = 0.0


class WikipediaFetchError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# HTTP with caching, throttling and backoff
# ---------------------------------------------------------------------------


def _cache_path(params: dict[str, str]) -> Path:
    key = urllib.parse.urlencode(sorted(params.items()))
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return cache_dir("news", "wikipedia_raw") / f"{digest}.json"


def _api_get(params: dict[str, str], *, use_cache: bool = True) -> dict[str, Any]:
    global _last_request_at
    full = dict(params)
    full.setdefault("format", "json")
    full.setdefault("formatversion", "2")

    path = _cache_path(full)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    url = API + "?" + urllib.parse.urlencode(full)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            _last_request_at = time.monotonic()
            if "error" in payload:
                info = payload["error"].get("info", str(payload["error"]))
                # maxlag is a transient server-load signal, not a bad request.
                if payload["error"].get("code") == "maxlag":
                    raise urllib.error.URLError(f"maxlag: {info}")
                raise WikipediaFetchError(info)
            path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            _last_request_at = time.monotonic()
            # Honour Retry-After when the server tells us how long to wait.
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after and str(retry_after).isdigit():
                delay = min(float(retry_after) + 1.0, MAX_BACKOFF)
            else:
                delay = min(BACKOFF_BASE * (2**attempt), MAX_BACKOFF)
            time.sleep(delay)
        except (
            urllib.error.URLError,
            TimeoutError,
            # RemoteDisconnected is an HTTPException/ConnectionResetError and is
            # not a URLError, so it escaped the handler above and killed a window
            # outright on the first full run.
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as exc:
            last_error = exc
            _last_request_at = time.monotonic()
            time.sleep(min(BACKOFF_BASE * (2**attempt), MAX_BACKOFF))

    raise WikipediaFetchError(f"failed after {MAX_RETRIES} attempts: {last_error}")


# ---------------------------------------------------------------------------
# wikitext parsing
# ---------------------------------------------------------------------------

_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S | re.I)
_RE_EXTLINK = re.compile(r"\[(https?://[^\s\]]+)\s+\(?([^\]]*?)\)?\]")
_RE_BARE_EXTLINK = re.compile(r"\[(https?://[^\s\]]+)\]")
_RE_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_HTMLTAG = re.compile(r"<[^>]+>")
_RE_BOLDITAL = re.compile(r"'{2,5}")
_RE_WS = re.compile(r"\s+")


@dataclass
class ParsedItem:
    section: str
    topic: str
    text: str
    sources: list[tuple[str, str]]   # (publisher, url)


def _clean_text(raw: str) -> tuple[str, list[tuple[str, str]]]:
    """Strip wiki markup, returning plain text and the external citations."""
    sources: list[tuple[str, str]] = []

    text = _RE_COMMENT.sub(" ", raw)
    text = _RE_REF.sub(" ", text)

    def _take_link(m: re.Match) -> str:
        url, label = m.group(1), (m.group(2) or "").strip()
        sources.append((label or _domain(url), url))
        return " "

    text = _RE_EXTLINK.sub(_take_link, text)
    text = _RE_BARE_EXTLINK.sub(lambda m: (sources.append((_domain(m.group(1)), m.group(1))) or " "), text)

    # Collapse templates repeatedly -- they nest.
    for _ in range(4):
        new = _RE_TEMPLATE.sub(" ", text)
        if new == text:
            break
        text = new

    text = _RE_WIKILINK.sub(lambda m: (m.group(2) or m.group(1)), text)
    text = _RE_HTMLTAG.sub(" ", text)
    text = _RE_BOLDITAL.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&ndash;", "-").replace("&mdash;", "-")
    text = text.replace("&amp;", "&").replace("&quot;", '"')
    text = _RE_WS.sub(" ", text).strip(" *;:-")
    return text, sources


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def parse_current_events(wikitext: str) -> list[ParsedItem]:
    """Turn one day's portal wikitext into leaf news items.

    Structure of these pages:
        ;Section heading
        *Topic or standalone item
        **Specific news item with citations

    Leaf bullets are the news; a ``*`` line with ``**`` children is a topic
    header and is carried as context rather than emitted on its own.
    """
    lines = wikitext.split("\n")
    section = ""
    topic = ""
    parsed: list[tuple[int, str, str, str]] = []   # (depth, section, topic, raw)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            sec, _ = _clean_text(stripped.lstrip(";"))
            section = sec
            topic = ""
            continue
        if stripped.startswith("*"):
            depth = len(stripped) - len(stripped.lstrip("*"))
            body = stripped.lstrip("*").strip()
            if depth == 1:
                cleaned, _ = _clean_text(body)
                topic = cleaned
            parsed.append((depth, section, topic if depth > 1 else "", body))

    # A depth-1 line is emitted only when it has no depth-2 children after it.
    items: list[ParsedItem] = []
    for i, (depth, sec, top, body) in enumerate(parsed):
        if depth == 1:
            has_child = i + 1 < len(parsed) and parsed[i + 1][0] > 1
            if has_child:
                continue
        text, sources = _clean_text(body)
        if len(text) < 25:      # drop stubs and stray markup
            continue
        items.append(ParsedItem(section=sec, topic=top, text=text, sources=sources))

    return items


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def _page_title(day: date) -> str:
    return f"Portal:Current events/{day.year} {day.strftime('%B')} {day.day}"


def fetch_day(day: date, *, boundary: date, use_cache: bool = True) -> tuple[list[NewsItem], dict]:
    """Items for one day, from the last revision at or before the boundary."""
    title = _page_title(day)
    cutoff = boundary_instant(boundary)

    payload = _api_get(
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "content|ids|timestamp",
            "rvslots": "main",
            "rvlimit": "1",
            "rvdir": "older",
            "rvstart": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        use_cache=use_cache,
    )

    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return [], {"title": title, "status": "no_page"}
    page = pages[0]
    if page.get("missing") or not page.get("revisions"):
        return [], {"title": title, "status": "missing_or_no_revision_before_boundary"}

    rev = page["revisions"][0]
    rev_ts = datetime.strptime(rev["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )

    # The knowledge date must also precede the boundary. If MediaWiki ever
    # returns a later revision, that is a leak and we refuse it outright.
    if rev_ts > cutoff:
        raise AssertionError(
            f"pinned revision {rev['revid']} for {title} is dated {rev_ts.isoformat()}, "
            f"after the boundary {cutoff.isoformat()}"
        )

    content = rev["slots"]["main"]["content"]
    parsed = parse_current_events(content)

    # Event timestamp: midday UTC on the day the page covers. The precise hour is
    # unknown, so it is flagged as assumed; midday keeps it unambiguously inside
    # the day and therefore inside the window.
    published = datetime.combine(day, dtime(12, 0), tzinfo=timezone.utc)

    items = [
        NewsItem(
            source=SOURCE_NAME,
            published=published,
            title=(p.topic or p.section or "Current events")[:200],
            text=p.text,
            url=p.sources[0][1] if p.sources else "",
            tz_assumed=True,
            extra={
                "section": p.section,
                "topic": p.topic,
                "event_date": day.isoformat(),
                "revision_id": rev["revid"],
                "revision_timestamp": rev_ts.isoformat(),
                "citations": [{"publisher": s, "url": u} for s, u in p.sources],
            },
        )
        for p in parsed
    ]

    meta = {
        "title": title,
        "status": "ok",
        "revision_id": rev["revid"],
        "revision_timestamp": rev_ts.isoformat(),
        "content_chars": len(content),
        "n_items": len(items),
    }
    return items, meta


def fetch_window(
    boundary: date,
    *,
    window_days: int,
    max_items: int | None = None,
    use_cache: bool = True,
    sections: Iterable[str] | None = None,
) -> NewsWindow:
    """A boundary-enforced window of Current Events items.

    ``sections`` optionally restricts to portal sections such as
    "Business and economy"; the default keeps everything, because restricting to
    business news would pre-select for market relevance and quietly make the
    placebo test easier to pass.
    """
    start = window_start_instant(boundary, window_days).date()
    days = [start + timedelta(days=n) for n in range((boundary - start).days + 1)]

    all_items: list[NewsItem] = []
    page_meta: list[dict] = []
    for day in days:
        items, meta = fetch_day(day, boundary=boundary, use_cache=use_cache)
        page_meta.append(meta)
        if sections:
            wanted = {s.lower() for s in sections}
            items = [i for i in items if str(i.extra.get("section", "")).lower() in wanted]
        all_items.extend(items)

    window = build_window(
        boundary_date=boundary,
        window_days=window_days,
        source=SOURCE_NAME,
        candidate_items=all_items,
        query="Portal:Current events, all sections",
        provenance={
            "api": API,
            "revision_pinned": True,
            "pinned_to_utc": boundary_instant(boundary).isoformat(),
            "pages": page_meta,
            "n_pages_ok": sum(1 for m in page_meta if m.get("status") == "ok"),
            "n_pages_missing": sum(1 for m in page_meta if m.get("status") != "ok"),
        },
    )

    if max_items is not None and len(window) > max_items:
        window = _truncate_evenly(window, max_items)
    return window


def _truncate_evenly(window: NewsWindow, max_items: int) -> NewsWindow:
    """Cap item count while keeping coverage spread across the whole window.

    Taking the first N would bias every window toward its earliest days, which
    would systematically starve the days closest to the transition -- exactly
    the ones most likely to carry the driver.
    """
    import numpy as np

    n = len(window.items)
    keep_idx = sorted(set(np.linspace(0, n - 1, max_items).round().astype(int).tolist()))
    kept = tuple(window.items[i] for i in keep_idx)
    prov = dict(window.provenance)
    prov["truncated_from"] = n
    prov["truncation"] = "evenly spaced across window"
    return NewsWindow(
        boundary_date=window.boundary_date,
        window_days=window.window_days,
        source=window.source,
        items=kept,
        retrieved_at=window.retrieved_at,
        dropped_post_boundary=window.dropped_post_boundary,
        dropped_pre_window=window.dropped_pre_window,
        query=window.query,
        provenance=prov,
    )


def measure_hindsight_leak(day: date, boundary: date) -> dict:
    """Quantify how much of a page was written after the boundary.

    This is evidence for the report, not a retrieval path: it is the number that
    justifies revision pinning to a sceptical reader.
    """
    title = _page_title(day)
    pinned = _api_get(
        {
            "action": "query", "prop": "revisions", "titles": title,
            "rvprop": "ids|timestamp|size", "rvslots": "main", "rvlimit": "1",
            "rvdir": "older",
            "rvstart": boundary_instant(boundary).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    current = _api_get(
        {
            "action": "query", "prop": "revisions", "titles": title,
            "rvprop": "ids|timestamp|size", "rvslots": "main", "rvlimit": "1",
        }
    )
    p_pages = pinned.get("query", {}).get("pages", [{}])[0]
    c_pages = current.get("query", {}).get("pages", [{}])[0]
    if not p_pages.get("revisions") or not c_pages.get("revisions"):
        return {"page": title, "status": "unavailable"}

    p, c = p_pages["revisions"][0], c_pages["revisions"][0]
    return {
        "page": title,
        "status": "ok",
        "pinned_revision": p["revid"],
        "pinned_timestamp": p["timestamp"],
        "pinned_size_bytes": p["size"],
        "current_revision": c["revid"],
        "current_timestamp": c["timestamp"],
        "current_size_bytes": c["size"],
        "bytes_added_after_boundary": c["size"] - p["size"],
        "pct_written_after_boundary": (
            round(100 * (c["size"] - p["size"]) / c["size"], 1) if c["size"] else None
        ),
    }
