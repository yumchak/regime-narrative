"""Stage 13 -- measure the video script instead of trusting its own header.

The script asserts a runtime and a speaking pace. Both are easy to break: any
reworded cell changes its word count, and any retimed cell can silently leave a
gap or an overlap in the timeline. A 7:00 limit is a hard submission rule, so
the claim is worth checking rather than believing.

Reads the timings straight out of the markdown table and fails loudly on:
    * total runtime over 7:00
    * a gap or overlap between consecutive cells
    * any cell that needs more than 120 words per minute to deliver
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "VIDEO_SCRIPT.md"
LIMIT_S = 7 * 60
MAX_WPM = 120

ROW = re.compile(r"^\|\s*\*\*(\d):(\d\d)[\u2013-](\d):(\d\d)\*\*\s*\|(.*?)\|(.*?)\|\s*$", re.M)


def main() -> None:
    txt = SCRIPT.read_text(encoding="utf-8")
    rows, prev_end, gaps = [], None, []

    for m in ROW.finditer(txt):
        a = int(m.group(1)) * 60 + int(m.group(2))
        b = int(m.group(3)) * 60 + int(m.group(4))
        show, say = m.group(5), m.group(6)
        words = len(re.sub(r"\*\*|\*|`|\u2026", " ", say).split())
        dur = b - a
        # a written-in silent beat is staging, not speaking
        if "seconds of silence" in show:
            dur -= 3
        if prev_end is not None and prev_end != a:
            gaps.append(f"{m.group(1)}:{m.group(2)}")
        prev_end = b
        rows.append((f"{m.group(1)}:{m.group(2)}-{m.group(3)}:{m.group(4)}",
                     words, dur, words * 60 / dur if dur > 0 else 1e9))

    if not rows:
        print("no timed rows found -- did the table format change?")
        sys.exit(1)

    for label, words, dur, wpm in rows:
        flag = "  <-- TOO FAST" if wpm > MAX_WPM else ""
        print(f"{label}  {words:3d}w / {dur:3d}s = {wpm:5.1f} wpm{flag}")

    fast = [(r[0], round(r[3])) for r in rows if r[3] > MAX_WPM]
    print(f"\ntotal runtime {prev_end // 60}:{prev_end % 60:02d}")

    problems = []
    if prev_end > LIMIT_S:
        problems.append(f"over the {LIMIT_S // 60}:00 limit by {prev_end - LIMIT_S}s")
    if gaps:
        problems.append(f"gap or overlap before {gaps}")
    if fast:
        problems.append(f"cells above {MAX_WPM} wpm: {fast}")
    if problems:
        for q in problems:
            print("FAIL:", q)
        sys.exit(1)

    print(f"{len(rows)} cells, contiguous, none above {MAX_WPM} wpm, "
          f"{LIMIT_S - prev_end}s under the limit")


if __name__ == "__main__":
    main()
