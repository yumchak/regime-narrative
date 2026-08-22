"""Guards on credential handling.

The failure mode here is permanent in a way most bugs are not: once a key is
committed, it is in git history for good, and rewriting history does not help
if anyone has already cloned or if the repo was ever pushed. So this is checked
by a test rather than by remembering.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from regime_narrative.config import ENV_PATH, PROJECT_ROOT, load_env_file

# Real keys look like sk-ant-<something>-<long body>. The pattern deliberately
# requires a long tail so that documentation placeholders ("sk-ant-...",
# "sk-ant-api03-REPLACE-ME") do not trip it.
KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9]+-[A-Za-z0-9_\-]{40,}")


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git not available")
    if out.returncode != 0:  # pragma: no cover
        pytest.skip("not a git repository")
    return [PROJECT_ROOT / line for line in out.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# the guard that matters
# ---------------------------------------------------------------------------


def test_no_api_key_in_any_tracked_file():
    """Nothing git knows about may contain something shaped like a real key."""
    offenders = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover
            continue
        if KEY_PATTERN.search(text):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, (
        "an API key appears in tracked files: "
        + ", ".join(offenders)
        + " -- remove it, then rotate the key, because git remembers"
    )


def test_env_file_is_ignored_by_git():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert any(line.strip() == ".env" for line in gitignore.splitlines()), (
        ".env must be listed in .gitignore before anyone puts a key in it"
    )


def test_env_example_exists_and_holds_no_real_key():
    example = PROJECT_ROOT / ".env.example"
    assert example.exists(), "there must be a template telling people what to fill in"
    text = example.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" in text
    assert not KEY_PATTERN.search(text), ".env.example must hold a placeholder only"


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------


def test_loader_reads_a_simple_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    f = tmp_path / ".env"
    f.write_text("DEMO_TOKEN=abc123\n", encoding="utf-8")
    assert load_env_file(f) == ["DEMO_TOKEN"]
    assert os.environ["DEMO_TOKEN"] == "abc123"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("DEMO_TOKEN=plain", "plain"),
        ('DEMO_TOKEN="double quoted"', "double quoted"),
        ("DEMO_TOKEN='single quoted'", "single quoted"),
        ("export DEMO_TOKEN=with-export", "with-export"),
        ("  DEMO_TOKEN =  padded  ", "padded"),
        ("DEMO_TOKEN=has=equals=inside", "has=equals=inside"),
    ],
)
def test_loader_handles_the_shapes_people_actually_write(line, expected, tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    f = tmp_path / ".env"
    f.write_text(line + "\n", encoding="utf-8")
    load_env_file(f)
    assert os.environ["DEMO_TOKEN"] == expected


def test_loader_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    f = tmp_path / ".env"
    f.write_text("# a comment\n\n   \nDEMO_TOKEN=value\n", encoding="utf-8")
    assert load_env_file(f) == ["DEMO_TOKEN"]


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """A CI secret or a one-off inline value must not be silently overridden."""
    monkeypatch.setenv("DEMO_TOKEN", "from-the-environment")
    f = tmp_path / ".env"
    f.write_text("DEMO_TOKEN=from-the-file\n", encoding="utf-8")
    assert load_env_file(f) == []
    assert os.environ["DEMO_TOKEN"] == "from-the-environment"


def test_override_is_available_when_explicitly_asked_for(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "from-the-environment")
    f = tmp_path / ".env"
    f.write_text("DEMO_TOKEN=from-the-file\n", encoding="utf-8")
    assert load_env_file(f, override=True) == ["DEMO_TOKEN"]
    assert os.environ["DEMO_TOKEN"] == "from-the-file"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_malformed_lines_are_skipped_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    f = tmp_path / ".env"
    f.write_text("this line has no equals sign\n=novalue\nDEMO_TOKEN=ok\n", encoding="utf-8")
    assert load_env_file(f) == ["DEMO_TOKEN"]
