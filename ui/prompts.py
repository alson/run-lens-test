"""Free-text prompts with persistent readline history + slug helper."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

# Lazy-imported readline so non-Linux test runs don't trip on it.
try:
    import readline as _readline
except ImportError:                                  # pragma: no cover
    _readline = None


@contextmanager
def readline_history(path: Path | str | None):
    """Load and save readline history around the with-block.

    No-op if ``path`` is ``None`` or the readline module isn't
    available. Safe if the file doesn't exist yet — first run creates
    it on save.
    """
    if path is None or _readline is None:
        yield
        return
    p = Path(path)
    if p.exists():
        try:
            _readline.read_history_file(str(p))
        except OSError:
            pass
    try:
        yield
    finally:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            _readline.write_history_file(str(p))
        except OSError:
            pass


# ---- slug -----------------------------------------------------------------

_SLUG_REPLACEMENTS = {":": "_", "/": "_"}


def slugify_lens_config(text: str) -> str:
    """Replace ``:`` and ``/`` with ``_``. No other transformation —
    the user's text is the source of truth for the directory name and
    log entries."""
    out = text
    for src, dst in _SLUG_REPLACEMENTS.items():
        out = out.replace(src, dst)
    # Strip leading/trailing whitespace; preserve internal spaces.
    return out.strip()


# ---- input ----------------------------------------------------------------

InputFunc = Callable[[str], str]


def prompt_lens_config(
    prompt: str = "Lens config (free text, blank to quit): ",
    *,
    history_path: Path | str | None = None,
    input_func: InputFunc = input,
    min_length: int = 5,
) -> str | None:
    """Prompt for the next lens-config line.

    Returns ``None`` (signalling "quit") when:
    * the user enters an empty line, or EOF;
    * the trimmed input is shorter than ``min_length`` characters
      (catches stray ``q`` / ``quit`` / typos that would otherwise
      slugify to a junk lens-config name and `mkdir` it before
      aborting at the jog).

    Otherwise returns the trimmed text.
    """
    with readline_history(history_path):
        try:
            text = input_func(prompt)
        except EOFError:
            return None
    text = text.strip()
    if len(text) < min_length:
        return None
    return text


def prompt_step_count(
    prompt: str = "Number of rail steps for this lens config (>=2): ",
    *,
    input_func: InputFunc = input,
    minimum: int = 2,
) -> int:
    """Prompt for an integer ≥ ``minimum``. Re-prompts on bad input."""
    while True:
        try:
            text = input_func(prompt)
        except EOFError:
            raise ValueError("EOF while reading step count")
        try:
            value = int(text.strip())
        except ValueError:
            print(f"  not an integer: {text!r}")
            continue
        if value < minimum:
            print(f"  must be >= {minimum}")
            continue
        return value


def confirm(
    prompt: str,
    *,
    input_func: InputFunc = input,
) -> bool:
    """Press ENTER to confirm, ``q`` (or empty + EOF) to abort.

    Returns ``True`` if the user pressed ENTER.
    """
    try:
        text = input_func(prompt)
    except EOFError:
        return False
    return text.strip().lower() not in ("q", "quit", "abort")


def prompt_range_inadequate(
    prompt: str,
    *,
    input_func: InputFunc = input,
) -> str:
    """Three-way prompt for the range-still-inadequate decision.

    Returns one of:
    * ``"extend"`` — empty input / ENTER (default: extend the sweep
      by one more standard increment on the failing side).
    * ``"skip"`` — ``s`` / ``skip`` / ``n``: accept the current
      data as-is and continue to the lens-change prompt.
    * ``"abort"`` — ``q`` / ``quit`` / ``abort`` / EOF: abort the
      session.
    """
    try:
        text = input_func(prompt)
    except EOFError:
        return "abort"
    t = text.strip().lower()
    if t in ("q", "quit", "abort"):
        return "abort"
    if t in ("s", "skip", "n", "no"):
        return "skip"
    return "extend"


def prompt_off_target(
    prompt: str,
    *,
    input_func: InputFunc = input,
) -> str:
    """Three-way prompt for the all-off-target decision.

    Returns one of:
    * ``"rejog"`` — ENTER (default): re-jog START / END and re-sweep
      this lens config; calibration is preserved.
    * ``"skip"`` — ``s`` / ``skip``: accept the flat data as-is and
      continue to the lens-change prompt anyway.
    * ``"abort"`` — ``q`` / EOF.
    """
    try:
        text = input_func(prompt)
    except EOFError:
        return "abort"
    t = text.strip().lower()
    if t in ("q", "quit", "abort"):
        return "abort"
    if t in ("s", "skip", "n", "no"):
        return "skip"
    return "rejog"
