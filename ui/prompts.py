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
) -> str | None:
    """Prompt for the next lens-config line.

    Returns ``None`` when the user enters an empty line (signalling
    "quit") and the trimmed text otherwise.
    """
    with readline_history(history_path):
        try:
            text = input_func(prompt)
        except EOFError:
            return None
    text = text.strip()
    if not text:
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
