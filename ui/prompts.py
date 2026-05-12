"""Free-text prompts with persistent readline history + slug helper."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

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


def prompt_magnification(
    prompt: str = (
        "Magnification (e.g. 1:1, 1.5:1, 1:1.5, or a number like 0.5): "
    ),
    *,
    input_func: InputFunc = input,
) -> float:
    """Prompt for a magnification, accepting ratio or float forms.

    Uses :func:`sharpness.planner.parse_magnification` for the actual
    parsing; re-prompts on any :class:`ValueError`. EOF re-raises so
    callers can distinguish "user gave up" from "user gave junk".
    """
    from sharpness.planner import parse_magnification

    while True:
        try:
            text = input_func(prompt)
        except EOFError:
            raise
        try:
            return parse_magnification(text)
        except ValueError as exc:
            print(f"  {exc}; try e.g. 1:1, 1.5:1, 0.5")


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


# ---- Manual-aperture mode prompts -----------------------------------------


def prompt_magnification_indicated(
    prompt: str = (
        "Magnification indicated on lens barrel "
        "(e.g. 1:2 or 0.5; blank = same as total magnification): "
    ),
    *,
    default: float,
    input_func: InputFunc = input,
) -> float:
    """Prompt for ``m_indicated``, accepting ratio or float forms.

    The lens barrel reading is the lens's own contribution to
    magnification (helix-only). With no external tube, this equals
    ``m_total``; with a tube, it's smaller. Empty input returns the
    ``default`` (typically ``m_total``).

    Uses :func:`sharpness.planner.parse_magnification` for parsing.
    """
    from sharpness.planner import parse_magnification

    while True:
        try:
            text = input_func(prompt)
        except EOFError:
            raise
        text = text.strip()
        if not text:
            return default
        try:
            return parse_magnification(text)
        except ValueError as exc:
            print(f"  {exc}; try e.g. 1:2, 0.5, 1/2")


def prompt_focal_length(
    prompt: str = "Lens focal length (mm, e.g. 55): ",
    *,
    input_func: InputFunc = input,
) -> float:
    """Prompt for a focal length in mm; re-prompts on bad input.

    Used as a fallback when the slug regex doesn't find a ``<N>mm``
    token. EOF re-raises so callers can distinguish "user gave up"
    from "user gave junk".
    """
    while True:
        try:
            text = input_func(prompt)
        except EOFError:
            raise
        try:
            value = float(text.strip())
        except ValueError:
            print(f"  not a number: {text!r}")
            continue
        if value <= 0:
            print(f"  must be > 0: {value}")
            continue
        return value


def prompt_max_aperture(
    prompt: str = (
        "Indicated max aperture marked on lens "
        "(e.g. 2.8 or f/2.8): "
    ),
    *,
    input_func: InputFunc = input,
) -> float:
    """Prompt for the lens's marked widest aperture (the value on the
    ring). Accepts ``"2.8"`` or ``"f/2.8"`` forms. Re-prompts on bad
    input.

    Used as a fallback when the slug regex doesn't find an ``f/N``
    token. This is *not* the EXIF effective aperture — it's the
    ring's marked widest, before any bellows-factor correction.
    """
    while True:
        try:
            text = input_func(prompt)
        except EOFError:
            raise
        cleaned = text.strip().lower().lstrip("f").lstrip("/")
        try:
            value = float(cleaned)
        except ValueError:
            print(f"  not a number: {text!r}")
            continue
        if value <= 0:
            print(f"  must be > 0: {value}")
            continue
        return value


def prompt_turn_ring(
    aperture_label: str,
    *,
    ring_stop: float,
    m_indicated: float,
    input_func: InputFunc = input,
) -> bool:
    """Block on ENTER while the user turns the lens ring to a target.

    ``ring_stop`` is the **actual geometric f-number marked on the
    lens ring** the user should dial to (e.g. 2.8, 4, 5.6 — picked
    by the runner from the WHOLE_STOP_SCALE sequence walk).
    ``aperture_label`` is the **effective aperture** the rig stores
    in the file label (e.g. ``"4"`` for the 55 AIS at ring f/2.8 +
    PK-13 at 1:1). The prompt names both so the user knows what to
    dial and what the rig is recording.

    Returns ``True`` on ENTER, ``False`` on ``q`` / EOF.
    """
    # Only the ring stop is operator-relevant during capture; the
    # effective label exists for the file path / analysis side and
    # would just clutter the prompt.
    del aperture_label, m_indicated
    msg = (
        f"Turn lens ring to f/{_format_ring_stop(ring_stop)}, "
        f"then press [ENTER]. [q] to abort: "
    )
    return confirm(msg, input_func=input_func)


def _format_ring_stop(ring_stop: float) -> str:
    """Format a geometric ring f-number for display.

    Whole-stop values render without trailing decimals (``"4"``,
    ``"8"``); thirds / halves keep one decimal (``"2.8"``, ``"5.6"``,
    ``"3.5"``).
    """
    if abs(ring_stop - round(ring_stop)) < 0.05:
        return f"{int(round(ring_stop))}"
    return f"{ring_stop:.1f}".rstrip("0").rstrip(".")


def prompt_set_flash_power(
    *,
    power_c_label: str,
    power_a_label: str,
    context: str = "",
    input_func: InputFunc = input,
) -> bool:
    """Ask the user to dial flash power on the SB-5000 commander.

    Manual-aperture mode keeps the remote groups in M (manual power)
    on the commander, since the body can't communicate aperture to
    the flash for non-CPU lenses on Z7. The rig pre-computes the
    target power per aperture (see
    :func:`session.flash_power.target_for_next_aperture`) and prompts
    the user to dial it. Group C is the left flash; group A is the
    right flash and runs 1/3 stop lower (subject-distance offset).

    ``context`` is a short string included in the prompt header so
    the user can see which aperture / phase they're setting power
    for (e.g. ``"coarse search step 3"`` or ``"f/5.6 calibration"``).

    Returns ``True`` if the user pressed ENTER, ``False`` on ``q`` /
    EOF.
    """
    prefix = f"[{context}] " if context else ""
    # SB-5000 commander displays group A before C; match that order
    # so the operator reads the prompt top-to-bottom in the same
    # direction they're scrolling on the flash.
    msg = (
        f"{prefix}Set flash power on SB-5000 commander:\n"
        f"  group A (right) : {power_a_label}\n"
        f"  group C (left)  : {power_c_label}\n"
        "[ENTER] to confirm, [q] to abort: "
    )
    return confirm(msg, input_func=input_func)




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


OTHER = "OTHER"
"""Sentinel returned by :func:`prompt_lens_config_from_plan` when the
user chooses "Other (free text)" — signals the caller to fall back to
:func:`prompt_lens_config` + :func:`prompt_magnification`."""


def _is_f16_only(entry: Any) -> bool:
    """True if this entry's status flags it as F16-only-truncated.

    Status text is whatever the plan author put in the markdown table;
    we match the ``(F16-only)`` substring case-insensitively so other
    annotations (``(est.)``, ``(NEW)``, etc.) don't false-positive.
    """
    return "f16-only" in (getattr(entry, "status", "") or "").lower()


def _format_choice_title(entry: Any, completed: set[str], slugify: Callable[[str], str]) -> str:
    if slugify(entry.lens_config) in completed:
        leading = "[done]"
    elif _is_f16_only(entry):
        leading = "[F16-opt]"
    else:
        leading = "      "
    return f"{leading}  {entry.lens_config}  —  {entry.magnification_text}"


def prompt_lens_config_from_plan(
    entries: list,
    completed: set[str],
    *,
    ask: Callable[[list], Any] | None = None,
    slugify: Callable[[str], str] = slugify_lens_config,
):
    """Show an arrow-key menu of preset lens configs.

    Parameters
    ----------
    entries:
        ``list[ui.reshoot_plan.PlanEntry]`` — the curated presets.
    completed:
        Slugs already successfully acquired in past sessions; rendered
        with a ``[done]`` marker. Re-acquiring a completed preset is a
        normal pick, not blocked.
    ask:
        Injectable hook for tests. Takes the choice list and returns
        the chosen value. Default constructs a ``questionary.select``
        and calls ``.ask()`` on it.
    slugify:
        Slugifier used to compare entry names against ``completed``.

    Returns
    -------
    The picked :class:`PlanEntry`, the sentinel :data:`OTHER` for the
    free-text path, or ``None`` to quit (Quit choice / Ctrl-C / EOF).
    """
    import questionary

    # Stable per-group sort: F16-only entries sink to the bottom of
    # their group because they're optional (only F16 truncated; user
    # can skip if F16 completeness isn't required). Within each
    # category (normal vs F16-only) the original plan-file order
    # is preserved.
    by_group: dict[str, list] = {}
    group_order: list[str] = []
    for entry in entries:
        if entry.group not in by_group:
            by_group[entry.group] = []
            group_order.append(entry.group)
        by_group[entry.group].append(entry)
    sorted_entries: list = []
    for group in group_order:
        bucket = by_group[group]
        # Python sort is stable: original order preserved within each key.
        bucket.sort(key=lambda e: 1 if _is_f16_only(e) else 0)
        sorted_entries.extend(bucket)

    choices: list = []
    last_group: str | None = None
    for entry in sorted_entries:
        if entry.group != last_group:
            choices.append(questionary.Separator(f"── {entry.group} ──"))
            last_group = entry.group
        choices.append(
            questionary.Choice(
                title=_format_choice_title(entry, completed, slugify),
                value=entry,
            )
        )
    choices.append(questionary.Separator(" "))
    choices.append(questionary.Choice(title="Other (free text)…", value=OTHER))
    choices.append(questionary.Choice(title="Quit", value=None))

    if ask is None:
        def _default_ask(cs):
            return questionary.select(
                "Pick a lens config:",
                choices=cs,
                use_arrow_keys=True,
                use_shortcuts=False,
            ).ask()
        ask = _default_ask

    try:
        result = ask(choices)
    except (KeyboardInterrupt, EOFError):
        return None
    return result


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
