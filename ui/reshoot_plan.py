"""Parser for the curated reshoot/new-shoot Markdown plan.

The plan lives outside this repo (e.g.
``../analyse-lens-test/reshoot-plan-2026-05.md``) and lists the
lens configurations the user wants to acquire next, grouped by
mounting style. Each table row gives a lens-config name, a
magnification (sometimes marked ``(est.)``), and a status.

This module turns those rows into :class:`PlanEntry` records that
the lens-config menu can offer as presets.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from sharpness.planner import parse_magnification


@dataclass(frozen=True)
class PlanEntry:
    group: str
    number: int
    lens_config: str
    magnification: float
    magnification_text: str
    status: str
    estimated: bool


_GROUP_RE = re.compile(r"^##\s+(Group\s+\d+\b.*?)\s*$")
_EST_RE = re.compile(r"\s*\(est\.?\)\s*$", re.IGNORECASE)


def load_plan(path: Path | str) -> list[PlanEntry]:
    """Parse the Markdown file at ``path`` and return entries in order.

    Raises :class:`FileNotFoundError` if the file is missing — callers
    typically handle this by falling through to the free-text prompt.
    Rows with unparseable magnifications are skipped with a warning to
    stderr; one bad row never breaks the whole menu.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    entries: list[PlanEntry] = []
    current_group: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = _GROUP_RE.match(line)
        if m:
            current_group = m.group(1)
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        # Skip header / separator rows.
        if cells[0].lower() == "#" or set(cells[0]) <= {"-", ":"}:
            continue
        try:
            number = int(cells[0])
        except ValueError:
            continue

        lens_config, mag_text, status = cells[1], cells[2], cells[3]
        estimated = bool(_EST_RE.search(mag_text))
        mag_clean = _EST_RE.sub("", mag_text).strip()
        try:
            mag = parse_magnification(mag_clean)
        except ValueError as exc:
            print(
                f"reshoot_plan: skipping row {number} "
                f"({lens_config!r}): {exc}",
                file=sys.stderr,
            )
            continue

        entries.append(
            PlanEntry(
                group=current_group or "",
                number=number,
                lens_config=lens_config,
                magnification=mag,
                magnification_text=mag_text,
                status=status,
                estimated=estimated,
            )
        )
    return entries
