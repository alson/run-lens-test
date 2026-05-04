"""Session log writers.

Two artefacts per session:

* ``shots.csv`` — one row per captured frame, appended atomically as
  the session runs. Columns are fixed at session start; the schema
  is the source of truth for downstream sharpness analysis.
* ``session.json`` — written at the start and rewritten at the end
  with full metadata.

Plus per-lens-config files: ``calibration.json`` (FEC/ISO chosen at
step 0) and ``range_check.json`` (range-check decision and probe
data).

Atomic write pattern: write to ``<path>.tmp`` and rename. CSV
appends use a small helper that re-reads + rewrites the whole file
through a tmp; volume is small (one row per shot) so this is fine.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


# ---- Phase enum ------------------------------------------------------------


class Phase(str, Enum):
    """Allowed values of the ``phase`` column in ``shots.csv``."""

    CALIBRATION = "calibration"
    SWEEP = "sweep"
    PROBE = "probe"
    EXTENSION = "extension"
    ABORTED = "aborted"


# ---- Shot row --------------------------------------------------------------


SHOT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "session_id",
    "lens_config_slug",
    "rail_step",
    "rail_position_mm",
    "aperture",
    "iso",
    "shutter_speed",
    "fec",
    "median_central_luminance",
    "median_ev_relative_to_baseline",
    "filename",
    "phase",
    "notes",
)


@dataclass
class ShotRow:
    timestamp: str
    session_id: str
    lens_config_slug: str
    rail_step: int
    rail_position_mm: float
    aperture: str
    iso: str | int
    shutter_speed: str
    fec: float
    median_central_luminance: float
    median_ev_relative_to_baseline: float
    filename: str
    phase: Phase
    notes: str = ""

    def to_csv_row(self) -> dict[str, str]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return {col: _csv_str(d[col]) for col in SHOT_COLUMNS}


def _csv_str(value: Any) -> str:
    if isinstance(value, float):
        # Stable formatting; downstream analysis parses these.
        return f"{value:.6g}"
    return str(value)


# ---- ShotsCSV --------------------------------------------------------------


class ShotsCSV:
    """Append-only CSV writer for shot rows.

    Atomic: stages every append to a ``<path>.tmp`` and renames into
    place so concurrent readers (e.g. an analysis script) never see
    a half-written line.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self._path.exists():
            self._atomic_write(self._render([], header=True))

    def append(self, row: ShotRow) -> None:
        existing = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        line = self._render([row.to_csv_row()], header=not existing)
        if existing:
            payload = existing + line
        else:
            payload = line
        self._atomic_write(payload)

    def _atomic_write(self, payload: str) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self._path)

    @staticmethod
    def _render(rows: Iterable[Mapping[str, str]], *, header: bool) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=SHOT_COLUMNS)
        if header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    def read_rows(self) -> list[dict[str, str]]:
        """Return all rows (used by tests)."""
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


# ---- JSON writer ----------------------------------------------------------


def write_json(path: Path | str, payload: Any) -> None:
    """Atomic JSON write; supports dataclasses and Enums via a custom
    serialiser."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default),
                   encoding="utf-8")
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"unserialisable type {type(obj).__name__}")


# ---- session.json metadata ------------------------------------------------


@dataclass
class SessionMetadata:
    session_id: str
    started_at: str
    ended_at: str | None = None
    output_root: str = ""
    ntfy_topic: str = ""
    stackshot_software_id: int | None = None
    stackshot_hardware_id: int | None = None
    baseline_median_luminance: float | None = None
    lens_configs: list[str] = field(default_factory=list)
    aborted: bool = False
    notes: str = ""
