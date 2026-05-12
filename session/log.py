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
    RESHOOT = "reshoot"


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
    # ---- Manual-aperture / non-CPU mode metadata ------------------
    # Populated only when the session is run with --manual-aperture
    # and only for the lens-configs in that session. Auto-mode
    # (CPU lens) rows leave these blank.
    "magnification_indicated",
    "f_ring",
    "focal_length_mm",
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
    # ---- Manual-aperture metadata (None for auto mode) ------------
    magnification_indicated: float | None = None
    f_ring: float | None = None
    focal_length_mm: float | None = None

    def to_csv_row(self) -> dict[str, str]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return {col: _csv_str(d[col]) for col in SHOT_COLUMNS}


def _csv_str(value: Any) -> str:
    if value is None:
        # Blank cell — used by the manual-aperture columns when the
        # session is auto-mode, so the column stays present but the
        # value is unambiguously "n/a".
        return ""
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

    def drop_rows(self, predicate) -> int:
        """Atomically rewrite the file removing rows where
        ``predicate(row_dict)`` returns truthy. Returns the number of
        rows dropped.

        Used to keep shots.csv clean of stale entries pointing at
        files that have been overwritten on disk:
        * after a droop reshoot, the original ``phase=sweep`` rows
          (and any redundant ``phase=aborted`` rows) for the reshot
          step range describe images that no longer exist.
        * after an off-target re-jog, the discarded sweep's rows
          point at JPEGs that the new sweep is about to overwrite.
        """
        rows = self.read_rows()
        kept = [r for r in rows if not predicate(r)]
        n_dropped = len(rows) - len(kept)
        if n_dropped == 0:
            return 0
        payload = self._render(kept, header=True)
        self._atomic_write(payload)
        return n_dropped


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


def completed_slugs(output_root: Path | str) -> set[str]:
    """Return the union of `lens_configs[*].slug` across every
    ``<output_root>/*/session.json``. Used by the lens-config menu to
    show a `[done]` marker against presets that were already
    successfully acquired in a past session.

    A slug only lands in `lens_configs` after the sweep + range-check
    succeed (see :meth:`SessionRunner._run_one_lens_config`), so this
    is interruption-safe — an aborted session contributes only the
    configs it managed to finish before aborting.

    Best-effort: a missing or malformed `session.json` is skipped
    silently, never raised.
    """
    root = Path(output_root)
    if not root.exists():
        return set()
    out: set[str] = set()
    for child in root.iterdir():
        meta_path = child / "session.json"
        if not meta_path.is_file():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for lc in payload.get("lens_configs", []) or []:
            slug = lc.get("slug") if isinstance(lc, dict) else None
            if isinstance(slug, str) and slug:
                out.add(slug)
    return out


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
    lens_configs: list[dict[str, Any]] = field(default_factory=list)
    """Per-completed-lens-config metadata. Each entry is a dict with
    keys ``slug``, ``magnification``, ``widest_aperture``, and
    ``sample_plan`` (the full SamplePlan as a dict)."""
    aborted: bool = False
    notes: str = ""
