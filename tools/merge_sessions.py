"""Merge macro-lens-test sessions into one folder with a unified
coordinate frame and metadata. Useful when the rail's mechanical START
position differs between sessions but the underlying optical setup
(wafer, lens, magnification) is the same — e.g. you covered the inner
crops' bells in one session, then re-jogged with a deeper START to
catch the outer corners' bells in the second.

Output is a *new* directory; the source sessions stay untouched.
JPEGs and crops are symlinked rather than copied, so the merge is
nearly free in disk space.

Two CLIs are supported.

**Two-session merge** (legacy)::

    python -m tools.merge_sessions \\
        --prev ~/macro-lens-test-output/2026-05-08_175556 \\
        --cur  ~/macro-lens-test-output/2026-05-08_201054 \\
        --cur-offset-mm -2.93 \\
        --lens-config-slug "NIKKOR Z 24-120mm f_4 S + 61mm extension" \\
        --output ~/macro-lens-test-output/merged_24-120_2026-05-08 \\
        --overlap-strategy cur

``--cur-offset-mm``: value to *add* to a CUR ``rail_position_mm`` to
land in the PREV frame. ``cur`` strategy keeps CUR in the overlap.

**N-session merge** (any number of sessions, each with its own
lens-config slug and rail-zero offset)::

    python -m tools.merge_sessions \\
        --source ~/.../session_A:"slug A":0.0:A \\
        --source ~/.../session_B:"slug B":-5.0:B \\
        --source ~/.../session_C:"slug C":-5.0:C \\
        --source ~/.../session_D:"slug D":-3.5:D \\
        --output-slug "Viltrox 85mm merged" \\
        --output ~/macro-lens-test-output/merged_viltrox_85_2026-05-10

``--source`` format: ``PATH:SLUG:OFFSET_MM[:LABEL]`` (repeatable).
Use ``\\:`` to escape literal colons in the slug. Priority is the
order of ``--source`` flags: the *first* source wins in overlap
zones, the next wins where it adds new coverage past the first, etc.
Each source's positions are translated by its ``OFFSET_MM`` into the
merged frame; pass ``0.0`` for the anchor session whose native
frame should be the canonical merged frame. Priority is independent
of frame choice.

``--output-slug`` is the lens-config directory name in the merged
output. Source slugs vary per session (one session per tail
extension typically uses a distinct slug); the merged dataset
collapses to one.

Filenames are prefixed with the source's ``LABEL`` (or its session
timestamp when label is omitted) so the four sessions' identical
``step_NNNN.jpg`` filenames don't collide in the merged folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---- Schema ----------------------------------------------------------------

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


OVERLAP_STRATEGIES: tuple[str, ...] = ("cur", "prev", "both")


# ---- Helpers ---------------------------------------------------------------

def _read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_pos(row: dict) -> float | None:
    try:
        return float(row["rail_position_mm"])
    except (TypeError, ValueError, KeyError):
        return None


def _captured_range(rows: list[dict], offset_mm: float = 0.0):
    """Min / max rail position across non-calibration rows, with offset
    applied. Returns ``None`` if the dataset is empty."""
    positions = []
    for r in rows:
        if r.get("phase") == "calibration":
            continue
        pos = _parse_pos(r)
        if pos is not None:
            positions.append(pos + offset_mm)
    if not positions:
        return None
    return (min(positions), max(positions))


def _symlink(target: Path, link_path: Path) -> bool:
    """Best-effort relative-path symlink. Skips silently if the target
    is missing. Replaces an existing link / file. Relative paths keep
    the merged folder portable across moves of either the source or
    the merged tree, as long as their *relative* layout stays the
    same."""
    if not target.exists():
        return False
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    import os
    rel_target = os.path.relpath(target.resolve(), link_path.parent.resolve())
    link_path.symlink_to(rel_target)
    return True


def _link_jpg_and_crops(
    *,
    source_session: str,
    source_lens_dir: Path,
    output_lens_dir: Path,
    relative_filename: str,
) -> str | None:
    """Symlink one source JPEG plus its 17 per-crop crops into the
    merged folder. Filenames are prefixed with ``source_session_`` so
    PREV and CUR don't collide on each session's identical
    ``step_NNNN.jpg`` filenames.

    Returns the merged-relative filename (e.g. ``f_4/prev_step_0100.jpg``)
    that the merged ``shots.csv`` should point at, or ``None`` when
    the input was blank or the source JPEG is missing.
    """
    if not relative_filename:
        return None
    source_jpg = source_lens_dir / relative_filename
    if not source_jpg.exists():
        return None
    # The captured-shot path is always ``<aperture>/<leaf>``.
    if "/" not in relative_filename:
        # Defensive: shouldn't happen for sweep/probe/extension rows.
        return None
    aperture_dir, leaf = relative_filename.split("/", 1)
    new_leaf = f"{source_session}_{leaf}"
    new_relative = f"{aperture_dir}/{new_leaf}"
    output_jpg = output_lens_dir / new_relative
    if not _symlink(source_jpg, output_jpg):
        return None
    # Crops: ``<aperture>/cropped/<crop_name>/<leaf>`` per aperture.
    source_crop_root = source_lens_dir / aperture_dir / "cropped"
    if source_crop_root.is_dir():
        for crop_subdir in sorted(source_crop_root.iterdir()):
            if not crop_subdir.is_dir():
                continue
            src_crop = crop_subdir / leaf
            if src_crop.exists():
                dst_crop = (
                    output_lens_dir
                    / aperture_dir
                    / "cropped"
                    / crop_subdir.name
                    / new_leaf
                )
                _symlink(src_crop, dst_crop)
    return new_relative


def _merge_one_row(
    *,
    row: dict,
    source_session: str,
    source_lens_dir: Path,
    output_lens_dir: Path,
    rail_offset_mm: float,
) -> dict:
    """Translate one shots.csv row's coordinates + symlink its JPEG.
    Returns the merged row as a new dict."""
    out = dict(row)
    out["source_session"] = source_session
    pos = _parse_pos(row)
    if pos is not None:
        out["rail_position_mm"] = f"{pos + rail_offset_mm:.6g}"
    new_filename = _link_jpg_and_crops(
        source_session=source_session,
        source_lens_dir=source_lens_dir,
        output_lens_dir=output_lens_dir,
        relative_filename=row.get("filename", ""),
    )
    if new_filename is not None:
        out["filename"] = new_filename
    return out


# ---- Filter logic ---------------------------------------------------------

def _filter_by_overlap(
    prev_rows: list[dict],
    cur_rows: list[dict],
    cur_offset_mm: float,
    strategy: str,
) -> tuple[list[dict], list[dict], tuple[float, float] | None]:
    """Apply ``strategy`` to drop rows whose rail position falls
    inside the *other* session's captured range. Calibration rows are
    always kept (they're not part of the analysis sweep, just per-
    aperture metadata)."""
    cur_range = _captured_range(cur_rows, offset_mm=cur_offset_mm)
    prev_range = _captured_range(prev_rows)

    if strategy == "cur":
        # Drop PREV rows that fall inside CUR's captured range.
        if cur_range is None:
            kept_prev = list(prev_rows)
        else:
            lo, hi = cur_range
            kept_prev = []
            for r in prev_rows:
                if r.get("phase") == "calibration":
                    kept_prev.append(r)
                    continue
                pos = _parse_pos(r)
                if pos is None or pos < lo or pos > hi:
                    kept_prev.append(r)
        kept_cur = list(cur_rows)
        return kept_prev, kept_cur, cur_range

    if strategy == "prev":
        if prev_range is None:
            kept_cur = list(cur_rows)
        else:
            lo, hi = prev_range
            kept_cur = []
            for r in cur_rows:
                if r.get("phase") == "calibration":
                    kept_cur.append(r)
                    continue
                pos = _parse_pos(r)
                if pos is None:
                    kept_cur.append(r)
                    continue
                prev_frame_pos = pos + cur_offset_mm
                if prev_frame_pos < lo or prev_frame_pos > hi:
                    kept_cur.append(r)
        kept_prev = list(prev_rows)
        return kept_prev, kept_cur, prev_range

    if strategy == "both":
        return list(prev_rows), list(cur_rows), None

    raise ValueError(f"unknown overlap_strategy: {strategy!r}")


# ---- Top-level entry ------------------------------------------------------

def merge_sessions(
    *,
    prev_dir: Path,
    cur_dir: Path,
    cur_offset_mm: float,
    lens_config_slug: str,
    output_dir: Path,
    overlap_strategy: str = "cur",
) -> dict:
    """Merge two sessions for one lens config into ``output_dir``.
    Returns a summary dict (which is also persisted as the merged
    ``session.json``)."""
    prev_csv = prev_dir / "shots.csv"
    cur_csv = cur_dir / "shots.csv"
    prev_lens_dir = prev_dir / lens_config_slug
    cur_lens_dir = cur_dir / lens_config_slug

    for path, label in (
        (prev_csv, "PREV shots.csv"),
        (cur_csv, "CUR shots.csv"),
        (prev_lens_dir, "PREV lens dir"),
        (cur_lens_dir, "CUR lens dir"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"required {label} missing: {path}")

    prev_rows = [
        r for r in _read_rows(prev_csv)
        if r.get("lens_config_slug") == lens_config_slug
    ]
    cur_rows = [
        r for r in _read_rows(cur_csv)
        if r.get("lens_config_slug") == lens_config_slug
    ]

    kept_prev, kept_cur, overlap_range = _filter_by_overlap(
        prev_rows, cur_rows, cur_offset_mm, overlap_strategy,
    )

    output_lens_dir = output_dir / lens_config_slug
    output_lens_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict] = []
    for r in kept_prev:
        merged.append(_merge_one_row(
            row=r,
            source_session="prev",
            source_lens_dir=prev_lens_dir,
            output_lens_dir=output_lens_dir,
            rail_offset_mm=0.0,
        ))
    for r in kept_cur:
        merged.append(_merge_one_row(
            row=r,
            source_session="cur",
            source_lens_dir=cur_lens_dir,
            output_lens_dir=output_lens_dir,
            rail_offset_mm=cur_offset_mm,
        ))

    # Stable sort: aperture first (so each crop's curve is contiguous),
    # then rail_position_mm ascending.
    def _sort_key(r: dict):
        pos = _parse_pos(r)
        return (r.get("aperture", ""), pos if pos is not None else 0.0)

    merged.sort(key=_sort_key)

    # Write merged shots.csv with the new source_session column.
    fieldnames = SHOT_COLUMNS + ("source_session",)
    out_csv_path = output_dir / "shots.csv"
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in merged:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Compose merged session.json.
    prev_session = json.loads((prev_dir / "session.json").read_text())
    cur_session = json.loads((cur_dir / "session.json").read_text())
    summary = {
        "merged": True,
        "merge_created_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_sessions": [
            {
                "session_id": prev_session.get("session_id"),
                "path": str(prev_dir.resolve()),
                "rail_offset_mm": 0.0,
                "role": "anchor",
            },
            {
                "session_id": cur_session.get("session_id"),
                "path": str(cur_dir.resolve()),
                "rail_offset_mm": cur_offset_mm,
                "role": "offset",
            },
        ],
        "overlap_strategy": overlap_strategy,
        "overlap_range_prev_mm": (
            list(overlap_range) if overlap_range is not None else None
        ),
        "lens_configs": [lens_config_slug],
        "baseline_median_luminance": cur_session.get(
            "baseline_median_luminance"
        ) or prev_session.get("baseline_median_luminance"),
        "ntfy_topic": cur_session.get("ntfy_topic"),
        "row_counts": {
            "prev_input": len(prev_rows),
            "prev_kept": len(kept_prev),
            "cur_input": len(cur_rows),
            "cur_kept": len(kept_cur),
            "merged_total": len(merged),
        },
        "notes": (
            f"Merged {prev_session.get('session_id')} (anchor) and "
            f"{cur_session.get('session_id')} (rail offset "
            f"{cur_offset_mm:+.3f} mm into PREV frame). "
            f"Overlap strategy: {overlap_strategy}. "
            "JPEGs and crops are symlinks into the source sessions; "
            "both originals are retained on disk."
        ),
    }
    (output_dir / "session.json").write_text(
        json.dumps(summary, indent=2)
    )

    # CUR is post the recent calibrator + replica + crop-layout fixes,
    # so its calibration.json / range_check.json are the better
    # representative for the merged dataset. Symlink them; the originals
    # under each source session remain available for cross-check.
    for fname in ("calibration.json", "range_check.json"):
        src = cur_lens_dir / fname
        if src.exists():
            _symlink(src, output_lens_dir / fname)

    return summary


# ---- N-source merge -------------------------------------------------------


def _parse_source_spec(spec: str) -> dict:
    """``PATH:SLUG:OFFSET_MM[:LABEL]`` → dict. ``\\:`` escapes colons
    inside SLUG."""
    # Replace escaped colons with a sentinel so split('/').replace works.
    sentinel = "\x00"
    safe = spec.replace("\\:", sentinel)
    parts = safe.split(":")
    if not (3 <= len(parts) <= 4):
        raise argparse.ArgumentTypeError(
            f"--source needs PATH:SLUG:OFFSET_MM[:LABEL], got {spec!r}"
        )
    path, slug, offset_str, *label_rest = parts
    label = label_rest[0] if label_rest else ""
    try:
        offset_mm = float(offset_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--source OFFSET_MM must be a number, got {offset_str!r}"
        ) from exc
    return {
        "path": Path(path.replace(sentinel, ":")),
        "slug": slug.replace(sentinel, ":"),
        "offset_mm": offset_mm,
        "label": (label or "").replace(sentinel, ":"),
    }


def merge_sessions_n(
    *,
    sources: list[dict],
    output_dir: Path,
    output_slug: str,
) -> dict:
    """Merge N sessions into ``output_dir``. ``sources`` is a list of
    dicts (PATH/SLUG/OFFSET_MM/LABEL); the first source has the
    highest priority — its rows survive intact, subsequent sources
    contribute only rows that don't overlap any earlier source's
    captured range.

    Each source's ``rail_position_mm`` values are translated by its
    ``offset_mm`` into the merged coordinate frame. The source whose
    offset is 0.0 (typically the "anchor" session, often the
    oldest) keeps its native frame as canonical — but the merge
    doesn't require it to be listed first; priority is independent
    of frame choice. If no source has offset 0.0, the merged frame
    is whatever the offsets implicitly define; positions are
    consistent across the merged dataset regardless.
    """
    if not sources:
        raise ValueError("need at least one --source")

    output_lens_dir = output_dir / output_slug
    output_lens_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict] = []
    # Each source's captured-range (in canonical frame) — used to drop
    # subsequent sources' rows that fall inside higher-priority sources'
    # ranges.
    earlier_ranges: list[tuple[float, float]] = []
    row_counts: list[dict] = []
    for src in sources:
        src_csv = src["path"] / "shots.csv"
        if not src_csv.exists():
            raise FileNotFoundError(f"missing shots.csv: {src_csv}")
        src_lens_dir = src["path"] / src["slug"]
        if not src_lens_dir.exists():
            raise FileNotFoundError(
                f"missing source lens dir: {src_lens_dir}"
            )

        slug = src["slug"]
        all_rows = [
            r for r in _read_rows(src_csv)
            if r.get("lens_config_slug") == slug
        ]
        kept = []
        for r in all_rows:
            if r.get("phase") == "calibration":
                kept.append(r)
                continue
            pos = _parse_pos(r)
            if pos is None:
                kept.append(r)
                continue
            canonical_pos = pos + src["offset_mm"]
            # Drop if any earlier source covered this position.
            inside_earlier = any(
                lo <= canonical_pos <= hi for lo, hi in earlier_ranges
            )
            if inside_earlier:
                continue
            kept.append(r)

        # Compute this source's captured range BEFORE filtering — so
        # later sources see the full coverage, not the post-filter
        # subset (a later source can still contribute rows past this
        # range, which is the whole point of the merge).
        full_range = _captured_range(all_rows, offset_mm=src["offset_mm"])
        if full_range is not None:
            earlier_ranges.append(full_range)

        label = src["label"] or _safe_label_from_path(src["path"])
        for r in kept:
            merged.append(_merge_one_row(
                row=r,
                source_session=label,
                source_lens_dir=src_lens_dir,
                output_lens_dir=output_lens_dir,
                rail_offset_mm=src["offset_mm"],
            ))
        row_counts.append({
            "label": label,
            "session_id": _read_session_id(src["path"]),
            "slug": slug,
            "offset_mm": src["offset_mm"],
            "captured_range_canonical_mm": (
                list(full_range) if full_range else None
            ),
            "input_rows": len(all_rows),
            "kept_rows": len(kept),
        })

    # Sort by aperture, then rail_position_mm.
    def _sort_key(r: dict):
        pos = _parse_pos(r)
        return (r.get("aperture", ""), pos if pos is not None else 0.0)

    merged.sort(key=_sort_key)

    fieldnames = SHOT_COLUMNS + ("source_session",)
    out_csv_path = output_dir / "shots.csv"
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in merged:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Symlink calibration.json + range_check.json from the
    # highest-priority source.
    primary_lens_dir = sources[0]["path"] / sources[0]["slug"]
    for fname in ("calibration.json", "range_check.json"):
        src_file = primary_lens_dir / fname
        if src_file.exists():
            _symlink(src_file, output_lens_dir / fname)

    # Compose merged session.json.
    primary_session = json.loads(
        (sources[0]["path"] / "session.json").read_text()
    )
    summary = {
        "merged": True,
        "merge_created_at": datetime.now(tz=timezone.utc).isoformat(),
        "merge_mode": "n-source",
        "source_sessions": row_counts,
        "overlap_strategy": "priority-first-wins",
        "lens_configs": [output_slug],
        "output_slug": output_slug,
        "baseline_median_luminance": primary_session.get(
            "baseline_median_luminance"
        ),
        "ntfy_topic": primary_session.get("ntfy_topic"),
        "row_counts": {
            **{src["label"] or _safe_label_from_path(src["path"]):
               {"input": rc["input_rows"], "kept": rc["kept_rows"]}
               for src, rc in zip(sources, row_counts)},
            "merged_total": len(merged),
        },
        "notes": (
            f"N-source merge of {len(sources)} sessions into "
            f"slug {output_slug!r}. Highest-priority source first: "
            + ", ".join(
                f"{rc['label']} ({rc['session_id']})"
                for rc in row_counts
            )
            + ". JPEGs and crops are symlinks into the source sessions; "
            "originals are retained on disk."
        ),
    }
    (output_dir / "session.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def _safe_label_from_path(p: Path) -> str:
    """Fallback label from a session path — strip the timestamp dir
    name. Replace any path separators with ``_``."""
    return p.name.replace("/", "_").replace(":", "_") or "src"


def _read_session_id(session_dir: Path) -> str | None:
    js = session_dir / "session.json"
    if not js.exists():
        return None
    try:
        return json.loads(js.read_text()).get("session_id")
    except (OSError, ValueError):
        return None


# ---- CLI ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--prev", type=Path,
        help="(2-session) Anchor session dir — its frame is preserved",
    )
    parser.add_argument(
        "--cur", type=Path,
        help="(2-session) Offset session dir",
    )
    parser.add_argument(
        "--cur-offset-mm", type=float,
        help=(
            "(2-session) Offset to ADD to CUR rail_position_mm so it "
            "lands in PREV coordinates."
        ),
    )
    parser.add_argument(
        "--lens-config-slug", type=str,
        help="(2-session) Lens-config slug shared by both sessions",
    )
    parser.add_argument(
        "--overlap-strategy", choices=OVERLAP_STRATEGIES, default="cur",
        help=(
            "(2-session) Where both sessions captured the same rail-mm "
            "range, which session's data wins. Default 'cur'."
        ),
    )
    parser.add_argument(
        "--source", action="append", default=[], metavar="SPEC",
        help=(
            "(N-session) PATH:SLUG:OFFSET_MM[:LABEL] — repeat for each "
            "session. First --source = highest priority, defines the "
            "canonical coordinate frame (offset must be 0.0). Escape "
            "literal colons inside SLUG as ``\\:``."
        ),
    )
    parser.add_argument(
        "--output-slug", type=str,
        help=(
            "(N-session) Lens-config directory name in the merged "
            "output. Sources may have different per-session slugs."
        ),
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output merged session directory (created if absent)",
    )
    args = parser.parse_args()

    if args.source:
        if not args.output_slug:
            parser.error("--source mode requires --output-slug")
        sources = [_parse_source_spec(s) for s in args.source]
        summary = merge_sessions_n(
            sources=sources,
            output_dir=args.output,
            output_slug=args.output_slug,
        )
    else:
        required = [
            ("--prev", args.prev),
            ("--cur", args.cur),
            ("--cur-offset-mm", args.cur_offset_mm),
            ("--lens-config-slug", args.lens_config_slug),
        ]
        missing = [name for name, v in required if v is None]
        if missing:
            parser.error(
                f"2-session mode requires {missing}; or use --source for "
                "N-session merge"
            )
        summary = merge_sessions(
            prev_dir=args.prev,
            cur_dir=args.cur,
            cur_offset_mm=args.cur_offset_mm,
            lens_config_slug=args.lens_config_slug,
            output_dir=args.output,
            overlap_strategy=args.overlap_strategy,
        )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
