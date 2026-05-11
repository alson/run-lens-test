"""Tests for tools.merge_sessions.

Builds a pair of synthetic session directories on a tmp filesystem, runs
the merge, and asserts the resulting shots.csv + symlink layout matches
expectations. The session-level fixtures are deliberately tiny (a few
sweep + calibration rows per session) so each test reads cleanly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.merge_sessions import (
    SHOT_COLUMNS,
    _filter_by_overlap,
    _parse_pos,
    merge_sessions,
)


# ---- Fixture builders ------------------------------------------------------

LENS = "Test Lens 50mm"


def _row(
    *,
    session_id: str,
    rail_step: int,
    rail_position_mm: float,
    aperture: str,
    filename: str,
    phase: str,
    median: float = 100.0,
    ev: float = 0.0,
):
    return {
        "timestamp": f"2026-05-01T12:00:0{rail_step % 10}",
        "session_id": session_id,
        "lens_config_slug": LENS,
        "rail_step": str(rail_step),
        "rail_position_mm": f"{rail_position_mm:.6g}",
        "aperture": aperture,
        "iso": "100",
        "shutter_speed": "1/200",
        "fec": "0",
        "median_central_luminance": str(median),
        "median_ev_relative_to_baseline": str(ev),
        "filename": filename,
        "phase": phase,
        "notes": "",
    }


def _write_session(
    root: Path,
    *,
    session_id: str,
    rows: list[dict],
    baseline_median: float = 129.0,
) -> Path:
    """Create a minimal session directory with shots.csv +
    session.json + a per-lens-config subdir + dummy source JPEGs +
    one center crop per source."""
    session_dir = root / session_id
    session_dir.mkdir()
    # shots.csv
    csv_path = session_dir / "shots.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHOT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SHOT_COLUMNS})
    # session.json
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": session_id,
        "baseline_median_luminance": baseline_median,
        "ntfy_topic": "test-topic",
    }))
    # Lens config subdir + dummy JPEGs + center crops
    lens_dir = session_dir / LENS
    lens_dir.mkdir()
    for r in rows:
        if not r["filename"]:
            continue
        # Calibration rows live at the lens-config top level (no
        # aperture sub-dir). Sweep/probe/etc. rows have an
        # ``<aperture>/<leaf>`` layout.
        if "/" not in r["filename"]:
            (lens_dir / r["filename"]).write_bytes(
                b"fake " + r["filename"].encode()
            )
            continue
        aperture_dir, leaf = r["filename"].split("/", 1)
        src_dir = lens_dir / aperture_dir
        src_dir.mkdir(exist_ok=True)
        (src_dir / leaf).write_bytes(b"fake jpeg " + leaf.encode())
        crop_dir = src_dir / "cropped" / "center"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / leaf).write_bytes(b"fake crop " + leaf.encode())
    return session_dir


# ---- Tests on the filter logic only ---------------------------------------

class TestFilterByOverlap:
    def test_cur_strategy_drops_prev_in_overlap(self):
        # PREV captured -3.0 to -1.0; CUR (after offset -2.0) is at
        # -3.0 to -1.0 (same range). 'cur' strategy → drop all PREV
        # non-calibration rows.
        prev = [
            _row(session_id="P", rail_step=0, rail_position_mm=-3.0,
                 aperture="f/4", filename="f_4/step_0100.jpg", phase="sweep"),
            _row(session_id="P", rail_step=1, rail_position_mm=-2.0,
                 aperture="f/4", filename="f_4/step_0101.jpg", phase="sweep"),
            _row(session_id="P", rail_step=2, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0102.jpg", phase="sweep"),
            _row(session_id="P", rail_step=0, rail_position_mm=0.0,
                 aperture="f/4", filename="cal_f_4_00.jpg", phase="calibration"),
        ]
        cur = [
            _row(session_id="C", rail_step=0, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg", phase="sweep"),
            _row(session_id="C", rail_step=1, rail_position_mm=-0.5,
                 aperture="f/4", filename="f_4/step_0101.jpg", phase="sweep"),
            _row(session_id="C", rail_step=2, rail_position_mm=1.0,
                 aperture="f/4", filename="f_4/step_0102.jpg", phase="sweep"),
        ]
        kept_p, kept_c, overlap = _filter_by_overlap(prev, cur, -2.0, "cur")
        # Calibration row survives; all sweep rows in [-3.0, -1.0]
        # range (CUR's translated range) are dropped.
        kept_phases = [r["phase"] for r in kept_p]
        assert kept_phases == ["calibration"]
        assert len(kept_c) == 3
        # Overlap range in PREV frame
        assert overlap == pytest.approx((-3.0, -1.0))

    def test_cur_strategy_keeps_prev_outside_cur_range(self):
        # PREV has rows outside CUR's translated range.
        prev = [
            _row(session_id="P", rail_step=0, rail_position_mm=+1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg", phase="sweep"),
            _row(session_id="P", rail_step=1, rail_position_mm=-0.5,
                 aperture="f/4", filename="f_4/step_0101.jpg", phase="sweep"),
            _row(session_id="P", rail_step=2, rail_position_mm=-5.0,
                 aperture="f/4", filename="f_4/step_0102.jpg", phase="sweep"),
        ]
        cur = [
            _row(session_id="C", rail_step=0, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg", phase="sweep"),
            _row(session_id="C", rail_step=1, rail_position_mm=0.0,
                 aperture="f/4", filename="f_4/step_0101.jpg", phase="sweep"),
        ]
        # CUR translates to [-3.0, -2.0]. Only PREV row at -0.5 falls
        # outside; rows at +1.0 and -5.0 are outside too. The -2.0
        # offset means CUR translated to [-3.0, -2.0], so PREV at
        # -0.5, +1.0, -5.0 are all outside.
        kept_p, kept_c, overlap = _filter_by_overlap(prev, cur, -2.0, "cur")
        assert len(kept_p) == 3
        # Translated CUR range
        assert overlap == pytest.approx((-3.0, -2.0))

    def test_both_strategy_keeps_everything(self):
        prev = [_row(session_id="P", rail_step=0, rail_position_mm=-1.0,
                     aperture="f/4", filename="f_4/step_0100.jpg",
                     phase="sweep")]
        cur = [_row(session_id="C", rail_step=0, rail_position_mm=0.0,
                    aperture="f/4", filename="f_4/step_0100.jpg",
                    phase="sweep")]
        kept_p, kept_c, _ = _filter_by_overlap(prev, cur, -1.0, "both")
        assert len(kept_p) == 1
        assert len(kept_c) == 1

    def test_prev_strategy_drops_cur_in_overlap(self):
        prev = [
            _row(session_id="P", rail_step=0, rail_position_mm=-2.0,
                 aperture="f/4", filename="f_4/step_0100.jpg", phase="sweep"),
            _row(session_id="P", rail_step=1, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0101.jpg", phase="sweep"),
        ]
        cur = [
            _row(session_id="C", rail_step=0, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg", phase="sweep"),
            _row(session_id="C", rail_step=1, rail_position_mm=-3.0,
                 aperture="f/4", filename="f_4/step_0101.jpg", phase="sweep"),
        ]
        # CUR + offset -1.0 → CUR positions in PREV frame: -2.0, -4.0.
        # PREV range: -2.0..-1.0. Drop CUR row at -2.0 (inside), keep
        # CUR row at -4.0 (outside).
        kept_p, kept_c, overlap = _filter_by_overlap(prev, cur, -1.0, "prev")
        assert len(kept_p) == 2
        kept_cur_positions = [_parse_pos(r) for r in kept_c]
        assert -3.0 in kept_cur_positions  # the one outside
        # The -1.0 PREV-frame -2.0 row is dropped


# ---- End-to-end merge with synthetic sessions ----------------------------


class TestEndToEndMerge:
    def test_merged_csv_has_translated_positions(self, tmp_path: Path):
        prev_rows = [
            _row(session_id="P", rail_step=0, rail_position_mm=0.0,
                 aperture="f/4", filename="cal_f_4_00.jpg",
                 phase="calibration"),
            _row(session_id="P", rail_step=10, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0110.jpg", phase="sweep"),
            _row(session_id="P", rail_step=20, rail_position_mm=-2.0,
                 aperture="f/4", filename="f_4/step_0120.jpg", phase="sweep"),
        ]
        # CUR jog uses its own frame; positions 0, -1, -2 mm in CUR.
        # With offset -3.0, those map to PREV -3.0, -4.0, -5.0.
        cur_rows = [
            _row(session_id="C", rail_step=0, rail_position_mm=0.0,
                 aperture="f/4", filename="f_4/step_0100.jpg", phase="sweep"),
            _row(session_id="C", rail_step=10, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0110.jpg", phase="sweep"),
            _row(session_id="C", rail_step=20, rail_position_mm=-2.0,
                 aperture="f/4", filename="f_4/step_0120.jpg", phase="sweep"),
        ]
        prev_dir = _write_session(tmp_path, session_id="prev_sess",
                                  rows=prev_rows)
        cur_dir = _write_session(tmp_path, session_id="cur_sess",
                                 rows=cur_rows)
        out_dir = tmp_path / "merged"

        merge_sessions(
            prev_dir=prev_dir, cur_dir=cur_dir, cur_offset_mm=-3.0,
            lens_config_slug=LENS, output_dir=out_dir,
            overlap_strategy="cur",
        )

        merged_csv = list(csv.DictReader((out_dir / "shots.csv").open()))
        # Sweep rows in merged CSV.
        sweep = [r for r in merged_csv if r["phase"] == "sweep"]
        positions = sorted(float(r["rail_position_mm"]) for r in sweep)
        # PREV (no overlap with CUR's translated [-5..-3]): -2.0, -1.0
        # CUR (translated): -5.0, -4.0, -3.0
        assert positions == pytest.approx([-5.0, -4.0, -3.0, -2.0, -1.0])

    def test_source_session_column_distinguishes_rows(
        self, tmp_path: Path,
    ):
        prev_rows = [
            _row(session_id="P", rail_step=0, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        cur_rows = [
            _row(session_id="C", rail_step=0, rail_position_mm=-2.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        prev_dir = _write_session(tmp_path, session_id="pp", rows=prev_rows)
        cur_dir = _write_session(tmp_path, session_id="cc", rows=cur_rows)
        out_dir = tmp_path / "m"

        merge_sessions(
            prev_dir=prev_dir, cur_dir=cur_dir, cur_offset_mm=0.0,
            lens_config_slug=LENS, output_dir=out_dir,
            overlap_strategy="both",
        )
        merged = list(csv.DictReader((out_dir / "shots.csv").open()))
        assert sorted(r["source_session"] for r in merged) == ["cur", "prev"]

    def test_jpegs_symlinked_with_session_prefix(self, tmp_path: Path):
        prev_rows = [
            _row(session_id="P", rail_step=0, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        cur_rows = [
            _row(session_id="C", rail_step=0, rail_position_mm=+2.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        prev_dir = _write_session(tmp_path, session_id="pp", rows=prev_rows)
        cur_dir = _write_session(tmp_path, session_id="cc", rows=cur_rows)
        out_dir = tmp_path / "m"

        merge_sessions(
            prev_dir=prev_dir, cur_dir=cur_dir, cur_offset_mm=-3.0,
            lens_config_slug=LENS, output_dir=out_dir,
            overlap_strategy="both",
        )
        # PREV's step_0100.jpg and CUR's step_0100.jpg both land in
        # f_4/ but with session-prefixed leaves.
        f4 = out_dir / LENS / "f_4"
        assert (f4 / "prev_step_0100.jpg").is_symlink()
        assert (f4 / "cur_step_0100.jpg").is_symlink()
        # Resolved targets point to the original session files.
        assert (f4 / "prev_step_0100.jpg").resolve() == (
            prev_dir / LENS / "f_4" / "step_0100.jpg"
        ).resolve()
        assert (f4 / "cur_step_0100.jpg").resolve() == (
            cur_dir / LENS / "f_4" / "step_0100.jpg"
        ).resolve()
        # CSV filenames reflect the new prefix.
        rows = list(csv.DictReader((out_dir / "shots.csv").open()))
        names = sorted(r["filename"] for r in rows)
        assert names == ["f_4/cur_step_0100.jpg", "f_4/prev_step_0100.jpg"]

    def test_per_crop_directories_get_symlinked_too(self, tmp_path: Path):
        prev_rows = [
            _row(session_id="P", rail_step=0, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        cur_rows = [
            _row(session_id="C", rail_step=0, rail_position_mm=+2.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        prev_dir = _write_session(tmp_path, session_id="pp", rows=prev_rows)
        cur_dir = _write_session(tmp_path, session_id="cc", rows=cur_rows)
        out_dir = tmp_path / "m"

        merge_sessions(
            prev_dir=prev_dir, cur_dir=cur_dir, cur_offset_mm=-3.0,
            lens_config_slug=LENS, output_dir=out_dir,
            overlap_strategy="both",
        )
        center = out_dir / LENS / "f_4" / "cropped" / "center"
        assert (center / "prev_step_0100.jpg").is_symlink()
        assert (center / "cur_step_0100.jpg").is_symlink()

    def test_session_json_records_provenance(self, tmp_path: Path):
        prev_rows = [
            _row(session_id="P", rail_step=0, rail_position_mm=-1.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        cur_rows = [
            _row(session_id="C", rail_step=0, rail_position_mm=-2.0,
                 aperture="f/4", filename="f_4/step_0100.jpg",
                 phase="sweep"),
        ]
        prev_dir = _write_session(tmp_path, session_id="P_001",
                                  rows=prev_rows)
        cur_dir = _write_session(tmp_path, session_id="C_002",
                                 rows=cur_rows)
        out_dir = tmp_path / "m"

        merge_sessions(
            prev_dir=prev_dir, cur_dir=cur_dir, cur_offset_mm=-3.0,
            lens_config_slug=LENS, output_dir=out_dir,
            overlap_strategy="cur",
        )
        meta = json.loads((out_dir / "session.json").read_text())
        assert meta["merged"] is True
        assert meta["overlap_strategy"] == "cur"
        ids = sorted(s["session_id"] for s in meta["source_sessions"])
        assert ids == ["C_002", "P_001"]
        # Anchor session has offset 0; offset session has the provided
        # value.
        offsets = {s["session_id"]: s["rail_offset_mm"]
                   for s in meta["source_sessions"]}
        assert offsets["P_001"] == 0.0
        assert offsets["C_002"] == pytest.approx(-3.0)
