"""Tests for session.log — atomic CSV append + JSON writer + Phase enum."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session.log import (
    Phase,
    SHOT_COLUMNS,
    SessionMetadata,
    ShotRow,
    ShotsCSV,
    write_json,
)


def _row(**overrides) -> ShotRow:
    base = dict(
        timestamp="2026-05-04T18:35:00Z",
        session_id="2026-05-04_183500",
        lens_config_slug="Sigma 105 macro",
        rail_step=0,
        rail_position_mm=0.0,
        aperture="f/4",
        iso="100",
        shutter_speed="1/200",
        fec=0.0,
        median_central_luminance=128.0,
        median_ev_relative_to_baseline=0.0,
        filename="step_000.jpg",
        phase=Phase.CALIBRATION,
        notes="",
    )
    base.update(overrides)
    return ShotRow(**base)


# ---- ShotsCSV --------------------------------------------------------------


class TestShotsCSV:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        path = tmp_path / "shots.csv"
        ShotsCSV(path)
        text = path.read_text("utf-8")
        first = text.splitlines()[0]
        assert first == ",".join(SHOT_COLUMNS)

    def test_append_writes_row(self, tmp_path: Path) -> None:
        path = tmp_path / "shots.csv"
        log = ShotsCSV(path)
        log.append(_row())
        rows = log.read_rows()
        assert len(rows) == 1
        assert rows[0]["aperture"] == "f/4"
        assert rows[0]["phase"] == "calibration"

    def test_multiple_appends_accumulate(self, tmp_path: Path) -> None:
        path = tmp_path / "shots.csv"
        log = ShotsCSV(path)
        for i in range(3):
            log.append(_row(rail_step=i, filename=f"step_{i:03d}.jpg"))
        rows = log.read_rows()
        assert len(rows) == 3
        assert [int(r["rail_step"]) for r in rows] == [0, 1, 2]

    def test_phase_enum_values_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "shots.csv"
        log = ShotsCSV(path)
        for phase in Phase:
            log.append(_row(phase=phase))
        rows = log.read_rows()
        assert [r["phase"] for r in rows] == [p.value for p in Phase]

    def test_atomic_write_no_tmp_left_behind(self, tmp_path: Path) -> None:
        path = tmp_path / "shots.csv"
        log = ShotsCSV(path)
        log.append(_row())
        # No <path>.tmp leaks after a normal append.
        assert not (tmp_path / "shots.csv.tmp").exists()
        # And the file is not empty.
        assert path.read_text("utf-8")

    def test_floats_formatted_compactly(self, tmp_path: Path) -> None:
        path = tmp_path / "shots.csv"
        log = ShotsCSV(path)
        log.append(_row(rail_position_mm=3.14159265358979,
                        median_ev_relative_to_baseline=-0.333333333))
        text = path.read_text("utf-8").splitlines()[1]
        # %.6g caps float width without unbounded trailing digits.
        assert "3.14159" in text
        assert "-0.333333" in text


# ---- write_json ------------------------------------------------------------


class TestWriteJson:
    def test_dict_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json(path, {"a": 1, "b": [2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}

    def test_dataclass_serialised(self, tmp_path: Path) -> None:
        path = tmp_path / "session.json"
        meta = SessionMetadata(
            session_id="abc",
            started_at="2026-05-04T18:35:00Z",
            output_root="/tmp/x",
            ntfy_topic="topic",
        )
        write_json(path, meta)
        loaded = json.loads(path.read_text())
        assert loaded["session_id"] == "abc"
        assert loaded["aborted"] is False

    def test_enum_serialised(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        write_json(path, {"phase": Phase.SWEEP})
        assert json.loads(path.read_text()) == {"phase": "sweep"}

    def test_path_serialised(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        write_json(path, {"dir": Path("/foo/bar")})
        assert json.loads(path.read_text()) == {"dir": "/foo/bar"}

    def test_unknown_type_raises(self, tmp_path: Path) -> None:
        class Weird: ...
        with pytest.raises(TypeError):
            write_json(tmp_path / "x.json", {"x": Weird()})

    def test_atomic_no_tmp_left_behind(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        write_json(path, {"a": 1})
        assert not (tmp_path / "out.json.tmp").exists()
