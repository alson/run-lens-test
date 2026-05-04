"""Tests for the range-check + auto-extension flow in session.runner.

Uses a scripted FakeRangeChecker (separate from the production stub)
to drive the runner through:
  * stub-style ok=True: no extra captures, range_check.json has
    extension_applied=False
  * needs_extension=True, peak_side='far', then peak_in_range=True:
    probe shots and a full extension sweep happen, files land in
    <lens>/probe and <lens>/f_*/step_NNN.jpg with NNN past the
    original sweep range
  * needs_extension=True, peak_side='near': probe + extension shoot
    on the negative side
  * peak_in_range_after_probe=False: user is prompted, no extension
    sweep, range_check.json reflects it
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from exposure.range_check import RangeChecker, RangeResult
from session.log import Phase
from session.runner import JogResult, SessionRunner
from stackshot.protocol import Direction
from tests.test_runner import FakeCamera, FakeRail, cfg, make_runner


# ---- Scripted RangeChecker ------------------------------------------------


class ScriptedRangeChecker(RangeChecker):
    """Returns a scripted sequence of RangeResults instead of running
    the real (TBD) detector. The runner calls evaluate() at most twice
    per lens config (once with no probe data, once with it)."""

    def __init__(self, results: list[RangeResult]) -> None:
        super().__init__(config=None)
        self._queue = deque(results)
        self.calls: list[dict] = []

    def evaluate(  # type: ignore[override]
        self,
        lens_dir,
        end_mm,
        step_size_mm,
        apertures,
        *,
        probe_data=None,
    ):
        self.calls.append({
            "lens_dir": lens_dir,
            "end_mm": end_mm,
            "step_size_mm": step_size_mm,
            "apertures": list(apertures),
            "probe_data": probe_data,
        })
        return self._queue.popleft()


# ---- Helpers --------------------------------------------------------------


def _run_one_lens_config(runner: SessionRunner, slug: str = "L") -> None:
    calls = {"n": 0}

    def _prompt() -> str | None:
        calls["n"] += 1
        return slug if calls["n"] == 1 else None

    runner.prompt_lens_config = _prompt
    runner.run()


# ---- Stub ok=True (default behaviour) --------------------------------------


class TestRangeOk:
    def test_no_extra_captures_and_json_records_ok(
        self, cfg, tmp_path
    ) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"steps": 3, "end_mm": 4.0},
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=True, peak_side="centered",
                        needs_extension=False, notes="ok"),
        ])
        _run_one_lens_config(runner)

        capture_count = sum(
            1 for k, _ in camera.history if k == "capture"
        )
        # 1 calibration + 3 sweep = 4 captures, no probe / extension.
        assert capture_count == 4

        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["ok"] is True
        assert rc["extension_applied"] is False


# ---- Far-side extension --------------------------------------------------


class TestFarExtension:
    def test_probe_then_full_extension(self, cfg, tmp_path) -> None:
        # end_mm=10 with 3 steps → step_size_mm = 5.
        # extension = end_mm × 0.5 = 5 → 1 extension step.
        # Captures: 1 cal + 3 sweep + 2 probe + 1 extension = 7.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 10.0},
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True,
                        notes="peak likely beyond end"),
            RangeResult(ok=True, peak_side="far",
                        needs_extension=False,
                        peak_in_range_after_probe=True,
                        notes="probe brackets peak"),
        ])
        _run_one_lens_config(runner)

        capture_count = sum(
            1 for k, _ in camera.history if k == "capture"
        )
        assert capture_count == 7

        probe_dir = runner.session_dir / "L" / "probe"
        probe_files = list(probe_dir.glob("*.jpg"))
        assert len(probe_files) == 2

        sweep_dir = runner.session_dir / "L" / "f_4"
        sweep_files = sorted(sweep_dir.glob("step_*.jpg"))
        # Original sweep created step_000..step_002; the 1-step
        # extension adds step_003.
        assert [p.name for p in sweep_files] == [
            "step_000.jpg", "step_001.jpg", "step_002.jpg",
            "step_003.jpg",
        ]

        rows = runner.shots_csv.read_rows()
        phases = [r["phase"] for r in rows]
        assert phases.count(Phase.PROBE.value) == 2
        assert phases.count(Phase.EXTENSION.value) == 1

        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["ok"] is False
        assert rc["peak_side"] == "far"
        assert rc["extension_applied"] is True
        assert rc["extension_step_count"] == 1


# ---- Near-side extension --------------------------------------------------


class TestNearExtension:
    def test_probe_and_extension_on_negative_side(self, cfg, tmp_path) -> None:
        # Same step / extension math as the far case: 3 sweep + 1
        # extension = 4 step_*.jpg files in the per-aperture dir.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 10.0},
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="near",
                        needs_extension=True, notes="peak before start"),
            RangeResult(ok=True, peak_side="near",
                        needs_extension=False,
                        peak_in_range_after_probe=True,
                        notes="probe ok"),
        ])
        _run_one_lens_config(runner)

        probe_dir = runner.session_dir / "L" / "probe"
        assert len(list(probe_dir.glob("*.jpg"))) == 2

        sweep_dir = runner.session_dir / "L" / "f_4"
        assert len(list(sweep_dir.glob("step_*.jpg"))) == 4

        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["peak_side"] == "near"
        assert rc["extension_applied"] is True


# ---- Probe says peak still outside ----------------------------------------


class TestProbeFails:
    def test_no_extension_user_prompted(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        confirms: list[str] = []

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 4.0},
        )
        runner.confirm = lambda msg: (confirms.append(msg), True)[1]
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True, notes="peak beyond end"),
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True,
                        peak_in_range_after_probe=False,
                        notes="probe still doesn't bracket"),
        ])
        _run_one_lens_config(runner)

        # No extension files in f_4 (only the original 3).
        sweep_dir = runner.session_dir / "L" / "f_4"
        assert len(list(sweep_dir.glob("step_*.jpg"))) == 3
        # Probe files are still present.
        probe_dir = runner.session_dir / "L" / "probe"
        assert len(list(probe_dir.glob("*.jpg"))) == 2

        # User got a "Range still inadequate" prompt before lens
        # change.
        assert any("Range still inadequate" in c for c in confirms)
        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["extension_applied"] is False
        assert "did not bracket peak" in rc["notes"]


# ---- Probe positions math --------------------------------------------------


class TestProbePositions:
    """Direct unit test of the probe-position helper (no runner state)."""

    def test_far_side_two_probes_evenly_spaced(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[],
        )
        positions = runner._probe_positions(
            extension_mm=6.0, probe_count=2, peak_side="far", end_mm=10.0
        )
        # Fractions 1/3 and 2/3 of 6 mm past 10 mm: 12, 14.
        assert positions == pytest.approx([12.0, 14.0])

    def test_near_side_two_probes_negative(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[],
        )
        positions = runner._probe_positions(
            extension_mm=6.0, probe_count=2, peak_side="near", end_mm=10.0
        )
        # Negative fractions of extension before 0: -2, -4.
        assert positions == pytest.approx([-2.0, -4.0])
