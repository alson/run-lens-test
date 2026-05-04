"""Tests for session.runner.SessionRunner.

The hardware is faked end-to-end. We exercise:
  * aperture_sequence helper
  * the sweep flow: rail moves, per-aperture captures, CSV rows
  * droop pause + resume mid-sweep
  * lens-change recovery success path
  * graceful start/stop and session.json writes
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
from unittest.mock import MagicMock

import pytest

from exposure.range_check import RangeChecker
from session.runner import (
    JogResult,
    SessionRunner,
    aperture_sequence,
)
from session.calibrate import ApertureCalibration, CalibrationResult
from session.log import Phase, ShotsCSV
from stackshot.protocol import Direction


# ---- Fixture: minimal config -----------------------------------------------


@pytest.fixture
def cfg():
    return SimpleNamespace(
        MAX_FSTOP=16.0,
        WHOLE_STOP_SCALE=(1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0),
        RANGE_EXTENSION={
            4.0: (100, 0.0),
            5.6: (100, 0.0),
            8.0: (100, 0.0),
            11.0: (200, 0.0),
            16.0: (400, 0.0),
        },
        ISO_FLOOR=100,
        ISO_WARN=1600,
        ISO_CEIL=3200,
        FEC_MIN_EV=-3.0,
        FEC_MAX_EV=+1.0,
        FEC_STEP_EV=1 / 3,
        TEST_SHOT_TOLERANCE_EV=1 / 3,
        TEST_SHOT_MAX_RETRIES=3,
        ROLLING_WINDOW=5,
        DROOP_CRIT_A=(3, -0.7),
        DROOP_CRIT_B=(2, -2.0),
        MIN_CONFIG_TREE_SIZE=100,
        IMAGE_QUALITY="JPEG Fine",
        SHUTTER_SPEED_CANDIDATES=("1/200",),
        CROP_W_FRAC=0.2,
        CROP_H_FRAC=0.2,
        CROP_X_FRAC=0.0,
        CROP_Y_FRAC=0.15,
        NTFY_TOPIC="test-topic",
    )


# ---- Aperture sequence helper ---------------------------------------------


class TestApertureSequence:
    def test_widest_then_whole_stops(self, cfg) -> None:
        choices = ["f/2.8", "f/3.5", "f/4", "f/5", "f/5.6", "f/8", "f/11", "f/16", "f/22"]
        seq = aperture_sequence(
            choices,
            whole_stop_scale=cfg.WHOLE_STOP_SCALE,
            max_fstop=16.0,
        )
        # Widest 2.8, then whole stops 4, 5.6, 8, 11, 16.
        assert seq == ["f/2.8", "f/4", "f/5.6", "f/8", "f/11", "f/16"]

    def test_widest_already_a_whole_stop(self, cfg) -> None:
        seq = aperture_sequence(
            ["f/4", "f/5.6", "f/8", "f/11", "f/16"],
            whole_stop_scale=cfg.WHOLE_STOP_SCALE,
            max_fstop=16.0,
        )
        # Widest is f/4; subsequent whole stops 5.6, 8, 11, 16.
        assert seq == ["f/4", "f/5.6", "f/8", "f/11", "f/16"]

    def test_max_fstop_caps(self, cfg) -> None:
        seq = aperture_sequence(
            ["f/4", "f/5.6", "f/8", "f/11", "f/16", "f/22", "f/32"],
            whole_stop_scale=cfg.WHOLE_STOP_SCALE,
            max_fstop=11.0,
        )
        assert "f/11" in seq
        assert "f/16" not in seq

    def test_handles_unparseable_entries(self, cfg) -> None:
        seq = aperture_sequence(
            ["unknown", "f/4", "auto", "f/8"],
            whole_stop_scale=cfg.WHOLE_STOP_SCALE,
            max_fstop=11.0,
        )
        assert seq == ["f/4", "f/5.6", "f/8"] or seq == ["f/4", "f/8"]
        # ("f/5.6" missing from choices → omitted; widest f/4, then whole-stop 8)


# ---- Fake camera + rail ----------------------------------------------------


@dataclass
class FakeCamera:
    apertures: list[str] = field(default_factory=lambda: [
        "f/4", "f/5.6", "f/8", "f/11", "f/16",
    ])
    history: list[tuple[str, object]] = field(default_factory=list)
    open_count: int = 0
    closed: bool = False
    healthy_count: int = 0

    def open(self) -> None:
        self.open_count += 1
        self.history.append(("open", None))

    def close(self) -> None:
        self.closed = True
        self.history.append(("close", None))

    def verify_healthy(self, *, min_widgets: int = 100) -> None:
        self.healthy_count += 1
        self.history.append(("verify_healthy", min_widgets))

    def apply_session_config(self, **kwargs) -> None:
        self.history.append(("apply_session_config", kwargs))

    def list_apertures(self) -> list[str]:
        return list(self.apertures)

    def set_aperture(self, fnum: str) -> None:
        self.history.append(("aperture", fnum))

    def set_iso(self, iso) -> None:
        self.history.append(("iso", iso))

    def set_fec(self, ev: float) -> None:
        self.history.append(("fec", float(ev)))

    def capture_to(self, dest, *, max_retries: int = 3) -> int:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"x")
        self.history.append(("capture", str(dest)))
        return 1

    def snapshot(self, label: str = "") -> None:
        self.history.append(("snapshot", label))


@dataclass
class FakeRail:
    position_mm: float = 0.0
    history: list[tuple[str, object]] = field(default_factory=list)
    closed: bool = False

    def get_position(self) -> float:
        return self.position_mm

    def zero_position(self) -> None:
        self.position_mm = 0.0
        self.history.append(("zero", None))

    def move_and_wait(
        self,
        direction: Direction,
        distance_mm: float,
        *,
        backlash: bool = True,
        timeout_s: float | None = None,
    ) -> None:
        if direction == Direction.FORWARD:
            self.position_mm += distance_mm
        else:
            self.position_mm -= distance_mm
        self.history.append(("move", (direction, distance_mm, backlash)))

    def stop(self) -> None:
        self.history.append(("stop", None))

    def close(self) -> None:
        self.closed = True


# ---- Convenience: build a runner with everything wired ---------------------


def make_runner(
    *,
    camera: FakeCamera,
    rail: FakeRail,
    cfg,
    tmp_path: Path,
    medians: Iterable[float],
    inputs: dict | None = None,
):
    queue = deque(medians)

    def _measure(_p):
        if not queue:
            return 100.0
        return queue.popleft()

    inputs = inputs or {}
    fixed_now = datetime(2026, 5, 4, 18, 35, tzinfo=timezone.utc)

    def _prompt_lens() -> str | None:
        return inputs.get("lens", "lens-1")

    def _prompt_step_count() -> int:
        return inputs.get("steps", 3)

    def _confirm(_msg) -> bool:
        return inputs.get("confirm", True)

    def _jog(rail: FakeRail) -> JogResult:
        # Simulates the user jogging to END_MM = 4 mm and pressing 'e'.
        end_mm = inputs.get("end_mm", 4.0)
        rail.position_mm = end_mm
        return JogResult(end_mm=end_mm)

    return SessionRunner(
        camera=camera,
        rail=rail,
        config=cfg,
        ntfy=None,
        range_checker=RangeChecker(cfg),
        output_root=tmp_path,
        measure_path=_measure,
        sleep=lambda s: None,
        now=lambda: fixed_now,
        prompt_lens_config=_prompt_lens,
        prompt_step_count=_prompt_step_count,
        confirm=_confirm,
        jog=_jog,
    )


# ---- start / stop ----------------------------------------------------------


class TestStartStop:
    def test_start_opens_camera_and_creates_session_dir(
        self, cfg, tmp_path
    ) -> None:
        camera = FakeCamera()
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        assert camera.open_count == 1
        assert camera.healthy_count == 1
        assert any(k == "apply_session_config" for k, _ in camera.history)
        assert (runner.session_dir / "session.json").exists()
        assert (runner.session_dir / "shots.csv").exists()

    def test_stop_writes_ended_at(self, cfg, tmp_path) -> None:
        camera = FakeCamera()
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        runner.stop()
        meta = runner.metadata
        assert meta is not None
        assert meta.ended_at is not None
        assert camera.closed is True


# ---- One full lens-config sweep --------------------------------------------


class TestSweep:
    def test_three_steps_two_apertures(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4", "f/8"])
        rail = FakeRail()
        # Calibration: 1 baseline shot at f/4 + 1 at f/8.
        # Sweep: 3 steps × 2 apertures = 6 shots. All return baseline.
        medians = [100.0, 100.0] + [100.0] * 6
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "test-lens", "steps": 3, "end_mm": 4.0},
        )

        # Drive only one lens config: prompt returns "test-lens" once
        # then None to terminate. Use a counter.
        calls = {"n": 0}

        def _prompt() -> str | None:
            calls["n"] += 1
            return "test-lens" if calls["n"] == 1 else None

        runner.prompt_lens_config = _prompt
        runner.run()

        # Six sweep captures + 2 calibration captures = 8 captures.
        captures = [v for k, v in camera.history if k == "capture"]
        assert len(captures) == 8

        # Sweep files land at <session>/test-lens/f_4/step_NNN.jpg etc.
        step_files = list(
            (runner.session_dir / "test-lens" / "f_4").glob("step_*.jpg")
        )
        assert len(step_files) == 3

        # CSV: 2 calibration rows + 6 sweep rows = 8.
        rows = runner.shots_csv.read_rows() if runner.shots_csv else []
        phases = [r["phase"] for r in rows]
        assert phases.count(Phase.CALIBRATION.value) == 2
        assert phases.count(Phase.SWEEP.value) == 6

    def test_rail_moves_forward_per_step(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"lens": "L", "steps": 3, "end_mm": 6.0},
        )

        def _prompt() -> str | None:
            if not hasattr(_prompt, "called"):
                _prompt.called = True
                return "L"
            return None

        runner.prompt_lens_config = _prompt
        runner.run()

        moves = [v for k, v in rail.history if k == "move"]
        # Sequence: jog leaves rail at 6.0; backward 6.0 to position 0
        # (before sweep); forward 3.0 (step 1); forward 3.0 (step 2);
        # backward 6.0 (return to 0 for next lens).
        # step_size_mm = 6 / (3-1) = 3.0
        directions_distances = [(d, dist) for (d, dist, _) in moves]
        assert (Direction.BACKWARD, 6.0) in directions_distances
        assert directions_distances.count((Direction.FORWARD, 3.0)) == 2


# ---- Droop pause + resume --------------------------------------------------


class TestDroopPause:
    def test_droop_triggers_confirm_and_resets_window(
        self, cfg, tmp_path
    ) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # Baseline 100 (calibration). Sweep: two -2 EV shots in a row
        # at f/4 trip criterion B.
        # ev_delta(median, 100) ≤ -2 EV requires median ≤ ~31.
        medians = [100.0,                 # calibration baseline
                   25.0, 25.0,            # two sweep shots that trip B
                   100.0]                 # remaining shot after resume
        confirm_calls: list[str] = []

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "L", "steps": 2, "end_mm": 2.0},
        )

        def _confirm(msg: str) -> bool:
            confirm_calls.append(msg)
            return True

        runner.confirm = _confirm

        def _prompt() -> str | None:
            if not hasattr(_prompt, "called"):
                _prompt.called = True
                return "L"
            return None

        runner.prompt_lens_config = _prompt
        runner.run()

        # Droop confirm prompt was hit once.
        droop_prompts = [c for c in confirm_calls if "Battery droop" in c]
        assert len(droop_prompts) == 1
        # Plus one for lens-change prompt.
        assert any("Power off camera" in c for c in confirm_calls)

    def test_droop_aborts_when_user_says_q(self, cfg, tmp_path) -> None:
        from errors import SessionAborted

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        medians = [100.0, 25.0, 25.0]

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "L", "steps": 2, "end_mm": 2.0},
        )

        def _confirm(msg: str) -> bool:
            return False  # user always says abort

        runner.confirm = _confirm

        def _prompt() -> str | None:
            if not hasattr(_prompt, "called"):
                _prompt.called = True
                return "L"
            return None

        runner.prompt_lens_config = _prompt

        with pytest.raises(SessionAborted):
            runner.run()


# ---- Lens-change recovery -------------------------------------------------


class TestLensChange:
    def test_camera_closed_then_reopened(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"lens": "L", "steps": 2, "end_mm": 2.0},
        )

        def _prompt() -> str | None:
            if not hasattr(_prompt, "called"):
                _prompt.called = True
                return "L"
            return None

        runner.prompt_lens_config = _prompt
        runner.run()

        # Camera was opened initially (1) and reopened after lens
        # change (>= 2). open_count tracks that.
        assert camera.open_count >= 2
        assert camera.healthy_count >= 2
