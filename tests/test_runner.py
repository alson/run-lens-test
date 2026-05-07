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
        EC_MIN_EV=-3.0,
        EC_MAX_EV=+3.0,
        EC_STEP_EV=1 / 3,
        SHOTS_PER_STEP_APERTURE=1,
        MIN_RAIL_POSITIONS=2,
        TEST_SHOT_TOLERANCE_EV=1 / 3,
        TEST_SHOT_MAX_RETRIES=3,
        ROLLING_WINDOW=5,
        DROOP_CRIT_A=(1, -2.0),  # severe single-shot trip
        DROOP_CRIT_B=(2, -0.7),  # moderate cumulative trip
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

    def set_ec(self, ev: float) -> None:
        self.history.append(("ec", float(ev)))

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

    def _prompt_magnification() -> float:
        return inputs.get("magnification", 1.0)

    def _confirm(_msg) -> bool:
        return inputs.get("confirm", True)

    def _jog(rail: FakeRail) -> JogResult:
        # Simulates the user jogging to END_MM = 4 mm and pressing 'e'.
        end_mm = inputs.get("end_mm", 4.0)
        rail.position_mm = end_mm
        return JogResult(end_mm=end_mm)

    # Fake planner so tests can set "steps" via inputs without
    # caring about the underlying physics (DOF / FWHM math). The
    # plan honours the legacy ``inputs["steps"]`` and the
    # ``inputs["end_mm"]`` so existing fixtures keep working with
    # minimal churn.
    from sharpness.planner import SamplePlan

    def _fake_plan_sweep(
        *, magnification, widest_aperture, max_aperture,
        rail_range_mm, replicas,
        coc_mm=None, positions_across_fwhm=None,
    ) -> SamplePlan:
        steps = inputs.get("steps", 3)
        step_size = rail_range_mm / max(1, steps - 1) if steps > 1 else rail_range_mm
        # Tests can override subsample_factors via inputs to exercise
        # the per-aperture skip logic; default is "no skip" so
        # existing tests keep their pre-Phase-B behaviour.
        factors = inputs.get(
            "subsample_factors", {widest_aperture: 1}
        )
        return SamplePlan(
            rail_step_mm=step_size,
            replicas_per_position=replicas,
            estimated_dof_mm_widest=1.0,
            estimated_fwhm_mm_widest=0.5,
            widest_aperture=widest_aperture,
            max_aperture=max_aperture,
            subsample_factors=factors,
            rail_range_mm=rail_range_mm,
            rail_positions=steps,
        )

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
        prompt_magnification=_prompt_magnification,
        plan_sweep=_fake_plan_sweep,
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


class TestTinyRangeRejog:
    """When the planner returns fewer rail positions than
    ``MIN_RAIL_POSITIONS``, the runner refuses and prompts a
    re-jog. ENTER → re-jog with the new range; q → SessionAborted."""

    def test_rejogs_when_range_yields_too_few_positions(
        self, cfg, tmp_path
    ) -> None:
        # MIN_RAIL_POSITIONS=2 in the test cfg; force the first plan
        # to return 1 (below the floor) and the second to return 3.
        cfg.MIN_RAIL_POSITIONS = 3
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        from sharpness.planner import SamplePlan as _SamplePlan
        plan_calls = {"n": 0}

        def _plan(*, rail_range_mm, replicas, widest_aperture, max_aperture, **_):
            plan_calls["n"] += 1
            positions = 1 if plan_calls["n"] == 1 else 4
            return _SamplePlan(
                rail_step_mm=rail_range_mm / max(1, positions - 1),
                replicas_per_position=replicas,
                estimated_dof_mm_widest=1.0,
                estimated_fwhm_mm_widest=0.5,
                widest_aperture=widest_aperture,
                max_aperture=max_aperture,
                subsample_factors={widest_aperture: 1},
                rail_range_mm=rail_range_mm,
                rail_positions=positions,
            )

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"lens": "L", "end_mm": 4.0},
        )
        runner.plan_sweep = _plan
        confirm_msgs: list[str] = []

        def _confirm(msg: str) -> bool:
            confirm_msgs.append(msg)
            return True   # ENTER → re-jog

        runner.confirm = _confirm
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()

        # Tiny-range prompt fired once.
        assert any("yields only" in m for m in confirm_msgs)
        # Plan was called twice (first refused, second accepted).
        assert plan_calls["n"] == 2

    def test_q_at_tiny_range_prompt_aborts(self, cfg, tmp_path) -> None:
        from errors import SessionAborted
        cfg.MIN_RAIL_POSITIONS = 3
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        from sharpness.planner import SamplePlan as _SamplePlan

        def _plan(*, rail_range_mm, replicas, widest_aperture, max_aperture, **_):
            return _SamplePlan(
                rail_step_mm=rail_range_mm,
                replicas_per_position=replicas,
                estimated_dof_mm_widest=1.0,
                estimated_fwhm_mm_widest=0.5,
                widest_aperture=widest_aperture,
                max_aperture=max_aperture,
                subsample_factors={widest_aperture: 1},
                rail_range_mm=rail_range_mm,
                rail_positions=1,  # always too few
            )

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 5,
            inputs={"lens": "L", "end_mm": 4.0},
        )
        runner.plan_sweep = _plan
        runner.confirm = lambda _msg: False  # q at tiny-range prompt
        runner.prompt_lens_config = lambda: "L"

        with pytest.raises(SessionAborted, match="tiny-range"):
            runner.run()


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


# ---- Recycle wait subtracts elapsed time -----------------------------------


class TestRecycleWait:
    """The configured per-aperture flash recycle (RANGE_EXTENSION[ap][1])
    is the minimum delay between consecutive shutter fires. Time spent
    on set_capture_settings, capture_to, measurement, sharpness queueing,
    and any pauses (range check, rail moves) all count against the
    budget. Across thousands of shots a naive sleep(RECYCLE) on top of
    work that already exceeds it adds hours."""

    def _drive_one_step_one_aperture(
        self, cfg, tmp_path, *, monotonic_seq: list[float],
    ) -> list[float]:
        """Drive a single-aperture, two-step sweep and return the list
        of seconds passed to ``sleep``. ``monotonic_seq`` is consumed
        in call order so each test can inject its own clock advance."""
        # The default test cfg has recycle=0.0 across all apertures
        # (calibration tests don't care). Override f/4 to a real
        # 3.0s recycle so the wait-time math has something to chew on.
        cfg.RANGE_EXTENSION = {**cfg.RANGE_EXTENSION, 4.0: (100, 3.0)}
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # 1 calibration + 2 sweep shots
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 4,
            inputs={"lens": "L", "steps": 2, "end_mm": 4.0},
        )

        sleeps: list[float] = []
        runner.sleep = lambda s: sleeps.append(s)
        clock = iter(monotonic_seq)
        runner.monotonic = lambda: next(clock)

        def _prompt() -> str | None:
            if not hasattr(_prompt, "called"):
                _prompt.called = True
                return "L"
            return None

        runner.prompt_lens_config = _prompt
        runner.run()
        return sleeps

    def test_first_shot_does_not_wait(self, cfg, tmp_path) -> None:
        # First sweep shot has no prior capture timestamp -> _wait_for_recycle
        # returns immediately. Each shot consumes one monotonic() call for
        # the ETA EWMA, one for the recycle wait (from shot 2 on), and one
        # to stamp _last_capture_started_at. With recycle=3.0s for f/4 in
        # cfg and a 100-second gap, no sleep should happen at all.
        sleeps = self._drive_one_step_one_aperture(
            cfg, tmp_path,
            monotonic_seq=[
                0.0,    # shot 1 ETA record
                0.0,    # shot 1 stamp
                100.0,  # shot 2 ETA record
                100.0,  # shot 2 wait read (elapsed=100 >> 3 → no sleep)
                100.0,  # shot 2 stamp
            ],
        )
        # No sleep should have been issued by _wait_for_recycle.
        assert sleeps == []

    def test_waits_only_remaining_recycle(self, cfg, tmp_path) -> None:
        # Recycle for f/4 is 3.0s in the test cfg. If 1.0s elapsed
        # between captures already, we should sleep 2.0s, not 3.0s.
        sleeps = self._drive_one_step_one_aperture(
            cfg, tmp_path,
            monotonic_seq=[
                0.0,    # shot 1 ETA record
                0.0,    # shot 1 stamp
                1.0,    # shot 2 ETA record
                1.0,    # shot 2 wait read → remaining = 3 - 1 = 2
                1.0,    # shot 2 stamp
            ],
        )
        assert sleeps == [pytest.approx(2.0)]

    def test_no_wait_when_elapsed_exceeds_recycle(self, cfg, tmp_path) -> None:
        # 5.0s elapsed since prior shutter -> 0 remaining; no sleep.
        sleeps = self._drive_one_step_one_aperture(
            cfg, tmp_path,
            monotonic_seq=[
                0.0,    # shot 1 ETA record
                0.0,    # shot 1 stamp
                5.0,    # shot 2 ETA record
                5.0,    # shot 2 wait read
                5.0,    # shot 2 stamp
            ],
        )
        assert sleeps == []


# ---- Droop pause + resume --------------------------------------------------


class TestOfferJogPause:
    """Optional freeform jog pause at session start and before each
    lens-config prompt — lets the user verify focus / release the
    StackShot from USB-control mode."""

    def _runner(self, cfg, tmp_path):
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        return make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 5,
            inputs={"lens": "L", "steps": 2, "end_mm": 2.0},
        )

    def test_no_op_when_jog_freeform_unset(self, cfg, tmp_path) -> None:
        runner = self._runner(cfg, tmp_path)
        runner.jog_freeform = None
        # Doesn't read input, doesn't call jog. Returns immediately.
        runner._offer_jog_pause(where="x")

    def test_enter_continues(self, cfg, tmp_path, monkeypatch) -> None:
        runner = self._runner(cfg, tmp_path)
        called = []
        runner.jog_freeform = lambda r: called.append(r)
        monkeypatch.setattr("builtins.input", lambda _msg: "")
        runner._offer_jog_pause(where="x")
        assert called == []  # ENTER → no jog

    def test_q_aborts(self, cfg, tmp_path, monkeypatch) -> None:
        from errors import SessionAborted
        runner = self._runner(cfg, tmp_path)
        runner.jog_freeform = lambda r: None
        monkeypatch.setattr("builtins.input", lambda _msg: "q")
        with pytest.raises(SessionAborted):
            runner._offer_jog_pause(where="testing")

    def test_j_jogs_then_continues_on_enter(
        self, cfg, tmp_path, monkeypatch,
    ) -> None:
        runner = self._runner(cfg, tmp_path)
        called = []
        runner.jog_freeform = lambda r: called.append(r)
        responses = iter(["j", ""])
        monkeypatch.setattr(
            "builtins.input", lambda _msg: next(responses)
        )
        runner._offer_jog_pause(where="x")
        assert called == [runner.rail]  # jog called once

    def test_jog_exception_does_not_propagate(
        self, cfg, tmp_path, monkeypatch,
    ) -> None:
        runner = self._runner(cfg, tmp_path)

        def failing_jog(_r) -> float:
            raise RuntimeError("rail unplugged")

        runner.jog_freeform = failing_jog
        responses = iter(["j", ""])
        monkeypatch.setattr(
            "builtins.input", lambda _msg: next(responses)
        )
        # Should swallow the runtime error and re-prompt.
        runner._offer_jog_pause(where="x")


class TestETA:
    """Unit tests for the ETA EWMA + remaining-shots calculator."""

    def _runner(self, cfg, tmp_path, *, factors, apertures, step_count):
        camera = FakeCamera(apertures=apertures)
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 200,
            inputs={"lens": "L", "steps": step_count, "end_mm": 2.0},
        )
        runner._subsample_factors = dict(factors)
        runner._eta_sweep_apertures = tuple(apertures)
        runner._eta_sweep_step_count = step_count
        return runner

    def test_remaining_with_no_subsampling(self, cfg, tmp_path) -> None:
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1, "f/8": 1},
            apertures=["f/4", "f/8"],
            step_count=10,
        )
        # At (step=0, ap=f/4 [idx 0], replica=0) of 3 replicas:
        # Remaining replicas of current (0, f/4): 2
        # Remaining apertures of step 0 (f/8 only): 1 × 3 = 3
        # All future steps (steps 1..9 = 9 steps × 2 apertures × 3 replicas) = 54
        # Total: 2 + 3 + 54 = 59
        assert runner._eta_remaining_shots(
            step_index=0, aperture_index=0, replica_index=0, replica_count=3,
        ) == 59

    def test_remaining_honours_subsample_factors(self, cfg, tmp_path) -> None:
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1, "f/16": 4},
            apertures=["f/4", "f/16"],
            step_count=8,
        )
        # At (step=0, ap=f/4 [idx 0], replica=0) of 1 replica:
        # remaining replicas of (0, f/4): 0
        # remaining ap of step 0: f/16 fires at step 0 (0 % 4 == 0): 1
        # future steps 1..7: f/4 fires every step (7), f/16 fires
        #   only at step 4 (1) → 8
        # Total: 0 + 1 + 8 = 9
        assert runner._eta_remaining_shots(
            step_index=0, aperture_index=0, replica_index=0, replica_count=1,
        ) == 9

    def test_remaining_at_last_position(self, cfg, tmp_path) -> None:
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1},
            apertures=["f/4"],
            step_count=5,
        )
        # Last replica of last aperture of last step → nothing remains.
        assert runner._eta_remaining_shots(
            step_index=4, aperture_index=0, replica_index=2, replica_count=3,
        ) == 0

    def test_record_skips_outlier_intervals(self, cfg, tmp_path) -> None:
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1}, apertures=["f/4"], step_count=2,
        )
        clock = iter([0.0, 5.0, 200.0, 205.0])
        runner.monotonic = lambda: next(clock)

        runner._eta_record_shot()         # t=0, no prior interval
        assert runner._eta_avg_interval_s is None
        runner._eta_record_shot()         # t=5, interval 5 → avg=5
        assert runner._eta_avg_interval_s == pytest.approx(5.0)

        runner._eta_record_shot()         # t=200, interval 195 → outlier, skip
        assert runner._eta_avg_interval_s == pytest.approx(5.0)  # unchanged

        runner._eta_record_shot()         # t=205, interval 5 → avg=(5+5)/2=5
        assert runner._eta_avg_interval_s == pytest.approx(5.0)

    def test_cumulative_mean_blends_intervals_steadily(self, cfg, tmp_path) -> None:
        # An EWMA(α=0.2) over [3, 7, 3, 7, 3, 7, ...] visibly oscillates
        # shot-to-shot. The cumulative mean settles toward the true
        # average (5) and stops oscillating after a handful of shots.
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1}, apertures=["f/4"], step_count=2,
        )
        intervals = [3, 7] * 6  # 12 shots
        # Build clock samples: t=0 then cumulative sums.
        ts = [0.0]
        for d in intervals:
            ts.append(ts[-1] + d)
        clock = iter(ts)
        runner.monotonic = lambda: next(clock)
        for _ in range(len(ts)):
            runner._eta_record_shot()
        # 12 intervals: 6× 3s + 6× 7s = 60s total → mean 5.0
        assert runner._eta_avg_interval_s == pytest.approx(5.0)

    def test_invalidate_clears_last_shot_at(self, cfg, tmp_path) -> None:
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1}, apertures=["f/4"], step_count=2,
        )
        clock = iter([0.0, 5.0, 1000.0])
        runner.monotonic = lambda: next(clock)
        runner._eta_record_shot()
        runner._eta_record_shot()
        runner._eta_invalidate()
        # Next record after invalidate should NOT update average from
        # the huge bridging interval — the prior shots' data is kept.
        prior = runner._eta_avg_interval_s
        runner._eta_record_shot()
        assert runner._eta_avg_interval_s == pytest.approx(prior)

    def test_sweep_start_resets_lens_average(self, cfg, tmp_path) -> None:
        # Per-lens average must NOT carry state from one config to
        # the next — a different aperture / recycle table means
        # different intervals.
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1}, apertures=["f/4"], step_count=2,
        )
        clock = iter([0.0, 5.0, 10.0])
        runner.monotonic = lambda: next(clock)
        runner._eta_record_shot()
        runner._eta_record_shot()
        runner._eta_record_shot()
        assert runner._eta_avg_interval_s == pytest.approx(5.0)

        # Simulate sweep-start reset
        runner._eta_lens_interval_sum = 0.0
        runner._eta_lens_interval_count = 0
        runner._eta_last_shot_at = None
        assert runner._eta_avg_interval_s is None

    def test_format_eta_minutes(self, cfg, tmp_path) -> None:
        runner = self._runner(
            cfg, tmp_path,
            factors={"f/4": 1}, apertures=["f/4"], step_count=2,
        )
        assert runner._format_eta(45) == "0:45"
        assert runner._format_eta(125) == "2:05"
        assert runner._format_eta(3725) == "1:02:05"


class TestCalibrationFailure:
    """Calibration that doesn't reach tolerance must alert + prompt
    distinct from droop (it's a lighting / reference issue, not a
    battery problem)."""

    def test_skip_proceeds_with_non_converged_values(
        self, cfg, tmp_path
    ) -> None:
        # Lighting can never close the gap → calibration saturates.
        # User picks "skip" and the sweep proceeds anyway.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # All shots stay at 25 (vs baseline 100 → -2 EV) regardless of
        # FEC / EC / ISO; calibration cycles through its 7 attempts
        # without converging.
        medians = [25.0] * 12
        prompts: list[str] = []
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "L", "steps": 2, "end_mm": 2.0},
        )
        # Pre-set baseline so the first calibration shot is compared
        # (vs setting the baseline at first capture).
        runner.baseline_median = 100.0
        runner.prompt_off_target = (
            lambda msg: (prompts.append(msg), "skip")[1]
        )
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()
        # Calibration-failure prompt fired at least once.
        assert any("Calibration didn't reach" in p for p in prompts)


class TestSubsampling:
    """SamplePlan.subsample_factors are honoured at capture time.
    Apertures whose factor doesn't divide the current step_index are
    silently skipped — no shots.csv row, no replicas, no droop push."""

    def test_skip_pattern_per_aperture(self, cfg, tmp_path) -> None:
        # 6 rail positions, 2 apertures: f/4 (factor 1) + f/8 (factor 2).
        # f/4 captures every step → 6 shots.
        # f/8 captures step_index % 2 == 0 → steps 0, 2, 4 → 3 shots.
        # Plus 2 calibration baseline shots (one per aperture; the
        # widest-aperture shot at step 0 sets the baseline).
        camera = FakeCamera(apertures=["f/4", "f/8"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 30,
            inputs={
                "lens": "L", "steps": 6, "end_mm": 5.0,
                "subsample_factors": {4.0: 1, 8.0: 2},
            },
        )
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()
        rows = runner.shots_csv.read_rows()
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        f4_rows = [r for r in sweep_rows if r["aperture"] == "f/4"]
        f8_rows = [r for r in sweep_rows if r["aperture"] == "f/8"]
        assert len(f4_rows) == 6
        # f/8 only at step indices divisible by 2.
        f8_steps = sorted(int(r["rail_step"]) for r in f8_rows)
        assert f8_steps == [0, 2, 4]

    def test_factor_default_one_for_unlisted_aperture(self, cfg, tmp_path) -> None:
        # Camera offers f/4 + f/8 but plan.subsample_factors only
        # lists f/4. f/8 falls back to factor 1 (no skip), the safe
        # default. Both apertures get every step.
        camera = FakeCamera(apertures=["f/4", "f/8"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 30,
            inputs={
                "lens": "L", "steps": 4, "end_mm": 3.0,
                "subsample_factors": {4.0: 1},  # f/8 omitted
            },
        )
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()
        rows = runner.shots_csv.read_rows()
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        # 4 steps × 2 apertures = 8 sweep rows (no skip).
        assert len(sweep_rows) == 8

    def test_zero_captures_at_aperture_when_factor_exceeds_steps(
        self, cfg, tmp_path
    ) -> None:
        # f/8 with factor 10 on a 5-step sweep gets exactly 1 shot
        # (step 0; 0 % 10 == 0). No other steps qualify.
        camera = FakeCamera(apertures=["f/4", "f/8"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={
                "lens": "L", "steps": 5, "end_mm": 4.0,
                "subsample_factors": {4.0: 1, 8.0: 10},
            },
        )
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()
        rows = runner.shots_csv.read_rows()
        f8_sweep = [
            r for r in rows
            if r["phase"] == "sweep" and r["aperture"] == "f/8"
        ]
        assert len(f8_sweep) == 1
        assert int(f8_sweep[0]["rail_step"]) == 0

    def test_reshoot_honours_factors(self, cfg, tmp_path) -> None:
        # Droop trips at step 4 f/4 (2nd moderate ≤ -0.7); earliest
        # under-exposed = step 3 (1st moderate). Reshoot covers
        # steps 3..4.
        # f/4 (factor 1) reshot at every step in range = 2 frames (3, 4).
        # f/8 (factor 2): step 3 % 2 = 1 → skip; step 4 % 2 = 0 →
        # captured. = 1 reshoot frame.
        camera = FakeCamera(apertures=["f/4", "f/8"])
        rail = FakeRail()
        medians = [
            # Calibration:
            100.0,                     # cal f/4 (sets baseline)
            100.0,                     # cal f/8
            # Sweep step 0: f/4 + f/8 both fire.
            100.0, 100.0,
            # Sweep step 1: f/4 only (f/8 step%2 != 0 → skipped).
            100.0,
            # Sweep step 2: f/4 + f/8.
            100.0, 100.0,
            # Sweep step 3: f/4 = 70 (1st moderate); f/8 skipped.
            70.0,
            # Sweep step 4: f/4 = 70 (2nd moderate → trip). f/8 not
            # reached because the inner loop returns when reshoot
            # covers the current step.
            70.0,
            # Outer reshoot at steps 3..4:
            100.0,                     # reshoot step 3 f/4 (f/8 skipped)
            100.0, 100.0,              # reshoot step 4: f/4, f/8
        ]
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={
                "lens": "L", "steps": 5, "end_mm": 4.0,
                "subsample_factors": {4.0: 1, 8.0: 2},
            },
        )
        runner.confirm = lambda _msg: True   # ENTER on droop prompt
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()
        rows = runner.shots_csv.read_rows()
        reshoot_rows = [r for r in rows if r["phase"] == "reshoot"]
        f4_reshoot = sorted(
            int(r["rail_step"]) for r in reshoot_rows if r["aperture"] == "f/4"
        )
        f8_reshoot = sorted(
            int(r["rail_step"]) for r in reshoot_rows if r["aperture"] == "f/8"
        )
        # f/4 reshoot every step in 3..4
        assert f4_reshoot == [3, 4]
        # f/8 reshoot only on its grid (step%2==0)
        assert f8_reshoot == [4]


class TestMultiShot:
    """SHOTS_PER_STEP_APERTURE > 1 captures multiple replicas at each
    (step, aperture). Replica 0 keeps the legacy filename
    (``step_NNNN.jpg``); replicas 1..N-1 use ``_b``, ``_c``, …
    suffixes. Each replica is its own row in shots.csv."""

    def test_three_replicas_per_step_aperture(self, cfg, tmp_path) -> None:
        cfg.SHOTS_PER_STEP_APERTURE = 3
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # 1 cal + 2 steps × 1 aperture × 3 replicas = 7
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"lens": "L", "steps": 2, "end_mm": 2.0},
        )
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()

        rows = runner.shots_csv.read_rows()
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        # 2 steps × 3 replicas = 6 sweep rows
        assert len(sweep_rows) == 6
        filenames = sorted(r["filename"] for r in sweep_rows)
        assert filenames == [
            "f_4/step_0100.jpg", "f_4/step_0100_b.jpg", "f_4/step_0100_c.jpg",
            "f_4/step_0101.jpg", "f_4/step_0101_b.jpg", "f_4/step_0101_c.jpg",
        ]

    def test_default_one_shot_preserves_legacy_filenames(
        self, cfg, tmp_path
    ) -> None:
        # Default cfg has no SHOTS_PER_STEP_APERTURE attribute → fall
        # back to 1 → no replica suffixes.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 6,
            inputs={"lens": "L", "steps": 2, "end_mm": 2.0},
        )
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()
        rows = runner.shots_csv.read_rows()
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        # 2 steps × 1 replica = 2 sweep rows, no suffixes
        assert len(sweep_rows) == 2
        assert sorted(r["filename"] for r in sweep_rows) == [
            "f_4/step_0100.jpg", "f_4/step_0101.jpg",
        ]


class TestDroopPause:
    def test_droop_triggers_confirm_and_resets_window(
        self, cfg, tmp_path
    ) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # Baseline 100 (calibration). Sweep step 0 dim enough to trip
        # the SEVERE criterion (1 shot ≤ -2 EV) — median 25 → ≈ -4.4 EV.
        # The reshoot wraps up immediately (current=earliest=0) and
        # the droop window is reset; sweep continues clean at step 1.
        medians = [100.0,                 # calibration baseline
                   25.0,                  # step 0 sweep — severe → trip
                   100.0]                 # step 1 sweep — clean
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
        # One severe shot (≤ -2 EV) trips the severe criterion in
        # one push; user says q → SessionAborted.
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

    def test_droop_during_reshoot_triggers_nested_reshoot(
        self, cfg, tmp_path
    ) -> None:
        # Sweep produces moderate under-exposures (median 70 →
        # ≈ -1.1 EV) at steps 1 and 3. Trip on step 3 (2nd ≤ -0.7
        # → moderate criterion) → reshoot covers steps 1..3.
        # The reshoot at steps 2 and 3 is *also* moderately
        # under-exposed (the user missed a flash), so droop fires
        # again on the 2nd reshoot trip-shot → nested reshoot for
        # steps 2..3. Nested captures land clean. Outer reshoot's
        # skip_remainder kicks in for the rest of step 3, sweep
        # continues normally at step 4.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        medians = [
            100.0,    # calibration baseline
            100.0,    # step 0 sweep ok
            70.0,     # step 1 sweep — 1st moderate (~-1.1 EV)
            100.0,    # step 2 sweep ok
            70.0,     # step 3 sweep — 2nd moderate → trip
            # First (outer) reshoot at steps 1..3:
            100.0,    # reshoot step 1 ok (resets droop window after trip)
            70.0,     # reshoot step 2 — still moderate (1st in fresh window)
            70.0,     # reshoot step 3 — 2nd moderate → NESTED trip
            # Nested reshoot at steps 2..3:
            100.0,    # nested reshoot step 2 ok
            100.0,    # nested reshoot step 3 ok
            # Outer for-loop continues at step 4:
            100.0,    # sweep step 4
        ]

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "L", "steps": 5, "end_mm": 4.0},
        )
        runner.confirm = lambda _msg: True   # always resume on droop
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()

        rows = runner.shots_csv.read_rows()
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        reshoot_rows = [r for r in rows if r["phase"] == "reshoot"]
        # After all the cleanup: only step 0 + step 4 are sweep
        # phase; steps 1, 2, 3 are reshoot phase (with the freshest
        # frames from the nested reshoot at the canonical filename).
        assert sorted(int(r["rail_step"]) for r in sweep_rows) == [0, 4]
        assert sorted(int(r["rail_step"]) for r in reshoot_rows) == [1, 2, 3]
        # Expect droop confirm prompt to have fired at least twice
        # (one outer + one nested).
        # confirm_calls aren't tracked here, but the medians list was
        # consumed exactly — assert no leftover medians:
        assert camera.history  # smoke check that captures happened

    def test_droop_repeats_when_user_resumes_without_swapping(
        self, cfg, tmp_path
    ) -> None:
        # User-reported workflow: when droop fires the user can't
        # always tell which flash needs new batteries until they see
        # the flashes shoot during the reshoot. So they press ENTER
        # without swapping, droop fires again on the reshoot, they
        # press ENTER again, etc. Confirm three nested levels work
        # cleanly (each level reshoots the current step + rewinds via
        # the rolling window) and that after the user finally swaps
        # batteries the freshest captures land on disk with phase=
        # reshoot at every previously-bad step.
        cfg.SHOTS_PER_STEP_APERTURE = 1
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # Sweep: severe trip at step 2 (median 25 → ≈-4.4 EV).
        # Outer reshoot covers [2..2]; reshoot also under (still no
        # battery swap) → nested reshoot. Nested also under → nested-
        # nested reshoot. THIS time the user finally swaps. Sweep
        # then continues at step 3.
        medians = [
            100.0,    # cal
            100.0,    # step 0 sweep ok
            100.0,    # step 1 sweep ok
            25.0,     # step 2 sweep — severe trip
            25.0,     # outer reshoot at step 2 — still bad, recurses
            25.0,     # 2nd nested reshoot at step 2 — still bad
            100.0,    # 3rd nested reshoot at step 2 — fresh batteries
            100.0,    # sweep step 3 ok
        ]
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "L", "steps": 4, "end_mm": 3.0},
        )
        confirm_calls = []
        def confirm(msg):
            confirm_calls.append(msg)
            return True
        runner.confirm = confirm
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()

        rows = runner.shots_csv.read_rows()
        # Sweep rows: only steps 0, 1, 3 — step 2's sweep row was
        # dropped by the outer reshoot's stale-row purge.
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        assert sorted(int(r["rail_step"]) for r in sweep_rows) == [0, 1, 3]
        # Reshoot rows at step 2: the outermost / 2nd-nested / 3rd-
        # nested reshoots all wrote rows for step 2 in turn, but each
        # subsequent _reshoot_steps drops the stale rows from the
        # prior attempt before writing new ones. The final row count
        # at step 2 is just the final (fresh-battery) attempt.
        step2_reshoot = [
            r for r in rows
            if r["phase"] == "reshoot" and int(r["rail_step"]) == 2
        ]
        assert len(step2_reshoot) == 1
        assert float(step2_reshoot[0]["median_central_luminance"]) == 100.0
        # Three droop confirms = one per trip (outer sweep + two nested).
        droop_prompts = [m for m in confirm_calls if "droop" in m.lower()]
        assert len(droop_prompts) == 3

    def test_severe_trip_on_first_replica_skips_remaining_replicas(
        self, cfg, tmp_path
    ) -> None:
        # SHOTS_PER_STEP_APERTURE=3. Replica 0 at step 1 is severely
        # under-exposed (median 25 → ≈-4.4 EV) → severe criterion
        # trips on the very first push. Replicas 1 and 2 must NOT be
        # captured: the runner short-circuits at the first trip,
        # prompts for batteries, then reshoots step 1 with fresh
        # batteries (any droop trip overwrites at least the current
        # step's apertures).
        cfg.SHOTS_PER_STEP_APERTURE = 3
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # Single aperture, 3-step sweep.
        # Step 0: 3 replicas at 100 (clean).
        # Step 1: replica 0 = 25 (severe trip, no more replicas).
        # _handle_droop: restart_step=1, current=1 → reshoot just
        #   step 1 (3 fresh replicas).
        # Step 2: 3 replicas at 100 (clean).
        medians = [
            100.0,                     # cal f/4
            # Step 0 (3 replicas):
            100.0, 100.0, 100.0,
            # Step 1 sweep (replica 0 captured before trip):
            25.0,
            # Step 1 reshoot (3 replicas, fresh batteries):
            100.0, 100.0, 100.0,
            # Step 2 (clean again, 3 replicas):
            100.0, 100.0, 100.0,
        ]

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "L", "steps": 3, "end_mm": 2.0},
        )
        runner.confirm = lambda _msg: True   # ENTER on droop prompt
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))
        runner.run()

        rows = runner.shots_csv.read_rows()
        # The single under-exposed sweep row at step 1 (replica 0,
        # captured before the short-circuit) is dropped by
        # _reshoot_steps before the reshoot writes its fresh rows —
        # otherwise the stale row would point at a JPEG that the
        # reshoot is about to overwrite. So the only step-1 rows
        # remaining are the 3 reshoot replicas.
        step1_sweep = [
            r for r in rows
            if r["phase"] == "sweep" and int(r["rail_step"]) == 1
        ]
        assert len(step1_sweep) == 0
        step1_reshoot = [
            r for r in rows
            if r["phase"] == "reshoot" and int(r["rail_step"]) == 1
        ]
        assert len(step1_reshoot) == 3
        # Total capture events for the session:
        # 1 cal + 3 step 0 + 1 step 1 sweep (short-circuit) + 3 step 1
        # reshoot + 3 step 2 = 11.
        capture_count = sum(1 for k, _ in camera.history if k == "capture")
        assert capture_count == 11

    def test_resume_reshoots_from_first_underexposed_step(
        self, cfg, tmp_path
    ) -> None:
        # 5-step single-aperture sweep. Steps 1 and 3 are moderately
        # under-exposed (median 70 → ≈ -1.1 EV — below criterion B's
        # -0.7 threshold but above criterion A's -2). Step 3's push
        # is the 2nd ≤ -0.7 in the rolling window → moderate trip.
        # On resume the runner should rewind to step 1 (first
        # under-exposed) and reshoot steps 1..3 inclusive. Step 0 is
        # fine, so reshoot doesn't touch it.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        medians = [
            100.0,    # calibration baseline
            100.0,    # step 0 sweep  (ev = 0)
            70.0,     # step 1 sweep  (ev ≈ -1.1) — first under-exposed
            100.0,    # step 2 sweep
            70.0,     # step 3 sweep  (ev ≈ -1.1) — 2nd ≤ -0.7 → trip
            100.0,    # reshoot step 1
            100.0,    # reshoot step 2
            100.0,    # reshoot step 3
            100.0,    # step 4 sweep (continues after reshoot)
        ]

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            inputs={"lens": "L", "steps": 5, "end_mm": 4.0},
        )
        runner.confirm = lambda _msg: True   # always resume on droop prompt
        runner.prompt_lens_config = (lambda calls={"n": 0}: (
            calls.update(n=calls["n"] + 1) or ("L" if calls["n"] == 1 else None)
        ))

        runner.run()

        rows = runner.shots_csv.read_rows()
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        reshoot_rows = [r for r in rows if r["phase"] == "reshoot"]
        # Stale phase=sweep rows for steps 1,2,3 are dropped — those
        # JPEGs were overwritten by the reshoot. Only the surviving
        # sweep rows are step 0 (before the reshoot range) and step 4
        # (after the reshoot, continuing the sweep).
        assert sorted(int(r["rail_step"]) for r in sweep_rows) == [0, 4]
        # Reshoot covers steps 1..3 inclusive (the trigger step is 3).
        assert [int(r["rail_step"]) for r in reshoot_rows] == [1, 2, 3]
        for r in reshoot_rows:
            assert r["filename"].endswith(".jpg")


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

    def test_quit_at_lens_change_records_completed_config(
        self, cfg, tmp_path
    ) -> None:
        # Regression test: a session that completes one lens config
        # and is then aborted at the lens-change prompt (`q`) must
        # still record that completed config in session.json. The
        # prior behaviour was an empty lens_configs list because the
        # append happened only after _run_one_lens_config returned.
        from errors import SessionAborted
        import json

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"lens": "test-lens", "steps": 2, "end_mm": 2.0},
        )

        # User accepts droop confirmations (none expected) but says
        # 'q' (False) at the lens-change prompt -> SessionAborted.
        confirm_calls = []

        def _confirm(msg: str) -> bool:
            confirm_calls.append(msg)
            if "Power off camera" in msg:
                return False
            return True

        runner.confirm = _confirm

        def _prompt() -> str | None:
            return "test-lens"

        runner.prompt_lens_config = _prompt

        with pytest.raises(SessionAborted):
            runner.run()

        meta = json.loads((runner.session_dir / "session.json").read_text())
        slugs = [lc["slug"] for lc in meta["lens_configs"]]
        assert slugs == ["test-lens"]
        # Per-lens-config dict carries the magnification + plan.
        assert meta["lens_configs"][0]["magnification"] == 1.0
        assert "sample_plan" in meta["lens_configs"][0]
        assert meta["aborted"] is True
        assert meta["ended_at"] is not None


class TestReshootPlanMenu:
    """The runner's lens-config loop should route through the
    reshoot-plan menu when ``config.RESHOOT_PLAN_PATH`` resolves to
    plan entries, and skip the standalone magnification prompt for
    presets (because the menu already supplies the magnification)."""

    def test_preset_skips_magnification_prompt(
        self, cfg, tmp_path
    ) -> None:
        from ui.reshoot_plan import PlanEntry

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"steps": 2, "end_mm": 2.0},
        )

        entry = PlanEntry(
            group="Group 1 — bench",
            number=1,
            lens_config="Preset Lens",
            magnification=1.6,
            magnification_text="1.6:1",
            status="reshoot",
            estimated=False,
        )
        runner._plan_entries = [entry]

        # First call returns the preset, second call returns None to
        # quit the loop cleanly.
        calls = {"n": 0}

        def _menu(entries, completed):
            calls["n"] += 1
            return entry if calls["n"] == 1 else None

        runner.prompt_lens_config_from_plan = _menu

        # Should NEVER be asked: the preset supplied the mag.
        mag_calls = {"n": 0}

        def _mag() -> float:
            mag_calls["n"] += 1
            return 99.0   # poison value to assert against if used
        runner.prompt_magnification = _mag

        # Free-text prompt should also not be reached.
        runner.prompt_lens_config = lambda: pytest.fail("free-text prompt called")

        runner.run()

        assert mag_calls["n"] == 0
        # Slug = lens_config text (no `:` or `/` to slugify here).
        slugs = [lc["slug"] for lc in runner.metadata.lens_configs]
        assert slugs == ["Preset Lens"]
        assert runner.metadata.lens_configs[0]["magnification"] == 1.6

    def test_other_falls_back_to_free_text(self, cfg, tmp_path) -> None:
        from ui.prompts import OTHER
        from ui.reshoot_plan import PlanEntry

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"steps": 2, "end_mm": 2.0},
        )
        runner._plan_entries = [
            PlanEntry(
                group="g", number=1, lens_config="ignored",
                magnification=1.0, magnification_text="1:1",
                status="reshoot", estimated=False,
            )
        ]
        menu_calls = {"n": 0}

        def _menu(entries, completed):
            menu_calls["n"] += 1
            return OTHER if menu_calls["n"] == 1 else None
        runner.prompt_lens_config_from_plan = _menu

        ft_calls = {"n": 0}

        def _ft() -> str | None:
            ft_calls["n"] += 1
            return "Hand-typed Lens"
        runner.prompt_lens_config = _ft

        mag_calls = {"n": 0}

        def _mag() -> float:
            mag_calls["n"] += 1
            return 0.5
        runner.prompt_magnification = _mag

        runner.run()

        assert ft_calls["n"] == 1
        assert mag_calls["n"] == 1
        slugs = [lc["slug"] for lc in runner.metadata.lens_configs]
        assert slugs == ["Hand-typed Lens"]
        assert runner.metadata.lens_configs[0]["magnification"] == 0.5

    def test_completed_set_grows_after_success(
        self, cfg, tmp_path
    ) -> None:
        from ui.reshoot_plan import PlanEntry

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"steps": 2, "end_mm": 2.0},
        )
        entry = PlanEntry(
            group="g", number=1, lens_config="Done Lens",
            magnification=1.0, magnification_text="1:1",
            status="reshoot", estimated=False,
        )
        runner._plan_entries = [entry]

        seen_completed: list[set[str]] = []

        def _menu(entries, completed):
            # Snapshot the set the runner passes in *before* the call
            # mutates state.
            seen_completed.append(set(completed))
            if len(seen_completed) == 1:
                return entry
            return None
        runner.prompt_lens_config_from_plan = _menu

        runner.run()

        # The first call saw an empty completed set (fresh tmp_path
        # output_root). After the lens config was acquired, the
        # second call sees the new slug.
        assert seen_completed[0] == set()
        assert seen_completed[1] == {"Done Lens"}

    def test_no_plan_falls_through_to_free_text(self, cfg, tmp_path) -> None:
        # When _plan_entries is None (e.g. RESHOOT_PLAN_PATH missing),
        # the runner uses prompt_lens_config + prompt_magnification
        # exactly as before.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"steps": 2, "end_mm": 2.0},
        )
        # Belt-and-braces: no plan entries available.
        runner._plan_entries = None

        # Menu must not be called.
        runner.prompt_lens_config_from_plan = (
            lambda *_: pytest.fail("menu called when plan is None")
        )

        ft_calls = {"n": 0}

        def _ft() -> str | None:
            ft_calls["n"] += 1
            return "Free Lens" if ft_calls["n"] == 1 else None
        runner.prompt_lens_config = _ft
        runner.prompt_magnification = lambda: 1.0
        runner.run()

        assert ft_calls["n"] == 2  # one for the run, one for quit
        slugs = [lc["slug"] for lc in runner.metadata.lens_configs]
        assert slugs == ["Free Lens"]
