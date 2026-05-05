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
        # returns immediately. Each shot consumes 1 monotonic() call (to
        # stamp _last_capture_started_at) plus 1 for _wait_for_recycle from
        # the second shot onward. With recycle=3.0s for f/4 in cfg and a
        # 100-second gap, no sleep should happen at all.
        sleeps = self._drive_one_step_one_aperture(
            cfg, tmp_path,
            # _wait_for_recycle reads monotonic on the 2nd sweep shot;
            # _last_capture_started_at was stamped at t=0 on shot 1.
            # With t=100 on the wait, elapsed=100 >> recycle=3.
            monotonic_seq=[
                0.0,    # shot 1 stamp
                100.0,  # shot 2 wait read
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
                0.0,   # shot 1 stamp
                1.0,   # shot 2 wait read -> remaining = 3 - 1 = 2
                1.0,   # shot 2 stamp
            ],
        )
        assert sleeps == [pytest.approx(2.0)]

    def test_no_wait_when_elapsed_exceeds_recycle(self, cfg, tmp_path) -> None:
        # 5.0s elapsed since prior shutter -> 0 remaining; no sleep.
        sleeps = self._drive_one_step_one_aperture(
            cfg, tmp_path,
            monotonic_seq=[
                0.0,
                5.0,
                5.0,
            ],
        )
        assert sleeps == []


# ---- Droop pause + resume --------------------------------------------------


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

    def test_resume_reshoots_from_first_underexposed_step(
        self, cfg, tmp_path
    ) -> None:
        # 5-step single-aperture sweep. Step 1 is the first
        # under-exposed shot (median 30 → ≈ -3.8 EV). Step 3 is the
        # second sample ≤ -2 EV which trips criterion B. On resume
        # the runner should rewind to step 1 and reshoot steps
        # 1..3 inclusive (the trigger step). Step 0 is fine, so
        # reshoot doesn't touch it.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        medians = [
            100.0,    # calibration baseline
            100.0,    # step 0 sweep  (ev = 0)
            30.0,     # step 1 sweep  (ev ≈ -3.8) — first under-exposed
            100.0,    # step 2 sweep
            25.0,     # step 3 sweep  (ev ≈ -4.4) — 2nd ≤ -2 EV → trip
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
        assert len(sweep_rows) == 5
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
        assert meta["lens_configs"] == ["test-lens"]
        assert meta["aborted"] is True
        assert meta["ended_at"] is not None
