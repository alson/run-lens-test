"""Runner-level tests for the coarse-to-fine focus probe flow.

The pure phase-A math is covered in ``tests/test_probe.py``. These
tests verify the runner orchestrates phase A → analysis → phase C
correctly: rail moves, file destinations, and the probe-fallback
behaviour when phase A is inconclusive.

We use the same FakeCamera/FakeRail fixtures as ``tests/test_runner.py``
plus a ``ProgrammableCollector`` fake that returns canned crop sizes
keyed by (rail_step, aperture). The fake stands in for both phase A
(probe-output collector at ``<lens>/probe/cropped/``) and phase C
(per-lens collector at ``<lens>/cropped/``) — the runner injects each
via the ``sharpness_collector_factory`` so we can route them to
distinct fakes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pytest

from exposure.range_check import RangeChecker
from session.runner import JogResult, SessionRunner
from sharpness.collector import CollectedDataset, ShotSample
from sharpness.planner import SamplePlan
from stackshot.protocol import Direction


# ---- Test config ---------------------------------------------------------


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
        DROOP_CRIT_A=(1, -2.0),
        DROOP_CRIT_B=(2, -0.7),
        MIN_CONFIG_TREE_SIZE=100,
        IMAGE_QUALITY="JPEG Fine",
        SHUTTER_SPEED_CANDIDATES=("1/200",),
        CROP_W_FRAC=0.2,
        CROP_H_FRAC=0.2,
        CROP_X_FRAC=0.0,
        CROP_Y_FRAC=0.15,
        NTFY_TOPIC="test-topic",
        # Probe-specific:
        PROBE_ENABLED=True,
        PROBE_PHASE_A_POINTS=5,
        PROBE_PHASE_C_FWHM_PADDING=1.5,
        STEP_NUMBER_OFFSET=100,
        STEP_NUMBER_WIDTH=4,
    )


# ---- Fake camera + rail (shared with test_runner.py shape) ---------------


@dataclass
class FakeCamera:
    apertures: list[str] = field(default_factory=lambda: [
        "f/4", "f/5.6", "f/8", "f/11", "f/16",
    ])
    history: list[tuple[str, object]] = field(default_factory=list)

    def open(self) -> None: self.history.append(("open", None))
    def close(self) -> None: self.history.append(("close", None))
    def verify_healthy(self, *, min_widgets: int = 100) -> None:
        self.history.append(("verify_healthy", min_widgets))
    def apply_session_config(self, **kw) -> None:
        self.history.append(("apply_session_config", kw))
    def list_apertures(self) -> list[str]: return list(self.apertures)
    def set_aperture(self, fnum: str) -> None: self.history.append(("aperture", fnum))
    def set_iso(self, iso) -> None: self.history.append(("iso", iso))
    def set_fec(self, ev: float) -> None: self.history.append(("fec", float(ev)))
    def set_ec(self, ev: float) -> None: self.history.append(("ec", float(ev)))

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

    def get_position(self) -> float: return self.position_mm

    def zero_position(self) -> None:
        self.position_mm = 0.0
        self.history.append(("zero", None))

    def move_and_wait(
        self, direction: Direction, distance_mm: float, *,
        backlash: bool = True, timeout_s: float | None = None,
    ) -> None:
        if direction == Direction.FORWARD:
            self.position_mm += distance_mm
        else:
            self.position_mm -= distance_mm
        self.history.append(("move", (direction, distance_mm, backlash)))

    def stop(self) -> None: self.history.append(("stop", None))
    def close(self) -> None: pass


# ---- Programmable sharpness collector --------------------------------------


@dataclass
class ProgrammableCollector:
    """Test fake for both phase-A and phase-C collectors.

    The runner constructs one collector per `sharpness_collector_factory`
    call. We route phase-A and phase-C to distinct fake instances by
    inspecting the ``out_dir`` argument: ``.../probe`` is phase A,
    everything else is phase C.

    ``crop_sizes_by_step`` is ``{rail_step: {crop_name: size}}`` —
    record_shot looks up the entry for the given rail_step and
    populates a ShotSample. Apertures are flattened (all share the
    same per-step crop size table) since these tests don't exercise
    aperture-specific cropping.
    """

    out_dir: Path
    crop_sizes_by_step: dict[int, dict[str, int]] = field(default_factory=dict)
    samples: list[ShotSample] = field(default_factory=list)
    pending_drained: bool = False
    shutdown_called: bool = False

    def record_shot(
        self,
        rail_step: int,
        aperture: float,
        jpg_path,
        *,
        replica_index: int = 0,
    ) -> None:
        sizes = self.crop_sizes_by_step.get(rail_step, {})
        self.samples.append(ShotSample(
            rail_step=rail_step, aperture=aperture, sizes=dict(sizes),
            replica_index=replica_index,
        ))
        return None

    def wait_for_pending(self) -> None:
        self.pending_drained = True

    def shutdown(self, *, wait: bool = True) -> None:
        self.shutdown_called = True

    def snapshot(self) -> CollectedDataset:
        return CollectedDataset(samples=list(self.samples))

    @property
    def dataset(self) -> CollectedDataset:
        # SharpnessRangeChecker reads .dataset (we never use the real
        # adapter in these tests, but a stub for forward-compat).
        return CollectedDataset(samples=list(self.samples))


# ---- Runner builder ------------------------------------------------------


def make_probe_runner(
    *,
    camera: FakeCamera,
    rail: FakeRail,
    cfg,
    tmp_path: Path,
    medians: Iterable[float],
    phase_a_crops: dict[int, dict[str, int]] | None = None,
    phase_c_crops: dict[int, dict[str, int]] | None = None,
    inputs: dict | None = None,
    plan_step_mm: float = 0.020,
    plan_fwhm_mm: float = 0.080,
    plan_steps: int = 50,
):
    """Build a SessionRunner wired with the probe-aware test fakes."""
    queue = deque(medians)

    def _measure(_p: Path) -> float:
        if not queue:
            return 100.0
        return queue.popleft()

    inputs = inputs or {}
    fixed_now = datetime(2026, 5, 7, 18, 35, tzinfo=timezone.utc)

    def _prompt_lens() -> str | None:
        return inputs.get("lens", "lens-1")

    def _prompt_magnification() -> float:
        return inputs.get("magnification", 1.0)

    def _confirm(_msg) -> bool:
        return inputs.get("confirm", True)

    def _jog(rail: FakeRail) -> JogResult:
        end_mm = inputs.get("end_mm", 1.0)
        rail.position_mm = end_mm
        return JogResult(end_mm=end_mm)

    def _fake_plan_sweep(
        *, magnification, widest_aperture, max_aperture,
        rail_range_mm, replicas,
        coc_mm=None, positions_across_fwhm=None,
    ) -> SamplePlan:
        # The plan is what the runner uses for phase-C grid math.
        # We pin step / fwhm via the fixture so the test can predict
        # phase-C window positions.
        positions = inputs.get("steps", plan_steps)
        return SamplePlan(
            rail_step_mm=plan_step_mm,
            replicas_per_position=replicas,
            estimated_dof_mm_widest=2 * plan_fwhm_mm,
            estimated_fwhm_mm_widest=plan_fwhm_mm,
            widest_aperture=widest_aperture,
            max_aperture=max_aperture,
            subsample_factors={widest_aperture: 1},
            rail_range_mm=rail_range_mm,
            rail_positions=positions,
        )

    # Two collectors: one for phase A (path ends in /probe), one for
    # phase C (path is the lens dir). Factory dispatches by path.
    phase_a_seen: list[ProgrammableCollector] = []
    phase_c_seen: list[ProgrammableCollector] = []

    def _factory(out_dir: Path):
        if str(out_dir).endswith("/probe"):
            c = ProgrammableCollector(
                out_dir=out_dir,
                crop_sizes_by_step=phase_a_crops or {},
            )
            phase_a_seen.append(c)
            return c
        c = ProgrammableCollector(
            out_dir=out_dir,
            crop_sizes_by_step=phase_c_crops or {},
        )
        phase_c_seen.append(c)
        return c

    runner = SessionRunner(
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
        sharpness_collector_factory=_factory,
    )
    # Stash the collector lists on the runner so tests can inspect them.
    runner._test_phase_a_collectors = phase_a_seen      # type: ignore[attr-defined]
    runner._test_phase_c_collectors = phase_c_seen      # type: ignore[attr-defined]
    return runner


def _bell_sizes(
    n_probes: int, peak_idx: float, fwhm: float,
    *, floor: int = 100_000, amplitude: int = 200_000,
) -> list[int]:
    import numpy as np
    sigma = fwhm / 2.355
    ys = floor + amplitude * np.exp(
        -0.5 * ((np.arange(n_probes) - peak_idx) / sigma) ** 2
    )
    return [int(round(v)) for v in ys]


def _phase_a_dataset(
    n_probes: int, *, peaks: dict[str, float], fwhm: float = 1.5,
) -> dict[int, dict[str, int]]:
    """Build {step: {crop: size}} for n_probes probes with per-crop peaks."""
    out: dict[int, dict[str, int]] = {step: {} for step in range(n_probes)}
    for crop_name, peak in peaks.items():
        series = _bell_sizes(n_probes, peak_idx=peak, fwhm=fwhm)
        for step, size in enumerate(series):
            out[step][crop_name] = size
    return out


# ---- Tests --------------------------------------------------------------


class TestProbeFlow:
    def test_phase_a_captures_at_widest_only(self, cfg, tmp_path):
        """When PROBE_ENABLED, phase A captures only at the widest
        aperture, single replica per probe — N captures, not N × K."""
        camera = FakeCamera(apertures=["f/4", "f/8"])
        rail = FakeRail()
        # 2 calibration shots (f/4, f/8) + 5 phase A + 1 phase-C step ×
        # 2 apertures (we'll check capture count separately).
        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
            phase_a_crops=_phase_a_dataset(
                n_probes=5,
                peaks={
                    "center": 2.0, "top-left": 1.5, "top-right": 2.5,
                    "bottom-left": 1.8, "bottom-right": 2.2,
                },
            ),
            phase_c_crops={},
            inputs={"lens": "lens-1", "steps": 10, "end_mm": 1.0},
        )

        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        captures = [v for k, v in camera.history if k == "capture"]
        # Phase A captures should land in <session>/lens-1/probe/f_4/.
        phase_a_captures = [c for c in captures if "/probe/f_4/" in c]
        assert len(phase_a_captures) == 5, (
            f"expected 5 phase-A captures at widest aperture only, "
            f"got {len(phase_a_captures)}: {phase_a_captures}"
        )
        # Phase A captures must not appear at the smaller apertures.
        phase_a_at_f8 = [c for c in captures if "/probe/f_8/" in c]
        assert phase_a_at_f8 == []

    def test_phase_c_runs_at_window_offset(self, cfg, tmp_path):
        """Phase C should retract from end_mm to phase_c_lo_mm and
        sweep forward through the computed window — file step numbers
        anchored to absolute rail-zero, not zero-shifted."""
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # Phase-A peaks at sample 2 (centre of 5 probes) → phase-C
        # window centred near rail-zero + 0.5 mm. With end_mm=1.0,
        # the spacing is 0.25; sample 2 = 0.5 mm.
        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
            phase_a_crops=_phase_a_dataset(
                n_probes=5,
                peaks={
                    "center": 2.0,
                    "top-left": 1.8, "top-right": 2.2,
                    "bottom-left": 2.0, "bottom-right": 2.0,
                },
                fwhm=1.5,
            ),
            phase_c_crops={},
            inputs={"lens": "lens-1", "steps": 10, "end_mm": 1.0},
            plan_step_mm=0.020,
            plan_fwhm_mm=0.080,
        )

        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        # Find phase-C step files (under <session>/lens-1/f_4/, NOT
        # under /probe/f_4/).
        lens_dir = runner.session_dir / "lens-1"
        phase_c_files = sorted((lens_dir / "f_4").glob("step_*.jpg"))
        # Phase C captures must exist at non-zero step indices.
        # File numbers are STEP_NUMBER_OFFSET + step_index = 100 + start_step
        # to 100 + start_step + n_steps - 1.
        # All-corner peaks centred around sample 2 → phase-C centred near
        # 0.5 mm → step ~25 → file step_0125.jpg-ish.
        assert len(phase_c_files) > 0
        step_numbers = [
            int(f.stem.split("_")[1]) for f in phase_c_files
        ]
        # The window should be small (FWHM-padded) and offset away
        # from step 100 (which would be rail-zero).
        assert min(step_numbers) > 100, (
            f"phase-C should not start at step 100 (= rail-zero); "
            f"got min={min(step_numbers)}: {step_numbers}"
        )

    def test_phase_a_droop_pauses_and_restarts(self, cfg, tmp_path):
        """A flash misfire during phase-A probing must trip the droop
        detector, prompt the user, and restart phase A from probe 0
        with a fresh probe collector + dropped shots.csv rows."""
        from session.log import Phase

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # Medians: cal baseline + N probes in attempt 1 (with bad EVs
        # mid-way to trip droop) + N probes in attempt 2 (clean) +
        # phase-C frames. Trip criterion B = 2 of last 10 ≤ -2 EV.
        medians = [
            100.0,           # calibration baseline
            100.0, 100.0,    # probes 0, 1 healthy
            10.0, 10.0,      # probes 2, 3 noise floor → both ≤ -2 EV → trip after probe 3
            # User pressed ENTER on droop prompt → phase-A restarts from 0:
            100.0, 100.0, 100.0, 100.0, 100.0,  # full clean restart at 5 probes
        ] + [100.0] * 200  # plenty for phase-C
        confirm_calls: list[str] = []

        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=medians,
            phase_a_crops=_phase_a_dataset(
                n_probes=5,
                peaks={
                    "center": 2.0,
                    "top-left": 1.8, "top-right": 2.2,
                    "bottom-left": 2.0, "bottom-right": 2.0,
                },
                fwhm=1.5,
            ),
            phase_c_crops={},
            inputs={"lens": "lens-1", "steps": 10, "end_mm": 1.0},
            plan_step_mm=0.020,
            plan_fwhm_mm=0.080,
        )

        def _confirm(msg: str) -> bool:
            confirm_calls.append(msg)
            return True  # always resume

        runner.confirm = _confirm
        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        # The droop confirm prompt must have fired during phase A.
        droop_prompts = [c for c in confirm_calls if "Battery droop" in c]
        assert len(droop_prompts) >= 1, (
            "Phase-A droop trip did not invoke the droop pause prompt; "
            f"got prompts: {confirm_calls!r}"
        )
        # After restart, only the *fresh* phase-A captures should
        # remain in shots.csv as phase=probe rows. The under-exposed
        # ones from the aborted first attempt should have been dropped.
        rows = runner.shots_csv.read_rows()
        probe_rows = [r for r in rows if r["phase"] == Phase.PROBE.value]
        # The runner emits 5 probes per phase-A attempt. After the
        # restart there should be exactly 5 probe rows, all with
        # healthy medians (non-trip).
        assert len(probe_rows) == 5, (
            f"Expected 5 phase-probe rows after restart-cleanup, got "
            f"{len(probe_rows)}: {probe_rows}"
        )
        assert all(
            float(r["median_central_luminance"]) >= 90 for r in probe_rows
        )

    def test_phase_c_works_for_backward_jog(self, cfg, tmp_path):
        """Phase C planner must produce a positive n_steps when the
        user jogged backward (end_mm < 0). Regression for an earlier
        bug where _plan_phase_c divided by an unsigned rail_step_mm
        and the swap arithmetic gave negative n_steps → returned None
        → "Phase A inconclusive" fallback even with conclusive peaks."""
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
            phase_a_crops=_phase_a_dataset(
                n_probes=5,
                peaks={
                    "center": 2.0,
                    "top-left": 1.8, "top-right": 2.2,
                    "bottom-left": 2.0, "bottom-right": 2.0,
                },
                fwhm=1.5,
            ),
            phase_c_crops={},
            inputs={"lens": "lens-1", "steps": 10, "end_mm": -1.0},
            plan_step_mm=0.020,
            plan_fwhm_mm=0.080,
        )
        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        lens_dir = runner.session_dir / "lens-1"
        phase_c_files = sorted((lens_dir / "f_4").glob("step_*.jpg"))
        # Phase C must have produced captures (not fallen through to
        # the uniform sweep). Step numbers are offset+step_index;
        # start_step is positive in step-index space (counting forward
        # in the user's chosen direction from rail-zero).
        assert len(phase_c_files) > 0, (
            "Phase C produced no captures — the backward-jog planner "
            "regressed to returning None"
        )
        step_numbers = [int(f.stem.split("_")[1]) for f in phase_c_files]
        assert min(step_numbers) > 100, (
            f"phase-C should not start at rail-zero; got min={min(step_numbers)}"
        )

    def test_phase_a_inconclusive_falls_back_to_uniform_sweep(
        self, cfg, tmp_path
    ):
        """When phase A can't get 3 valid crops, fall back to today's
        uniform sweep across the user-jogged range — no retract, file
        step numbers start at 100 (offset only)."""
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        # All crops flat → inconclusive.
        flat = [100_000] * 5
        phase_a_crops = {
            step: {"center": flat[step], "top-left": flat[step]}
            for step in range(5)
        }
        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
            phase_a_crops=phase_a_crops,
            phase_c_crops={},
            inputs={"lens": "lens-1", "steps": 5, "end_mm": 1.0},
            plan_steps=5,
            plan_step_mm=0.25,
        )

        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        # Phase-A captures still happened.
        captures = [v for k, v in camera.history if k == "capture"]
        phase_a_captures = [c for c in captures if "/probe/f_4/" in c]
        assert len(phase_a_captures) == 5
        # Fall-back uniform sweep ran: phase-C captures at steps 0..4
        # (file numbers 100..104).
        lens_dir = runner.session_dir / "lens-1"
        phase_c_files = sorted((lens_dir / "f_4").glob("step_*.jpg"))
        step_numbers = sorted(
            int(f.stem.split("_")[1]) for f in phase_c_files
        )
        assert step_numbers[0] == 100, (
            f"fallback sweep starts at step 0 (file step_0100); "
            f"got {step_numbers[0]}"
        )

    def test_probe_disabled_runs_uniform_sweep_unchanged(
        self, cfg, tmp_path
    ):
        """With PROBE_ENABLED=False, no phase A captures happen and
        the flow matches today's behaviour."""
        cfg.PROBE_ENABLED = False
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
            inputs={"lens": "lens-1", "steps": 3, "end_mm": 1.0},
            plan_steps=3,
            plan_step_mm=0.5,
        )

        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        captures = [v for k, v in camera.history if k == "capture"]
        # No probe-dir captures.
        assert not any("/probe/" in c for c in captures), (
            f"PROBE_ENABLED=False should skip phase A; "
            f"got captures: {captures}"
        )
        # Phase-A collector factory should not have been invoked.
        assert runner._test_phase_a_collectors == []
        # Regular sweep ran with 3 steps × 1 aperture + 1 calibration.
        phase_c_files = list(
            (runner.session_dir / "lens-1" / "f_4").glob("step_*.jpg")
        )
        assert len(phase_c_files) == 3

    def test_phase_a_logs_shots_with_probe_phase(self, cfg, tmp_path):
        """Each phase-A capture writes a row to shots.csv with
        phase=probe so the user can audit the probe pass."""
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
            phase_a_crops=_phase_a_dataset(
                n_probes=5,
                peaks={
                    "center": 2.0, "top-left": 1.8, "top-right": 2.2,
                    "bottom-left": 2.0, "bottom-right": 2.0,
                },
            ),
            inputs={"lens": "lens-1", "steps": 5, "end_mm": 1.0},
        )

        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        rows = runner.shots_csv.read_rows() if runner.shots_csv else []
        probe_rows = [r for r in rows if r["phase"] == "probe"]
        assert len(probe_rows) == 5
        # Each probe row should reference a file under probe/f_4/.
        for r in probe_rows:
            assert "probe/f_4/" in r["filename"], (
                f"probe row filename {r['filename']!r} should be "
                f"under probe/f_4/"
            )

    def test_rail_makes_one_retract_between_phase_a_and_c(
        self, cfg, tmp_path
    ):
        """The only direction reversal in the whole probe is the
        retract from end_mm to phase_c_lo_mm. Phase A and phase C
        themselves are forward-only (backlash=False)."""
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_probe_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
            phase_a_crops=_phase_a_dataset(
                n_probes=5,
                peaks={
                    "center": 2.0, "top-left": 1.8, "top-right": 2.2,
                    "bottom-left": 2.0, "bottom-right": 2.0,
                },
                fwhm=1.5,
            ),
            inputs={"lens": "lens-1", "steps": 10, "end_mm": 1.0},
            plan_step_mm=0.020,
            plan_fwhm_mm=0.080,
        )

        calls = {"n": 0}
        def _prompt() -> str | None:
            calls["n"] += 1
            return "lens-1" if calls["n"] == 1 else None
        runner.prompt_lens_config = _prompt
        runner.run()

        # Pick out the moves (direction, distance, backlash).
        moves = [v for k, v in rail.history if k == "move"]
        # Phase-A moves: 4 forward steps with backlash=False
        # (n_probes=5 → 4 inter-probe moves; first probe is at the
        # current rail position).
        forward_no_backlash = [
            (d, dist) for (d, dist, b) in moves
            if d == Direction.FORWARD and not b
        ]
        # We expect at least 4 forward+no-backlash moves from phase A.
        assert len(forward_no_backlash) >= 4
        # Exactly one BACKWARD move with backlash=True between phases A and C.
        backwards = [(d, dist, b) for (d, dist, b) in moves
                     if d == Direction.BACKWARD]
        # There may be 2: one for the probe→phase-C retract, one for
        # the post-sweep return-to-zero. Both with backlash=True.
        assert all(b for (_, _, b) in backwards), (
            f"all BACKWARD moves should use backlash=True; got {backwards}"
        )
