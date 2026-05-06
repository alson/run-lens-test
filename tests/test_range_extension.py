"""Tests for the range-check + auto-extension flow in session.runner.

Uses a scripted FakeRangeChecker (separate from the production stub)
to drive the runner through:
  * stub-style ok=True: no extra captures, range_check.json has
    extension_applied=False
  * needs_extension=True, peak_side='far' -> grid-aligned probe,
    optional extension follow-up; phases marked PROBE / EXTENSION
  * peak_side='near' -> probe + extension at negative step indices
    (filenames stay positive thanks to STEP_NUMBER_OFFSET)
  * peak_side='both' -> bilateral probes + bilateral extensions
  * peak_in_range_after_probe=False -> user prompted; no extension

The grid-aligned design means probes and extension shots share the
same per-aperture directory and the same step_NNNN.jpg filename
convention; only shots.csv's ``phase`` column distinguishes them.
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


# ---- Far-side extension ---------------------------------------------------


class TestFarExtension:
    """end_mm=20, steps=3 -> step_size_mm=10. Default 0.5 fraction
    -> extension_mm=10 -> extension_steps=1. Default RANGE_CHECK_PROBE_STEPS=2
    is clamped to 1 (one grid slot in the extension). So we get exactly
    1 probe shot at step_index=3 and zero extension follow-up."""

    def test_one_grid_probe_when_extension_is_one_step(
        self, cfg, tmp_path
    ) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 20.0},
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

        # 1 calibration + 3 sweep + 1 probe + 0 extension = 5 captures.
        capture_count = sum(
            1 for k, _ in camera.history if k == "capture"
        )
        assert capture_count == 5

        # Probes share the per-aperture directory with the sweep —
        # they are simply step indices past the end of the original
        # sweep, with the source filename derived from the step
        # index plus STEP_NUMBER_OFFSET.
        sweep_dir = runner.session_dir / "L" / "f_4"
        sweep_files = sorted(sweep_dir.glob("step_*.jpg"))
        assert [p.name for p in sweep_files] == [
            "step_0100.jpg",  # main sweep step 0
            "step_0101.jpg",  # main sweep step 1
            "step_0102.jpg",  # main sweep step 2 (= end_mm)
            "step_0103.jpg",  # probe at step_index=3 (= end_mm + step_size)
        ]
        # No /probe/ subdir: phase distinction lives in the CSV.
        assert not (runner.session_dir / "L" / "probe").exists()

        rows = runner.shots_csv.read_rows()
        phases = [r["phase"] for r in rows]
        assert phases.count(Phase.PROBE.value) == 1
        assert phases.count(Phase.EXTENSION.value) == 0

        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["ok"] is False
        assert rc["peak_side"] == "far"
        assert rc["extension_applied"] is True
        # Per-side counters: 0 extension follow-up shots on far side
        # (the lone slot was already covered by the probe).
        assert rc["extension_step_count"] == {"far": 0, "near": 0}

    def test_probe_then_extension_follow_up(self, cfg, tmp_path) -> None:
        """Bigger range so the extension fits the probe count plus an
        extra step. end_mm=60, steps=7 -> step_size=10, ext_mm=30,
        ext_steps=3, probe=2, ext follow-up=1."""
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 30,
            inputs={"steps": 7, "end_mm": 60.0},
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True, notes="far"),
            RangeResult(ok=True, peak_side="far",
                        needs_extension=False,
                        peak_in_range_after_probe=True,
                        notes="bracket"),
        ])
        _run_one_lens_config(runner)

        # 1 cal + 7 sweep + 2 probe + 1 extension = 11 captures.
        capture_count = sum(
            1 for k, _ in camera.history if k == "capture"
        )
        assert capture_count == 11

        sweep_dir = runner.session_dir / "L" / "f_4"
        sweep_files = sorted(sweep_dir.glob("step_*.jpg"))
        # 7 sweep + 3 extension grid positions.
        assert len(sweep_files) == 10
        # Filenames: 0100..0109 (steps 0..9 with offset 100).
        assert sweep_files[-1].name == "step_0109.jpg"

        rows = runner.shots_csv.read_rows()
        phases = [r["phase"] for r in rows]
        assert phases.count(Phase.PROBE.value) == 2
        assert phases.count(Phase.EXTENSION.value) == 1

        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["extension_step_count"] == {"far": 1, "near": 0}


# ---- Near-side extension --------------------------------------------------


class TestNearExtension:
    def test_negative_step_indices_have_positive_filenames(
        self, cfg, tmp_path
    ) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 20.0},
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="near",
                        needs_extension=True, notes="near"),
            RangeResult(ok=True, peak_side="near",
                        needs_extension=False,
                        peak_in_range_after_probe=True,
                        notes="probe ok"),
        ])
        _run_one_lens_config(runner)

        sweep_dir = runner.session_dir / "L" / "f_4"
        sweep_files = sorted(sweep_dir.glob("step_*.jpg"))
        # Probe at step_index = -1 -> filename step_0099.jpg, sorts
        # *before* the main sweep's step_0100..step_0102 in any
        # standard listing tool.
        assert [p.name for p in sweep_files] == [
            "step_0099.jpg",
            "step_0100.jpg",
            "step_0101.jpg",
            "step_0102.jpg",
        ]
        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["peak_side"] == "near"
        assert rc["extension_applied"] is True


# ---- Both-side extension --------------------------------------------------


class TestBothExtension:
    def test_bilateral_probes_and_filenames_sort_by_rail_position(
        self, cfg, tmp_path
    ) -> None:
        """peak_side='both' shoots up to (probe_count // 2 + 1) probes
        on each side, capped by extension_steps."""
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 20.0},
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="both",
                        needs_extension=True,
                        notes="focus shift across apertures"),
            RangeResult(ok=True, peak_side="both",
                        needs_extension=False,
                        peak_in_range_after_probe=True,
                        notes="bracket on both sides"),
        ])
        _run_one_lens_config(runner)

        # 1 cal + 3 sweep + 1 far probe + 1 near probe = 6.
        capture_count = sum(
            1 for k, _ in camera.history if k == "capture"
        )
        assert capture_count == 6

        sweep_dir = runner.session_dir / "L" / "f_4"
        sweep_files = sorted(sweep_dir.glob("step_*.jpg"))
        assert [p.name for p in sweep_files] == [
            "step_0099.jpg",  # near probe (step_index = -1)
            "step_0100.jpg",  # main sweep step 0
            "step_0101.jpg",  # main sweep step 1
            "step_0102.jpg",  # main sweep step 2 (= end_mm)
            "step_0103.jpg",  # far probe (step_index = +3)
        ]

        rows = runner.shots_csv.read_rows()
        phases = [r["phase"] for r in rows]
        assert phases.count(Phase.PROBE.value) == 2

        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["peak_side"] == "both"
        assert rc["probe_step_count"] == {"far": 1, "near": 1}


# ---- Probe says peak still outside ----------------------------------------


class TestProbeFails:
    def test_no_extension_user_prompted(self, cfg, tmp_path) -> None:
        # User picks 'skip' at the range-inadequate prompt → keep
        # the data we have and continue to the lens-change prompt.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        prompts: list[str] = []

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 20.0},
        )
        runner.prompt_range_inadequate = (
            lambda msg: (prompts.append(msg), "skip")[1]
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True, notes="peak beyond end"),
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True,
                        peak_in_range_after_probe=False,
                        notes="probe still doesn't bracket"),
        ])
        _run_one_lens_config(runner)

        sweep_dir = runner.session_dir / "L" / "f_4"
        # 3 sweep + 1 probe = 4 files; no extension follow-up.
        assert len(list(sweep_dir.glob("step_*.jpg"))) == 4

        # Range-still-inadequate prompt was shown.
        assert any("Range still inadequate" in p for p in prompts)
        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["extension_applied"] is False
        assert "did not bracket peak" in rc["notes"]
        assert rc["extend_rounds"] == 0

    def test_extend_more_loops_until_bracketed(self, cfg, tmp_path) -> None:
        # User says 'extend' twice. Second extend round brings the
        # peak into range. Then the runner fills any remaining grid
        # slots inside the original extension (none here — extension
        # is one step and the probe used it) and writes a successful
        # range_check.json.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        prompts: list[str] = []
        choices = iter(["extend", "extend"])

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 30,
            inputs={"steps": 3, "end_mm": 20.0},
        )
        runner.prompt_range_inadequate = (
            lambda msg: (prompts.append(msg), next(choices))[1]
        )
        runner.range_checker = ScriptedRangeChecker([
            # 1) first call after main sweep
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True,
                        notes="peak beyond end"),
            # 2) after probe — still no bracket
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True,
                        peak_in_range_after_probe=False,
                        notes="still beyond end"),
            # 3) after first extend-more round — still no
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True,
                        peak_in_range_after_probe=False,
                        notes="still beyond end"),
            # 4) after second extend-more round — bracket!
            RangeResult(ok=True, peak_side="far",
                        needs_extension=False,
                        peak_in_range_after_probe=True,
                        notes="bracket"),
        ])
        _run_one_lens_config(runner)

        # 1 cal + 3 sweep + 1 probe + 2 extend-more rounds × 1 step = 7
        capture_count = sum(1 for k, _ in camera.history if k == "capture")
        assert capture_count == 7
        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["extension_applied"] is True
        assert rc["extend_rounds"] == 2
        # Two prompts: one per extend choice (no third because the
        # 4th evaluate returned bracket).
        assert len(prompts) == 2

    def test_abort_at_inadequate_prompt(self, cfg, tmp_path) -> None:
        # User picks 'abort' at the prompt → SessionAborted bubbles
        # out and the session.json reflects it.
        from errors import SessionAborted

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 20.0},
        )
        runner.prompt_range_inadequate = lambda msg: "abort"
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True, notes="peak beyond"),
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True,
                        peak_in_range_after_probe=False,
                        notes="still beyond"),
        ])
        with pytest.raises(SessionAborted):
            _run_one_lens_config(runner)
        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert "user aborted" in rc["notes"]


class TestOffTarget:
    """All apertures' curves are flat — the runner must alert and
    prompt for a re-jog instead of treating the result as OK."""

    def test_skip_records_off_target_marker(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        prompts: list[str] = []
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"steps": 3, "end_mm": 20.0},
        )
        runner.prompt_off_target = (
            lambda msg: (prompts.append(msg), "skip")[1]
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="centered",
                        needs_extension=False,
                        all_off_target=True,
                        notes="all flat"),
        ])
        _run_one_lens_config(runner)

        assert any("off-target" in p.lower() for p in prompts)
        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc["all_off_target"] is True
        assert rc["rejog_requested"] is False
        assert "user accepted" in rc["notes"]

    def test_rejog_re_runs_sweep_with_new_range(self, cfg, tmp_path) -> None:
        # First sweep: 3 steps over 20mm. Off-target prompt fires;
        # user picks 'rejog'. Test jog injects a new 30mm range with
        # 4 steps. Second sweep is OK -> normal flow proceeds.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        prompts: list[str] = []

        # The custom jog: first call returns 20mm; on the re-jog it
        # returns 30mm. The make_runner's default jog isn't reused
        # here because we need it to change between calls.
        from session.runner import JogResult
        jog_calls = {"n": 0}

        def _jog(rail: FakeRail) -> JogResult:
            jog_calls["n"] += 1
            end = 20.0 if jog_calls["n"] == 1 else 30.0
            rail.position_mm = end
            return JogResult(end_mm=end)

        # Step counts: first plan call returns 3 positions, second
        # returns 4 (re-jog with new range).
        plan_calls = {"n": 0}

        from sharpness.planner import SamplePlan as _SamplePlan

        def _plan(*, rail_range_mm, replicas, widest_aperture, max_aperture, **_kw):
            plan_calls["n"] += 1
            steps = 3 if plan_calls["n"] == 1 else 4
            return _SamplePlan(
                rail_step_mm=rail_range_mm / max(1, steps - 1),
                replicas_per_position=replicas,
                estimated_dof_mm_widest=1.0,
                estimated_fwhm_mm_widest=0.5,
                widest_aperture=widest_aperture,
                max_aperture=max_aperture,
                subsample_factors={widest_aperture: 1},
                rail_range_mm=rail_range_mm,
                rail_positions=steps,
            )

        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 30,
            inputs={"lens": "L"},
        )
        runner.jog = _jog
        runner.plan_sweep = _plan
        runner.prompt_off_target = (
            lambda msg: (prompts.append(msg), "rejog")[1]
        )
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="centered",
                        needs_extension=False,
                        all_off_target=True,
                        notes="all flat (first sweep)"),
            RangeResult(ok=True, peak_side="centered",
                        needs_extension=False,
                        notes="ok (after re-jog)"),
        ])

        def _prompt_lens(calls={"n": 0}) -> str | None:
            calls["n"] += 1
            return "L" if calls["n"] == 1 else None

        runner.prompt_lens_config = _prompt_lens
        runner.run()

        # Two sweeps' worth of captures: 1 cal + 3 sweep + 1 cal-skip + 4 sweep
        # Calibration runs once (preserved across re-jog), so total =
        # 1 cal + 3 first sweep + 4 re-sweep = 8.
        capture_count = sum(1 for k, _ in camera.history if k == "capture")
        assert capture_count == 8
        # The off-target prompt fired once.
        assert len(prompts) == 1
        # range_check.json reflects the *second* run's state (final
        # ok=True, no off-target marker).
        rc = json.loads(
            (runner.session_dir / "L" / "range_check.json").read_text()
        )
        assert rc.get("all_off_target") is not True
        assert rc["ok"] is True

    def test_abort_raises_session_aborted(self, cfg, tmp_path) -> None:
        from errors import SessionAborted
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
            inputs={"steps": 3, "end_mm": 20.0},
        )
        runner.prompt_off_target = lambda msg: "abort"
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="centered",
                        needs_extension=False,
                        all_off_target=True,
                        notes="all flat"),
        ])
        with pytest.raises(SessionAborted):
            _run_one_lens_config(runner)


# ---- Subsampling on the probe / extension path ---------------------------


class TestProbeWithSubsampling:
    """Probes / extensions also honour per-aperture subsample factors.
    A probe at step_index=N is shot only by apertures whose factor
    divides N. Smaller-aperture series stay sparse end-to-end."""

    def test_smaller_aperture_skips_probes_off_its_grid(
        self, cfg, tmp_path
    ) -> None:
        # 5-step main sweep at f/4 + f/8 with f/8 factor=2.
        # Original sweep f/4 indices: 0,1,2,3,4 (factor 1).
        # Original sweep f/8 indices: 0,2,4 (factor 2).
        # Far-side probe shoots at step_index=5 first.
        # f/4: 5 % 1 == 0 → captured.
        # f/8: 5 % 2 == 1 → SKIPPED.
        # Second probe at step_index=6.
        # f/4: 6 % 1 == 0 → captured.
        # f/8: 6 % 2 == 0 → captured.
        # → expect 2 f/4 probe rows, 1 f/8 probe row.
        camera = FakeCamera(apertures=["f/4", "f/8"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 30,
            inputs={
                "lens": "L", "steps": 5, "end_mm": 20.0,
                "subsample_factors": {4.0: 1, 8.0: 2},
            },
        )
        # Force a probe by scripting peak_side='far' once, then ok.
        runner.range_checker = ScriptedRangeChecker([
            RangeResult(ok=False, peak_side="far",
                        needs_extension=True, notes="far"),
            RangeResult(ok=True, peak_side="far",
                        needs_extension=False,
                        peak_in_range_after_probe=True,
                        notes="bracket"),
        ])
        _run_one_lens_config(runner)
        rows = runner.shots_csv.read_rows()
        probe_rows = [r for r in rows if r["phase"] == "probe"]
        f4_probe_steps = sorted(
            int(r["rail_step"]) for r in probe_rows if r["aperture"] == "f/4"
        )
        f8_probe_steps = sorted(
            int(r["rail_step"]) for r in probe_rows if r["aperture"] == "f/8"
        )
        # Both probe step indices captured at f/4.
        assert f4_probe_steps == [5, 6]
        # Only step 6 captured at f/8 (step 5 % 2 != 0 → skipped).
        assert f8_probe_steps == [6]


# ---- Step-index helper math -----------------------------------------------


class TestExtensionStepIndex:
    def test_far_side_continues_past_main_sweep(self) -> None:
        # 3-step main sweep (indices 0..2). Far probe i=0 is index 3.
        assert SessionRunner._extension_step_index(
            side="far", original_steps=3, offset_in_extension=0
        ) == 3
        assert SessionRunner._extension_step_index(
            side="far", original_steps=3, offset_in_extension=2
        ) == 5

    def test_near_side_descends_below_zero(self) -> None:
        # Near probe i=0 is index -1 (one step *before* main sweep).
        assert SessionRunner._extension_step_index(
            side="near", original_steps=3, offset_in_extension=0
        ) == -1
        assert SessionRunner._extension_step_index(
            side="near", original_steps=3, offset_in_extension=2
        ) == -3
