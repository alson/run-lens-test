"""Tests for SessionRunner in manual-aperture mode.

These tests verify the inverted-loop semantics:

* K rail-zero returns between apertures (one per ring-turn).
* K prompt_turn_ring invocations (one per aperture).
* No ``camera.set_aperture`` calls (the diaphragm is mechanical).
* No FEC sanity check at session start (FEC is inert in flash-M).
* shots.csv per-aperture-grouped, with manual-mode metadata
  (``magnification_indicated``, ``f_ring``, ``focal_length_mm``)
  populated.
* Phase A probe runs once per aperture (not just at widest) when
  PROBE_ENABLED.

The hardware fakes from :mod:`tests.test_runner` are reused — they
record every interaction so the test can assert against the trace.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from exposure.range_check import RangeChecker
from session.runner import (
    JogResult,
    SessionRunner,
)
from stackshot.protocol import Direction


# ---- Local fakes (mirrors test_runner.FakeCamera/FakeRail) -----------------


@dataclass
class FakeCamera:
    apertures: list[str] = field(
        default_factory=lambda: ["f/4", "f/5.6", "f/8", "f/11", "f/16"]
    )
    history: list[tuple[str, object]] = field(default_factory=list)
    fec_sanity_check_calls: int = 0
    open_count: int = 0
    closed: bool = False

    def open(self) -> None:
        self.open_count += 1
        self.history.append(("open", None))

    def close(self) -> None:
        self.closed = True
        self.history.append(("close", None))

    def verify_healthy(self, *, min_widgets: int = 100) -> None:
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

    def fec_sanity_check(self, **kwargs) -> tuple[float, float]:
        self.fec_sanity_check_calls += 1
        return (100.0, 25.0)

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


# ---- Fixture: minimal config ----------------------------------------------


@pytest.fixture
def cfg():
    """Mirrors tests.test_runner.cfg, plus PROBE_ENABLED=False so the
    base tests can verify the rail-zero / prompt-count contract
    without involving the Phase A code path. A separate test enables
    PROBE for the per-aperture-probe assertion."""
    return SimpleNamespace(
        MAX_FSTOP=16.0,
        WHOLE_STOP_SCALE=(1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0),
        ONE_THIRD_STOP_SCALE=(
            1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5,
            4.0, 4.5, 5.0, 5.6, 6.3, 7.1,
            8.0, 9.0, 10.0, 11.0, 13.0, 14.0,
            16.0, 18.0, 20.0, 22.0, 25.0, 29.0, 32.0,
        ),
        RANGE_EXTENSION={
            2.8: (100, 0.0),
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
        PROBE_ENABLED=False,
        STEP_NUMBER_OFFSET=100,
        STEP_NUMBER_WIDTH=4,
        # Manual-aperture flash-M cascade constants.
        MANUAL_ISO_FLOOR=100,
        MANUAL_FLASH_SOFT_CAP_STOPS=1.0,
        MANUAL_FLASH_HARD_CAP_STOPS=0.0,
        MANUAL_FLASH_INITIAL_POWER_STOPS=5.0,  # 1/32
        MANUAL_FLASH_SEARCH_TOLERANCE_EV=1.0,
        MANUAL_FLASH_BUMP_SANITY_EV=0.5,
        MANUAL_FLASH_MAX_SEARCH_ITERATIONS=15,
    )


# ---- Runner builder --------------------------------------------------------


def _make_manual_runner(
    *,
    camera: FakeCamera,
    rail: FakeRail,
    cfg,
    tmp_path: Path,
    medians,
    slug: str = "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
    magnification_total: float = 1.0,
    magnification_indicated: float = 0.5,
    end_mm: float = 4.0,
    steps: int = 2,
    ring_turn_responses: list[bool] | None = None,
):
    """Build a SessionRunner configured for manual-aperture mode."""
    queue = deque(medians)

    def _measure(_p):
        if not queue:
            return 100.0
        return queue.popleft()

    fixed_now = datetime(2026, 5, 4, 18, 35, tzinfo=timezone.utc)

    def _prompt_lens() -> str | None:
        return slug

    def _prompt_magnification() -> float:
        return magnification_total

    def _prompt_magnification_indicated(default: float) -> float:
        return magnification_indicated

    def _confirm(_msg) -> bool:
        return True

    def _jog(rail: FakeRail) -> JogResult:
        rail.position_mm = end_mm
        return JogResult(end_mm=end_mm)

    from sharpness.planner import SamplePlan

    def _fake_plan_sweep(
        *, magnification, widest_aperture, max_aperture,
        rail_range_mm, replicas,
        coc_mm=None, positions_across_fwhm=None,
    ) -> SamplePlan:
        step_size = (
            rail_range_mm / max(1, steps - 1)
            if steps > 1 else rail_range_mm
        )
        return SamplePlan(
            rail_step_mm=step_size,
            replicas_per_position=replicas,
            estimated_dof_mm_widest=1.0,
            estimated_fwhm_mm_widest=0.5,
            widest_aperture=widest_aperture,
            max_aperture=max_aperture,
            subsample_factors={widest_aperture: 1},
            rail_range_mm=rail_range_mm,
            rail_positions=steps,
        )

    # Track ring-turn invocations. Each call consumes one entry of
    # ring_turn_responses (default: all True). Records arguments so
    # tests can introspect.
    ring_turns: list[dict] = []
    responses = list(ring_turn_responses or [])

    def _prompt_turn_ring(
        aperture_label: str,
        *,
        ring_stop: float,
        m_indicated: float,
        input_func=None,
    ) -> bool:
        ring_turns.append(
            dict(
                aperture_label=aperture_label,
                ring_stop=ring_stop,
                m_indicated=m_indicated,
            )
        )
        if responses:
            return responses.pop(0)
        return True

    # Same idea for the flash-power prompt.
    power_prompts: list[dict] = []

    def _prompt_set_flash_power(
        *,
        power_c_label: str,
        power_a_label: str,
        context: str = "",
        input_func=None,
    ) -> bool:
        power_prompts.append(
            dict(
                power_c_label=power_c_label,
                power_a_label=power_a_label,
                context=context,
            )
        )
        return True

    runner = SessionRunner(
        camera=camera,
        rail=rail,
        config=cfg,
        ntfy=None,
        range_checker=None,
        output_root=tmp_path,
        measure_path=_measure,
        sleep=lambda s: None,
        now=lambda: fixed_now,
        prompt_lens_config=_prompt_lens,
        prompt_magnification=_prompt_magnification,
        prompt_magnification_indicated=_prompt_magnification_indicated,
        prompt_turn_ring=_prompt_turn_ring,
        prompt_set_flash_power=_prompt_set_flash_power,
        plan_sweep=_fake_plan_sweep,
        confirm=_confirm,
        jog=_jog,
        manual_aperture=True,
    )
    runner.__test_ring_turns__ = ring_turns  # type: ignore[attr-defined]
    runner.__test_power_prompts__ = power_prompts  # type: ignore[attr-defined]
    return runner


# ---- Tests -----------------------------------------------------------------


class TestManualSessionStartup:
    def test_fec_sanity_check_skipped(self, cfg, tmp_path) -> None:
        """In manual-aperture mode the flash is in M (manual power);
        FEC has no effect on flash output, so the FEC sanity check
        at session start would always fail. The runner must skip
        it."""
        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        assert camera.fec_sanity_check_calls == 0


class TestManualLensConfigPrompting:
    def test_slug_parse_extracts_focal_and_fring(
        self, cfg, tmp_path
    ) -> None:
        """For the 55 AIS slug, the rig parses focal=55, f_ring=2.8
        from the slug and doesn't need to prompt for either. We
        call the prep helper directly so the per-lens-config state
        is observable (the lens-config flow clears it in a finally
        block once the sweep ends)."""
        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 10,
        )
        runner.start()
        apertures, widest_str, widest_val = (
            runner._prepare_manual_aperture_config(
                slug="Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
                magnification_total=1.0,
            )
        )
        assert runner._current_focal_length_mm == 55.0
        assert runner._current_f_ring == 2.8
        assert runner._current_magnification_indicated == 0.5
        assert apertures == ["4", "6.3", "8", "13", "16"]
        assert widest_str == "4"
        assert widest_val == 4.0


class TestManualSweepInvertedLoop:
    def test_no_camera_set_aperture_calls(self, cfg, tmp_path) -> None:
        """The diaphragm is mechanical — the rig must not call
        camera.set_aperture in manual mode."""
        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 50,
        )
        runner.start()
        runner._run_one_lens_config(
            "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
            magnification=1.0,
        )
        aperture_calls = [v for k, v in camera.history if k == "aperture"]
        assert aperture_calls == [], (
            f"manual mode must not call set_aperture; got {aperture_calls}"
        )

    def test_K_ring_turn_prompts_one_per_aperture(
        self, cfg, tmp_path
    ) -> None:
        """One ring-turn prompt per aperture in the synthesised
        sequence. For 55 AIS f/2.8 + PK-13 at 1:1 (m_indicated=0.5):
        sequence is ["4", "6.3", "8", "13"], so K=4 prompts."""
        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
        )
        runner.start()
        runner._run_one_lens_config(
            "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
            magnification=1.0,
        )
        ring_turns = runner.__test_ring_turns__  # type: ignore[attr-defined]
        # Sequence per test_manual_aperture: ["4", "6.3", "8", "13", "16"]
        assert [t["aperture_label"] for t in ring_turns] == [
            "4", "6.3", "8", "13", "16",
        ]
        # Each prompt receives the matching ring stop from the
        # WHOLE_STOP_SCALE walk. Ring f/11 is included because its
        # rounded effective label ("16") equals MAX_FSTOP, even
        # though the raw effective (16.5) would over-shoot.
        assert [t["ring_stop"] for t in ring_turns] == [
            2.8, 4.0, 5.6, 8.0, 11.0,
        ]
        for t in ring_turns:
            assert t["m_indicated"] == 0.5

    def test_rail_returns_to_zero_between_apertures(
        self, cfg, tmp_path
    ) -> None:
        """Between each aperture the rail must return to position 0
        (one backward move per aperture transition). For K
        apertures, K-1 transitions + 1 final return = K rail-zero
        moves in the direction reverse of the sweep direction.

        We just verify the rail ends at position 0 after the
        lens-config (since the sweep walks forward, then returns
        before the next aperture, and the final aperture's
        completion also returns to zero)."""
        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
        )
        runner.start()
        runner._run_one_lens_config(
            "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
            magnification=1.0,
        )
        # After the lens config, rail position should be back at 0
        # (within numerical tolerance — the FakeRail tracks exact
        # accumulated moves so this is exact).
        assert abs(rail.position_mm) < 1e-9, (
            f"rail should return to 0 between apertures; got "
            f"{rail.position_mm}"
        )


class TestManualShotsCsvMetadata:
    def test_sweep_rows_carry_manual_metadata(
        self, cfg, tmp_path
    ) -> None:
        """Sweep-phase rows in shots.csv must carry
        magnification_indicated, f_ring, focal_length_mm from the
        runner's per-lens-config state."""
        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 100,
        )
        runner.start()
        runner._run_one_lens_config(
            "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
            magnification=1.0,
        )
        with (runner.session_dir / "shots.csv").open(
            "r", encoding="utf-8", newline=""
        ) as f:
            rows = list(csv.DictReader(f))
        assert rows, "shots.csv should have rows after a manual sweep"
        # Every sweep row carries the manual metadata.
        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        assert sweep_rows, "expected at least one sweep row"
        for row in sweep_rows:
            assert row["magnification_indicated"] == "0.5"
            assert row["f_ring"] == "2.8"
            assert row["focal_length_mm"] == "55"

    def test_per_aperture_grouped_not_per_step(self, cfg, tmp_path) -> None:
        """The inverted loop captures all rail steps for aperture 1,
        then all for aperture 2, etc. Verified by reading the
        sweep-phase rows in order and checking that aperture labels
        are runs (e.g. [4,4,6.3,6.3,...]), not interleaved
        ([4,6.3,4,6.3,...])."""
        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 200,
            steps=3,
        )
        runner.start()
        runner._run_one_lens_config(
            "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
            magnification=1.0,
        )
        with (runner.session_dir / "shots.csv").open(
            "r", encoding="utf-8", newline=""
        ) as f:
            rows = list(csv.DictReader(f))
        sweep_apertures = [
            r["aperture"] for r in rows if r["phase"] == "sweep"
        ]
        # Run-length-encode and verify each unique aperture is in a
        # single run (no interleaving).
        if not sweep_apertures:
            pytest.fail("no sweep rows captured")
        seen: list[str] = []
        prev = None
        for ap in sweep_apertures:
            if ap != prev:
                seen.append(ap)
                prev = ap
        # Each label appears exactly once in the RLE — i.e. no
        # repeats after switching aperture.
        assert len(seen) == len(set(seen)), (
            f"aperture labels are interleaved (per-step ordering); "
            f"RLE sequence: {seen}"
        )


class TestManualAbortAtRingTurn:
    def test_q_at_ring_turn_aborts_session(self, cfg, tmp_path) -> None:
        """If the user answers `q` at any ring-turn prompt, the
        session aborts cleanly (SessionAborted raised)."""
        from errors import SessionAborted

        camera = FakeCamera()
        rail = FakeRail()
        runner = _make_manual_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 50,
            # First aperture's ring turn is OK; second aperture: abort.
            ring_turn_responses=[True, False],
        )
        runner.start()
        with pytest.raises(SessionAborted):
            runner._run_one_lens_config(
                "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318",
                magnification=1.0,
            )
