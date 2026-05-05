"""Tests for sharpness.planner — DOF estimation and rail-sweep planning."""

from __future__ import annotations

import math

import pytest

from sharpness.planner import (
    DEFAULT_BASE_POSITIONS,
    DEFAULT_COC_MM,
    DEFAULT_REPLICAS,
    SamplePlan,
    estimate_dof_mm,
    plan_sweep,
)


# --- estimate_dof_mm ----------------------------------------------------


class TestEstimateDofMm:
    def test_at_unity_magnification_and_f8(self) -> None:
        # DOF = 2 * 0.030 * 8 * (1+1) / 1² = 0.96
        dof = estimate_dof_mm(1.0, 8.0)
        assert dof == pytest.approx(0.96, rel=1e-6)

    def test_doubling_aperture_doubles_dof(self) -> None:
        a = estimate_dof_mm(1.0, 8.0)
        b = estimate_dof_mm(1.0, 16.0)
        assert b == pytest.approx(2 * a, rel=1e-6)

    def test_doubling_magnification_quarters_dof_approximately(self) -> None:
        # DOF ∝ (1+M)/M² → at M=1 we get 2/1, at M=2 we get 3/4.
        # Ratio 0.375 — narrower DOF at higher magnification.
        a = estimate_dof_mm(1.0, 8.0)
        b = estimate_dof_mm(2.0, 8.0)
        assert b / a == pytest.approx(0.375, rel=1e-6)

    def test_very_low_magnification_blows_up(self) -> None:
        # M=0.1 → DOF ∝ 1.1/0.01 = 110 × c × N. Should be order of
        # tens of millimetres at typical apertures.
        dof = estimate_dof_mm(0.1, 8.0)
        assert dof > 50.0  # very wide

    def test_custom_coc(self) -> None:
        # Halving c halves DOF.
        a = estimate_dof_mm(1.0, 8.0, coc_mm=0.030)
        b = estimate_dof_mm(1.0, 8.0, coc_mm=0.015)
        assert b == pytest.approx(a / 2, rel=1e-6)

    @pytest.mark.parametrize("m", [0, -0.1, -1])
    def test_non_positive_magnification_raises(self, m: float) -> None:
        with pytest.raises(ValueError, match="magnification"):
            estimate_dof_mm(m, 8.0)

    @pytest.mark.parametrize("n", [0, -1])
    def test_non_positive_aperture_raises(self, n: float) -> None:
        with pytest.raises(ValueError, match="aperture"):
            estimate_dof_mm(1.0, n)


# --- plan_sweep --------------------------------------------------------


class TestPlanSweep:
    def test_historical_baseline_at_m1_f8(self) -> None:
        # Calibration target: at M=1, N=8 we want ~25 positions × 3 replicas.
        p = plan_sweep(1.0, 8.0)
        assert p.rail_positions == 25
        assert p.replicas_per_position == 3
        assert p.total_shots == 75
        # Rail step is calibration-driven: sweep is 1 × DOF, so step
        # = DOF / 25 = 0.96 / 25 ≈ 0.0384 mm.
        assert p.rail_step_mm == pytest.approx(0.96 / 25, rel=1e-6)
        assert p.sweep_range_mm == pytest.approx(0.96, rel=1e-6)
        assert p.estimated_dof_mm == pytest.approx(0.96, rel=1e-6)
        assert p.estimated_fwhm_mm == pytest.approx(0.48, rel=1e-6)

    def test_low_magnification_bumps_positions(self) -> None:
        # M < 0.4 ⇒ +5 positions
        p = plan_sweep(0.3, 8.0)
        assert p.rail_positions == DEFAULT_BASE_POSITIONS + 5

    def test_small_aperture_bumps_positions(self) -> None:
        # N >= 11 ⇒ +3 positions
        p = plan_sweep(1.0, 11.0)
        assert p.rail_positions == DEFAULT_BASE_POSITIONS + 3

    def test_high_magnification_bumps_positions(self) -> None:
        # M >= 1.5 ⇒ +2 positions
        p = plan_sweep(2.0, 8.0)
        assert p.rail_positions == DEFAULT_BASE_POSITIONS + 2

    def test_bumps_stack(self) -> None:
        # M >= 1.5 (+2) AND N >= 11 (+3) ⇒ +5
        p = plan_sweep(2.0, 16.0)
        assert p.rail_positions == DEFAULT_BASE_POSITIONS + 2 + 3

    def test_low_mag_high_f_stack_fully(self) -> None:
        # M < 0.4 (+5) AND N >= 11 (+3) ⇒ +8
        p = plan_sweep(0.3, 16.0)
        assert p.rail_positions == DEFAULT_BASE_POSITIONS + 5 + 3

    def test_step_scales_with_dof(self) -> None:
        # At fixed positions, step ∝ DOF; at higher M (smaller DOF) step is finer.
        p_low_mag = plan_sweep(1.0, 8.0)
        p_high_mag = plan_sweep(2.0, 8.0)
        # M=2 → DOF = 2*0.03*8*3/4 = 0.36 → step = 0.36 / 27 ≈ 0.0133 mm
        assert p_high_mag.rail_step_mm < p_low_mag.rail_step_mm

    def test_sweep_equals_one_dof(self) -> None:
        # Sweep = 2 * FWHM = 1 * DOF by construction.
        p = plan_sweep(1.0, 8.0)
        assert p.sweep_range_mm == pytest.approx(p.estimated_dof_mm, rel=1e-9)

    def test_total_shots_reflects_replicas(self) -> None:
        p = plan_sweep(1.0, 8.0, replicas=4)
        assert p.replicas_per_position == 4
        assert p.total_shots == p.rail_positions * 4

    def test_custom_base_positions(self) -> None:
        p = plan_sweep(1.0, 8.0, base_positions=20)
        assert p.rail_positions == 20

    def test_invalid_inputs_propagate(self) -> None:
        with pytest.raises(ValueError, match="magnification"):
            plan_sweep(0, 8.0)


# --- SamplePlan ---------------------------------------------------------


class TestSamplePlan:
    def test_total_shots_property(self) -> None:
        p = SamplePlan(
            rail_positions=10,
            replicas_per_position=3,
            rail_step_mm=0.05,
            sweep_range_mm=0.5,
            estimated_dof_mm=1.0,
            estimated_fwhm_mm=0.5,
        )
        assert p.total_shots == 30

    def test_frozen(self) -> None:
        p = plan_sweep(1.0, 8.0)
        with pytest.raises(Exception):
            p.rail_positions = 99  # type: ignore[misc]


# --- Cross-check against the rail-coverage analysis ---------------------


class TestRegimeSanityChecks:
    """Spot-check that the planner reproduces the regimes in the test data."""

    def test_low_mag_gets_meaningfully_more_steps(self) -> None:
        """At M=0.5, F11 (the 'low (1:2 to 1:1.18)' band) the planner should
        recommend more positions than at M=1, F8 — that band showed more
        truncation in the rail-coverage data."""
        p_baseline = plan_sweep(1.0, 8.0)
        p_lowband = plan_sweep(0.5, 11.0)
        # At M=0.5 we get the +0 bump (M=0.5 is not <0.4) but +3 for f-stop
        assert p_lowband.rail_positions == p_baseline.rail_positions + 3

    def test_very_high_mag_step_under_30_microns(self) -> None:
        """At M=2, F4 the rail step should be very fine (under 30 µm) because
        the DOF is razor-thin."""
        p = plan_sweep(2.0, 4.0)
        assert p.rail_step_mm < 0.030

    def test_very_low_mag_step_over_100_microns(self) -> None:
        """At M=0.2, F11 (very low mag) the rail step should be coarse
        (over 100 µm) because the DOF is wide; otherwise we'd waste shots
        oversampling."""
        p = plan_sweep(0.2, 11.0)
        assert p.rail_step_mm > 0.100

    def test_total_shots_stays_within_battery_budget(self) -> None:
        """Across the realistic (M, N) grid, the planner should never
        recommend more than 35 positions × 3 replicas = 105 shots — beyond
        that the user's battery droop architecture starts to bite."""
        for m in [0.2, 0.3, 0.5, 1.0, 1.5, 2.0]:
            for n in [4.0, 5.6, 8.0, 11.0, 16.0]:
                p = plan_sweep(m, n)
                assert p.total_shots <= 35 * 3, (
                    f"too many shots at M={m}, N={n}: {p.total_shots}"
                )
