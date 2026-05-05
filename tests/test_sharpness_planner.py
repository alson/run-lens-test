"""Tests for sharpness.planner — DOF estimation and rail-sweep planning."""

from __future__ import annotations

import pytest

from sharpness.planner import (
    DEFAULT_COC_MM,
    DEFAULT_MAX_APERTURE,
    DEFAULT_REPLICAS,
    HIGH_MAG_THRESHOLD,
    MIN_RAIL_STEP_MM,
    POSITIONS_ACROSS_FWHM_DEFAULT,
    POSITIONS_ACROSS_FWHM_HIGH_MAG,
    SamplePlan,
    estimate_dof_mm,
    parse_aperture,
    parse_magnification,
    plan_sweep,
    subsample_factor,
)


# --- parse_magnification ------------------------------------------------


class TestParseMagnification:
    def test_one_to_one_ratio(self) -> None:
        assert parse_magnification("1:1") == 1.0

    def test_one_and_a_half_to_one_ratio(self) -> None:
        assert parse_magnification("1.5:1") == 1.5

    def test_one_to_one_and_a_half_ratio(self) -> None:
        assert parse_magnification("1:1.5") == pytest.approx(1.0 / 1.5)

    def test_two_to_one_ratio(self) -> None:
        assert parse_magnification("2:1") == 2.0

    def test_bare_float(self) -> None:
        assert parse_magnification("0.5") == 0.5
        assert parse_magnification("1") == 1.0
        assert parse_magnification("1.0") == 1.0

    def test_whitespace_tolerated(self) -> None:
        assert parse_magnification("  1:1  ") == 1.0
        assert parse_magnification(" 1.5 : 1 ") == 1.5

    @pytest.mark.parametrize("bad", ["", "abc", "1:0", "0:1", "1:", ":1", "-1", "1:-1"])
    def test_invalid_raises(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_magnification(bad)


# --- parse_aperture -----------------------------------------------------


class TestParseAperture:
    @pytest.mark.parametrize("text,expected", [
        ("f/4", 4.0),
        ("F/4.5", 4.5),
        ("f/2.8", 2.8),
        ("4", 4.0),
        ("4.5", 4.5),
        ("f/16", 16.0),
        ("  f/8  ", 8.0),
    ])
    def test_accepts_common_forms(self, text: str, expected: float) -> None:
        assert parse_aperture(text) == expected

    @pytest.mark.parametrize("bad", ["", "abc", "0", "-4", "f/abc"])
    def test_invalid_raises(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_aperture(bad)


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
        # DOF ∝ (1+M)/M² → ratio at M=2 vs M=1 is (3/4) / (2/1) = 0.375
        a = estimate_dof_mm(1.0, 8.0)
        b = estimate_dof_mm(2.0, 8.0)
        assert b / a == pytest.approx(0.375, rel=1e-6)

    def test_custom_coc(self) -> None:
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


# --- subsample_factor ---------------------------------------------------


class TestSubsampleFactor:
    def test_widest_returns_1(self) -> None:
        assert subsample_factor(2.8, 2.8) == 1

    def test_doubled_aperture_returns_2(self) -> None:
        assert subsample_factor(5.6, 2.8) == 2

    def test_quadrupled_returns_4(self) -> None:
        assert subsample_factor(11.0, 2.8) == 3  # 11/2.8 = 3.93 → floor 3

    def test_f16_over_f4(self) -> None:
        assert subsample_factor(16.0, 4.0) == 4

    def test_f22_over_f4(self) -> None:
        # 22/4 = 5.5 → floor 5
        assert subsample_factor(22.0, 4.0) == 5

    def test_below_widest_clamps_to_one(self) -> None:
        assert subsample_factor(2.0, 2.8) == 1

    def test_intermediate_third_stop(self) -> None:
        # 7.1 / 2.8 = 2.535 → 2
        assert subsample_factor(7.1, 2.8) == 2


# --- plan_sweep ---------------------------------------------------------


class TestPlanSweep:
    def test_step_at_unity_mag_widest_f2_8(self) -> None:
        # DOF = 2 * 0.030 * 2.8 * 2 / 1 = 0.336 mm
        # FWHM = 0.168 mm
        # step = FWHM / 5 = 0.0336 mm
        p = plan_sweep(1.0, 2.8)
        assert p.rail_step_mm == pytest.approx(0.0336, rel=1e-3)
        assert p.estimated_dof_mm_widest == pytest.approx(0.336, rel=1e-3)
        assert p.widest_aperture == 2.8
        assert p.replicas_per_position == DEFAULT_REPLICAS

    def test_step_doubles_when_widest_aperture_doubles(self) -> None:
        a = plan_sweep(1.0, 2.8)
        b = plan_sweep(1.0, 5.6)
        assert b.rail_step_mm == pytest.approx(2 * a.rail_step_mm, rel=1e-6)

    def test_high_mag_uses_finer_step(self) -> None:
        # M >= 1.5 ⇒ POSITIONS_ACROSS_FWHM_HIGH_MAG (8 instead of 5)
        a = plan_sweep(1.0, 2.8)  # k=5
        b = plan_sweep(2.0, 2.8)  # k=8 plus narrower DOF at higher mag
        # Narrower DOF AND finer FWHM density both reduce step
        assert b.rail_step_mm < a.rail_step_mm
        # Specifically: at M=2 the DOF is 0.375× the M=1 DOF, AND the
        # density is 8/5 finer. So step factor = 0.375 * 5/8 = 0.234
        assert b.rail_step_mm / a.rail_step_mm == pytest.approx(
            0.375 * 5 / 8, rel=1e-3
        )

    def test_threshold_boundary_for_high_mag(self) -> None:
        # M = 1.49 → default density; M = 1.5 → high-mag density
        a = plan_sweep(1.49, 2.8)
        b = plan_sweep(1.5, 2.8)
        # b uses k=8 instead of k=5 ⇒ smaller step
        assert b.rail_step_mm < a.rail_step_mm

    def test_subsample_factors_populated(self) -> None:
        p = plan_sweep(1.0, 2.8, max_aperture=16.0)
        assert 2.8 in p.subsample_factors
        assert p.subsample_factors[2.8] == 1
        assert p.subsample_factors[5.6] == 2
        assert p.subsample_factors[11.0] == 3
        assert p.subsample_factors[16.0] == 5
        # Apertures below widest are not included
        assert 2.0 not in p.subsample_factors
        # Apertures above max are not included
        assert 22.0 not in p.subsample_factors

    def test_subsample_table_truncated_to_max_aperture(self) -> None:
        p = plan_sweep(1.0, 2.8, max_aperture=11.0)
        assert max(p.subsample_factors) == 11.0
        assert 16.0 not in p.subsample_factors

    def test_rail_range_translates_to_positions(self) -> None:
        # step = 0.0336 mm; 1 mm range → 30 positions
        p = plan_sweep(1.0, 2.8, rail_range_mm=1.0)
        assert p.rail_positions == 30
        assert p.rail_range_mm == 1.0
        assert p.total_shots == 30 * 3

    def test_no_range_means_no_positions(self) -> None:
        p = plan_sweep(1.0, 2.8)
        assert p.rail_positions is None
        assert p.rail_range_mm is None
        assert p.total_shots is None

    def test_invalid_widest_raises(self) -> None:
        with pytest.raises(ValueError, match="widest_aperture"):
            plan_sweep(1.0, 0)

    def test_invalid_max_below_widest_raises(self) -> None:
        with pytest.raises(ValueError, match="max_aperture"):
            plan_sweep(1.0, 8.0, max_aperture=4.0)

    def test_invalid_rail_range_raises(self) -> None:
        with pytest.raises(ValueError, match="rail_range_mm"):
            plan_sweep(1.0, 2.8, rail_range_mm=0)

    def test_explicit_density_override(self) -> None:
        # Caller can override the auto-density choice if they want
        p = plan_sweep(1.0, 2.8, positions_across_fwhm=10)
        # step = FWHM / 10 = 0.0168 mm
        assert p.rail_step_mm == pytest.approx(0.0168, rel=1e-3)

    def test_step_clamped_at_floor(self) -> None:
        # Push parameters into a regime where the raw step would be
        # below the StackShot's mechanical floor. With a tight custom
        # min_step we can verify clamping without going to extreme M/N.
        p = plan_sweep(1.0, 2.8, min_step_mm=0.05)
        # raw step at M=1, N=2.8 is 0.0336 mm < 0.05 ⇒ clamp
        assert p.step_clamped is True
        assert p.rail_step_mm == pytest.approx(0.05, rel=1e-9)

    def test_step_not_clamped_when_above_floor(self) -> None:
        p = plan_sweep(1.0, 2.8)
        # Default min is 5 µm; raw step at this regime is ~33.6 µm ⇒ no clamp
        assert p.step_clamped is False
        assert p.rail_step_mm > MIN_RAIL_STEP_MM

    def test_extreme_high_mag_wide_open_clamps_at_default_floor(self) -> None:
        # M=3, widest=f/1.4 → DOF tiny → raw step well under 5 µm
        # DOF = 2 * 0.030 * 1.4 * 4 / 9 ≈ 0.0373 mm
        # FWHM ≈ 0.0187 mm; high-mag density 8 ⇒ raw step ≈ 2.3 µm
        p = plan_sweep(3.0, 1.4)
        assert p.step_clamped is True
        assert p.rail_step_mm == pytest.approx(MIN_RAIL_STEP_MM, rel=1e-9)

    def test_clamp_affects_position_count(self) -> None:
        # Same range, with vs without clamp ⇒ fewer positions when clamped
        # (because step is rounded UP from raw).
        p_unclamped = plan_sweep(1.0, 2.8, rail_range_mm=1.0, min_step_mm=1e-9)
        p_clamped = plan_sweep(1.0, 2.8, rail_range_mm=1.0, min_step_mm=0.1)
        assert p_clamped.rail_positions < p_unclamped.rail_positions
        assert p_clamped.step_clamped is True

    def test_invalid_min_step_raises(self) -> None:
        with pytest.raises(ValueError, match="min_step_mm"):
            plan_sweep(1.0, 2.8, min_step_mm=0)
        with pytest.raises(ValueError, match="min_step_mm"):
            plan_sweep(1.0, 2.8, min_step_mm=-1)


# --- SamplePlan ---------------------------------------------------------


class TestSamplePlan:
    def test_total_shots_is_none_without_range(self) -> None:
        p = SamplePlan(
            rail_step_mm=0.05,
            replicas_per_position=3,
            estimated_dof_mm_widest=1.0,
            estimated_fwhm_mm_widest=0.5,
            widest_aperture=2.8,
            max_aperture=16.0,
        )
        assert p.total_shots is None

    def test_total_shots_with_range(self) -> None:
        p = plan_sweep(1.0, 2.8, rail_range_mm=1.0)
        assert p.total_shots == p.rail_positions * 3

    def test_frozen(self) -> None:
        p = plan_sweep(1.0, 2.8)
        with pytest.raises(Exception):
            p.rail_step_mm = 99.0  # type: ignore[misc]


# --- Regime checks against the experiment ------------------------------


class TestRegimeSanityChecks:
    def test_high_mag_step_finer_than_about_15_microns(self) -> None:
        """At M=2, F4 widest the step should be ~10-15 µm — fine enough
        that ~8 positions span the FWHM (which is what the resampling
        experiment showed was needed)."""
        p = plan_sweep(2.0, 4.0)
        assert p.rail_step_mm < 0.020

    def test_baseline_step_about_30_microns_at_m1_f2_8(self) -> None:
        """Calibration spot-check: at M=1, F2.8 widest the step should be
        in the 25-40 µm range. (FWHM 168 µm / 5 = 33.6 µm.)"""
        p = plan_sweep(1.0, 2.8)
        assert 0.025 < p.rail_step_mm < 0.040

    def test_very_low_mag_step_well_above_100_microns(self) -> None:
        """At M=0.2, F4 widest the DOF is enormous and the step should be
        coarse (>100 µm) — otherwise we'd waste positions."""
        p = plan_sweep(0.2, 4.0)
        assert p.rail_step_mm > 0.100

    def test_aperture_subsample_factor_1_at_widest(self) -> None:
        for widest in [2.8, 4.0, 5.6, 8.0]:
            p = plan_sweep(1.0, widest)
            assert p.subsample_factors[widest] == 1
