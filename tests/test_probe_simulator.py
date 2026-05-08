"""Tests for the dry-run coarse-to-fine probe simulator.

The simulator (``tools/probe_simulator.py``) is the validation harness
the algorithm was calibrated against — these tests pin down its core
math (phase-A peak picking, parabolic refinement, asymmetric bracket
window) on synthetic inputs so refactors don't silently change the
calibration numbers in the plan doc.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.probe_simulator import (
    PHASE_A_POINTS,
    PHASE_C_FWHM_PADDING,
    ApertureData,
    CropSeries,
    SessionData,
    simulate_phase_a,
    simulate_session,
)


def _make_gaussian_series(
    n_steps: int,
    peak_step: float,
    fwhm_steps: float,
    *,
    floor: int = 100_000,
    amplitude: int = 200_000,
) -> dict[int, int]:
    """Synth a Gaussian-shaped JPEG-size series centred on `peak_step`."""
    sigma = fwhm_steps / 2.355
    xs = np.arange(n_steps)
    ys = floor + amplitude * np.exp(-0.5 * ((xs - peak_step) / sigma) ** 2)
    return {int(s): int(round(v)) for s, v in zip(xs, ys, strict=True)}


def _session_with_peaks(
    crop_peaks: dict[str, float],
    *,
    n_steps: int = 200,
    fwhm_steps: float = 8.0,
    rail_step_mm: float = 0.011,
    n_apertures: int = 1,
) -> SessionData:
    """Build a SessionData where each crop has a Gaussian centred on its peak.

    By default a single aperture (f/4.5) so the simulator's "widest
    aperture only" phase A logic has something to grab.
    """
    apertures: dict[str, ApertureData] = {}
    for ap_idx in range(n_apertures):
        ap_value = 4.5 + ap_idx * 1.0  # f/4.5, f/5.5, ...
        ap_text = f"f/{ap_value:g}"
        crops: dict[str, CropSeries] = {}
        for name, pk in crop_peaks.items():
            crops[name] = CropSeries(
                name=name,
                by_step=_make_gaussian_series(n_steps, pk, fwhm_steps),
            )
        apertures[ap_text] = ApertureData(
            aperture_text=ap_text, aperture_value=ap_value, crops=crops,
        )
    from pathlib import Path
    return SessionData(
        path=Path("/synthetic"),
        lens_config="synthetic",
        rail_step_mm=rail_step_mm,
        fwhm_widest_mm=fwhm_steps * rail_step_mm,
        widest_aperture=4.5,
        subsample_factors={4.5: 1},
        apertures=apertures,
    )


class TestPhaseA:
    """Phase A picks evenly spaced points and finds per-crop peaks."""

    def test_picks_n_points_evenly_spaced(self):
        sess = _session_with_peaks({"center": 100.0}, n_steps=200)
        result = simulate_phase_a(sess, n_points=PHASE_A_POINTS)
        assert len(result.sample_steps) == PHASE_A_POINTS
        # First and last should be the captured-range endpoints.
        assert result.sample_steps[0] == 0
        assert result.sample_steps[-1] == 199
        # Spacing should be ~uniform.
        diffs = np.diff(result.sample_steps)
        assert max(diffs) - min(diffs) <= 1

    def test_centre_peak_recovered_within_one_phase_a_spacing(self):
        """A single-crop session puts its peak at step 100; phase A should
        recover that with sub-sample resolution well below the spacing."""
        sess = _session_with_peaks({"center": 100.0}, n_steps=200)
        result = simulate_phase_a(sess, n_points=7)
        assert result.centre_peak_step is not None
        # Phase-A spacing = 200 / 6 ≈ 33; the parabolic refinement
        # should land within a few steps of the true peak.
        assert abs(result.centre_peak_step - 100.0) < 5.0

    def test_corner_spread_matches_input_spread(self):
        """When corners peak in a known band, phase A should report a
        spread that approximately matches the input.

        Uses 15 phase-A points to keep spacing well under FWHM — the
        7-point default works on real rig data because the corner
        bells are wider than the centre FWHM, but a clean Gaussian
        synthetic with FWHM=20 and spacing=33 attenuates side samples
        to near-floor and washes out parabolic refinement."""
        peaks = {
            "center": 120.0,
            "top-left": 100.0,
            "top-right": 100.0,
            "bottom-left": 105.0,
            "bottom-right": 105.0,
        }
        sess = _session_with_peaks(peaks, n_steps=200, fwhm_steps=20.0)
        result = simulate_phase_a(sess, n_points=15)
        true_spread = max(peaks.values()) - min(peaks.values())  # 20
        # With phase-A spacing < FWHM and parabolic refinement,
        # observed spread should match truth within ~30 %.
        assert 0.7 * true_spread <= result.corner_spread_steps <= 1.5 * true_spread

    def test_inconclusive_when_too_few_valid_crops(self):
        """If fewer than 3 crops yield a valid peak, phase A flags
        inconclusive."""
        peaks = {"center": 100.0, "top-left": 100.0}  # only 2 crops
        sess = _session_with_peaks(peaks, n_steps=200)
        result = simulate_phase_a(sess, n_points=7)
        assert result.inconclusive

    def test_min_max_peak_steps_populated(self):
        peaks = {
            "center": 120.0,
            "top-left": 100.0,
            "bottom-right": 130.0,
        }
        sess = _session_with_peaks(peaks, n_steps=200)
        result = simulate_phase_a(sess, n_points=7)
        assert result.min_peak_step is not None
        assert result.max_peak_step is not None
        # min should be near 100, max near 130 (within phase-A spacing).
        assert abs(result.min_peak_step - 100.0) < 5.0
        assert abs(result.max_peak_step - 130.0) < 5.0


class TestPhaseCWindow:
    """Phase C window is an asymmetric min-max bracket around crop peaks."""

    def test_bracket_not_centred_on_centre_peak(self):
        """High-FC case — corners on one side of centre. Window must
        bracket the corner band, not be symmetric around centre."""
        # Mirrors the 231805 pattern: centre peaks late, corners peak early.
        peaks = {
            "center": 200.0,
            "top-left": 150.0,
            "top-right": 145.0,
            "bottom-left": 155.0,
            "bottom-right": 150.0,
            "left-edge": 160.0,
            "right-edge": 165.0,
        }
        sess = _session_with_peaks(peaks, n_steps=300, fwhm_steps=20.0)
        result = simulate_session(sess)
        # Compute expected bracket from the inputs.
        pa = result.phase_a
        fwhm_steps = sess.fwhm_widest_mm / sess.rail_step_mm
        pad = PHASE_C_FWHM_PADDING * fwhm_steps
        expected_lo = pa.min_peak_step - pad
        expected_hi = pa.max_peak_step + pad
        ap = next(iter(result.by_aperture.values()))
        assert ap.phase_c_master_steps[0] >= int(round(expected_lo)) - 1
        assert ap.phase_c_master_steps[-1] <= int(round(expected_hi)) + 1
        # Window must reach the centre peak (200): centre is the
        # max-peak crop, so the upper bracket = 200 + 2 × FWHM.
        assert ap.phase_c_master_steps[-1] >= 200 - 5
        # And reach the leftmost corner (top-right at 145): lower bracket
        # = 145 - 2 × FWHM ≈ 105. So the window starts well below the centre.
        assert ap.phase_c_master_steps[0] < 145

    def test_no_data_loss_on_clean_well_jogged_session(self):
        """If every crop's peak is interior and well-spaced from edges,
        every crop's full-sweep peak file size should be preserved."""
        peaks = {
            "center": 150.0,
            "top-left": 130.0,
            "top-right": 135.0,
            "bottom-left": 140.0,
            "bottom-right": 145.0,
        }
        sess = _session_with_peaks(peaks, n_steps=300, fwhm_steps=20.0)
        result = simulate_session(sess)
        for ap_result in result.by_aperture.values():
            for v in ap_result.crop_verdicts:
                assert v.rel_loss_pct < 0.5, (
                    f"{v.crop_name} rel-loss {v.rel_loss_pct:.2f}% "
                    f"(in_window={v.in_phase_c_window})"
                )

    def test_savings_positive_when_user_jog_is_wide(self):
        """If the user jogged way past the bell on both sides, phase C
        should fit inside that range and produce a positive savings."""
        peaks = {
            "center": 200.0,
            "top-left": 195.0,
            "top-right": 195.0,
            "bottom-left": 205.0,
            "bottom-right": 205.0,
        }
        # 600-step capture range, peaks all at ~200. FWHM=40 keeps the
        # FWHM/phase-A-spacing ratio (~0.4) within the rig's regime;
        # phase A spacing is 100 across 600 steps with 7 samples.
        sess = _session_with_peaks(peaks, n_steps=600, fwhm_steps=40.0)
        result = simulate_session(sess)
        # Savings should be substantial — narrow phase C inside a wide
        # original range. Window ≈ 10 (spread) + 2 × 2 × 40 (padding)
        # = 170 steps out of 600 → ~70 % savings minus phase A overhead.
        assert result.savings_pct > 50

    def test_extreme_outlier_corner_still_in_window(self):
        """If a corner's peak is way out at one end of the capture range
        (e.g. severe field tilt), the bracket should still extend out
        to catch it — that's the whole point of the asymmetric design.

        Uses dense phase A (n=21) so the outlier crop's bell registers
        on the dynamic-range gate; the 7-point default would miss a
        bell sitting between two phase-A samples on a clean Gaussian."""
        peaks = {
            "center": 300.0,
            "top-left": 305.0,
            "top-right": 295.0,
            "bottom-left": 290.0,
            "bottom-right": 295.0,
            "right-edge": 80.0,
        }
        sess = _session_with_peaks(peaks, n_steps=600, fwhm_steps=20.0)
        from tools import probe_simulator as ps
        ps.PHASE_A_POINTS = 21
        try:
            result = simulate_session(sess)
        finally:
            ps.PHASE_A_POINTS = 7
        # Bracket should extend down to ~80 - 2 × FWHM = 40, catching
        # the right-edge crop. Verify no peak data is lost.
        for ap_result in result.by_aperture.values():
            for v in ap_result.crop_verdicts:
                assert v.rel_loss_pct < 1.0


class TestSubSamplePeakRefinement:
    """Parabolic fit gives sub-sample resolution beyond the discrete argmax."""

    def test_peak_between_two_samples_is_refined(self):
        """A true peak at step 100.7 — phase A samples land at integer
        steps (e.g. 95, 105). Discrete argmax = 105; parabolic should
        refine toward 100.7."""
        peaks = {
            "center": 100.7,
            "top-left": 105.3,
            "top-right": 95.5,
        }
        # Need ≥ 3 valid crops to avoid the "inconclusive" gate.
        sess = _session_with_peaks(peaks, n_steps=200, fwhm_steps=20.0)
        # Force phase A to 21 points so spacing = 10 across 200 steps.
        result = simulate_phase_a(sess, n_points=21)
        assert result.centre_peak_step is not None
        # With spacing 10 and parabolic refinement on a Gaussian, the
        # error should be much smaller than 1 sample (~10 steps).
        assert abs(result.centre_peak_step - 100.7) < 2.0


def test_smoke_three_reference_sessions_results(monkeypatch):
    """If real session data is available, smoke-test that the savings
    numbers stay in the calibrated bands. Otherwise skip — these
    tests run in CI without the macro-test-output directory."""
    from pathlib import Path
    from tools.probe_simulator import DEFAULT_REFERENCE_SESSIONS, load_session

    root = Path.home() / "macro-lens-test-output"
    if not root.is_dir():
        pytest.skip("no macro-lens-test-output directory")
    # Only check if all three sessions exist.
    targets = [root / s for s in DEFAULT_REFERENCE_SESSIONS]
    if not all(t.is_dir() for t in targets):
        pytest.skip("reference sessions not present")

    expected_savings_band = {
        "2026-05-06_175727": (10, 30),   # truncated p99; 17 % at N=9, padding=1.5
        "2026-05-06_231805": (35, 55),   # clean max-FC; 44 % at N=9, padding=1.5
        "2026-05-06_200150": (30, 50),   # diopter p95; 39 % at N=9, padding=1.5
    }
    for t in targets:
        sess = load_session(t)
        result = simulate_session(sess)
        lo, hi = expected_savings_band[t.name]
        assert lo <= result.savings_pct <= hi, (
            f"{t.name}: savings {result.savings_pct:.0f}% outside "
            f"calibrated band [{lo}, {hi}]"
        )
        # And no material peak data loss. At padding=1.5 we accept up
        # to 3 % rel-loss on a single corner of a p95+ config; the
        # phase-D extension flow catches those in production.
        for ap_result in result.by_aperture.values():
            for v in ap_result.crop_verdicts:
                assert v.rel_loss_pct < 3.0, (
                    f"{t.name} {ap_result.aperture_text} {v.crop_name}: "
                    f"rel-loss {v.rel_loss_pct:.2f}%"
                )
