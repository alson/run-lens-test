"""Tests for the pure-function probe math in :mod:`session.probe`.

Mirrors the harness used by ``tools/probe_simulator.py`` so changes
to one stay in sync with the other. The runner-side integration is
covered by ``tests/test_runner_probe.py`` with fakes; this module
sticks to deterministic Gaussian synthetics.
"""

from __future__ import annotations

import numpy as np

from session.probe import PhaseAResult, phase_a_peaks, phase_c_window_mm


def _gaussian_series(
    n_points: int,
    peak_idx: float,
    fwhm: float,
    *,
    floor: int = 100_000,
    amplitude: int = 200_000,
) -> list[int]:
    sigma = fwhm / 2.355
    xs = np.arange(n_points)
    ys = floor + amplitude * np.exp(-0.5 * ((xs - peak_idx) / sigma) ** 2)
    return [int(round(v)) for v in ys]


class TestPhaseAPeaks:
    def test_recovers_peak_within_subsample_resolution(self):
        """Peak at sample index 3.7; parabolic fit should land near 3.7."""
        sizes = _gaussian_series(7, peak_idx=3.7, fwhm=2.5)
        result = phase_a_peaks({
            "center": sizes,
            "top-left": _gaussian_series(7, peak_idx=3.0, fwhm=2.5),
            "top-right": _gaussian_series(7, peak_idx=4.0, fwhm=2.5),
        })
        assert result.centre_peak_sample is not None
        assert abs(result.centre_peak_sample - 3.7) < 0.5

    def test_inconclusive_when_too_few_valid_crops(self):
        """Fewer than 3 crops with usable bells → inconclusive."""
        flat = [100_000] * 7
        sizes = _gaussian_series(7, peak_idx=3, fwhm=2.5)
        result = phase_a_peaks({
            "center": sizes,
            "top-left": flat,
            "top-right": flat,
        })
        assert result.inconclusive

    def test_drops_low_dynamic_range_crops(self):
        """A nearly-flat series (dr < 0.25) is dropped; remaining crops
        still produce a usable spread."""
        sizes_full = _gaussian_series(7, peak_idx=3, fwhm=2.5)
        # 5% dynamic range — well below the 0.25 floor.
        sizes_flat = [100_000 + 50 * i for i in range(7)]
        result = phase_a_peaks({
            "center": sizes_full,
            "top-left": _gaussian_series(7, peak_idx=2.5, fwhm=2.5),
            "bottom-right": _gaussian_series(7, peak_idx=3.5, fwhm=2.5),
            "left-edge": sizes_flat,
        })
        assert "left-edge" not in result.crop_peaks
        assert not result.inconclusive
        assert result.min_peak_sample is not None
        assert result.max_peak_sample is not None
        assert result.max_peak_sample > result.min_peak_sample

    def test_corner_spread_is_max_minus_min(self):
        result = phase_a_peaks({
            "center": _gaussian_series(7, peak_idx=3.0, fwhm=2.5),
            "top-left": _gaussian_series(7, peak_idx=1.5, fwhm=2.5),
            "top-right": _gaussian_series(7, peak_idx=4.5, fwhm=2.5),
        })
        assert not result.inconclusive
        assert result.min_peak_sample is not None
        assert result.max_peak_sample is not None
        # Spread should approximate the input (max-min = 3.0).
        assert abs(result.corner_spread_samples - 3.0) < 1.0

    def test_returns_min_max_separately_from_centre(self):
        """When corners all sit on one side of centre, min/max should
        bracket the corners, not be symmetric around centre."""
        result = phase_a_peaks({
            "center": _gaussian_series(7, peak_idx=5.5, fwhm=2.5),
            "top-left": _gaussian_series(7, peak_idx=1.5, fwhm=2.5),
            "top-right": _gaussian_series(7, peak_idx=2.0, fwhm=2.5),
            "bottom-left": _gaussian_series(7, peak_idx=2.5, fwhm=2.5),
        })
        assert result.centre_peak_sample is not None
        assert result.min_peak_sample is not None
        assert result.max_peak_sample is not None
        # Bracket reaches both the leftmost corner (~1.5) and centre (~5.5).
        assert result.min_peak_sample < 2.5
        assert result.max_peak_sample > 5.0

    def test_empty_input_is_inconclusive(self):
        result = phase_a_peaks({})
        assert result.inconclusive
        assert result.centre_peak_sample is None
        assert result.min_peak_sample is None
        assert result.max_peak_sample is None


class TestPhaseCWindowMm:
    def _build_result(self, min_s: float, max_s: float) -> PhaseAResult:
        return PhaseAResult(
            crop_peaks={"center": (min_s + max_s) / 2,
                        "left-edge": min_s,
                        "right-edge": max_s},
            centre_peak_sample=(min_s + max_s) / 2,
            min_peak_sample=min_s,
            max_peak_sample=max_s,
            inconclusive=False,
        )

    def test_forward_jog_window(self):
        """7 probes across [0, 2.0] mm, peaks at sample 1 and 4 →
        peaks at 0.333 mm and 1.333 mm in rail-mm. Padding 1.5 ×
        FWHM on each side."""
        result = self._build_result(min_s=1.0, max_s=4.0)
        window = phase_c_window_mm(
            result,
            phase_a_start_mm=0.0,
            phase_a_end_mm=2.0,
            n_probes=7,
            fwhm_mm=0.10,
            padding_x_fwhm=1.5,
        )
        assert window is not None
        lo, hi = window
        # spacing = 2.0 / 6 ≈ 0.333; lo_peak = 0 + 1 * 0.333 = 0.333
        # hi_peak = 0 + 4 * 0.333 = 1.333; pad = 0.15 each side.
        assert abs(lo - (0.333 - 0.15)) < 0.01
        assert abs(hi - (1.333 + 0.15)) < 0.01
        assert lo < hi

    def test_backward_jog_window(self):
        """User jogged backward: end_mm = -2.0. Sample 0 is at 0 mm,
        sample 6 is at -2.0 mm. min_peak in sample space (smallest
        sample idx) is at the *largest* rail-mm; bracket inverts."""
        result = self._build_result(min_s=1.0, max_s=4.0)
        window = phase_c_window_mm(
            result,
            phase_a_start_mm=0.0,
            phase_a_end_mm=-2.0,
            n_probes=7,
            fwhm_mm=0.10,
            padding_x_fwhm=1.5,
        )
        assert window is not None
        lo, hi = window
        # The bracket should still be a valid (lo, hi) where lo ≤ hi
        # in signed rail-mm. For backward jog, both bounds are negative.
        assert lo <= hi
        assert lo < 0
        assert hi < 0

    def test_inconclusive_returns_none(self):
        result = PhaseAResult(inconclusive=True)
        assert phase_c_window_mm(
            result,
            phase_a_start_mm=0.0,
            phase_a_end_mm=2.0,
            n_probes=7,
            fwhm_mm=0.10,
        ) is None

    def test_too_few_probes_returns_none(self):
        result = self._build_result(min_s=1.0, max_s=4.0)
        assert phase_c_window_mm(
            result,
            phase_a_start_mm=0.0,
            phase_a_end_mm=2.0,
            n_probes=1,
            fwhm_mm=0.10,
        ) is None

    def test_padding_widens_window(self):
        """Doubling padding doubles the slack on each side."""
        result = self._build_result(min_s=2.0, max_s=4.0)
        window_a = phase_c_window_mm(
            result, phase_a_start_mm=0.0, phase_a_end_mm=2.0,
            n_probes=7, fwhm_mm=0.10, padding_x_fwhm=1.0,
        )
        window_b = phase_c_window_mm(
            result, phase_a_start_mm=0.0, phase_a_end_mm=2.0,
            n_probes=7, fwhm_mm=0.10, padding_x_fwhm=2.0,
        )
        assert window_a is not None and window_b is not None
        span_a = window_a[1] - window_a[0]
        span_b = window_b[1] - window_b[0]
        # Bigger padding → longer span. Each unit of padding × fwhm
        # adds 2 × 0.10 = 0.20 mm to the span (one on each side).
        assert abs((span_b - span_a) - 0.20) < 0.01
