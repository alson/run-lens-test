"""Tests for sharpness.detect.

Mostly numpy-on-list-of-ints, so unit tests are fast. The parabola-fit
extension estimator is exercised with synthetic clipped half-bells so
we can verify the recommended extension lands close to the true peak.
"""

from __future__ import annotations

import numpy as np
import pytest

from sharpness.detect import (
    DEFAULT_DYNAMIC_RANGE_MIN,
    CropStatus,
    analyze_crop,
    analyze_crops_for_aperture,
    dynamic_range,
    estimate_peak_vertex,
    extra_steps_needed,
    is_flat,
    off_target_by_cross_compare,
    peak_step,
    smooth,
    truncation_status,
)


# Synthetic stepped bell modelled on Sigma 150mm @ 1:1 @ F8 / center
# but compressed to one shot per rail step (which is what the live
# collector produces with the rail-step-outside / aperture-inside
# architecture from DESIGN_NOTES §2.6).
BELL = [
    80, 90, 100, 110, 120, 128,
    132, 130, 124, 116, 106, 96,
    87, 80, 75, 72, 70, 69,
    68,
]


# --- smoothing / peak ----------------------------------------------------


class TestSmooth:
    def test_basic_centred_average(self) -> None:
        out = smooth([1, 2, 3, 4, 5], window=3)
        np.testing.assert_allclose(out, [2.0, 3.0, 4.0])

    def test_short_input_returns_copy(self) -> None:
        out = smooth([5, 5], window=3)
        np.testing.assert_allclose(out, [5, 5])


class TestPeakStep:
    def test_finds_centre_of_synthetic_bell(self) -> None:
        # peak at idx 6 (value 132)
        assert peak_step(BELL) == 6

    def test_monotonic_peaks_at_end(self) -> None:
        assert peak_step(list(range(10))) == 8


# --- truncation flags ----------------------------------------------------


class TestTruncationStatus:
    @pytest.mark.parametrize("idx", [0, 1])
    def test_first_two_steps_are_truncated_start(self, idx: int) -> None:
        assert truncation_status(idx, 30) == CropStatus.TRUNCATED_START

    @pytest.mark.parametrize("idx", [28, 29])
    def test_last_two_steps_are_truncated_end(self, idx: int) -> None:
        assert truncation_status(idx, 30) == CropStatus.TRUNCATED_END

    def test_idx_2_is_ok_with_default_edge_steps(self) -> None:
        # Default edge_steps=2 ⇒ idx 0,1 truncated, idx 2 fine
        assert truncation_status(2, 30) == CropStatus.OK

    def test_centre_is_ok(self) -> None:
        assert truncation_status(15, 30) == CropStatus.OK

    def test_empty(self) -> None:
        assert truncation_status(0, 0) == CropStatus.EMPTY

    def test_custom_edge_steps(self) -> None:
        # With edge_steps=4, indices 0..3 should be flagged
        for idx in range(4):
            assert truncation_status(idx, 20, edge_steps=4) == CropStatus.TRUNCATED_START
        assert truncation_status(4, 20, edge_steps=4) == CropStatus.OK


# --- flat / off-target ----------------------------------------------------


class TestIsFlat:
    def test_bell_is_not_flat(self) -> None:
        assert not is_flat(BELL)

    def test_constant_is_flat(self) -> None:
        assert is_flat([72] * 20)

    def test_low_variance_is_flat(self) -> None:
        # max/min ≈ 1.05 — well below 1.15
        assert is_flat([100, 102, 104, 102, 100, 101, 103, 105])

    def test_empty_returns_false(self) -> None:
        assert not is_flat([])


class TestDynamicRange:
    def test_bell_has_substantial_range(self) -> None:
        # BELL: max=132, min=68 → (132 - 68) / 132 ≈ 0.485
        assert dynamic_range(BELL) == pytest.approx(0.485, abs=0.01)

    def test_constant_is_zero(self) -> None:
        assert dynamic_range([72] * 10) == 0.0

    def test_empty_is_zero(self) -> None:
        assert dynamic_range([]) == 0.0

    def test_negative_max_is_zero(self) -> None:
        # Defensive: corrupt sizes shouldn't propagate as a real range.
        assert dynamic_range([0, 0, 0]) == 0.0


class TestOffTargetByCrossCompare:
    def test_flags_low_outliers_against_best(self) -> None:
        maxes = {
            "center": 130,
            "left-edge": 122,
            "top-left": 70,
            "top-edge": 71,
        }
        flagged = off_target_by_cross_compare(maxes)
        # 0.65 × 130 = 84.5; both 70 and 71 fall below
        assert flagged == {"top-left", "top-edge"}

    def test_uniform_no_flags(self) -> None:
        assert (
            off_target_by_cross_compare({"a": 100, "b": 100, "c": 100, "d": 100})
            == set()
        )

    def test_singleton(self) -> None:
        assert off_target_by_cross_compare({"a": 100}) == set()


# --- parabola-fit vertex --------------------------------------------------


class TestEstimatePeakVertex:
    def test_clean_parabola(self) -> None:
        # y = -(x-7)^2 + 100; vertex is at x=7
        xs = np.arange(15, dtype=float)
        ys = -((xs - 7) ** 2) + 100
        vertex = estimate_peak_vertex(ys.tolist())
        assert vertex == pytest.approx(7.0, abs=0.5)

    def test_clipped_half_bell_extrapolates_past_end(self) -> None:
        # Full parabola peak would be at x=30, but only sample x=0..14
        xs = np.arange(15, dtype=float)
        ys = -((xs - 30) ** 2) + 1000
        vertex = estimate_peak_vertex(ys.tolist())
        assert vertex is not None
        assert vertex > 14  # past the end
        assert 25 < vertex < 35  # close to the true peak

    def test_clipped_half_bell_extrapolates_before_start(self) -> None:
        xs = np.arange(15, dtype=float)
        ys = -((xs - (-10)) ** 2) + 1000  # peak at x = -10
        vertex = estimate_peak_vertex(ys.tolist())
        assert vertex is not None
        assert vertex < 0
        assert -15 < vertex < -5

    def test_concave_up_returns_none(self) -> None:
        # U-shape isn't a bell curve — no meaningful vertex
        ys = list(range(20, 0, -1)) + list(range(20))
        assert estimate_peak_vertex(ys) is None

    def test_too_few_points_returns_none(self) -> None:
        assert estimate_peak_vertex([1, 2]) is None


# --- extra_steps_needed ---------------------------------------------------


class TestExtraStepsNeeded:
    def test_ok_status_returns_none(self) -> None:
        assert extra_steps_needed(BELL, CropStatus.OK, len(BELL)) is None

    def test_truncated_end_with_known_peak(self) -> None:
        # Sample the rising side of a parabola that peaks at x=20
        xs = np.arange(15, dtype=float)
        ys = -((xs - 20) ** 2) + 1000
        n = len(ys)
        steps = extra_steps_needed(ys.tolist(), CropStatus.TRUNCATED_END, n)
        # True gap = 20 - 14 = 6 plus buffer of 2 = 8
        assert steps is not None
        assert 6 <= steps <= 12

    def test_truncated_start_with_known_peak(self) -> None:
        xs = np.arange(15, dtype=float)
        ys = -((xs - (-10)) ** 2) + 1000  # peak at x=-10 (10 steps before start)
        steps = extra_steps_needed(ys.tolist(), CropStatus.TRUNCATED_START, len(ys))
        # True gap = 10 plus buffer 2 ≈ 12
        assert steps is not None
        assert 9 <= steps <= 14

    def test_capped_at_series_length(self) -> None:
        # Pathological: a parabola whose vertex is enormously far away
        xs = np.arange(8, dtype=float)
        ys = -((xs - 1000) ** 2) + 1_000_000
        steps = extra_steps_needed(ys.tolist(), CropStatus.TRUNCATED_END, len(ys))
        assert steps is not None
        assert steps == len(ys)  # capped

    def test_minimum_buffer_when_vertex_close(self) -> None:
        # Sample a parabola that peaks just past the end (gap < buffer)
        xs = np.arange(15, dtype=float)
        ys = -((xs - 14.5) ** 2) + 100
        steps = extra_steps_needed(ys.tolist(), CropStatus.TRUNCATED_END, len(ys))
        # Even with a sub-step gap, we should recommend at least edge_steps
        assert steps is not None
        assert steps >= 2

    def test_falls_back_when_no_vertex(self) -> None:
        # Concave-up data — no meaningful vertex; estimator falls back
        # to half the range.
        ys = list(range(20, 0, -1)) + list(range(20))
        steps = extra_steps_needed(ys, CropStatus.TRUNCATED_START, len(ys))
        assert steps == len(ys) // 2


# --- analyze_crop ---------------------------------------------------------


class TestAnalyzeCrop:
    def test_bell_curve_is_ok(self) -> None:
        a = analyze_crop("center", BELL, flagged_off_target=False)
        assert a.status == CropStatus.OK
        assert a.peak_step is not None
        assert a.extra_steps_needed is None
        assert a.max_size == max(BELL)
        assert a.valid is True
        assert a.dynamic_range > DEFAULT_DYNAMIC_RANGE_MIN

    def test_truncated_end_recommends_extension(self) -> None:
        rising = [int(50 + i * 5) for i in range(15)]  # 50, 55, ..., 120
        a = analyze_crop("center", rising, flagged_off_target=False)
        assert a.status == CropStatus.TRUNCATED_END
        assert a.extra_steps_needed is not None
        assert a.extra_steps_needed >= 2

    def test_truncated_start_recommends_extension(self) -> None:
        falling = [int(120 - i * 5) for i in range(15)]
        a = analyze_crop("center", falling, flagged_off_target=False)
        assert a.status == CropStatus.TRUNCATED_START
        assert a.extra_steps_needed is not None

    def test_off_target_flag_short_circuits_analysis(self) -> None:
        a = analyze_crop("top-left", BELL, flagged_off_target=True)
        assert a.status == CropStatus.OFF_TARGET
        assert a.peak_step is None
        assert a.extra_steps_needed is None
        # Cross-compare flagged (peak weaker than peers), but BELL has
        # plenty of dynamic range, so the crop stays valid=True so the
        # advisor can distinguish it from "fell off the wafer".
        assert a.valid is True

    def test_flat_profile_is_off_target_and_invalid(self) -> None:
        a = analyze_crop("top-left", [70] * 15, flagged_off_target=False)
        assert a.status == CropStatus.OFF_TARGET
        assert a.valid is False
        assert a.dynamic_range == 0.0

    def test_low_dynamic_range_is_invalid(self) -> None:
        # Window sits on low-contrast background — JPEG sizes wobble a
        # tiny bit due to noise but well below the 25% floor.
        # max=104, min=100 → dynamic range ≈ 3.8%.
        noisy_off_wafer = [100, 101, 102, 103, 104, 103, 102, 101, 100, 102]
        a = analyze_crop("top-left", noisy_off_wafer, flagged_off_target=False)
        assert a.valid is False
        assert a.status == CropStatus.OFF_TARGET
        assert 0.0 < a.dynamic_range < DEFAULT_DYNAMIC_RANGE_MIN
        # The dynamic-range gate should *not* try to find a peak or
        # recommend an extension on a series we've declared invalid.
        assert a.peak_step is None
        assert a.extra_steps_needed is None

    def test_dynamic_range_threshold_can_be_overridden(self) -> None:
        # Same flat-ish series as above. The OFF_TARGET floor (0.05 by
        # default) is what gates validity; lowering it lets the crop
        # pass through to peak / truncation detection. With 0.05 default
        # and dr ≈ 0.038 the crop is OFF_TARGET; with 0.01 it gets
        # classified by its peak (here it sits below DYNAMIC_RANGE_MIN
        # too, so the in-range peak becomes LOW_SIGNAL).
        noisy = [100, 101, 102, 103, 104, 103, 102, 101, 100, 102]
        a = analyze_crop(
            "top-left",
            noisy,
            flagged_off_target=False,
            off_target_range=0.01,
        )
        assert a.valid is True

    def test_empty(self) -> None:
        a = analyze_crop("center", [], flagged_off_target=False)
        assert a.status == CropStatus.EMPTY
        assert a.valid is False


# --- LOW_SIGNAL classifier (three-region threshold) ----------------------


def _bell_at_scale(min_v: int, max_v: int, n: int = 19) -> list[int]:
    """Build a bell-shaped series with the exact requested min and max.

    Triangular profile so the peak lands at a deterministic centre
    index — sufficient for status classification tests without
    smearing into truncation edges.
    """
    mid = (n - 1) // 2
    out: list[int] = []
    for i in range(n):
        # Distance from centre, normalised to [0, 1].
        d = abs(i - mid) / mid
        out.append(int(round(max_v - (max_v - min_v) * d)))
    # Pin the exact extrema in case rounding nudged them.
    out[mid] = max_v
    out[0] = min_v
    out[-1] = min_v
    return out


class TestThreeRegionClassifier:
    """The dynamic-range classifier has three regions:

    * dr < 0.05  → OFF_TARGET (window off the wafer)
    * 0.05 ≤ dr < 0.25 → LOW_SIGNAL (real but weak bell)
    * dr ≥ 0.25 → OK (high-signal bell)
    """

    def test_dr_below_off_target_floor_is_off_target(self) -> None:
        # dr ≈ 0.03 — below the 0.05 floor.
        series = _bell_at_scale(min_v=100, max_v=103)
        assert (max(series) - min(series)) / max(series) < 0.05
        a = analyze_crop("top-left", series, flagged_off_target=False)
        assert a.status == CropStatus.OFF_TARGET
        assert a.valid is False
        # OFF_TARGET short-circuits — no peak detection runs.
        assert a.peak_step is None
        assert a.extra_steps_needed is None

    def test_dr_in_low_signal_band_yields_low_signal(self) -> None:
        # dr ≈ 0.10 — between OFF_TARGET (0.05) and OK (0.25).
        series = _bell_at_scale(min_v=100, max_v=111)
        dr = (max(series) - min(series)) / max(series)
        assert 0.05 <= dr < 0.25
        a = analyze_crop("top-left", series, flagged_off_target=False)
        assert a.status == CropStatus.LOW_SIGNAL
        # Peak detection ran — the centred bell is in-range.
        assert a.peak_step is not None
        # LOW_SIGNAL crops are *valid* contributors to the verdict.
        assert a.valid is True
        assert a.dynamic_range == pytest.approx(dr, abs=0.005)

    def test_dr_above_ok_floor_yields_ok(self) -> None:
        # dr ≈ 0.30 — at or above the 0.25 OK ceiling.
        series = _bell_at_scale(min_v=70, max_v=100)
        dr = (max(series) - min(series)) / max(series)
        assert dr >= 0.25
        a = analyze_crop("center", series, flagged_off_target=False)
        assert a.status == CropStatus.OK
        assert a.valid is True

    def test_truncation_takes_precedence_over_low_signal(self) -> None:
        # Rising half-bell in the LOW_SIGNAL dr band. Should be reported
        # as TRUNCATED_END (a truncated weak peak still needs more rail),
        # not LOW_SIGNAL.
        rising = [100 + i for i in range(15)]   # max=114, min=100, dr≈0.123
        dr = (max(rising) - min(rising)) / max(rising)
        assert 0.05 <= dr < 0.25
        a = analyze_crop("top-left", rising, flagged_off_target=False)
        assert a.status == CropStatus.TRUNCATED_END
        # Truncation also wins the extra_steps projection — the soft
        # bell is still a peak the runner can chase by extending rail.
        assert a.extra_steps_needed is not None
        assert a.extra_steps_needed >= 2

    def test_truncated_start_in_low_signal_band(self) -> None:
        falling = [114 - i for i in range(15)]   # max=114, min=100
        dr = (max(falling) - min(falling)) / max(falling)
        assert 0.05 <= dr < 0.25
        a = analyze_crop("bottom-left", falling, flagged_off_target=False)
        assert a.status == CropStatus.TRUNCATED_START

    def test_off_target_floor_can_be_overridden(self) -> None:
        # Caller can loosen the off-target floor if they want every
        # series classified by its peak (e.g. a soft-lens campaign).
        series = _bell_at_scale(min_v=100, max_v=103)  # dr ≈ 0.03
        a = analyze_crop(
            "top-left", series,
            flagged_off_target=False, off_target_range=0.01,
        )
        # Now the dr lands in the LOW_SIGNAL band (the OK floor is
        # still the default 0.25, well above 0.03).
        assert a.valid is True
        assert a.status == CropStatus.LOW_SIGNAL


# --- analyze_crops_for_aperture ------------------------------------------


class TestAnalyzeCropsForAperture:
    def test_runs_cross_compare_then_per_crop(self) -> None:
        # Cross-compare flags any valid crop whose peak < factor × best.
        # With the default factor of 0.25, "bottom-right" needs to peak
        # well below 25 % of BELL's max to be flagged — 0.20 lands at
        # peak ratio ≈ 0.20, below 0.25.
        crop_sizes = {
            "center": BELL,
            "left-edge": BELL,
            "top-left": [70] * len(BELL),  # off-target by flat
            "bottom-right": [int(b * 0.20) for b in BELL],  # off-target by cross
        }
        results = analyze_crops_for_aperture(crop_sizes)
        by_name = {r.crop_name: r for r in results}
        assert by_name["center"].status == CropStatus.OK
        assert by_name["left-edge"].status == CropStatus.OK
        assert by_name["top-left"].status == CropStatus.OFF_TARGET
        assert by_name["bottom-right"].status == CropStatus.OFF_TARGET

    def test_empty_input_returns_empty_list(self) -> None:
        assert analyze_crops_for_aperture({}) == []
