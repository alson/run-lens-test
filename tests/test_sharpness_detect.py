"""Tests for sharpness.detect.

Mostly numpy-on-list-of-ints, so unit tests are fast. The parabola-fit
extension estimator is exercised with synthetic clipped half-bells so
we can verify the recommended extension lands close to the true peak.
"""

from __future__ import annotations

import numpy as np
import pytest

from sharpness.detect import (
    CropStatus,
    analyze_crop,
    analyze_crops_for_aperture,
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

    def test_flat_profile_is_off_target(self) -> None:
        a = analyze_crop("top-left", [70] * 15, flagged_off_target=False)
        assert a.status == CropStatus.OFF_TARGET

    def test_empty(self) -> None:
        a = analyze_crop("center", [], flagged_off_target=False)
        assert a.status == CropStatus.EMPTY


# --- analyze_crops_for_aperture ------------------------------------------


class TestAnalyzeCropsForAperture:
    def test_runs_cross_compare_then_per_crop(self) -> None:
        crop_sizes = {
            "center": BELL,
            "left-edge": BELL,
            "top-left": [70] * len(BELL),  # off-target by flat
            "bottom-right": [int(b * 0.5) for b in BELL],  # off-target by cross
        }
        results = analyze_crops_for_aperture(crop_sizes)
        by_name = {r.crop_name: r for r in results}
        assert by_name["center"].status == CropStatus.OK
        assert by_name["left-edge"].status == CropStatus.OK
        assert by_name["top-left"].status == CropStatus.OFF_TARGET
        assert by_name["bottom-right"].status == CropStatus.OFF_TARGET

    def test_empty_input_returns_empty_list(self) -> None:
        assert analyze_crops_for_aperture({}) == []
