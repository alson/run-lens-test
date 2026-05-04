"""Tests for the range-check stub.

The real detector is the user's later contribution. These tests
pin down the public contract that the runner depends on:

  * ``evaluate(...)`` returns a :class:`RangeResult` with sensible
    defaults so the happy path through the runner is unchanged.
  * Disabling via ``RANGE_CHECK_ENABLED = False`` always reports ok.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from exposure.range_check import (
    ApertureCurve,
    RangeChecker,
    RangeResult,
)


def _config(*, enabled: bool = True):
    return SimpleNamespace(RANGE_CHECK_ENABLED=enabled)


class TestStubReturnsOk:
    def test_first_call_reports_ok(self, tmp_path: Path) -> None:
        checker = RangeChecker(_config())
        result = checker.evaluate(
            lens_dir=tmp_path,
            end_mm=20.0,
            step_size_mm=1.0,
            apertures=["f/4", "f/8"],
        )
        assert isinstance(result, RangeResult)
        assert result.ok is True
        assert result.needs_extension is False
        assert result.peak_side == "centered"

    def test_with_probe_data_still_ok(self, tmp_path: Path) -> None:
        checker = RangeChecker(_config())
        probe = [
            ApertureCurve(
                aperture="f/4",
                step_indices=[0, 1],
                rail_positions_mm=[21.0, 23.0],
                sharpness_values=[0.5, 0.6],
            ),
        ]
        result = checker.evaluate(
            lens_dir=tmp_path,
            end_mm=20.0,
            step_size_mm=1.0,
            apertures=["f/4"],
            probe_data=probe,
        )
        assert result.ok is True


class TestDisabledByConfig:
    def test_disabled_returns_ok(self, tmp_path: Path) -> None:
        checker = RangeChecker(_config(enabled=False))
        result = checker.evaluate(
            lens_dir=tmp_path,
            end_mm=20.0,
            step_size_mm=1.0,
            apertures=["f/4"],
        )
        assert result.ok is True
        assert "disabled" in result.notes


class TestApertureCurve:
    def test_round_trip(self) -> None:
        curve = ApertureCurve(
            aperture="f/8",
            step_indices=[0, 1, 2],
            rail_positions_mm=[0.0, 1.0, 2.0],
            sharpness_values=[0.3, 0.7, 0.4],
        )
        assert curve.aperture == "f/8"
        assert len(curve.sharpness_values) == 3


class TestPeakSideAccepts:
    """The contract permits four peak_side values; ensure the dataclass
    holds each unchanged. No semantic checks — the runner is responsible
    for reacting to each value (§8.3.3 in the plan)."""

    def test_near(self) -> None:
        result = RangeResult(ok=False, peak_side="near", needs_extension=True)
        assert result.peak_side == "near"

    def test_far(self) -> None:
        result = RangeResult(ok=False, peak_side="far", needs_extension=True)
        assert result.peak_side == "far"

    def test_both(self) -> None:
        result = RangeResult(ok=False, peak_side="both", needs_extension=True)
        assert result.peak_side == "both"

    def test_centered(self) -> None:
        result = RangeResult(ok=True, peak_side="centered", needs_extension=False)
        assert result.peak_side == "centered"
