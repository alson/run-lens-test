"""Tests for sharpness.range_check.SharpnessRangeChecker.

The cropper is bypassed entirely — tests construct a synthetic
:class:`CollectedDataset` and feed it to the adapter via a
``FakeCollector`` that mimics the LiveCollector surface.

Synthetic JPEG-size series let us assert what each truncation pattern
produces:

* a complete bell on every aperture -> verdict OK -> ``RangeResult``
  ``ok=True``, ``peak_side='centered'``, ``needs_extension=False``.
* a strictly-rising series (peak is past the rail's far end) ->
  ``Direction.EXTEND_FORWARD`` -> ``peak_side='far'``.
* a strictly-falling series (peak is before the rail's start) ->
  ``Direction.EXTEND_BACKWARD`` -> ``peak_side='near'``.
* mixed across apertures -> ``peak_side='both'``.
* on the second call, if probes complete the bell -> verdict OK
  again -> ``peak_in_range_after_probe=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from sharpness.collector import CollectedDataset, ShotSample
from sharpness.crops import CROP_NAMES
from sharpness.range_check import SharpnessRangeChecker, aperture_to_float


# ---- Fake collector --------------------------------------------------------


@dataclass
class FakeCollector:
    """Stands in for sharpness.collector.LiveCollector. Tests assemble
    the dataset directly and the fake's ``dataset`` property serves it."""

    samples: list[ShotSample] = field(default_factory=list)
    waits: int = 0

    def wait_for_pending(self) -> None:
        self.waits += 1

    @property
    def dataset(self) -> CollectedDataset:
        # Match LiveCollector.dataset's snapshot semantics — return a
        # copy so callers can't mutate the source.
        return CollectedDataset(samples=list(self.samples))


# ---- Synthetic JPEG-size series helpers ------------------------------------


def _bell(n: int, peak_step: int, *, base: int = 1000, peak: int = 5000) -> list[int]:
    """Symmetric triangular bell across n steps, peaking at peak_step."""
    out = []
    half_width = max(1, max(peak_step, n - 1 - peak_step))
    for i in range(n):
        d = abs(i - peak_step) / half_width
        out.append(int(base + (peak - base) * (1 - d)))
    return out


def _rising(n: int, *, base: int = 1000, top: int = 5000) -> list[int]:
    """Strictly increasing — peak is past the end of the captured range."""
    return [base + (top - base) * i // max(1, n - 1) for i in range(n)]


def _falling(n: int, *, base: int = 1000, top: int = 5000) -> list[int]:
    return list(reversed(_rising(n, base=base, top=top)))


def _samples_for_aperture(
    aperture: float,
    sizes: list[int],
    *,
    rail_step_start: int = 0,
    crops: tuple[str, ...] = CROP_NAMES,
) -> list[ShotSample]:
    """Build N ShotSamples each carrying the same ``sizes[i]`` value
    on every crop. Real data has variation across crops, but for
    contract testing equal sizes are sufficient — the detector treats
    each crop independently."""
    out = []
    for i, size in enumerate(sizes):
        out.append(ShotSample(
            rail_step=rail_step_start + i,
            aperture=aperture,
            sizes={name: size for name in crops},
        ))
    return out


@pytest.fixture
def cfg():
    return SimpleNamespace()


# ---- Aperture-to-float helper ---------------------------------------------


class TestApertureToFloat:
    @pytest.mark.parametrize("text,expected", [
        ("f/4", 4.0),
        ("f/2.8", 2.8),
        ("f/16", 16.0),
        ("F/5.6 ", 5.6),
    ])
    def test_parses(self, text: str, expected: float) -> None:
        assert aperture_to_float(text) == pytest.approx(expected)


# ---- First call: aggregate per-aperture into RangeResult ------------------


class TestFirstCall:
    def test_centered_bell_means_ok(self, cfg) -> None:
        dataset = FakeCollector(
            samples=_samples_for_aperture(8.0, _bell(7, peak_step=3))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert result.ok is True
        assert result.peak_side == "centered"
        assert result.needs_extension is False
        # per_aperture is populated even on the OK path so the runner
        # can persist the curve.
        assert len(result.per_aperture) == 1
        assert result.per_aperture[0].aperture == "f/8"
        assert len(result.per_aperture[0].sharpness_values) == 7

    def test_rising_series_means_far(self, cfg) -> None:
        # Strictly-rising = peak past the far end -> EXTEND_FORWARD.
        dataset = FakeCollector(
            samples=_samples_for_aperture(8.0, _rising(7))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert result.ok is False
        assert result.peak_side == "far"
        assert result.needs_extension is True

    def test_falling_series_means_near(self, cfg) -> None:
        dataset = FakeCollector(
            samples=_samples_for_aperture(8.0, _falling(7))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert result.peak_side == "near"

    def test_mixed_apertures_mean_both(self, cfg) -> None:
        # f/4 wants forward extension, f/16 wants backward → BOTH.
        samples = (
            _samples_for_aperture(4.0, _rising(7))
            + _samples_for_aperture(16.0, _falling(7))
        )
        dataset = FakeCollector(samples=samples)
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/4", "f/16"],
        )
        assert result.peak_side == "both"
        assert result.needs_extension is True

    def test_aperture_curve_uses_signed_step_indices(self, cfg) -> None:
        # Mix near (-1) and main (0..2) shots to confirm signed
        # rail_step values come through unmodified into ApertureCurve.
        samples = _samples_for_aperture(
            8.0, sizes=[100, 200, 300, 400], rail_step_start=-1,
        )
        dataset = FakeCollector(samples=samples)
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=2.0, apertures=["f/8"],
        )
        curve = result.per_aperture[0]
        assert curve.step_indices == [-1, 0, 1, 2]
        assert curve.rail_positions_mm == pytest.approx(
            [-2.0, 0.0, 2.0, 4.0]
        )
        # Sizes come from the canonical center crop.
        assert curve.sharpness_values == [100.0, 200.0, 300.0, 400.0]

    def test_evaluate_drains_pending_crops(self, cfg) -> None:
        dataset = FakeCollector(
            samples=_samples_for_aperture(8.0, _bell(5, peak_step=2))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        checker.evaluate(
            lens_dir=Path("."), end_mm=4.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        # Adapter calls wait_for_pending() exactly once per evaluate.
        assert dataset.waits == 1


class TestUnsetCollector:
    def test_evaluate_without_collector_raises(self, cfg) -> None:
        checker = SharpnessRangeChecker(cfg, collector=None)
        with pytest.raises(RuntimeError, match="collector"):
            checker.evaluate(
                lens_dir=Path("."), end_mm=4.0,
                step_size_mm=1.0, apertures=["f/8"],
            )


# ---- Second call: peak_in_range_after_probe ------------------------------


class TestSecondCall:
    def test_completed_bell_means_in_range(self, cfg) -> None:
        # First call shows a rising curve (only the start of the bell
        # captured). Probes added at rail_step 5,6 complete the bell;
        # second call sees the merged dataset and reports OK.
        first_dataset = FakeCollector(
            samples=_samples_for_aperture(8.0, [1000, 2000, 3000, 4000, 5000])
        )
        checker = SharpnessRangeChecker(cfg, collector=first_dataset)
        first = checker.evaluate(
            lens_dir=Path("."), end_mm=4.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert first.peak_side == "far"

        # Now extend the dataset with two probes at rail_step 5, 6
        # whose sizes complete the bell (peak ~5500 at step 6).
        first_dataset.samples.extend(
            _samples_for_aperture(
                8.0, sizes=[5500, 4000], rail_step_start=5,
            )
        )
        second = checker.evaluate(
            lens_dir=Path("."), end_mm=4.0,
            step_size_mm=1.0, apertures=["f/8"],
            probe_data=[],  # adapter ignores probe_data; data is in the
                            # dataset already (runner streamed probes
                            # through the same collector)
        )
        assert second.ok is True
        assert second.peak_in_range_after_probe is True

    def test_probe_did_not_bracket(self, cfg) -> None:
        # First call: rising. Probe data leaves it still rising — still
        # truncated, still NEEDS_RESHOOT, peak_in_range_after_probe stays
        # False.
        first_dataset = FakeCollector(
            samples=_samples_for_aperture(8.0, _rising(5))
        )
        checker = SharpnessRangeChecker(cfg, collector=first_dataset)
        checker.evaluate(
            lens_dir=Path("."), end_mm=4.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        # Probes at rail_step 5,6 keep climbing.
        first_dataset.samples.extend(
            _samples_for_aperture(8.0, sizes=[6000, 7000], rail_step_start=5)
        )
        second = checker.evaluate(
            lens_dir=Path("."), end_mm=4.0,
            step_size_mm=1.0, apertures=["f/8"],
            probe_data=[],
        )
        assert second.peak_in_range_after_probe is False


# ---- Notes -----------------------------------------------------------------


class TestNotes:
    def test_lists_per_aperture_verdicts(self, cfg) -> None:
        samples = (
            _samples_for_aperture(4.0, _bell(7, peak_step=3))
            + _samples_for_aperture(8.0, _rising(7))
        )
        dataset = FakeCollector(samples=samples)
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/4", "f/8"],
        )
        assert "f/4" in result.notes
        assert "f/8" in result.notes
        assert "OK" in result.notes
        assert "NEEDS_RESHOOT" in result.notes
