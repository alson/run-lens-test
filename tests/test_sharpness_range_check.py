"""Tests for sharpness.range_check.SharpnessRangeChecker.

The cropper is bypassed entirely — tests construct a synthetic
:class:`CollectedDataset` and feed it to the adapter via a
``FakeCollector`` that mimics the LiveCollector surface.

Synthetic JPEG-size series let us assert what each truncation pattern
produces:

* a complete bell on every aperture -> verdict OK -> ``RangeResult``
  ``ok=True``, ``peak_side='centered'``, ``needs_extension=False``.
* a strictly-rising series (peak is past the rail's far end) ->
  ``Direction.EXTEND_FORWARD`` -> ``peak_side='end_side'``.
* a strictly-falling series (peak is before the rail's start) ->
  ``Direction.EXTEND_BACKWARD`` -> ``peak_side='start_side'``.
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


def _samples_with_per_crop_series(
    aperture: float,
    per_crop_sizes: dict[str, list[int]],
    *,
    rail_step_start: int = 0,
) -> list[ShotSample]:
    """Build samples where each crop has its own size series. All
    series must have the same length; missing crops are filled with
    a constant flat value high enough to keep their dynamic range
    above the validity floor unless the test deliberately omits them.
    """
    n = len(next(iter(per_crop_sizes.values())))
    for name, series in per_crop_sizes.items():
        if len(series) != n:
            raise ValueError(f"crop {name} has length {len(series)}, expected {n}")
    out = []
    for i in range(n):
        sizes = {name: series[i] for name, series in per_crop_sizes.items()}
        out.append(ShotSample(
            rail_step=rail_step_start + i,
            aperture=aperture,
            sizes=sizes,
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
        assert result.peak_side == "end_side"
        assert result.needs_extension is True

    def test_all_flat_yields_all_off_target(self, cfg) -> None:
        # Every crop's series is flat (max/min < FLAT_RATIO) at every
        # aperture. The aggregated RangeResult must surface
        # all_off_target=True so the runner can prompt for re-jog
        # instead of silently accepting a meaningless OK.
        flat = [1000] * 7
        samples = []
        for ap in (4.0, 8.0, 16.0):
            samples.extend(_samples_for_aperture(ap, flat))
        dataset = FakeCollector(samples=samples)
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/4", "f/8", "f/16"],
        )
        assert result.all_off_target is True
        # ok stays False (the data is *not* good); no extension is
        # warranted since extending wouldn't add signal.
        assert result.ok is False
        assert result.needs_extension is False

    def test_falling_series_means_near(self, cfg) -> None:
        dataset = FakeCollector(
            samples=_samples_for_aperture(8.0, _falling(7))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert result.peak_side == "start_side"

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


class TestMultiCropFieldCurvature:
    """Per-crop verdicts disagree within a single aperture (field
    curvature): the adapter must surface that at the
    :class:`RangeResult` level so the runner extends bilaterally."""

    def test_corners_disagree_yields_both(self, cfg) -> None:
        # Centre is OK, one corner truncates forward, another backward.
        n = 11
        bell = _bell(n, peak_step=5)
        rising = _rising(n)   # peak past the far end
        falling = _falling(n)  # peak before the start
        # Build a full crop dict so all 17 crops are present and well
        # above the validity floor; the three "field-curvature" crops
        # diverge from the bell.
        per_crop = {name: list(bell) for name in CROP_NAMES}
        per_crop["top-left"] = rising
        per_crop["bottom-right"] = falling
        samples = _samples_with_per_crop_series(8.0, per_crop)
        checker = SharpnessRangeChecker(cfg, collector=FakeCollector(samples=samples))
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        # A single aperture with mixed-direction valid crops collapses
        # to Direction.BOTH at the advisor level; the adapter maps
        # that straight onto peak_side="both".
        assert result.peak_side == "both"
        assert result.ok is False
        assert result.needs_extension is True

    def test_invalid_corners_dont_perturb_centered_verdict(self, cfg) -> None:
        # Centre + edges are clean bells; the four corner crops fall
        # off the wafer (low dynamic range). The aperture should land
        # on OK / centered — invalid crops must not influence the
        # verdict, even when their constant series would otherwise
        # look "non-truncated peak in range" purely numerically.
        n = 11
        bell = _bell(n, peak_step=5)
        # Off-wafer noise: dynamic range ≈ 4%, well below 25% floor.
        noise = [100, 101, 102, 103, 104, 103, 102, 101, 100, 102, 100]
        per_crop = {name: list(bell) for name in CROP_NAMES}
        for corner in ("top-left", "top-right", "bottom-left", "bottom-right"):
            per_crop[corner] = list(noise)
        samples = _samples_with_per_crop_series(8.0, per_crop)
        checker = SharpnessRangeChecker(cfg, collector=FakeCollector(samples=samples))
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert result.ok is True
        assert result.peak_side == "centered"

    def test_all_crops_invalid_marks_all_off_target(self, cfg) -> None:
        # No crop has enough dynamic range — runner should treat as
        # "no signal at all", not "extend the rail".
        noise = [100, 101, 102, 103, 104, 103, 102, 101, 100, 102, 100]
        per_crop = {name: list(noise) for name in CROP_NAMES}
        samples = _samples_with_per_crop_series(8.0, per_crop)
        checker = SharpnessRangeChecker(cfg, collector=FakeCollector(samples=samples))
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert result.all_off_target is True
        assert result.ok is False
        assert result.needs_extension is False

    def test_all_low_signal_aperture_yields_ok_centered(self, cfg) -> None:
        # Soft-lens regime: every crop has a real but weak bell with dr
        # in the LOW_SIGNAL band (0.05–0.25). The advisor classifies
        # them LOW_SIGNAL, the aggregator treats those like OK for the
        # verdict, and the range-checker therefore reports
        # peak_side='centered' / ok=True. This is the Fix-3 interaction
        # path: LOW_SIGNAL crops do not trigger the "centered but not
        # OK" off-target exit.
        n = 11
        # Bell with min=100 max=111 — dr ≈ 0.099, in the LOW_SIGNAL band.
        soft_bell = _bell(n, peak_step=5, base=100, peak=111)
        per_crop = {name: list(soft_bell) for name in CROP_NAMES}
        samples = _samples_with_per_crop_series(8.0, per_crop)
        checker = SharpnessRangeChecker(cfg, collector=FakeCollector(samples=samples))
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=1.0, apertures=["f/8"],
        )
        assert result.ok is True
        assert result.peak_side == "centered"
        assert result.needs_extension is False
        assert result.all_off_target is False
        # Notes call out the LOW_SIGNAL count so the operator can spot
        # the soft-rig regime at a glance.
        assert "LOW_SIGNAL" in result.notes


class TestSubsampleFactorWiring:
    """Subsample factors injected via :meth:`set_subsample_factors`
    must reach the advisor and scale ``extra_steps_max`` into rail-step
    units. Without this the runner extends by sample-index units at
    sparse apertures and the reshoot probes land off-grid (the f/16
    infinite-loop bug)."""

    def test_sparse_aperture_extra_steps_max_in_rail_steps(self, cfg) -> None:
        # f/16 strictly rising — TRUNCATED_END. With stride=4 the
        # advisor's projection must be at least 2 sample-strides past
        # the projected vertex, expressed in rail steps. Without the
        # wiring fix, extra_steps_max would be the raw sample-index
        # count (~2), and the runner's probes at rail steps not
        # divisible by 4 would never fire for f/16.
        dataset = FakeCollector(
            samples=_samples_for_aperture(16.0, _rising(10))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        checker.set_subsample_factors({16.0: 4})
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=0.05, apertures=["f/16"],
        )
        assert result.peak_side == "end_side"
        assert result.needs_extension is True
        # At stride 4, the floor is at least 2 sample-strides past
        # the projected vertex — 2 × 4 = 8 rail steps minimum.
        assert result.extra_steps_max >= 2 * 4

    def test_dense_aperture_unaffected_by_wiring(self, cfg) -> None:
        # f/4 stride=1: extra_steps_max stays in rail-step units (same
        # as legacy behaviour). The new floor adds a few extra rail
        # steps but doesn't introduce a 4× jump.
        dataset = FakeCollector(
            samples=_samples_for_aperture(4.0, _rising(10))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        checker.set_subsample_factors({4.0: 1})
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=0.05, apertures=["f/4"],
        )
        # Sane: the dense recommendation is bounded by the input
        # series length (10), not stride-amplified.
        assert result.extra_steps_max >= 2
        assert result.extra_steps_max <= 10

    def test_unset_factors_falls_back_to_dense(self, cfg) -> None:
        # Default state (set_subsample_factors never called) — adapter
        # falls back to subsample_factors=None, advisor treats every
        # aperture as stride 1. Backwards compatible with existing
        # tests / fixtures.
        dataset = FakeCollector(
            samples=_samples_for_aperture(16.0, _rising(10))
        )
        checker = SharpnessRangeChecker(cfg, collector=dataset)
        # No set_subsample_factors call.
        result = checker.evaluate(
            lens_dir=Path("."), end_mm=10.0,
            step_size_mm=0.05, apertures=["f/16"],
        )
        # Without stride info we treat this as a dense aperture, so
        # the extra_steps_max is bounded by the input length.
        assert result.extra_steps_max <= 10


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
        assert first.peak_side == "end_side"

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
