"""Adapter: drive ``exposure.range_check.RangeChecker`` from the
sharpness pipeline.

The session runner streams every captured JPEG through a
:class:`sharpness.collector.LiveCollector`, so by the time
:meth:`evaluate` is called the cropping is (mostly) done and a
:class:`CollectedDataset` is available. This adapter:

1. Drains pending crops via the collector's barrier.
2. Calls :func:`sharpness.advise` on the dataset.
3. Aggregates the per-aperture :class:`Recommendation`\\ s into a
   single :class:`RangeResult` (peak_side, ok, needs_extension).
4. On the second call (with probe data already in the same dataset)
   re-runs ``advise()`` and answers ``peak_in_range_after_probe``.

No thread pool of its own — the runner-owned collector handles all
the heavy lifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from exposure.range_check import (
    ApertureCurve,
    PeakSide,
    RangeChecker,
    RangeResult,
)
from sharpness.advisor import (
    Direction,
    Recommendation,
    Verdict,
    advise,
)
from sharpness.collector import CollectedDataset, LiveCollector
from sharpness.detect import DEFAULT_DYNAMIC_RANGE_MIN


# Crop name used to derive the canonical per-aperture sharpness curve
# stored in :attr:`RangeResult.per_aperture`. Centre-crop file size is
# the most stable on-target proxy across apertures; the user retains
# the full 17-crop dataset on disk for offline analysis (per
# §8.3.5 of the plan).
_CANONICAL_CURVE_CROP = "center"


def aperture_to_float(text: str) -> float:
    """``"f/4"`` -> ``4.0``. Mirrors :func:`session.runner._parse_fnum`
    so the adapter doesn't import from session/."""
    cleaned = text.strip().lstrip("fF").lstrip("/")
    return float(cleaned)


class SharpnessRangeChecker(RangeChecker):
    """Maps :func:`sharpness.advise` output onto the
    :class:`RangeChecker` contract.

    Parameters
    ----------
    config :
        The :mod:`config` module or duck-typed equivalent. The adapter
        reads no attributes today but accepts it for forward
        compatibility (e.g. ``cross_compare_factor`` could be plumbed
        through later).
    collector :
        A :class:`sharpness.collector.LiveCollector`-like object with
        ``wait_for_pending()`` and ``dataset``. The runner creates
        one per lens config and passes it in (or the runner sets
        :attr:`collector` between lens configs).
    """

    def __init__(self, config, collector: LiveCollector | None = None) -> None:
        super().__init__(config=config)
        self.collector = collector

    def set_collector(self, collector) -> None:  # type: ignore[override]
        self.collector = collector

    # ---- RangeChecker contract ------------------------------------------

    def evaluate(  # type: ignore[override]
        self,
        lens_dir: Path,
        end_mm: float,
        step_size_mm: float,
        apertures: list[str],
        *,
        probe_data=None,
    ) -> RangeResult:
        if self.collector is None:
            raise RuntimeError(
                "SharpnessRangeChecker.collector is not set; the session "
                "runner must inject the per-lens-config LiveCollector "
                "before calling evaluate()."
            )
        # Drain pending crops so the dataset reflects every shot
        # taken so far. Idempotent — the runner also does this for
        # safety.
        self.collector.wait_for_pending()
        # Aggregate replicas (multiple shots at the same (step, ap))
        # before advising so peak / truncation primitives see one
        # point per rail step. We pass the aggregated dataset to the
        # downstream curve-builders too — per_aperture in the
        # RangeResult is one point per step.
        dataset = self.collector.dataset.aggregated()
        # Pull the validity floor from config when available so a
        # campaign tuning the dynamic-range gate doesn't need to edit
        # detect.py defaults; SimpleNamespace test fixtures fall
        # through to the detector default.
        dr_min = getattr(
            self.config,
            "SHARPNESS_CROP_DYNAMIC_RANGE_MIN",
            DEFAULT_DYNAMIC_RANGE_MIN,
        )
        recs = advise(dataset, dynamic_range_min=dr_min)

        if probe_data is None:
            # First call: classify the main sweep.
            return self._aggregate(
                recs=recs,
                dataset=dataset,
                apertures=apertures,
                step_size_mm=step_size_mm,
                is_second_call=False,
            )
        # Second call: dataset already includes the probe shots.
        return self._aggregate(
            recs=recs,
            dataset=dataset,
            apertures=apertures,
            step_size_mm=step_size_mm,
            is_second_call=True,
        )

    # ---- Aggregation -----------------------------------------------------

    def _aggregate(
        self,
        *,
        recs: dict[float, Recommendation],
        dataset: CollectedDataset,
        apertures: Iterable[str],
        step_size_mm: float,
        is_second_call: bool,
    ) -> RangeResult:
        verdict_ok = all(r.verdict == Verdict.OK for r in recs.values())
        all_off_target = (
            len(recs) > 0
            and all(r.verdict == Verdict.OFF_TARGET for r in recs.values())
        )
        peak_side = self._aggregate_peak_side(recs)
        per_aperture = self._build_curves(
            apertures=apertures, dataset=dataset, step_size_mm=step_size_mm
        )
        notes = self._summarise(recs)
        return RangeResult(
            ok=verdict_ok,
            peak_side=peak_side,
            needs_extension=(not verdict_ok) and peak_side != "centered",
            peak_in_range_after_probe=is_second_call and verdict_ok,
            notes=notes,
            all_off_target=all_off_target,
            per_aperture=per_aperture,
        )

    @staticmethod
    def _aggregate_peak_side(
        recs: dict[float, Recommendation],
    ) -> PeakSide:
        """Reduce per-aperture :class:`Direction`\\ s to a single
        :class:`PeakSide` value.

        * No truncated apertures -> ``"centered"``.
        * Single direction across all truncated apertures -> that side.
        * Any aperture with :data:`Direction.BOTH` or two apertures
          disagreeing on direction -> ``"both"``.
        """
        sides_seen: set[str] = set()
        for rec in recs.values():
            d = rec.direction
            if d == Direction.NONE:
                continue
            if d == Direction.BOTH:
                return "both"
            if d == Direction.EXTEND_FORWARD:
                sides_seen.add("end_side")
            elif d == Direction.EXTEND_BACKWARD:
                sides_seen.add("start_side")
        if not sides_seen:
            return "centered"
        if len(sides_seen) > 1:
            return "both"
        return next(iter(sides_seen))  # type: ignore[return-value]

    def _build_curves(
        self,
        *,
        apertures: Iterable[str],
        dataset: CollectedDataset,
        step_size_mm: float,
    ) -> list[ApertureCurve]:
        out: list[ApertureCurve] = []
        for aperture in apertures:
            ap_float = aperture_to_float(aperture)
            series = dataset.crop_series_for(ap_float, _CANONICAL_CURVE_CROP)
            # series is sorted [(rail_step, size), ...]. The signed
            # rail_step is the runner's filename key; multiply by
            # step_size_mm to recover the rail position.
            indices = [step for step, _ in series]
            positions = [step * step_size_mm for step in indices]
            sizes = [float(size) for _, size in series]
            out.append(
                ApertureCurve(
                    aperture=aperture,
                    step_indices=indices,
                    rail_positions_mm=positions,
                    sharpness_values=sizes,
                )
            )
        return out

    @staticmethod
    def _summarise(recs: dict[float, Recommendation]) -> str:
        if not recs:
            return "no apertures in dataset"
        # Per-aperture summary uses Recommendation.summary() so the
        # persisted notes pick up per-crop detail when interesting:
        # "(N off-target crops ignored)" when valid corners disagreed
        # with centre, "[N invalid crops: top-left 4%, ...]" when
        # corners fell off the wafer. Keeps the happy path terse and
        # the failure-path informative.
        return "; ".join(
            recs[ap].summary() for ap in sorted(recs.keys())
        )
