"""Per-aperture reshoot advisor.

Wraps the per-crop detection in a single ``advise()`` entry point that
the session runner can call after the rail sweep completes:

    report = advise(collector.snapshot())
    for aperture, rec in sorted(report.items()):
        print(rec.summary())

Each ``Recommendation`` aggregates all 17 per-crop verdicts into:

- a scenario-level status (``OK`` / ``NEEDS_RESHOOT`` / ``OFF_TARGET``);
- a direction (``EXTEND_FORWARD`` / ``EXTEND_BACKWARD`` / ``BOTH``)
  derived from the truncation direction(s) of the *valid* crops —
  field curvature can shift corner peaks past the captured range
  while the centre lands fine, so we want each valid crop to weigh
  in independently rather than collapsing into one synthetic peak;
- an estimated extension in rail steps (max across the truncated crops,
  so we cover the worst case at the centre or in any corner).

Validity. Corner crops can fall off the wafer onto a low-contrast
background, in which case their JPEG-size series barely moves across
the rail. We gate on the per-crop dynamic range
``(max - min) / max`` against ``SHARPNESS_CROP_DYNAMIC_RANGE_MIN``
(``DEFAULT_DYNAMIC_RANGE_MIN`` here when no value is plumbed
through). Crops below the floor are flagged ``valid=False`` and
excluded from the per-aperture verdict; the ``Recommendation``
exposes them via :meth:`Recommendation.invalid_crops` and the
rationale string calls them out individually so the user can tell
"corner fell off the wafer" apart from "everything is flat".

Note on direction semantics: ``EXTEND_FORWARD`` corresponds to
``TRUNCATED_END`` — the rail ran out of range before reaching the peak,
so the user should send the rail further along its current sweep
direction. ``EXTEND_BACKWARD`` corresponds to ``TRUNCATED_START`` — the
rail started already inside or past the focus zone, so the user should
move the rail's starting position back away from the lens (or
equivalently widen the start margin).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sharpness.collector import CollectedDataset
from sharpness.crops import CROP_NAMES
from sharpness.detect import (
    DEFAULT_DYNAMIC_RANGE_MIN,
    DEFAULT_EDGE_STEPS,
    DEFAULT_OFF_TARGET_FACTOR,
    DEFAULT_OFF_TARGET_RANGE,
    CropAnalysis,
    CropStatus,
    analyze_crops_for_aperture,
)


class Direction(str, Enum):
    NONE = "NONE"
    EXTEND_FORWARD = "EXTEND_FORWARD"
    EXTEND_BACKWARD = "EXTEND_BACKWARD"
    BOTH = "BOTH"


class Verdict(str, Enum):
    OK = "OK"
    NEEDS_RESHOOT = "NEEDS_RESHOOT"
    OFF_TARGET = "OFF_TARGET"
    """No *valid* crop in this aperture has a real bell curve — either
    every crop fell below the dynamic-range floor (window off the
    wafer / wholly out-of-focus area), or the cross-compare /
    is_flat flag rejected the few valid ones. The runner treats this
    as a re-jog signal at the session level when *all* apertures land
    here."""


@dataclass(frozen=True)
class Recommendation:
    aperture: float
    verdict: Verdict
    direction: Direction
    extra_steps: int  # 0 when verdict is OK
    crop_analyses: tuple[CropAnalysis, ...] = field(default_factory=tuple)

    @property
    def truncated_crops(self) -> tuple[CropAnalysis, ...]:
        return tuple(
            c for c in self.crop_analyses if c.valid and c.status.is_truncated
        )

    @property
    def off_target_crops(self) -> tuple[CropAnalysis, ...]:
        # Cross-compare / legacy is_flat off-targets — crops with real
        # dynamic range but a much weaker peak than their peers. Distinct
        # from invalid_crops (window fell off the wafer entirely).
        return tuple(
            c
            for c in self.crop_analyses
            if c.valid and c.status == CropStatus.OFF_TARGET
        )

    @property
    def low_signal_crops(self) -> tuple[CropAnalysis, ...]:
        """Crops with a real but weak bell — dynamic range between the
        OFF_TARGET floor and the OK ceiling, peak found in the middle
        of the rail series. Count as ``OK`` for the per-aperture
        verdict but surfaced separately so the operator can spot rigs
        where most corners are barely above the noise floor (e.g. the
        Viltrox 85 + 61 mm extension corner regime)."""
        return tuple(
            c
            for c in self.crop_analyses
            if c.status == CropStatus.LOW_SIGNAL
        )

    @property
    def invalid_crops(self) -> tuple[CropAnalysis, ...]:
        """Crops excluded from the verdict because their dynamic range
        was below ``SHARPNESS_CROP_DYNAMIC_RANGE_MIN`` — the window
        almost certainly fell off the wafer onto low-contrast
        background and has no usable focus signal."""
        return tuple(c for c in self.crop_analyses if not c.valid)

    @property
    def valid_crops(self) -> tuple[CropAnalysis, ...]:
        return tuple(c for c in self.crop_analyses if c.valid)

    def per_crop(self) -> dict[str, CropAnalysis]:
        """Convenience accessor: ``{crop_name: CropAnalysis}``."""
        return {c.crop_name: c for c in self.crop_analyses}

    def summary(self) -> str:
        """Single-line, human-readable recommendation suitable for the
        terminal output of the session runner.
        """
        invalid_note = self._invalid_note()
        if self.verdict == Verdict.OK:
            n_off = len(self.off_target_crops)
            n_low = len(self.low_signal_crops)
            base = f"f/{self.aperture:g}: OK"
            qualifiers: list[str] = []
            if n_low:
                qualifiers.append(f"{n_low} LOW_SIGNAL crops")
            if n_off:
                qualifiers.append(f"{n_off} off-target crops ignored")
            if qualifiers:
                base += f" ({', '.join(qualifiers)})"
            return base + invalid_note
        if self.verdict == Verdict.OFF_TARGET:
            base = (
                f"f/{self.aperture:g}: OFF_TARGET — no focus signal in "
                "this sweep (every valid crop flat or no crop valid)"
            )
            return base + invalid_note
        crops = ", ".join(c.crop_name for c in self.truncated_crops)
        if self.direction == Direction.EXTEND_FORWARD:
            arrow = "extend forward (further past start position)"
        elif self.direction == Direction.EXTEND_BACKWARD:
            arrow = "extend backward (start position further away from peak)"
        else:
            arrow = "extend in BOTH directions"
        return (
            f"f/{self.aperture:g}: NEEDS_RESHOOT — {arrow} by ~{self.extra_steps} "
            f"rail step(s); truncated crops: {crops}"
        ) + invalid_note

    def _invalid_note(self) -> str:
        invalid = self.invalid_crops
        if not invalid:
            return ""
        # Compact "(2 invalid crops: top-left 4%, bottom-right 6%)" so
        # the user knows at a glance which corners fell off the wafer.
        details = ", ".join(
            f"{c.crop_name} {int(round(c.dynamic_range * 100))}%"
            for c in invalid
        )
        return f" [{len(invalid)} invalid crops: {details}]"


# --- internals -----------------------------------------------------------


def _direction_for(crops: tuple[CropAnalysis, ...]) -> Direction:
    has_start = any(c.status == CropStatus.TRUNCATED_START for c in crops)
    has_end = any(c.status == CropStatus.TRUNCATED_END for c in crops)
    if has_start and has_end:
        return Direction.BOTH
    if has_end:
        return Direction.EXTEND_FORWARD
    if has_start:
        return Direction.EXTEND_BACKWARD
    return Direction.NONE


def _extra_steps_for(crops: tuple[CropAnalysis, ...]) -> int:
    """Worst-case extension across all truncated crops."""
    values = [c.extra_steps_needed for c in crops if c.extra_steps_needed is not None]
    return max(values) if values else 0


def _build_recommendation(
    aperture: float, analyses: list[CropAnalysis]
) -> Recommendation:
    # Only valid crops feed into the verdict / direction / extension.
    # Invalid (below the OFF_TARGET floor) crops fell off the wafer;
    # their near-constant series would otherwise look "non-truncated
    # peak in range" purely as a numerical artifact.
    valid = [c for c in analyses if c.valid]
    truncated = tuple(c for c in valid if c.status.is_truncated)
    if truncated:
        # Truncated crops always promote the aperture to
        # NEEDS_RESHOOT — even if the truncated crop's underlying
        # signal level was LOW_SIGNAL territory. A truncated weak
        # bell is still a peak the runner can chase by extending
        # rail.
        direction = _direction_for(truncated)
        extra = _extra_steps_for(truncated)
        return Recommendation(
            aperture=aperture,
            verdict=Verdict.NEEDS_RESHOOT,
            direction=direction,
            extra_steps=extra,
            crop_analyses=tuple(analyses),
        )
    # No truncated crops among the valid ones. "Real peak found"
    # includes both OK (high-signal) and LOW_SIGNAL (weak but bell-
    # shaped peak in range) — both signal that the rail bracketed the
    # peak, so the verdict is OK. If neither kind of in-range peak
    # is present (every valid crop is OFF_TARGET / EMPTY, or no crop
    # is valid at all) we return OFF_TARGET. That's the runner's
    # re-jog signal at the session level when *all* apertures land
    # here.
    has_real_peak = any(
        c.status in (CropStatus.OK, CropStatus.LOW_SIGNAL) for c in valid
    )
    if has_real_peak:
        return Recommendation(
            aperture=aperture,
            verdict=Verdict.OK,
            direction=Direction.NONE,
            extra_steps=0,
            crop_analyses=tuple(analyses),
        )
    return Recommendation(
        aperture=aperture,
        verdict=Verdict.OFF_TARGET,
        direction=Direction.NONE,
        extra_steps=0,
        crop_analyses=tuple(analyses),
    )


# --- public --------------------------------------------------------------


def advise(
    dataset: CollectedDataset,
    *,
    cross_compare_factor: float = DEFAULT_OFF_TARGET_FACTOR,
    edge_steps: int = DEFAULT_EDGE_STEPS,
    dynamic_range_min: float = DEFAULT_DYNAMIC_RANGE_MIN,
    off_target_range: float = DEFAULT_OFF_TARGET_RANGE,
    crop_names: tuple[str, ...] = CROP_NAMES,
) -> dict[float, Recommendation]:
    """Produce one ``Recommendation`` per aperture present in ``dataset``.

    Replicas at the same (step, aperture) are collapsed via per-crop
    median before peak / truncation detection so the bell-curve
    primitives see one point per rail step, not N.

    All crops above the ``off_target_range`` floor (default 0.05) feed
    independently into the per-aperture aggregate. Within that band,
    crops with dynamic range below ``dynamic_range_min`` (default 0.25)
    are still classified — an in-range peak becomes ``LOW_SIGNAL``,
    which the aggregator treats like ``OK`` for the verdict but
    surfaces in the summary string. Truncation in any direction at any
    valid crop promotes the aperture to ``NEEDS_RESHOOT``; if valid
    crops disagree on truncation direction (e.g. centre rolls back
    while a corner runs forward — classic field curvature) the
    aperture direction collapses to ``BOTH`` so the runner extends
    bilaterally.
    """
    aggregated = dataset.aggregated()
    out: dict[float, Recommendation] = {}
    for aperture, samples in aggregated.by_aperture().items():
        crop_sizes: dict[str, list[int]] = {}
        for name in crop_names:
            series = [s.sizes[name] for s in samples if name in s.sizes]
            if series:
                crop_sizes[name] = series
        analyses = analyze_crops_for_aperture(
            crop_sizes,
            cross_compare_factor=cross_compare_factor,
            edge_steps=edge_steps,
            dynamic_range_min=dynamic_range_min,
            off_target_range=off_target_range,
        )
        out[aperture] = _build_recommendation(aperture, analyses)
    return out
