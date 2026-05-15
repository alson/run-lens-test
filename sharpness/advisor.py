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
from sharpness.crops import CROP_NAMES, CROP_PEER_GROUPS
from sharpness.detect import (
    DEFAULT_DYNAMIC_RANGE_MIN,
    DEFAULT_EDGE_STEPS,
    DEFAULT_OFF_TARGET_FACTOR,
    DEFAULT_OFF_TARGET_RANGE,
    DEFAULT_PARTIAL_FRAME_FACTOR,
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
    def partial_frame_crops(self) -> tuple[CropAnalysis, ...]:
        """Crops where the wafer / target edge cuts through the crop
        window — peer-compare detected substantially lower absolute
        peak file size than positionally similar peers. Distinct from
        OFF_TARGET (window entirely off the wafer): partial-frame
        means re-frame, off-target means re-jog. ``valid=False`` so
        these crops don't contribute to truncation / extension
        decisions, but they're surfaced in the summary so the
        operator knows to recompose."""
        return tuple(
            c
            for c in self.crop_analyses
            if c.status == CropStatus.PARTIAL_FRAME
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
        partial_note = self._partial_frame_note()
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
            return base + partial_note + invalid_note
        if self.verdict == Verdict.OFF_TARGET:
            base = (
                f"f/{self.aperture:g}: OFF_TARGET — no focus signal in "
                "this sweep (every valid crop flat or no crop valid)"
            )
            return base + partial_note + invalid_note
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
        ) + partial_note + invalid_note

    def _invalid_note(self) -> str:
        # PARTIAL_FRAME crops are surfaced via _partial_frame_note;
        # don't double-count them in the "invalid" rollup.
        invalid = tuple(
            c for c in self.invalid_crops if c.status != CropStatus.PARTIAL_FRAME
        )
        if not invalid:
            return ""
        # Compact "(2 invalid crops: top-left 4%, bottom-right 6%)" so
        # the user knows at a glance which corners fell off the wafer.
        details = ", ".join(
            f"{c.crop_name} {int(round(c.dynamic_range * 100))}%"
            for c in invalid
        )
        return f" [{len(invalid)} invalid crops: {details}]"

    def _partial_frame_note(self) -> str:
        partial = self.partial_frame_crops
        if not partial:
            return ""
        # The operator action for partial-frame is "reframe" not
        # "re-jog", so the message is intentionally specific —
        # listing crop names tells them which side of the frame
        # extends past the wafer edge.
        names = ", ".join(c.crop_name for c in partial)
        return f" [PARTIAL_FRAME: {names} — reframe needed]"


# --- internals -----------------------------------------------------------


# Floor on the number of post-peak *aperture samples* the extension
# should reach in the truncated aperture's own sample grid. Two
# concerns drive this above the raw ``DEFAULT_EDGE_STEPS=2`` buffer
# that :func:`sharpness.detect.extra_steps_needed` already builds in:
#
# 1. The vertex projection is parabolic-fit on a noisy half-bell; the
#    true peak can drift a sample or two either way of the fit. We
#    want to land *past* the actual peak, not at the fit's vertex.
# 2. A single post-peak sample at a sparse aperture is one rail-grid
#    point that happens to be downhill — could still be a noise dip on
#    the way up. Two or three post-peak samples form a confirmed
#    decline, which is what the user wants before declaring the bell
#    captured.
#
# Three samples is the minimum that "confirmed decline" buys us; lower
# values revert to "barely past projected vertex" and reintroduce the
# noise-peak failure mode the bug fix is meant to close out.
MIN_POST_PEAK_SAMPLES = 3


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


def _extra_steps_for(crops: tuple[CropAnalysis, ...], stride: int) -> int:
    """Worst-case extension across all truncated crops, in **rail-step
    units**.

    Each ``CropAnalysis.extra_steps_needed`` is in per-aperture
    sample-index units (one index == one sample in the aperture's
    own sub-sampled series). For dense apertures (stride=1) that's
    already a rail-step count; for sparse apertures (e.g. f/16 with
    stride=4) one sample == 4 rail steps and we have to scale, or
    the runner's extension lands on rail positions that aren't on
    this aperture's capture grid and the reshot probes never fire.

    The returned count is also floored at
    ``MIN_POST_PEAK_SAMPLES * stride`` so the extension reaches a
    confirmed-decline region of the aperture's own grid (multiple
    post-peak samples) rather than barely clipping the projected
    vertex — the bell could otherwise be confirmed only by curve fit
    and the runner can re-extend chasing noise.
    """
    values = [c.extra_steps_needed for c in crops if c.extra_steps_needed is not None]
    if not values:
        return 0
    worst_samples = max(values)
    samples_with_floor = max(worst_samples, MIN_POST_PEAK_SAMPLES)
    return samples_with_floor * max(1, stride)


def _build_recommendation(
    aperture: float, analyses: list[CropAnalysis], *, stride: int = 1,
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
        extra = _extra_steps_for(truncated, stride=stride)
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
    partial_frame_factor: float = DEFAULT_PARTIAL_FRAME_FACTOR,
    edge_steps: int = DEFAULT_EDGE_STEPS,
    dynamic_range_min: float = DEFAULT_DYNAMIC_RANGE_MIN,
    off_target_range: float = DEFAULT_OFF_TARGET_RANGE,
    crop_names: tuple[str, ...] = CROP_NAMES,
    peer_groups: tuple[tuple[str, ...], ...] = CROP_PEER_GROUPS,
    subsample_factors: dict[float, int] | None = None,
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

    ``subsample_factors`` is the per-aperture rail-step stride
    (``SamplePlan.subsample_factors``); ``Recommendation.extra_steps``
    is rail-step count, so a sparse aperture's sample-index extra
    projection is scaled by its stride before being returned. Pass
    ``None`` (or omit) for dense-sampling tests where 1 sample == 1
    rail step at every aperture.
    """
    aggregated = dataset.aggregated()
    out: dict[float, Recommendation] = {}
    factors = subsample_factors or {}
    for aperture, samples in aggregated.by_aperture().items():
        crop_sizes: dict[str, list[int]] = {}
        for name in crop_names:
            series = [s.sizes[name] for s in samples if name in s.sizes]
            if series:
                crop_sizes[name] = series
        stride = max(1, int(factors.get(aperture, 1)))
        # ``edge_steps`` is the truncation edge-zone width in sample-
        # index units; the rail-step width is ``edge_steps * stride``.
        # The default (2) was tuned for stride=1, where 2 samples = 2
        # rail steps. At stride 2 those same 2 samples become 4 rail
        # steps — wide enough that a centred peak whose worst-case
        # corner crop legitimately sits 2 rail steps from the captured
        # edge gets falsely flagged TRUNCATED_END. Scale the zone down
        # so it expresses the same ~2-rail-step buffer regardless of
        # stride; floor at 1 sample so the last sample is always edge.
        effective_edge_steps = max(1, edge_steps // stride)
        analyses = analyze_crops_for_aperture(
            crop_sizes,
            cross_compare_factor=cross_compare_factor,
            partial_frame_factor=partial_frame_factor,
            peer_groups=peer_groups,
            edge_steps=effective_edge_steps,
            dynamic_range_min=dynamic_range_min,
            off_target_range=off_target_range,
        )
        out[aperture] = _build_recommendation(aperture, analyses, stride=stride)
    return out
