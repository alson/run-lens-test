"""Per-aperture reshoot advisor.

Wraps the per-crop detection in a single ``advise()`` entry point that
the session runner can call after the rail sweep completes:

    report = advise(collector.snapshot())
    for aperture, rec in sorted(report.items()):
        print(rec.summary())

Each ``Recommendation`` aggregates the 17 per-crop verdicts into:

- a scenario-level status (``OK`` / ``NEEDS_RESHOOT``);
- a direction (``EXTEND_FORWARD`` / ``EXTEND_BACKWARD`` / ``BOTH``)
  derived from the truncation direction(s) of the on-target crops;
- an estimated extension in rail steps (max across the truncated crops,
  so we cover the worst case at the centre or in any corner).

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
    DEFAULT_EDGE_STEPS,
    DEFAULT_OFF_TARGET_FACTOR,
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
    """No crop in this aperture has a real bell curve — every series
    is flat (max/min < DEFAULT_FLAT_RATIO) or the cross-compare flag
    rejected it. Either the rail spans an entirely out-of-focus
    region or the test target lacks micro-detail at every crop
    location. The runner treats this as a re-jog signal at the
    session level when *all* apertures land here."""


@dataclass(frozen=True)
class Recommendation:
    aperture: float
    verdict: Verdict
    direction: Direction
    extra_steps: int  # 0 when verdict is OK
    crop_analyses: tuple[CropAnalysis, ...] = field(default_factory=tuple)

    @property
    def truncated_crops(self) -> tuple[CropAnalysis, ...]:
        return tuple(c for c in self.crop_analyses if c.status.is_truncated)

    @property
    def off_target_crops(self) -> tuple[CropAnalysis, ...]:
        return tuple(c for c in self.crop_analyses if c.status == CropStatus.OFF_TARGET)

    def summary(self) -> str:
        """Single-line, human-readable recommendation suitable for the
        terminal output of the session runner.
        """
        if self.verdict == Verdict.OK:
            n_off = len(self.off_target_crops)
            base = f"f/{self.aperture:g}: OK"
            return base + (f" ({n_off} off-target crops ignored)" if n_off else "")
        if self.verdict == Verdict.OFF_TARGET:
            return (
                f"f/{self.aperture:g}: OFF_TARGET — every crop is flat; "
                "no focus signal in this sweep"
            )
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
        )


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
    truncated = tuple(c for c in analyses if c.status.is_truncated)
    if truncated:
        direction = _direction_for(truncated)
        extra = _extra_steps_for(truncated)
        return Recommendation(
            aperture=aperture,
            verdict=Verdict.NEEDS_RESHOOT,
            direction=direction,
            extra_steps=extra,
            crop_analyses=tuple(analyses),
        )
    # No truncated crops. Distinguish "every crop has a real peak in
    # range" (OK) from "every crop is flat / off-target" (OFF_TARGET).
    # If even one crop reports CropStatus.OK the aperture has signal.
    has_real_peak = any(c.status == CropStatus.OK for c in analyses)
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
    crop_names: tuple[str, ...] = CROP_NAMES,
) -> dict[float, Recommendation]:
    """Produce one ``Recommendation`` per aperture present in ``dataset``.

    Replicas at the same (step, aperture) are collapsed via per-crop
    median before peak / truncation detection so the bell-curve
    primitives see one point per rail step, not N.
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
        )
        out[aperture] = _build_recommendation(aperture, analyses)
    return out
