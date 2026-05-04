"""Bell-curve detection on per-crop file-size series.

The session shoots K apertures × N rail steps. For each (aperture, crop)
pair we get an N-point series of JPEG sizes; we want to know:

1. Is the bell curve clipped at the start or end of the rail range?
   (truncation → reshoot needed)
2. Did this crop window even land on the wafer? At low magnifications
   some crops fall outside the target; their series is flat noise
   and should be ignored. (off-target → exclude, but no reshoot)
3. If truncated, how far past the current edge does the actual peak
   sit? — used to suggest "extend the rail by N more steps in
   direction X".

The five numerical primitives (smooth / peak_index / truncation_status
/ is_flat / off_target_by_cross_compare) plus ``CropStatus`` come from
``lens-test-detect-core``, which the offline ``rail-coverage`` package
also depends on. Live-only logic — the parabola-fit extension
estimator and the ``CropAnalysis`` aggregate — stays in this module
because rail-coverage deliberately doesn't recommend a reshoot
direction (subject-distance ranges aren't reproducible offline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lens_test_detect_core import (
    CropStatus,
    is_flat,
    off_target_by_cross_compare,
    peak_index,
    smooth,
)
from lens_test_detect_core import truncation_status as _truncation_status_core

__all__ = [
    "CropAnalysis",
    "CropStatus",
    "DEFAULT_EDGE_STEPS",
    "DEFAULT_FLAT_RATIO",
    "DEFAULT_OFF_TARGET_FACTOR",
    "analyze_crop",
    "analyze_crops_for_aperture",
    "estimate_peak_vertex",
    "extra_steps_needed",
    "is_flat",
    "off_target_by_cross_compare",
    "peak_index",
    "smooth",
    "truncation_status",
]

# Live-collector defaults: one shot per rail step (rail-step-outside,
# aperture-inside architecture from DESIGN_NOTES §2.6), so an "edge
# zone" of 2 rail steps is one rail position of buffer.
DEFAULT_EDGE_STEPS = 2
DEFAULT_FLAT_RATIO = 1.15
DEFAULT_OFF_TARGET_FACTOR = 0.65


# Backwards-compatible alias — the live module historically called this
# ``peak_step`` to emphasise the rail-step granularity. Keep the alias
# so existing call sites (and tests) keep working.
peak_step = peak_index


def truncation_status(
    peak_idx: int, n: int, *, edge_steps: int = DEFAULT_EDGE_STEPS
) -> CropStatus:
    """Live-collector wrapper around the shared primitive.

    The shared primitive is unopinionated about the edge-zone size
    because the two consumers see different input granularities. This
    wrapper supplies the live-collector default (one shot per rail
    step ⇒ edge of 2 rail steps) so existing call sites and tests
    don't need to thread the threshold through.
    """
    return _truncation_status_core(peak_idx, n, edge_steps=edge_steps)


@dataclass(frozen=True)
class CropAnalysis:
    crop_name: str
    status: CropStatus
    n_steps: int
    peak_step: int | None  # rail-step index of the smoothed peak
    peak_pct: float | None  # 0..1, where 0=first step, 1=last step
    estimated_peak_step: float | None  # vertex of parabola fit (may be < 0 or > n-1)
    extra_steps_needed: int | None  # rail steps to extend in the truncation direction
    max_size: int | None
    min_size: int | None


def estimate_peak_vertex(sizes: np.ndarray | list[int]) -> float | None:
    """Quadratic-fit estimate of the bell-curve peak position.

    Returns the vertex ``x`` of the best-fit parabola through the
    smoothed series, in raw-step coordinates. Used for "where would the
    peak be if we'd swept further?".

    Returns ``None`` when:
    - too few points to fit a parabola (need ≥ 3 smoothed points), or
    - the best-fit parabola is concave-up (``a >= 0``), meaning the
      data isn't bell-shaped at all and an extrapolated peak is
      meaningless.
    """
    smoothed = smooth(sizes, window=3)
    if smoothed.size < 3:
        return None
    # Convert smoothed indices back into raw-step coordinates so the
    # vertex we return aligns with the input series.
    offset = 1  # centred 3-window
    xs = np.arange(smoothed.size, dtype=float) + offset
    a, b, _c = np.polyfit(xs, smoothed, deg=2)
    if a >= 0:
        return None
    return float(-b / (2 * a))


def extra_steps_needed(
    sizes: np.ndarray | list[int],
    status: CropStatus,
    n: int,
) -> int | None:
    """Suggested extension in rail-step units, given a truncated curve.

    Uses ``estimate_peak_vertex`` to guess where the actual peak sits,
    then returns the number of rail steps the user would need to add
    in the truncation direction to land on the peak with a
    ``DEFAULT_EDGE_STEPS`` buffer past it.

    The returned count is conservative: at minimum ``DEFAULT_EDGE_STEPS``
    (so a borderline truncation still earns an extension big enough to
    confirm the peak), at most ``n`` (so we never recommend more than
    doubling the current range from a single half-bell).
    """
    if not status.is_truncated:
        return None
    vertex = estimate_peak_vertex(sizes)
    if vertex is None:
        # No parabolic fit available — fall back to a half-range bump.
        return max(DEFAULT_EDGE_STEPS, n // 2)
    if status == CropStatus.TRUNCATED_END:
        # vertex is past the end at index >= n; extend by (vertex - (n-1)) + buffer
        gap = vertex - (n - 1)
    else:  # TRUNCATED_START
        # vertex is at index <= 0; extend backward by |vertex| + buffer
        gap = -vertex
    raw = max(DEFAULT_EDGE_STEPS, int(round(gap)) + DEFAULT_EDGE_STEPS)
    return min(raw, n)


def analyze_crop(
    crop_name: str,
    sizes: list[int],
    *,
    flagged_off_target: bool,
    edge_steps: int = DEFAULT_EDGE_STEPS,
) -> CropAnalysis:
    n = len(sizes)
    if n == 0:
        return CropAnalysis(
            crop_name=crop_name,
            status=CropStatus.EMPTY,
            n_steps=0,
            peak_step=None,
            peak_pct=None,
            estimated_peak_step=None,
            extra_steps_needed=None,
            max_size=None,
            min_size=None,
        )

    if flagged_off_target or is_flat(sizes):
        return CropAnalysis(
            crop_name=crop_name,
            status=CropStatus.OFF_TARGET,
            n_steps=n,
            peak_step=None,
            peak_pct=None,
            estimated_peak_step=None,
            extra_steps_needed=None,
            max_size=max(sizes),
            min_size=min(sizes),
        )

    pk = peak_index(sizes)
    pct = pk / max(1, n - 1)
    status = truncation_status(pk, n, edge_steps=edge_steps)
    vertex = estimate_peak_vertex(sizes) if status.is_truncated else None
    extra = extra_steps_needed(sizes, status, n)

    return CropAnalysis(
        crop_name=crop_name,
        status=status,
        n_steps=n,
        peak_step=pk,
        peak_pct=pct,
        estimated_peak_step=vertex,
        extra_steps_needed=extra,
        max_size=max(sizes),
        min_size=min(sizes),
    )


def analyze_crops_for_aperture(
    crop_sizes: dict[str, list[int]],
    *,
    cross_compare_factor: float = DEFAULT_OFF_TARGET_FACTOR,
    edge_steps: int = DEFAULT_EDGE_STEPS,
) -> list[CropAnalysis]:
    """Run the full per-crop analysis for one aperture.

    ``crop_sizes`` is ``{crop_name: [size_at_step_0, size_at_step_1, ...]}``.
    The returned list has one ``CropAnalysis`` per crop, in the
    iteration order of ``crop_sizes``.
    """
    crop_maxes: dict[str, int] = {
        name: max(sizes) for name, sizes in crop_sizes.items() if sizes
    }
    flagged = off_target_by_cross_compare(crop_maxes, factor=cross_compare_factor)
    return [
        analyze_crop(
            name,
            sizes,
            flagged_off_target=name in flagged,
            edge_steps=edge_steps,
        )
        for name, sizes in crop_sizes.items()
    ]
