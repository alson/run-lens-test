"""Pure-function probe math for the coarse-to-fine sweep.

The runner orchestrates rail moves and captures; this module just does
the math that turns a phase-A crop dataset into a phase-C window
recommendation. Keeping it pure means it's trivially unit-testable
without a fake camera / rail and it stays in lock-step with the
``tools/probe_simulator.py`` algorithm that calibrated the constants.

Algorithm (matches DESIGN_NOTES §6.26):

1. For each crop with a usable bell, find the discrete argmax of the
   3-window-smoothed series across the phase-A samples.
2. Refine to sub-sample resolution via a parabolic fit through the
   (argmax-1, argmax, argmax+1) triplet — phase A's spacing (~250 µm
   on a typical jog) is much larger than FWHM (~90 µm), so picking
   the discrete argmax would round each crop's peak to the nearest
   phase-A sample, inflating the apparent corner spread.
3. Drop crops whose dynamic range is below the validity floor or
   whose smoothed series doesn't have enough points — those are
   off-wafer or too noisy to contribute.
4. Return the asymmetric bracket
   ``[min(crop peaks) − padding × FWHM,
   max(crop peaks) + padding × FWHM]``.
   On high-FC configs all the corners can land on one side of the
   centre crop; a symmetric window centred on centre would miss
   half the corners.

The returned bracket is in *phase-A sample* coordinates: the caller
multiplies by phase-A step size to get rail-mm. The runner converts
that to phase-C grid steps using the planner's rail-step-mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sharpness.detect import DEFAULT_DYNAMIC_RANGE_MIN, smooth


__all__ = [
    "PhaseAResult",
    "phase_a_peaks",
    "phase_c_window_mm",
]


@dataclass(frozen=True)
class PhaseAResult:
    """Outcome of analysing a phase-A capture set.

    ``crop_peaks`` is keyed by crop name; each value is the per-crop
    sub-sample peak position in phase-A *sample* coordinates (0 = first
    phase-A probe, n_points-1 = last). Crops that didn't yield a usable
    peak are simply absent.

    ``inconclusive`` is True when fewer than 3 crops yielded a usable
    peak — the runner should fall back to the existing uniform sweep
    flow rather than try to size phase C off a bad estimate.
    """

    crop_peaks: dict[str, float] = field(default_factory=dict)
    centre_peak_sample: float | None = None
    min_peak_sample: float | None = None
    max_peak_sample: float | None = None
    inconclusive: bool = True

    @property
    def corner_spread_samples(self) -> float:
        if self.min_peak_sample is None or self.max_peak_sample is None:
            return 0.0
        return self.max_peak_sample - self.min_peak_sample


def _per_crop_peak(
    sizes: list[int],
    *,
    dynamic_range_min: float = DEFAULT_DYNAMIC_RANGE_MIN,
) -> float | None:
    """Sub-sample peak position in sample coordinates, or None.

    Returns the parabola-refined argmax of the smoothed series. Returns
    None when:
    - too few points to smooth (need ≥ 3),
    - dynamic range below the validity floor (off-wafer / flat),
    - all sizes ≤ 0 (corrupt input).
    """
    n = len(sizes)
    if n < 3:
        return None
    if min(sizes) <= 0 or max(sizes) <= 0:
        return None
    arr = np.asarray(sizes, dtype=float)
    if (arr.max() - arr.min()) / arr.max() < dynamic_range_min:
        return None
    sm = smooth(sizes, window=3)
    if sm.size == 0:
        return None
    # smooth() returns a centred 3-window; offset back to raw index space.
    sm_argmax = int(np.argmax(sm))
    raw_argmax = sm_argmax + 1
    if 0 < raw_argmax < n - 1:
        y0, y1, y2 = arr[raw_argmax - 1], arr[raw_argmax], arr[raw_argmax + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            shift = 0.5 * (y0 - y2) / denom
            # Clamp to ±1 so a wonky parabola can't jump us across a neighbour.
            shift = max(-1.0, min(1.0, shift))
        else:
            shift = 0.0
        return float(raw_argmax) + shift
    return float(raw_argmax)


def phase_a_peaks(
    crop_sizes: dict[str, list[int]],
    *,
    dynamic_range_min: float = DEFAULT_DYNAMIC_RANGE_MIN,
) -> PhaseAResult:
    """Analyse a phase-A capture set and return per-crop peak positions.

    ``crop_sizes`` is ``{crop_name: [size_at_probe_0, size_at_probe_1, ...]}``
    where each list has the same length (one entry per phase-A probe).
    The returned peaks are in phase-A *sample* coordinates: 0 is the
    first probe, len-1 is the last.
    """
    peaks: dict[str, float] = {}
    for name, sizes in crop_sizes.items():
        pk = _per_crop_peak(sizes, dynamic_range_min=dynamic_range_min)
        if pk is not None:
            peaks[name] = pk
    if len(peaks) < 3:
        return PhaseAResult(
            crop_peaks=peaks,
            centre_peak_sample=peaks.get("center"),
            min_peak_sample=min(peaks.values()) if peaks else None,
            max_peak_sample=max(peaks.values()) if peaks else None,
            inconclusive=True,
        )
    return PhaseAResult(
        crop_peaks=peaks,
        centre_peak_sample=peaks.get("center"),
        min_peak_sample=min(peaks.values()),
        max_peak_sample=max(peaks.values()),
        inconclusive=False,
    )


def phase_c_window_mm(
    result: PhaseAResult,
    *,
    phase_a_start_mm: float,
    phase_a_end_mm: float,
    n_probes: int,
    fwhm_mm: float,
    padding_x_fwhm: float = 1.5,
) -> tuple[float, float] | None:
    """Project a PhaseAResult onto a rail-mm window for phase C.

    Returns ``(lo_mm, hi_mm)`` — the rail positions (signed, same
    convention as the runner's ``end_mm``) bracketing the phase-C
    window, or ``None`` when phase A was inconclusive.

    The bracket is asymmetric: ``[min_peak_mm − padding × FWHM,
    max_peak_mm + padding × FWHM]``. On a high-FC config the corners
    all land on one side of the centre crop; a symmetric window
    centred on the centre crop loses half of them.
    """
    if result.inconclusive or result.min_peak_sample is None:
        return None
    if n_probes < 2:
        return None
    spacing_mm = (phase_a_end_mm - phase_a_start_mm) / (n_probes - 1)
    lo_mm = phase_a_start_mm + result.min_peak_sample * spacing_mm
    hi_mm = phase_a_start_mm + result.max_peak_sample * spacing_mm
    pad_mm = padding_x_fwhm * fwhm_mm
    if spacing_mm >= 0:
        return (lo_mm - pad_mm, hi_mm + pad_mm)
    # spacing_mm < 0 means phase A swept backward (user jogged END < START).
    # Sample 0 is at phase_a_start_mm (the larger value), sample n-1 at
    # the smaller phase_a_end_mm. Min/max in sample space become reversed
    # in mm space.
    return (hi_mm - pad_mm, lo_mm + pad_mm)
