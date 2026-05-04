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

The heuristics mirror ``analyse-lens-test/rail-coverage`` but with a
parabola-fit estimator added for (3) — the offline tool deliberately
omits direction information because subject-distance ranges can't be
reproduced between runs, but the live tool runs *while* the user is at
the rig and *can* extend the range, so direction is useful here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


# Match analyse-lens-test/rail-coverage defaults; see that package's
# detect.py for the empirical tuning notes.
DEFAULT_EDGE_STEPS = 2
DEFAULT_FLAT_RATIO = 1.15
DEFAULT_OFF_TARGET_FACTOR = 0.65


class CropStatus(str, Enum):
    OK = "OK"
    TRUNCATED_START = "TRUNCATED_START"
    TRUNCATED_END = "TRUNCATED_END"
    OFF_TARGET = "OFF_TARGET"
    EMPTY = "EMPTY"

    @property
    def is_truncated(self) -> bool:
        return self in (CropStatus.TRUNCATED_START, CropStatus.TRUNCATED_END)


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


# --- pure helpers ---------------------------------------------------------


def smooth(sizes: np.ndarray | list[int], window: int = 3) -> np.ndarray:
    """Centred moving average. Returns an array of length ``len(sizes) - window + 1``."""
    arr = np.asarray(sizes, dtype=float)
    if arr.size < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="valid")


def peak_step(sizes: np.ndarray | list[int], window: int = 3) -> int:
    """Rail-step index of the smoothed peak."""
    smoothed = smooth(sizes, window=window)
    if smoothed.size == 0:
        return 0
    offset = (window - 1) // 2
    return int(np.argmax(smoothed)) + offset


def truncation_status(
    peak_idx: int, n: int, *, edge_steps: int = DEFAULT_EDGE_STEPS
) -> CropStatus:
    """``edge_steps`` is in rail-step units (not raw shots).

    The live collector is one shot per rail step (one aperture per step,
    given the inverted shot-order architecture in DESIGN_NOTES §2.6),
    so peak inside the first/last ``edge_steps`` rail steps means we
    didn't sweep far enough past focus on that side.
    """
    if n == 0:
        return CropStatus.EMPTY
    if peak_idx < edge_steps:
        return CropStatus.TRUNCATED_START
    if peak_idx > n - 1 - edge_steps:
        return CropStatus.TRUNCATED_END
    return CropStatus.OK


def is_flat(
    sizes: np.ndarray | list[int], *, ratio_threshold: float = DEFAULT_FLAT_RATIO
) -> bool:
    """``max/min`` below threshold ⇒ no real focus modulation in this crop."""
    arr = np.asarray(sizes, dtype=float)
    if arr.size == 0 or arr.min() <= 0:
        return False
    return float(arr.max() / arr.min()) < ratio_threshold


def off_target_by_cross_compare(
    crop_maxes: dict[str, int],
    *,
    factor: float = DEFAULT_OFF_TARGET_FACTOR,
) -> set[str]:
    """Crops whose peak file size is below ``factor × best``.

    Reference is the *maximum* peak across the scenario (not the median):
    when more than half the 17 crops miss the wafer (common at low
    magnifications), a median-based cutoff sits inside the off-target
    cluster and flags nothing. See rail-coverage/detect.py for the
    empirical justification (Nikon 70-180 + 5T sample).
    """
    if len(crop_maxes) < 2:
        return set()
    cutoff = max(crop_maxes.values()) * factor
    return {name for name, m in crop_maxes.items() if m < cutoff}


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
    a, b, c = np.polyfit(xs, smoothed, deg=2)
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


# --- top-level ------------------------------------------------------------


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

    pk = peak_step(sizes)
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
