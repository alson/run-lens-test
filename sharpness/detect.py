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
    partial_frame_by_peer_compare,
    peak_index,
    smooth,
)
from lens_test_detect_core import truncation_status as _truncation_status_core

__all__ = [
    "CropAnalysis",
    "CropStatus",
    "DEFAULT_DYNAMIC_RANGE_MIN",
    "DEFAULT_EDGE_STEPS",
    "DEFAULT_FLAT_RATIO",
    "DEFAULT_OFF_TARGET_FACTOR",
    "DEFAULT_OFF_TARGET_RANGE",
    "DEFAULT_PARTIAL_FRAME_FACTOR",
    "analyze_crop",
    "analyze_crops_for_aperture",
    "dynamic_range",
    "estimate_peak_vertex",
    "extra_steps_needed",
    "is_flat",
    "off_target_by_cross_compare",
    "partial_frame_by_peer_compare",
    "peak_index",
    "smooth",
    "truncation_status",
]

# Live-collector defaults: one shot per rail step (rail-step-outside,
# aperture-inside architecture from DESIGN_NOTES §2.6), so an "edge
# zone" of 2 rail steps is one rail position of buffer.
#
# Unit: per-aperture sample-index. At ``subsample_factor=1`` one sample
# == one rail step, so the default of 2 expresses a 2-rail-step zone.
# At higher subsample factors one sample == N rail steps; a stride-
# aware caller (see :func:`sharpness.advisor.advise`) must rescale this
# value before handing it to :func:`truncation_status` so the edge zone
# stays at the same *rail-step* width and a sparse aperture's corner
# peak near the captured edge isn't falsely flagged TRUNCATED.
DEFAULT_EDGE_STEPS = 2
DEFAULT_FLAT_RATIO = 1.15
# Cross-compare cutoff: a crop's peak file size below
# ``factor × best_peak`` is flagged ``OFF_TARGET`` (i.e. "not even on
# the wafer"). 0.65 was inherited from the offline rail-coverage
# analyser, where the wafer often didn't cover the corners; in the
# live shooting context the wafer fills the frame at >1:1 mag, so
# corners are real measurements and only legitimate field-curvature
# / corner-softness signal. 0.65 false-positived essentially every
# corner on real rig data (per the post-session multi-crop recheck
# of 2026-05-06_175727 and 2026-05-06_200150 — corners with valid
# bell shapes peaked at 30-60 % of centre and were all flagged).
# 0.25 sits below the lowest legitimate corner peak ratio observed
# on this rig while still catching crops with peaks materially
# below the dynamic-range floor (windows landing on low-contrast
# background).
DEFAULT_OFF_TARGET_FACTOR = 0.25
# High-signal floor for per-crop dynamic range ``(max - min) / max``.
# Crops at or above this threshold with a peak in the middle of the
# rail series classify as :data:`CropStatus.OK`. Default matches
# ``config.SHARPNESS_CROP_DYNAMIC_RANGE_MIN`` but is duplicated here so
# the detector stays import-free of session config — callers wire the
# configured value through ``analyze_crops_for_aperture``.
DEFAULT_DYNAMIC_RANGE_MIN = 0.25
# Off-target floor. Crops with dynamic range strictly below this value
# are short-circuited to :data:`CropStatus.OFF_TARGET` (window fell off
# the wafer / featureless background). Crops between
# ``DEFAULT_OFF_TARGET_RANGE`` and ``DEFAULT_DYNAMIC_RANGE_MIN`` go
# through peak / truncation detection but their in-range result is
# marked :data:`CropStatus.LOW_SIGNAL` — a real but weak bell, common
# on soft lenses (e.g. fast primes with external extension tubes)
# where corner acutance lands at max/min ratios of 1.04–1.22.
DEFAULT_OFF_TARGET_RANGE = 0.05
# Partial-frame cutoff: a crop's peak file size below
# ``factor × peer-group median`` is flagged
# :data:`CropStatus.PARTIAL_FRAME` (window extends past the wafer edge,
# part of the crop is dark background). Peer groups (4 corners vs 4
# corners, 4 edges vs 4 edges) keep the comparison apples-to-apples so
# natural corner softness vs centre detail doesn't false-positive
# every corner.
#
# Calibrated against the 24-120 + 35 mm extension session
# (2026-05-11_185420) where bottom-left was confirmed partially
# off-frame: its ratio sat at 0.72-0.79 of peer-corner median across
# all five apertures, while normal non-partial corners on the same
# session and the sibling 24-120 + 45 mm session clustered at
# ≥ 0.95. 0.80 cleanly separates the two regimes; a one-off
# borderline aperture won't fire unless the geometry actually cuts
# off frame. See ``partial_frame_by_peer_compare`` for the
# full rationale.
DEFAULT_PARTIAL_FRAME_FACTOR = 0.80


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
    # Dynamic-range gate (the wafer-on-frame validity check). ``valid``
    # is the explicit signal callers should use to decide whether this
    # crop contributes to the aperture verdict. ``dynamic_range`` is
    # ``(max - min) / max`` over the rail-step series, exposed for
    # diagnostics / rationale strings (e.g. "top-left invalid: dynamic
    # range 4 % < 25 %").
    valid: bool = True
    dynamic_range: float = 0.0


def dynamic_range(values: list[int]) -> float:
    """Per-series dynamic range ``(max - min) / max``.

    Returns 0.0 when the series is empty or its max is non-positive
    (defensive against corrupt files); 0.0 always trips the validity
    gate so degenerate inputs are conservatively dropped.
    """
    if not values:
        return 0.0
    mx = max(values)
    if mx <= 0:
        return 0.0
    return (mx - min(values)) / mx


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
    """Suggested extension in **per-aperture sample-index units**,
    given a truncated curve.

    Returns the number of additional samples-in-this-series that
    would need to be added in the truncation direction to land on
    the projected peak with a ``DEFAULT_EDGE_STEPS`` buffer past it.

    Important: the unit is the input series' own sampling step, not
    the underlying rail step. Densely-sampled apertures
    (``subsample_factor=1``) have one sample per rail step so the two
    coincide; sparse apertures (e.g. f/16 with ``subsample_factor=4``)
    have one sample per N rail steps and the consumer must scale by
    that stride before issuing a rail extension. See
    :func:`sharpness.advisor._extra_steps_for` for the stride scaling.

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
    flagged_partial_frame: bool = False,
    edge_steps: int = DEFAULT_EDGE_STEPS,
    dynamic_range_min: float = DEFAULT_DYNAMIC_RANGE_MIN,
    off_target_range: float = DEFAULT_OFF_TARGET_RANGE,
) -> CropAnalysis:
    """Per-crop bell-curve verdict for one aperture.

    Per-crop dynamic range ``dr = (max - min) / max`` over the rail
    series is the primary classifier:

    * ``dr < off_target_range`` (default 0.05): short-circuit to
      :data:`CropStatus.OFF_TARGET`. The window almost certainly fell
      off the wafer onto featureless background; no peak detection is
      run. ``valid=False``.
    * ``off_target_range <= dr < dynamic_range_min`` (default 0.05 ≤
      dr < 0.25): a real but weak bell. Peak / truncation detection
      runs normally; an in-range peak becomes :data:`CropStatus.LOW_SIGNAL`
      (the operator-facing "soft lens" status). Truncation is reported
      as ``TRUNCATED_START`` / ``TRUNCATED_END`` exactly as for
      high-signal crops — truncation status takes precedence over the
      low-signal label so the runner still extends rail for these.
      ``valid=True``.
    * ``dr >= dynamic_range_min``: high signal. In-range peak becomes
      :data:`CropStatus.OK`; truncation handled as usual. ``valid=True``.

    ``flagged_off_target`` carries the cross-compare result and acts as
    a secondary ``OFF_TARGET`` signal for crops that *do* swing across
    the rail but whose peak is far weaker than the best crop in the
    aperture.

    ``flagged_partial_frame`` carries the absolute-size peer-compare
    result and routes to :data:`CropStatus.PARTIAL_FRAME` (the crop
    window extends past the wafer / target edge — distinct from
    OFF_TARGET in that the on-wafer portion still has real focus
    signal, but the dark-background filler dilutes the absolute size
    metric below peer-group median). Takes precedence over
    ``flagged_off_target`` because the diagnosis is more specific
    (re-frame vs re-jog), but only after the dr-floor OFF_TARGET
    check — a crop with no signal at all is still OFF_TARGET, even
    if it would also satisfy the peer-compare criterion.
    """
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
            valid=False,
            dynamic_range=0.0,
        )

    dr = dynamic_range(sizes)

    if dr < off_target_range:
        # Below the OFF_TARGET floor — window fell off the wafer.
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
            valid=False,
            dynamic_range=dr,
        )

    if flagged_partial_frame:
        # Absolute-size peer-compare flagged: the crop has real
        # dynamic range (it cleared the dr floor above) but its peak
        # absolute size is well below positionally similar peers,
        # consistent with the wafer / target edge cutting through the
        # crop window. Distinct from OFF_TARGET — the operator remedy
        # is reframing, not re-jogging.
        return CropAnalysis(
            crop_name=crop_name,
            status=CropStatus.PARTIAL_FRAME,
            n_steps=n,
            peak_step=None,
            peak_pct=None,
            estimated_peak_step=None,
            extra_steps_needed=None,
            max_size=max(sizes),
            min_size=min(sizes),
            valid=False,
            dynamic_range=dr,
        )

    if flagged_off_target:
        # Cross-compare flagged: crop has *some* dynamic range but its
        # peak is much weaker than the best crop in the aperture.
        # Treat as OFF_TARGET but valid=True so diagnostics distinguish
        # "fell off the wafer" (dynamic range below floor) from
        # "weaker than peers".
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
            valid=True,
            dynamic_range=dr,
        )

    # is_flat() is the legacy max/min < 1.15 detector. For high-signal
    # crops it's a useful safety net (a series that swings wildly but
    # has a max/min ratio just barely above 1.15 is rare and probably
    # noise); but in the LOW_SIGNAL band (dr < dynamic_range_min)
    # max/min will routinely fall below 1.15 *by definition*
    # (dr = 0.10 ⇔ max/min ≈ 1.11) — and we want those crops classified
    # as LOW_SIGNAL via the dr-based logic, not silently OFF_TARGET via
    # the ratio-based one. Only fall back to is_flat for crops that
    # cleared the high-signal floor.
    if dr >= dynamic_range_min and is_flat(sizes):
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
            valid=True,
            dynamic_range=dr,
        )

    pk = peak_index(sizes)
    pct = pk / max(1, n - 1)
    status = truncation_status(pk, n, edge_steps=edge_steps)
    # Vertex-projection fallback. When the caller narrowed
    # ``edge_steps`` (sparse aperture, stride > 1), two real-world
    # cases can still need classification past the basic peak-in-edge-
    # zone check:
    #
    # (1) Monotonic series. Smoothing trims the last sample so
    #     :func:`peak_index` always returns ``n-2`` for monotonic
    #     data; with a narrowed edge zone the truncation check would
    #     wrongly say OK. The parabolic fit either yields a vertex
    #     far past the captured end (for near-linear input,
    #     ``a`` slightly negative) or no vertex at all (for genuinely
    #     linear input, ``a >= 0``). The raw-argmax fallback covers
    #     the second case.
    # (2) Genuine bell whose peak just barely lands past the captured
    #     edge — e.g. corner field curvature with vertex within
    #     ``DEFAULT_EDGE_STEPS`` samples of ``n-1``. The narrowed
    #     edge zone correctly leaves this as OK; the vertex check
    #     confirms by not promoting it.
    #
    # The projection buffer is unscaled ``DEFAULT_EDGE_STEPS``
    # deliberately: vertex is in sample-index units and the
    # comparison is about how strongly the parabola extrapolates
    # past the captured range, independent of rail-step proximity.
    vertex = estimate_peak_vertex(sizes)
    if status == CropStatus.OK and n >= 2:
        if vertex is None:
            raw_max_idx = max(range(n), key=lambda i: sizes[i])
            if raw_max_idx == n - 1:
                status = CropStatus.TRUNCATED_END
            elif raw_max_idx == 0:
                status = CropStatus.TRUNCATED_START
        else:
            if vertex > (n - 1) + DEFAULT_EDGE_STEPS:
                status = CropStatus.TRUNCATED_END
            elif vertex < -DEFAULT_EDGE_STEPS:
                status = CropStatus.TRUNCATED_START
    # Demote an in-range OK to LOW_SIGNAL when the dynamic range is
    # below the high-signal floor. Truncation status (start/end/empty)
    # is untouched — truncated soft bells still need rail extension.
    if status == CropStatus.OK and dr < dynamic_range_min:
        status = CropStatus.LOW_SIGNAL
    if not status.is_truncated:
        vertex = None
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
        valid=True,
        dynamic_range=dr,
    )


def analyze_crops_for_aperture(
    crop_sizes: dict[str, list[int]],
    *,
    cross_compare_factor: float = DEFAULT_OFF_TARGET_FACTOR,
    partial_frame_factor: float = DEFAULT_PARTIAL_FRAME_FACTOR,
    peer_groups: list[list[str]] | tuple[tuple[str, ...], ...] = (),
    edge_steps: int = DEFAULT_EDGE_STEPS,
    dynamic_range_min: float = DEFAULT_DYNAMIC_RANGE_MIN,
    off_target_range: float = DEFAULT_OFF_TARGET_RANGE,
) -> list[CropAnalysis]:
    """Run the full per-crop analysis for one aperture.

    ``crop_sizes`` is ``{crop_name: [size_at_step_0, size_at_step_1, ...]}``.
    The returned list has one ``CropAnalysis`` per crop, in the
    iteration order of ``crop_sizes``.

    Cross-compare is computed only over the crops that clear the
    OFF_TARGET floor (``dynamic_range >= off_target_range``); a swarm
    of off-wafer corners can't drag down the reference max and
    falsely flag the few real crops. LOW_SIGNAL crops still
    participate in cross-compare — they are real signal, just weak.

    ``peer_groups`` enables the absolute-size partial-frame detector.
    Each group is a list of crop names at the same *position class*
    (e.g. the 4 full-frame corners). A crop whose peak absolute size
    falls below ``partial_frame_factor × group_median`` is routed to
    :data:`CropStatus.PARTIAL_FRAME` instead of OK/LOW_SIGNAL. Pass an
    empty tuple to disable (callers that don't know the crop layout
    — e.g. ad-hoc tests — can skip the detector this way).
    """
    valid_maxes: dict[str, int] = {
        name: max(sizes)
        for name, sizes in crop_sizes.items()
        if sizes and dynamic_range(sizes) >= off_target_range
    }
    flagged = off_target_by_cross_compare(valid_maxes, factor=cross_compare_factor)
    if peer_groups:
        flagged_partial = partial_frame_by_peer_compare(
            valid_maxes,
            [list(g) for g in peer_groups],
            factor=partial_frame_factor,
        )
    else:
        flagged_partial = set()
    return [
        analyze_crop(
            name,
            sizes,
            flagged_off_target=name in flagged,
            flagged_partial_frame=name in flagged_partial,
            edge_steps=edge_steps,
            dynamic_range_min=dynamic_range_min,
            off_target_range=off_target_range,
        )
        for name, sizes in crop_sizes.items()
    ]
