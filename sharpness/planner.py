"""Sample planner for rail-coverage sweeps.

The session captures every aperture at every rail step (the inverted
shot-order architecture in DESIGN_NOTES §2.6: outer rail loop, inner
aperture loop). One step size has to work for the entire aperture
sequence — there is no per-aperture step setting at capture time.

Given the lens's widest aperture (the sharpest peak; smallest DOF)
and the magnification, this module recommends:

- A single rail step in mm sized for the widest-aperture peak. At
  smaller apertures the bell is broader and you can analyse only
  every k-th position, but the *capture* still hits every step so
  the same data is reusable.
- An optional position count, derived if you also pass the rail
  range you intend to sweep.
- A per-aperture analysis subsample factor (integer ≥ 1) — at the
  smallest apertures (e.g. F16 with widest F4) the DOF is wide
  enough that you can analyse every 4th position with no peak-
  detection accuracy loss.

Inputs and why:

- ``magnification`` (M, e.g. 1.0 = 1:1) — required, drives DOF.
- ``widest_aperture`` (N_min, e.g. 2.8) — required, drives DOF at
  the worst case for sampling.
- ``max_aperture`` (N_max, default 16.0 to match
  ``run-lens-test/config.py:MAX_FSTOP``) — controls the per-aperture
  subsample table.
- ``rail_range_mm`` — optional. If supplied, the plan also reports a
  position count. The user normally jogs to a "before-focus" /
  "after-focus" pair so the range is determined by jogging, not by
  the planner; passing it just lets the planner translate step → count.
- ``coc_mm`` — circle of confusion. 30 µm matches the FX sensor and
  the historically-OK 25-position setup the user's existing rig used.
- ``replicas`` — replicas per rail position; default 3.

The historical default of 25 positions × 3 replicas at M=1, N=8 came
out at ~12 rail positions across the FWHM. Empirical resampling on
that data (see DESIGN_NOTES section on the experiment) shows that
**5 positions across FWHM** suffices to find the peak within ~1 % of
the fitted maximum at moderate magnifications, and **8** is enough at
high magnification (M ≥ 1.5) where the peak is narrow. We use those
two values as the two operating points; aperture and low-mag bumps
are NOT applied here because the experiment showed they don't help
in the intended sense (the broader-bell regimes are *more* tolerant
of sparse sampling, not less).

Example::

    >>> from sharpness.planner import plan_sweep
    >>> p = plan_sweep(magnification=1.0, widest_aperture=2.8)
    >>> round(p.rail_step_mm, 4)
    0.0672
    >>> p.subsample_factors[8.0]
    2
    >>> p.subsample_factors[16.0]
    5
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Circle of confusion at which sharpness peaks resolve. The standard
# 30 µm FX value matches the user's historically-OK setup.
DEFAULT_COC_MM = 0.030

DEFAULT_REPLICAS = 3

# Match config.py:MAX_FSTOP from run-lens-test.
DEFAULT_MAX_APERTURE = 16.0

# Aperture stops the planner reports subsample factors for. Picked to
# cover whole stops and ⅓ stops in the camera-control range.
KNOWN_APERTURE_STOPS: tuple[float, ...] = (
    1.4, 2.0, 2.5, 2.8, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0,
    9.0, 10.0, 11.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 29.0, 32.0,
)

# Number of rail positions to span the FWHM at the widest aperture.
# 5 came out of the resampling experiment as "enough for moderate
# magnification" — k=5 gave <2 % sharpness loss in F8/F11/F5.6
# scenarios at M=1.
POSITIONS_ACROSS_FWHM_DEFAULT = 5

# Bump for high magnification. The 200mm + 5T + 6T at 2:1 / F8 case
# needed k=15 across the full range (~7-8 across FWHM at F8) to find
# the peak within 1 %. We bump to 8 across FWHM whenever M ≥ 1.5.
POSITIONS_ACROSS_FWHM_HIGH_MAG = 8
HIGH_MAG_THRESHOLD = 1.5

# Practical lower bound on rail step in mm. The Cognisys StackShot
# accepts arbitrarily small distances over the wire but the rail's
# mechanical resolution (leadscrew pitch + motor backlash) is around
# 2–5 µm; below ~5 µm the requested move starts disappearing into
# mechanical noise. The planner clamps any recommended step at this
# floor and reports ``step_clamped=True`` so the caller knows the
# step would otherwise have been finer (and the actual sweep will
# under-sample the bell at that point).
MIN_RAIL_STEP_MM = 0.005


@dataclass(frozen=True)
class SamplePlan:
    """Recommended rail sweep settings.

    Distances are millimetres. ``subsample_factors`` is keyed by
    aperture (e.g. ``8.0``) and holds the integer skip factor for
    analysis at that aperture: 1 = use every captured step, 2 = use
    every other step, 3 = every third, etc.

    ``rail_range_mm`` and ``rail_positions`` are populated only when
    the caller supplied a rail range; otherwise they are ``None``
    because the planner has no opinion on how far to jog.
    """

    rail_step_mm: float
    replicas_per_position: int
    estimated_dof_mm_widest: float
    estimated_fwhm_mm_widest: float
    widest_aperture: float
    max_aperture: float
    subsample_factors: dict[float, int] = field(default_factory=dict)
    rail_range_mm: float | None = None
    rail_positions: int | None = None
    step_clamped: bool = False  # True when the StackShot floor was hit

    @property
    def total_shots(self) -> int | None:
        """Total shots = positions × replicas, or ``None`` if no range was passed."""
        if self.rail_positions is None:
            return None
        return self.rail_positions * self.replicas_per_position


def parse_magnification(text: str) -> float:
    """Parse a magnification expression to a float.

    Accepts:
    * Ratio form: ``"1:1"`` → 1.0, ``"1.5:1"`` → 1.5, ``"1:1.5"`` →
      ≈ 0.667, ``"2:1"`` → 2.0. Numerator and denominator must both
      be positive numbers.
    * Bare float: ``"1.0"`` → 1.0, ``"0.5"`` → 0.5, ``"1"`` → 1.0.

    Whitespace tolerant. Raises :class:`ValueError` on unparseable
    input or non-positive result.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty magnification")
    if ":" in cleaned:
        parts = cleaned.split(":")
        if len(parts) != 2:
            raise ValueError(f"bad ratio: {text!r}")
        num_text, den_text = parts[0].strip(), parts[1].strip()
        try:
            num = float(num_text)
            den = float(den_text)
        except ValueError as exc:
            raise ValueError(f"bad ratio: {text!r}") from exc
        if num <= 0 or den <= 0:
            raise ValueError(f"ratio must be positive: {text!r}")
        return num / den
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"bad magnification: {text!r}") from exc
    if value <= 0:
        raise ValueError(f"magnification must be positive, got {value}")
    return value


def parse_aperture(text: str) -> float:
    """Parse an aperture expression like ``"f/4"``, ``"F/4.5"``,
    ``"4"``, or ``"4.5"`` to a float.

    Strips an optional leading ``f`` / ``F`` and ``/`` so both the
    Z7 widget format (``f/4.5``) and bare numbers work. Raises
    :class:`ValueError` on bad input.
    """
    cleaned = text.strip().lstrip("fF").lstrip("/")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"bad aperture: {text!r}") from exc
    if value <= 0:
        raise ValueError(f"aperture must be positive, got {value}")
    return value


def estimate_dof_mm(
    magnification: float,
    aperture: float,
    *,
    coc_mm: float = DEFAULT_COC_MM,
) -> float:
    """Total depth of field in mm at the given magnification and aperture.

    Thin-lens macro form: ``DOF = 2 · c · N · (1 + M) / M²``.
    """
    if magnification <= 0:
        raise ValueError(f"magnification must be positive, got {magnification}")
    if aperture <= 0:
        raise ValueError(f"aperture must be positive, got {aperture}")
    return 2 * coc_mm * aperture * (1 + magnification) / (magnification**2)


def subsample_factor(aperture: float, widest_aperture: float) -> int:
    """Integer analysis-time subsample factor at ``aperture``.

    DOF scales linearly with aperture, so at ``N`` the bell is
    ``N / N_min`` times wider in mm than at ``N_min``. Analysing
    every ``floor(N / N_min)``-th captured step keeps the same
    bell-curve density at every aperture.
    """
    if aperture < widest_aperture:
        # Below the widest aperture the lens can't operate; defensive.
        return 1
    return max(1, int(aperture / widest_aperture))


def plan_sweep(
    magnification: float,
    widest_aperture: float,
    *,
    max_aperture: float = DEFAULT_MAX_APERTURE,
    rail_range_mm: float | None = None,
    coc_mm: float = DEFAULT_COC_MM,
    replicas: int = DEFAULT_REPLICAS,
    positions_across_fwhm: int | None = None,
    min_step_mm: float = MIN_RAIL_STEP_MM,
) -> SamplePlan:
    """Recommend a rail step for the whole aperture sweep.

    The step is sized for the widest aperture (the worst case for
    sampling — narrowest DOF, sharpest peak). Smaller apertures
    automatically have a broader bell over the same step set, so the
    *captured* data covers them adequately; the per-aperture
    ``subsample_factors`` table tells the analyser how aggressively
    to subsample at each aperture without losing peak precision.

    By default the step gives 5 positions across the FWHM at the
    widest aperture, bumped to 8 when ``M ≥ HIGH_MAG_THRESHOLD``
    (1.5). Both values were validated by resampling the existing
    25-position-per-scenario data: 5 was enough at moderate mag,
    and 8 was needed for the high-mag narrow-peak case.

    The recommended step is clamped at ``min_step_mm`` (default 5 µm,
    the StackShot's mechanical noise floor). When the clamp fires
    the returned plan has ``step_clamped=True`` and the caller
    should expect the bell to be under-sampled at the widest
    aperture — there's no way to physically do better with this
    rail.

    Pass ``rail_range_mm`` to also get a position count for that
    range; otherwise just step size is returned.
    """
    if widest_aperture <= 0:
        raise ValueError(f"widest_aperture must be positive, got {widest_aperture}")
    if max_aperture < widest_aperture:
        raise ValueError(
            f"max_aperture ({max_aperture}) must be ≥ widest_aperture ({widest_aperture})"
        )
    if min_step_mm <= 0:
        raise ValueError(f"min_step_mm must be positive, got {min_step_mm}")

    if positions_across_fwhm is None:
        if magnification >= HIGH_MAG_THRESHOLD:
            positions_across_fwhm = POSITIONS_ACROSS_FWHM_HIGH_MAG
        else:
            positions_across_fwhm = POSITIONS_ACROSS_FWHM_DEFAULT

    dof = estimate_dof_mm(magnification, widest_aperture, coc_mm=coc_mm)
    fwhm = dof / 2
    raw_step = fwhm / positions_across_fwhm
    if raw_step < min_step_mm:
        rail_step_mm = min_step_mm
        step_clamped = True
    else:
        rail_step_mm = raw_step
        step_clamped = False

    factors = {
        ap: subsample_factor(ap, widest_aperture)
        for ap in KNOWN_APERTURE_STOPS
        if widest_aperture <= ap <= max_aperture
    }

    rail_positions: int | None = None
    if rail_range_mm is not None:
        if rail_range_mm <= 0:
            raise ValueError(f"rail_range_mm must be positive, got {rail_range_mm}")
        rail_positions = max(1, round(rail_range_mm / rail_step_mm))

    return SamplePlan(
        rail_step_mm=rail_step_mm,
        replicas_per_position=replicas,
        estimated_dof_mm_widest=dof,
        estimated_fwhm_mm_widest=fwhm,
        widest_aperture=widest_aperture,
        max_aperture=max_aperture,
        subsample_factors=factors,
        rail_range_mm=rail_range_mm,
        rail_positions=rail_positions,
        step_clamped=step_clamped,
    )
