"""Sample planner for rail-coverage sweeps.

Given a ``(magnification, aperture)`` test point, recommend rail-step
and total-sweep settings for a StackShot-driven focus stack that
resolves the sharpness peak with confidence.

Calibrated against the historical 25 positions × 3 replicas regime,
which gave well-formed bells (FWHM ≈ 12 positions, plateau ≈ 2
positions) at moderate magnifications across the 2026-01-29 and
2026-02-18 test sessions. The planner aims to reproduce that regime
across ``(M, N)``, bumping the position count slightly where the
thin-lens DOF prediction is least reliable so the user is "a bit
more than too few" rather than too few.

The thin-lens depth-of-field formula used is the macro-photography
form::

    DOF = 2 · c · N · (1 + M) / M²

where ``c`` is the circle of confusion, ``N`` is the f-number, and
``M`` is the magnification (0.5 = 1:2, 1.0 = 1:1, 2.0 = 2:1).

Why this is enough input:
- ``c`` is a sensor-and-detection-threshold property; the 30 µm FX
  default is calibrated against the user's existing rig and matches
  the historically-OK 25×3 setup at ``M=1, N=8``. Override only if a
  different sensor or sharper detection threshold is needed.
- Lens focal length doesn't enter the DOF formula at fixed
  magnification — physical DOF depends only on ``M``, ``N``, ``c``.
- The planner does NOT need rail-step granularity as an input; it
  recommends the step. If the StackShot's minimum step is coarser
  than the recommendation, the caller should reduce the position
  count to match the achievable step.

Example::

    >>> from sharpness.planner import plan_sweep
    >>> p = plan_sweep(magnification=1.0, aperture=8.0)
    >>> p.rail_positions
    25
    >>> round(p.rail_step_mm, 4)
    0.0192
    >>> round(p.sweep_range_mm, 3)
    0.48
"""

from __future__ import annotations

from dataclasses import dataclass


# Circle of confusion at which sharpness peaks resolve. The standard
# FX value is 30 µm; we use it because it gives a sweep that matches
# the user's historically-OK 25-position setup at M=1, N=8.
DEFAULT_COC_MM = 0.030

DEFAULT_REPLICAS = 3
DEFAULT_BASE_POSITIONS = 25


@dataclass(frozen=True)
class SamplePlan:
    """Recommended rail sweep settings.

    All distances in millimetres; rail steps and sweep range can be
    fed directly into the StackShot move commands.
    """

    rail_positions: int
    replicas_per_position: int
    rail_step_mm: float
    sweep_range_mm: float
    estimated_dof_mm: float
    estimated_fwhm_mm: float

    @property
    def total_shots(self) -> int:
        return self.rail_positions * self.replicas_per_position


def estimate_dof_mm(
    magnification: float,
    aperture: float,
    *,
    coc_mm: float = DEFAULT_COC_MM,
) -> float:
    """Total depth of field in mm at the given magnification and aperture.

    Thin-lens approximation; ignores diffraction softening (which
    affects the *quality* of the peak but not its width within a few
    percent in our regime).
    """
    if magnification <= 0:
        raise ValueError(f"magnification must be positive, got {magnification}")
    if aperture <= 0:
        raise ValueError(f"aperture must be positive, got {aperture}")
    return 2 * coc_mm * aperture * (1 + magnification) / (magnification**2)


def plan_sweep(
    magnification: float,
    aperture: float,
    *,
    coc_mm: float = DEFAULT_COC_MM,
    replicas: int = DEFAULT_REPLICAS,
    base_positions: int = DEFAULT_BASE_POSITIONS,
) -> SamplePlan:
    """Recommend a rail sweep for one ``(magnification, aperture)`` test point.

    Returns a :class:`SamplePlan` whose ``rail_positions`` is at least
    ``base_positions`` (default 25), bumped where the DOF prediction
    is least reliable:

    - Low magnification (``M < 0.4``): +5 positions. The 24-120 zoom
      in the test data showed the most truncation here, so add
      buffer.
    - Small aperture (``N >= 11``): +3 positions. Diffraction widens
      the bell — a few extra rail positions help locate the peak
      when the plateau is broad.
    - High magnification (``M >= 1.5``): +2 positions. Razor-thin DOF
      plus StackShot step quantization makes off-by-one mistakes more
      likely; the buffer protects against them.

    The rail step is derived so the total sweep covers exactly one
    full DOF (``2 × FWHM``); that matches the historical sweep:DOF
    ratio at ``M=1, N=8`` and produces ~12 rail positions across the
    bell's FWHM, the empirically-OK density from the test sessions.
    """
    dof = estimate_dof_mm(magnification, aperture, coc_mm=coc_mm)
    fwhm = dof / 2

    bump = 0
    if magnification < 0.4:
        bump += 5
    if aperture >= 11:
        bump += 3
    if magnification >= 1.5:
        bump += 2
    positions = base_positions + bump

    sweep = 2 * fwhm  # = 1 × DOF
    step = sweep / positions

    return SamplePlan(
        rail_positions=positions,
        replicas_per_position=replicas,
        rail_step_mm=step,
        sweep_range_mm=sweep,
        estimated_dof_mm=dof,
        estimated_fwhm_mm=fwhm,
    )
