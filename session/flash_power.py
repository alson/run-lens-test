"""Flash-power formatting and target-computation helpers (manual mode).

Three pure functions used by the manual-aperture calibration flow:

* :func:`power_label` — render a stops-below-full float as the Nikon
  SB-5000 commander's display format (``"1/1"``, ``"1/2 -1/3"``,
  ``"1/128"``). Used by the rig's prompts so the user reads the
  exact label the commander shows.

* :func:`parse_power_label` — inverse, mostly for tests.

* :func:`target_for_next_aperture` — given the previous aperture's
  calibrated (flash_power, iso) and the aperture-stops difference
  to the next aperture, compute the (new_power, new_iso) the rig
  should request. Honours soft / hard power caps and ISO floor /
  warn / ceiling thresholds. See module docstring for the algorithm.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

import numpy as np


# ---- Power-label formatting (Nikon SB-5000 commander display) -------------


_LABEL_RE = re.compile(
    r"^\s*1\s*/\s*(?P<base>\d+)\s*(?P<frac>-\s*0\.3|-\s*0\.7)?\s*$"
)


def power_label(stops_below_full: float) -> str:
    """Render stops-below-full as the SB-5000 commander's display.

    Stops are counted from 1/1 (full power). The commander shows
    integer bases ``1/1, 1/2, 1/4, …, 1/128`` (7 stops total) and
    1/3-stop sub-steps as ``-0.3`` (1/3 stop) and ``-0.7`` (2/3 stop)
    below each base — matching the on-screen decimal format the
    operator reads.

    Examples::

        power_label(0)    -> "1/1"
        power_label(1/3)  -> "1/1 -0.3"
        power_label(2/3)  -> "1/1 -0.7"
        power_label(1)    -> "1/2"
        power_label(4/3)  -> "1/2 -0.3"
        power_label(2)    -> "1/4"
        power_label(7)    -> "1/128"

    Rounds the input to the nearest 1/3 stop before formatting.

    Raises :class:`ValueError` if the rounded value is negative
    (above 1/1) or below 1/128 (more than 7 stops below full).
    """
    third_stops = round(stops_below_full * 3)
    if third_stops < 0:
        raise ValueError(
            f"stops_below_full={stops_below_full} is above 1/1 (full power)"
        )
    if third_stops > 7 * 3:
        raise ValueError(
            f"stops_below_full={stops_below_full} is below 1/128 "
            f"(SB-5000 minimum)"
        )
    full_stops = third_stops // 3
    remainder = third_stops % 3
    base_denom = 2 ** full_stops
    if remainder == 0:
        return f"1/{base_denom}"
    fractional_label = "-0.3" if remainder == 1 else "-0.7"
    return f"1/{base_denom} {fractional_label}"


def parse_power_label(label: str) -> float:
    """Inverse of :func:`power_label`. Mostly used by tests."""
    match = _LABEL_RE.match(label)
    if not match:
        raise ValueError(f"unparseable power label: {label!r}")
    base = int(match.group("base"))
    if base <= 0 or (base & (base - 1)) != 0:
        # Not a power of two — invalid SB-5000 base.
        raise ValueError(f"invalid power base: 1/{base}")
    full_stops = int(round(math.log2(base)))
    frac = match.group("frac")
    if frac is None:
        return float(full_stops)
    third = 1 / 3 if "0.3" in frac else 2 / 3
    return full_stops + third


# ---- Per-aperture target computation --------------------------------------


class FlashTarget(NamedTuple):
    """Result of :func:`target_for_next_aperture`.

    Attributes
    ----------
    power_stops_below_full:
        Target flash power as stops below 1/1, rounded to 1/3 stop.
    iso:
        Target ISO. May be above the floor when the power cap forces
        ISO to carry the residual.
    """
    power_stops_below_full: float
    iso: int


# Score weights for the brute-force (power, iso) selection in
# :func:`target_for_next_aperture`. Tuned so that:
#
# * Primary objective is minimal |light_change − target|.
# * Going above the soft cap (brighter than 1/2 power) is mildly
#   discouraged but not forbidden — pays off when the alternative is
#   a larger error from snapping to the discrete ladder.
# * Going above the ISO floor is very mildly discouraged so that on
#   exact ties the algorithm prefers lower ISO.
_SOFT_CAP_PENALTY_PER_STOP = 0.10  # EV per stop above the soft cap
_ISO_PENALTY_PER_STOP = 0.02       # EV per stop above the ISO floor


def target_for_next_aperture(
    *,
    prev_power_stops: float,
    prev_iso: int,
    aperture_stops_diff: float,
    soft_cap_stops: float,
    hard_cap_stops: float,
    iso_floor: int,
    iso_warn: int,
    iso_ceil: int,
    prev_ev_delta: float = 0.0,
) -> FlashTarget:
    """Pick the (power, iso) tuple that lands closest to baseline
    after the next aperture step.

    Brute-force optimisation over the full ⅓-stop flash power ladder
    (0..7 stops below full) crossed with the camera ISO ladder
    ``[iso_floor, iso_ceil]``. Scoring:

    * ``error = |light_change − target_light_change|`` in EV, where
      ``target_light_change = aperture_stops_diff − prev_ev_delta``
      (i.e. we account for any leftover under/over-exposure from the
      previous aperture and aim to *correct* it on this aperture).
    * Soft-cap penalty: ``_SOFT_CAP_PENALTY_PER_STOP × max(0,
      soft_cap_stops − power)``. Going above the soft cap is allowed
      when the resulting error is small enough that the penalty pays
      for itself.
    * ISO penalty: ``_ISO_PENALTY_PER_STOP × log2(iso / iso_floor)``.
      Tiny — only breaks exact ties in favour of lower ISO.
    * Forbidden: ``iso > iso_warn`` unless ``power == hard_cap_stops``
      (full power). Reflects the spec: only push ISO past
      ``iso_warn`` after the flash is already at the hard cap.

    Vectorised over the (P × I) grid with NumPy; the grid is small
    (~22 × ~20) so the cost is negligible.

    ``prev_ev_delta`` is optional and defaults to ``0`` — when the
    previous aperture landed exactly at baseline (no overshoot),
    the algorithm just compensates for the aperture step itself.
    """
    powers = np.arange(0, 22, dtype=float) / 3.0  # 0, 1/3, 2/3, ..., 7
    iso_arr = np.array(
        [v for v in _ISO_LADDER if iso_floor <= v <= iso_ceil],
        dtype=float,
    )
    if iso_arr.size == 0:
        # Pathological: floor > ceil. Return a stub.
        return FlashTarget(soft_cap_stops, iso_floor)

    target_light_change = aperture_stops_diff - prev_ev_delta

    # Per-cell light change vs prev: (prev_power − power) + log2(iso / prev_iso).
    power_contrib = prev_power_stops - powers  # shape (P,)
    iso_contrib = np.log2(iso_arr / prev_iso)  # shape (I,)
    light_change = power_contrib[:, None] + iso_contrib[None, :]  # (P, I)

    error = np.abs(light_change - target_light_change)

    # Soft-cap penalty: positive when power is above soft cap (i.e.
    # power_stops < soft_cap_stops, meaning closer to full power).
    above_soft = np.clip(soft_cap_stops - powers, 0.0, None)
    cap_penalty = above_soft[:, None] * _SOFT_CAP_PENALTY_PER_STOP

    # ISO penalty (very small).
    iso_above_floor = np.log2(iso_arr / iso_floor)
    iso_penalty = iso_above_floor[None, :] * _ISO_PENALTY_PER_STOP

    score = error + cap_penalty + iso_penalty

    # Forbid ISO > warn when power is not at hard cap.
    iso_over_warn = iso_arr > iso_warn
    power_above_hard_cap = powers > hard_cap_stops + 1e-6
    forbidden = power_above_hard_cap[:, None] & iso_over_warn[None, :]
    score = np.where(forbidden, np.inf, score)

    flat_idx = int(np.argmin(score))
    p_idx, i_idx = divmod(flat_idx, len(iso_arr))
    return FlashTarget(
        power_stops_below_full=float(powers[p_idx]),
        iso=int(iso_arr[i_idx]),
    )


# ⅓-stop ISO ladder used by :func:`target_for_next_aperture`.
# Mirrors :data:`session.calibrate.ISO_LADDER` but lives here as a
# module-local copy so flash_power has no upward import on calibrate.
# Includes ISO 50 (Z7's Lo 1 extended-low) for completeness — the
# brute-force search filters to ``[iso_floor, iso_ceil]`` so callers
# control whether 50 is reachable.
_ISO_LADDER: tuple[int, ...] = (
    50,
    100, 125, 160,
    200, 250, 320,
    400, 500, 640,
    800, 1000, 1250,
    1600, 2000, 2500,
    3200, 4000, 5000, 6400,
)
