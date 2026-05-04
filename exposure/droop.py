"""Per-aperture flash-battery droop detector.

The wireless flashes weaken as their AA batteries deplete; recycle
slows first, then the flash starts under-exposing. The detector
watches a rolling window of EV-relative-to-baseline values per
aperture and trips when sustained underexposure shows up.

Two trip criteria — whichever fires first:

* Criterion A: ``DROOP_CRIT_A`` — e.g. 3 of last 10 shots ≤ -0.7 EV.
  Catches gradual drift before any single shot is catastrophically
  dim.
* Criterion B: ``DROOP_CRIT_B`` — e.g. 2 of last 10 shots ≤ -2.0 EV.
  Catches abrupt power loss faster than criterion A would.

Calibration-phase shots and shots taken during the aperture loop of
a step that is already aborted **must not** enter the window — those
are not representative of normal sweep performance. Only sweep-phase
shots feed the detector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


# ---- Trip criterion --------------------------------------------------------


@dataclass(frozen=True)
class TripCriterion:
    """``count`` of last ``window`` samples must be ``<= ev_threshold``."""

    count: int
    ev_threshold: float

    def fires(self, samples: deque) -> bool:
        # samples is a deque of EV values, most recent on the right.
        if len(samples) < self.count:
            return False
        triggered = sum(1 for v in samples if v <= self.ev_threshold)
        return triggered >= self.count


# ---- Result of a push ------------------------------------------------------


@dataclass(frozen=True)
class DroopEvent:
    """Returned when the rolling window trips."""

    aperture: str
    ev_value: float
    triggered_count: int
    threshold: float
    window_snapshot: tuple[float, ...]


# ---- Detector --------------------------------------------------------------


class DroopDetector:
    """Per-aperture rolling window of sweep-phase EV values.

    The window length and trip criteria are taken at construction
    time; pass tuples like ``(3, -0.7)`` exactly as configured in
    :mod:`config`.
    """

    def __init__(
        self,
        *,
        window: int,
        criteria: tuple[tuple[int, float], ...],
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._criteria = tuple(
            TripCriterion(count=c, ev_threshold=t) for c, t in criteria
        )
        self._values: dict[str, deque[float]] = {}

    def push(self, aperture: str, ev_value: float) -> DroopEvent | None:
        """Record a sweep shot's EV. Returns a :class:`DroopEvent` if
        any trip criterion fires after this push."""
        window = self._values.setdefault(
            aperture, deque(maxlen=self._window)
        )
        window.append(float(ev_value))
        for criterion in self._criteria:
            if criterion.fires(window):
                triggered = sum(
                    1 for v in window if v <= criterion.ev_threshold
                )
                return DroopEvent(
                    aperture=aperture,
                    ev_value=float(ev_value),
                    triggered_count=triggered,
                    threshold=criterion.ev_threshold,
                    window_snapshot=tuple(window),
                )
        return None

    def reset_all(self) -> None:
        """Clear every aperture's window.

        Called after the user resumes from a battery-droop pause so
        we don't immediately re-trip on the same low-EV samples.
        """
        for window in self._values.values():
            window.clear()

    def window_for(self, aperture: str) -> tuple[float, ...]:
        """Read-only snapshot of one aperture's window. Empty tuple if
        nothing pushed yet."""
        return tuple(self._values.get(aperture, ()))
