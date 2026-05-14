"""Session-level flash-battery droop detector.

The wireless flashes weaken as their AA batteries deplete; recycle
slows first, then the flash starts under-exposing. The detector
watches a *single* rolling window of EV-relative-to-baseline values
across the whole sweep — pooled across all apertures — and trips
when sustained off-target exposure shows up. Both directions trip:
under-exposure (battery droop) and over-exposure (rig-state shift,
e.g. flash position bumped during a battery swap).

Pooling across apertures is deliberate: a real flash droop tends to
spread across consecutive captures regardless of the aperture they
happened at. If two adjacent shots at f/5.6 and f/8 both come back
≤ −0.7 EV, that's already enough signal to act, even though
neither aperture has been pushed against twice. Per-aperture
isolation (the original design) was insensitive to that pattern.

Trip criteria — whichever fires first:

* Under, severe: a single shot ≤ −2 EV. Catches a flash hitting
  its mechanical-min on one capture.
* Under, moderate: two of the last ``ROLLING_WINDOW`` shots ≤ −0.7
  EV. Catches gradual battery drift before catastrophic.
* Over, severe: a single shot ≥ +2 EV. Catches gross rig-shift
  events (e.g. flash AA sensor view fully clipped).
* Over, moderate: two of the last ``ROLLING_WINDOW`` shots ≥ +0.7
  EV. Catches subtler rig-shifts (e.g. flash A bumped during a
  battery swap → AA over-fires by ~⅔ stop on subsequent frames).

The :class:`DroopEvent` carries a ``direction`` field ("under" or
"over") so the caller can branch: under → prompt battery swap,
over → silent re-calibration.

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
    """``count`` of last ``window`` samples must cross ``ev_threshold``
    in ``direction`` (``"under"`` = ``<=`` with a negative threshold,
    ``"over"`` = ``>=`` with a positive threshold).

    Direction is auto-derived from ``ev_threshold``'s sign when not
    supplied: negative → ``"under"``, positive → ``"over"``. The
    explicit ``direction`` arg lets tests build either-direction
    criteria with zero threshold without relying on sign.
    """

    count: int
    ev_threshold: float
    direction: str = ""

    def __post_init__(self) -> None:
        if not self.direction:
            object.__setattr__(
                self, "direction",
                "over" if self.ev_threshold > 0 else "under",
            )
        if self.direction not in ("under", "over"):
            raise ValueError(
                f"direction must be 'under' or 'over', got {self.direction!r}"
            )

    def fires(self, samples: deque) -> bool:
        # samples is a deque of EV values, most recent on the right.
        if len(samples) < self.count:
            return False
        if self.direction == "under":
            triggered = sum(1 for v in samples if v <= self.ev_threshold)
        else:
            triggered = sum(1 for v in samples if v >= self.ev_threshold)
        return triggered >= self.count


# ---- Window entry / DroopEvent --------------------------------------------


@dataclass(frozen=True)
class WindowEntry:
    """One sample in the rolling window — the EV delta and the
    rail step it was captured at, plus the aperture (so the runner's
    droop-recovery code can pick out which aperture(s) contributed
    to the trip).
    """

    step_index: int
    ev_value: float
    aperture: str = ""


@dataclass(frozen=True)
class DroopEvent:
    """Returned when the rolling window trips."""

    aperture: str
    """The aperture of the shot that *just* pushed the criterion
    over. With session-level pooling the trip can involve multiple
    apertures; this field reports the most recent one."""

    ev_value: float
    triggered_count: int
    threshold: float
    window_snapshot: tuple[WindowEntry, ...]
    direction: str = "under"
    """``"under"`` for battery-droop (negative-EV) trips —
    handler prompts the user to swap batteries. ``"over"`` for
    rig-state-shift (positive-EV) trips — handler re-calibrates
    silently and reshoots without a user pause."""


# ---- Detector --------------------------------------------------------------


class DroopDetector:
    """Session-level rolling window of sweep-phase samples.

    A single deque — not per-aperture — accumulates every push from
    the sweep. This pools cross-aperture signal: two consecutive
    under-exposed shots at *different* apertures still trip the
    moderate criterion even though no single aperture has two low
    shots in a row.
    """

    # Sentinel for callers (almost always tests) that don't carry a
    # rail step index. Production runner-side pushes always supply a
    # real step.
    UNKNOWN_STEP: int = -1

    def __init__(
        self,
        *,
        window: int,
        criteria: tuple,
    ) -> None:
        """``criteria`` is an iterable of either ``(count, threshold)``
        tuples (direction auto-derived from sign) or ``(count,
        threshold, direction)`` tuples, or :class:`TripCriterion`
        instances directly. Older tests pass 2-tuples for the
        under-only API; the auto-direction handles those without
        change."""
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        built: list[TripCriterion] = []
        for c in criteria:
            if isinstance(c, TripCriterion):
                built.append(c)
            elif len(c) == 2:
                count, threshold = c
                built.append(TripCriterion(count=count, ev_threshold=threshold))
            elif len(c) == 3:
                count, threshold, direction = c
                built.append(
                    TripCriterion(
                        count=count, ev_threshold=threshold, direction=direction,
                    )
                )
            else:
                raise ValueError(
                    f"criterion must be 2- or 3-tuple or TripCriterion, got {c!r}"
                )
        self._criteria = tuple(built)
        self._values: deque[WindowEntry] = deque(maxlen=window)

    def push(
        self,
        aperture: str,
        ev_value: float,
        *,
        step_index: int = UNKNOWN_STEP,
    ) -> DroopEvent | None:
        """Record a sweep shot's EV. Returns a :class:`DroopEvent` if
        any trip criterion fires after this push."""
        entry = WindowEntry(
            step_index=int(step_index),
            ev_value=float(ev_value),
            aperture=aperture,
        )
        self._values.append(entry)
        ev_window: deque[float] = deque(e.ev_value for e in self._values)
        for criterion in self._criteria:
            if criterion.fires(ev_window):
                if criterion.direction == "under":
                    triggered = sum(
                        1 for v in ev_window if v <= criterion.ev_threshold
                    )
                else:
                    triggered = sum(
                        1 for v in ev_window if v >= criterion.ev_threshold
                    )
                return DroopEvent(
                    aperture=aperture,
                    ev_value=float(ev_value),
                    triggered_count=triggered,
                    threshold=criterion.ev_threshold,
                    window_snapshot=tuple(self._values),
                    direction=criterion.direction,
                )
        return None

    def reset_all(self) -> None:
        """Clear the session-level window.

        Called after the user resumes from a battery-droop pause and
        any reshoot is done, so we don't immediately re-trip on the
        same low-EV samples.
        """
        self._values.clear()

    def window_for(self, aperture: str) -> tuple[float, ...]:
        """Read-only view of the EV values in the session-level
        window that came from one specific aperture. Empty tuple if
        nothing matches.

        Backwards-compat with the per-aperture API: callers that
        want a single aperture's series still get it, just filtered
        out of the shared window."""
        return tuple(
            e.ev_value for e in self._values if e.aperture == aperture
        )

    def window_entries_for(
        self, aperture: str
    ) -> tuple[WindowEntry, ...]:
        """Filtered (step, EV, aperture) view for one aperture."""
        return tuple(e for e in self._values if e.aperture == aperture)

    def session_window_entries(self) -> tuple[WindowEntry, ...]:
        """Full session-level rolling window — every aperture pooled
        in capture order, oldest first."""
        return tuple(self._values)

    def earliest_underexposed_step(
        self, threshold: float
    ) -> int | None:
        """Backwards-compat alias for ``earliest_anomalous_step`` with
        ``direction="under"``. ``threshold`` must be negative."""
        return self.earliest_anomalous_step(threshold, direction="under")

    def earliest_anomalous_step(
        self, threshold: float, *, direction: str = "under"
    ) -> int | None:
        """Smallest step_index in the session-level window whose
        ``ev_value`` crosses ``threshold`` in ``direction``. ``None``
        if no such sample exists. Step indices set to
        :attr:`UNKNOWN_STEP` are ignored (their position in the sweep
        isn't known).

        ``direction="under"`` → matches ``ev_value <= threshold``
        (threshold should be negative or zero).
        ``direction="over"`` → matches ``ev_value >= threshold``
        (threshold should be positive or zero).
        """
        if direction not in ("under", "over"):
            raise ValueError(
                f"direction must be 'under' or 'over', got {direction!r}"
            )
        earliest: int | None = None
        for entry in self._values:
            if entry.step_index == self.UNKNOWN_STEP:
                continue
            if direction == "under":
                matches = entry.ev_value <= threshold
            else:
                matches = entry.ev_value >= threshold
            if matches:
                if earliest is None or entry.step_index < earliest:
                    earliest = entry.step_index
        return earliest
