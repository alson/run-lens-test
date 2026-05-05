"""End-of-lens-config range-too-narrow check + auto-extension hook.

The detector decides whether the rail range covered the
peak-sharpness focus position for each aperture. When the peak is
outside the captured range it tells the runner which side to extend
toward; the runner shoots a small probe in that extension, calls back
into the detector, and either runs a full extension sweep or prompts
the user to re-jog.

This module currently ships a **stub** :class:`RangeChecker` that
always reports ``ok=True``. The real algorithm — sharpness via
multiple per-JPEG crops, with focus-shift handling per aperture — is
deferred until the user provides their data-driven implementation.
The runner consumes only the contract defined here, so swapping in
the real detector later is a single-file change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PeakSide = Literal["near", "far", "both", "centered"]
"""Side of the captured rail range the peak sharpness was detected on.

* ``"near"`` — peak is at the position-0 end; runner extends backwards.
* ``"far"`` — peak is at the END_MM end; runner extends forwards.
* ``"both"`` — different apertures peaked on different sides (focus
  shift); runner extends bilaterally.
* ``"centered"`` — peak is comfortably inside the captured range; no
  extension needed.
"""


@dataclass
class ApertureCurve:
    """Per-aperture sharpness curve, one point per rail step.

    The detector populates this; the runner persists it (in
    ``range_check.json``) for human review and for the offline
    sharpness analysis tool.
    """

    aperture: str                      # e.g. "f/4"
    step_indices: list[int]
    rail_positions_mm: list[float]
    sharpness_values: list[float]      # detector-defined metric


@dataclass
class RangeResult:
    """What the detector returns from :meth:`RangeChecker.evaluate`."""

    ok: bool
    """``True`` if the captured range comfortably covered peak sharpness
    for every aperture; the runner moves on to lens change."""

    peak_side: PeakSide | None
    """Which end of the captured range the peak is closest to. ``None``
    if not determinable."""

    needs_extension: bool
    """``True`` if the runner should shoot a probe in an extension on
    ``peak_side``."""

    peak_in_range_after_probe: bool = False
    """Only meaningful on the second call (with ``probe_data``). When
    ``True``, the runner runs a full extension sweep at the original
    step size and merges the results."""

    notes: str = ""
    """Free text for the persisted ``range_check.json`` and any
    user-visible messages."""

    all_off_target: bool = False
    """Every aperture's sharpness curve is flat — no real focus
    signal anywhere in the sweep. Distinct from ``ok=True`` (a real
    peak comfortably inside the range). The runner treats this as a
    re-jog request: extending won't help if there's no signal at
    all; the user almost certainly needs a different START/END."""

    per_aperture: list[ApertureCurve] = field(default_factory=list)
    """The curves the detector built; useful for diagnostics."""


class RangeChecker:
    """Decides whether a captured rail range bracketed the peak
    sharpness for each aperture.

    Parameters
    ----------
    config :
        The :mod:`config` module (or any duck-typed object exposing
        the same constants — currently only ``RANGE_CHECK_ENABLED``).

    The detector body is intentionally a stub. To wire in the real
    implementation later, override :meth:`evaluate`. The runner does
    not depend on anything beyond the public contract on this class.
    """

    def __init__(self, config) -> None:
        self.config = config

    def set_collector(self, collector) -> None:  # noqa: ARG002
        """Hook called by the session runner to hand over the per-
        lens-config sharpness collector. The default implementation
        ignores it (the stub doesn't need the cropped dataset);
        :class:`sharpness.range_check.SharpnessRangeChecker` overrides
        this to wire the collector through.
        """
        return None

    def evaluate(
        self,
        lens_dir: Path,
        end_mm: float,
        step_size_mm: float,
        apertures: list[str],
        *,
        probe_data: list[ApertureCurve] | None = None,
    ) -> RangeResult:
        """Return the decision.

        Parameters
        ----------
        lens_dir :
            Per-lens-config output directory; per-aperture sub-dirs
            named ``f<X>/`` contain ``step_NNN.jpg`` for the main
            sweep, and (after the runner has shot a probe)
            ``probe/step_PNN.jpg``.
        end_mm :
            END rail position relative to the (zeroed) START. Always
            non-negative.
        step_size_mm :
            Original sweep step size; the extension sweep reuses it.
        apertures :
            The ordered list of apertures actually shot.
        probe_data :
            ``None`` on the first call; populated on the second call
            with whatever curves the runner built from the probe
            shots. The detector is expected to use the probe data to
            confirm whether a full extension sweep would land on the
            true peak.

        Returns
        -------
        RangeResult
            See class docstring.
        """
        if not getattr(self.config, "RANGE_CHECK_ENABLED", True):
            return RangeResult(
                ok=True,
                peak_side="centered",
                needs_extension=False,
                notes="range check disabled by config",
            )
        # TODO: real implementation. The interface allows the
        # detector to:
        #   - load lens_dir/<aperture>/step_NNN.jpg per step
        #   - compute multi-crop sharpness per aperture
        #   - find the peak step per aperture
        #   - decide ok / peak_side / needs_extension
        #   - on the probe call, additionally use probe_data to
        #     confirm whether peak_in_range_after_probe is True
        return RangeResult(
            ok=True,
            peak_side="centered",
            needs_extension=False,
            notes="stub — detector not implemented; treating range as ok",
        )
