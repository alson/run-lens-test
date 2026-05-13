"""Per-lens-config FEC + ISO calibration at rail step 0.

For each aperture (widest first), the calibrator:

1. Sets the aperture and the per-aperture base ISO from
   :data:`config.RANGE_EXTENSION`.
2. Sets FEC = 0 and captures a test shot.
3. If this is the first lens config of the session and the widest
   aperture, the median of this shot becomes the session baseline.
4. Otherwise, computes the EV delta from the baseline and nudges FEC
   by ``-delta_ev`` (clamped to ``[FEC_MIN_EV, FEC_MAX_EV]``).
5. Re-shoots up to :data:`config.TEST_SHOT_MAX_RETRIES` times until
   ``|delta_ev| <= TEST_SHOT_TOLERANCE_EV``.
6. If FEC saturates while still under-exposing, bumps ISO by one ⅓
   stop (clamped to :data:`config.ISO_CEIL`) and continues.
7. Records the final (FEC, ISO) tuple per aperture.

The result is consumed by the sweep, which reuses the same FEC/ISO
for every rail step at that aperture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import math

from camera.nikon_z7 import NikonZ7
from errors import SessionAborted
from exposure.analyse import ev_delta
from session.flash_power import (
    FlashTarget,
    power_label,
    target_for_next_aperture,
)
from session.log import Phase, ShotRow, ShotsCSV


# ---- Standard ⅓-stop ISO ladder used for range-extension bumps ------------

ISO_LADDER: tuple[int, ...] = (
    100, 125, 160, 200, 250, 320, 400, 500, 640, 800,
    1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400,
)


def next_iso(current: int, *, ceiling: int) -> int:
    """Return the next ⅓-stop higher ISO, or ``current`` if the next
    step would exceed ``ceiling`` (or we're already past the ladder)."""
    for value in ISO_LADDER:
        if value > current and value <= ceiling:
            return value
    return current


def prev_iso(current: int, *, floor: int) -> int:
    """Return the next ⅓-stop lower ISO, or ``current`` if the previous
    step would drop below ``floor`` (or we're already at the bottom).

    Used by the manual-flash (ISO-only) cascade to drop ISO when the
    image is over-exposed, mirroring :func:`next_iso` in the other
    direction.
    """
    for value in reversed(ISO_LADDER):
        if value < current and value >= floor:
            return value
    return current


# ---- Result dataclasses ---------------------------------------------------


@dataclass
class ApertureCalibration:
    aperture: str
    fec_ev: float
    iso: int
    final_median: float
    final_ev_delta: float
    hit_tolerance: bool
    iso_bumped: bool
    test_shots_taken: int
    saturated_at_ceiling: bool = False
    ec_ev: float = 0.0
    ec_used: bool = False
    iso_headroom_applied: bool = False
    """``True`` when the post-convergence FEC/EC ↔ ISO swap landed
    within tolerance. The flash bias was reduced by
    ``ISO_HEADROOM_EV`` and ISO bumped equivalently — image stays
    at the same brightness, but the flash now fires below its
    mechanical max during the sweep, reducing heating, recycle
    times, and battery drain."""
    iso_headroom_attempted: bool = False
    """``True`` when the swap was attempted (regardless of whether it
    landed within tolerance). Combined with ``iso_headroom_applied``:
    attempted=True, applied=True → swap accepted. attempted=True,
    applied=False → swap reverted. attempted=False → swap not even
    tried (no plateau-triggered ISO bump in cascade)."""
    iso_headroom_attempt_ev: float | None = None
    """The EV delta from the verification capture taken after the
    swap. ``None`` if no swap attempt was made; otherwise the
    measured ev — useful for diagnosing reverts (the swap dropped
    exposure by this much)."""
    flash_power_stops_below_full: float | None = None
    """Manual-aperture mode only: the flash power the rig instructed
    the user to dial on the SB-5000 commander, as stops below 1/1
    (0.0 = full, 1.0 = 1/2, 2.0 = 1/4, …, 7.0 = 1/128). ``None`` in
    auto / AA mode where the flash auto-quenches and power isn't
    user-controlled. Stored per-aperture so downstream tooling can
    reconstruct the power sequence the operator dialled."""


@dataclass
class CalibrationResult:
    baseline_median: float
    per_aperture: dict[str, ApertureCalibration] = field(default_factory=dict)


# ---- Camera contract used by the calibrator -------------------------------


class CameraLike(Protocol):
    def set_aperture(self, fnum: str) -> None: ...
    def set_iso(self, iso) -> None: ...
    def set_fec(self, ev: float) -> None: ...
    def set_ec(self, ev: float) -> None: ...
    def capture_to(self, dest) -> int: ...


Measure = Callable[[Path], float]


# ---- Calibrator -----------------------------------------------------------


class Calibrator:
    """Walks one lens config's apertures and writes ``calibration.json``.

    Parameters
    ----------
    camera :
        Anything implementing :class:`CameraLike` (the production
        :class:`camera.nikon_z7.NikonZ7`, or a fake in tests).
    measure :
        Callable mapping a JPEG path to a median luminance
        (``measure_crop(...).p50`` in production).
    config :
        The :mod:`config` module or duck-typed equivalent. Required
        attributes: ``RANGE_EXTENSION``, ``ISO_CEIL``, ``ISO_WARN``,
        ``FEC_MIN_EV``, ``FEC_MAX_EV``, ``FEC_STEP_EV``,
        ``TEST_SHOT_TOLERANCE_EV``, ``TEST_SHOT_MAX_RETRIES``.
    shots_csv :
        Optional :class:`session.log.ShotsCSV` to receive test-shot
        rows. If ``None``, calibration runs without logging.
    now :
        Clock for timestamps; injectable for tests.
    """

    def __init__(
        self,
        *,
        camera: CameraLike,
        measure: Measure,
        config,
        shots_csv: ShotsCSV | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc),
        progress: Callable[..., None] | None = None,
    ) -> None:
        self._camera = camera
        self._measure = measure
        self._config = config
        self._csv = shots_csv
        self._now = now
        # progress(aperture, aperture_index, aperture_count, shot_index,
        # shot_count_max, fec, ec, iso, last_ev, label) — see
        # SessionRunner._calibration_progress for the production
        # renderer. Default no-op so tests stay opaque to UI.
        self._progress = progress or (lambda **kw: None)
        # Per-aperture context the cascade fills in for the progress
        # callback. ``Calibrator.run`` is the place that knows the
        # aperture index/count; the cascade just reads these.
        self._cal_aperture_index = 0
        self._cal_aperture_count = 1

    # ---- Public entry point ------------------------------------------------

    def run(
        self,
        *,
        lens_dir: Path,
        lens_slug: str,
        session_id: str,
        apertures: Iterable[str],
        baseline_median: float | None,
    ) -> CalibrationResult:
        """Calibrate FEC/ISO for each aperture at rail step 0.

        ``baseline_median`` is ``None`` on the first lens config of
        the session — the widest aperture's first capture sets it.
        """
        result = CalibrationResult(baseline_median=baseline_median or 0.0)
        baseline = baseline_median
        first_capture_dir = lens_dir / "calibration"
        first_capture_dir.mkdir(parents=True, exist_ok=True)
        aperture_list = list(apertures)
        self._cal_aperture_count = len(aperture_list)
        for ap_idx, aperture in enumerate(aperture_list):
            self._cal_aperture_index = ap_idx
            cal, baseline = self._calibrate_aperture(
                aperture=aperture,
                lens_slug=lens_slug,
                session_id=session_id,
                capture_dir=first_capture_dir,
                baseline=baseline,
            )
            result.per_aperture[aperture] = cal
        result.baseline_median = baseline or 0.0
        return result

    def calibrate_first_aperture_manual(
        self,
        *,
        aperture: str,
        lens_dir: Path,
        lens_slug: str,
        session_id: str,
        baseline_median: float | None,
        prompt_set_flash_power: Callable[..., bool],
    ) -> tuple[CalibrationResult, float, int]:
        """First-aperture cascade for manual-aperture mode.

        Four-phase protocol:

        1. **Coarse power search.** Start at
           :data:`config.MANUAL_FLASH_INITIAL_POWER_STOPS` (1/8 by
           default), ISO at :data:`config.MANUAL_ISO_FLOOR` (50).
           After each capture, halve / double the power one full
           stop based on the EV delta sign. Exit when within
           ``MANUAL_FLASH_SEARCH_TOLERANCE_EV`` (±1 stop). Safety:
           aborts if the search hits 1/1 or 1/128 still off-target,
           or exceeds ``MANUAL_FLASH_MAX_SEARCH_ITERATIONS``.
        2. **ISO fine-tune** at the settled power until within
           ``TEST_SHOT_TOLERANCE_EV`` (1/3 stop).
        3. **Back-calc** the precise 1/3-stop power that lands at
           ISO floor; if the calc exceeds the soft cap, clamp and
           let ISO carry the residual (per
           :func:`session.flash_power.target_for_next_aperture`).
        4. **Apply, verify, fine-tune ISO** at the new power.

        Returns ``(CalibrationResult, final_power_stops, final_iso)``
        so the runner can carry the (power, iso) state into
        subsequent apertures.

        The lens-ring aperture is set by the user before this method
        runs; the camera's ``set_aperture`` is never called.
        """
        cfg = self._config
        iso_floor = int(getattr(cfg, "MANUAL_ISO_FLOOR", 50))
        iso_warn = int(getattr(cfg, "ISO_WARN", 1600))
        iso_ceil = int(getattr(cfg, "ISO_CEIL", 3200))
        soft_cap = float(getattr(cfg, "MANUAL_FLASH_SOFT_CAP_STOPS", 1.0))
        hard_cap = float(getattr(cfg, "MANUAL_FLASH_HARD_CAP_STOPS", 0.0))
        initial_power = float(
            getattr(cfg, "MANUAL_FLASH_INITIAL_POWER_STOPS", 3.0)
        )
        search_tol = float(
            getattr(cfg, "MANUAL_FLASH_SEARCH_TOLERANCE_EV", 1.0)
        )
        max_search_iters = int(
            getattr(cfg, "MANUAL_FLASH_MAX_SEARCH_ITERATIONS", 15)
        )
        manual_tol = float(
            getattr(
                cfg, "MANUAL_FLASH_TOLERANCE_EV", cfg.TEST_SHOT_TOLERANCE_EV
            )
        )

        capture_dir = lens_dir / "calibration"
        capture_dir.mkdir(parents=True, exist_ok=True)

        # FEC stays at 0 throughout — inert in flash-M but the widget
        # wants a value.
        self._camera.set_fec(0.0)
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(0.0)
        self._camera.set_iso(iso_floor)

        baseline = baseline_median
        power_stops = initial_power
        shots_taken = 0
        last_median = 0.0
        last_delta = 0.0
        # Phase 1: coarse search.
        for iteration in range(max_search_iters):
            self._prompt_power(
                prompt_set_flash_power,
                power_stops,
                context=f"coarse search step {iteration + 1}",
            )
            dest = capture_dir / _calibration_filename(aperture, shots_taken)
            self._camera.capture_to(dest)
            median = self._measure(dest)
            last_median = median
            shots_taken += 1
            if baseline is None:
                # Establishing the session baseline — accept whatever
                # this capture yields at the current power. Skip the
                # rest of the protocol; the next lens config will
                # calibrate against this.
                baseline = median
                last_delta = 0.0
                self._log_shot(
                    aperture=aperture,
                    iso=iso_floor,
                    fec=0.0,
                    median=median,
                    ev=0.0,
                    filename=str(dest.name),
                    lens_slug=lens_slug,
                    session_id=session_id,
                    notes=f"baseline (manual first aperture, power {power_label(power_stops)})",
                )
                cal = ApertureCalibration(
                    aperture=aperture,
                    fec_ev=0.0,
                    iso=iso_floor,
                    final_median=median,
                    final_ev_delta=0.0,
                    hit_tolerance=True,
                    iso_bumped=False,
                    test_shots_taken=shots_taken,
                    flash_power_stops_below_full=power_stops,
                )
                result = CalibrationResult(baseline_median=baseline)
                result.per_aperture[aperture] = cal
                return result, power_stops, iso_floor
            delta = ev_delta(median, baseline)
            last_delta = delta
            self._log_shot(
                aperture=aperture,
                iso=iso_floor,
                fec=0.0,
                median=median,
                ev=delta,
                filename=str(dest.name),
                lens_slug=lens_slug,
                session_id=session_id,
                notes=f"coarse search step {iteration + 1}, power {power_label(power_stops)}",
            )
            if abs(delta) <= search_tol:
                break
            # Adjust power one full stop in the right direction.
            if delta > 0:
                # Too bright → drop power one stop (increase stops-below).
                new_power = power_stops + 1.0
                if new_power > 7.0:
                    raise SessionAborted(
                        f"coarse search hit 1/128 minimum but image is "
                        f"still over-exposed (delta={delta:+.2f} EV). "
                        f"Reduce ambient light or add ND filter."
                    )
                power_stops = new_power
            else:
                new_power = power_stops - 1.0
                if new_power < 0.0:
                    raise SessionAborted(
                        f"coarse search hit 1/1 full but image is still "
                        f"under-exposed (delta={delta:+.2f} EV). "
                        f"Reduce subject distance, increase ambient, or "
                        f"use a brighter flash."
                    )
                power_stops = new_power
        else:
            raise SessionAborted(
                f"coarse search exhausted {max_search_iters} iterations "
                f"without converging within ±{search_tol} EV"
            )

        # Phase 2: ISO fine-tune at the settled power.
        shots_taken, last_median, last_delta, fine_iso = self._iso_fine_tune(
            aperture=aperture,
            capture_dir=capture_dir,
            baseline=baseline,
            iso=iso_floor,
            iso_floor=iso_floor,
            iso_ceiling=iso_ceil,
            shots_taken=shots_taken,
            power_for_log=power_stops,
            lens_slug=lens_slug,
            session_id=session_id,
            note_prefix="iso fine-tune (post-search)",
            initial_median=last_median,
            initial_delta=last_delta,
            tol=manual_tol,
        )

        # Phase 3: back-calc target via target_for_next_aperture with
        # aperture_stops_diff=0 (no aperture change; just absorb the
        # iso overhang into power). Pass the post-fine-tune delta so
        # the brute-force search can also pull the residual error
        # toward zero.
        target = target_for_next_aperture(
            prev_power_stops=power_stops,
            prev_iso=fine_iso,
            aperture_stops_diff=0.0,
            soft_cap_stops=soft_cap,
            hard_cap_stops=hard_cap,
            iso_floor=iso_floor,
            iso_warn=iso_warn,
            iso_ceil=iso_ceil,
            prev_ev_delta=last_delta,
        )

        # Phase 4: prompt user to set the back-calculated power,
        # then verify + ISO fine-tune. Suppress the prompt when the
        # target equals the power dialled during Phase 1 — the user
        # already has it set, no reason to ask again.
        self._prompt_power(
            prompt_set_flash_power,
            target.power_stops_below_full,
            context=f"calibrated target for {aperture}",
            currently_set_power_stops=power_stops,
        )
        self._camera.set_iso(target.iso)
        shots_taken, last_median, last_delta, final_iso = self._iso_fine_tune(
            aperture=aperture,
            capture_dir=capture_dir,
            baseline=baseline,
            iso=target.iso,
            iso_floor=iso_floor,
            iso_ceiling=iso_ceil,
            shots_taken=shots_taken,
            power_for_log=target.power_stops_below_full,
            lens_slug=lens_slug,
            session_id=session_id,
            note_prefix="verification + iso fine-tune (post-back-calc)",
            initial_median=None,
            initial_delta=None,
            tol=manual_tol,
        )

        cal = ApertureCalibration(
            aperture=aperture,
            fec_ev=0.0,
            iso=final_iso,
            final_median=last_median,
            final_ev_delta=last_delta,
            hit_tolerance=abs(last_delta) <= manual_tol,
            iso_bumped=(final_iso != iso_floor),
            test_shots_taken=shots_taken,
            flash_power_stops_below_full=target.power_stops_below_full,
        )
        result = CalibrationResult(baseline_median=baseline)
        result.per_aperture[aperture] = cal
        return result, target.power_stops_below_full, final_iso

    def calibrate_subsequent_aperture_manual(
        self,
        *,
        aperture: str,
        lens_dir: Path,
        lens_slug: str,
        session_id: str,
        baseline_median: float,
        prev_power_stops: float,
        prev_iso: int,
        prev_ev_delta: float,
        aperture_stops_diff: float,
        prompt_set_flash_power: Callable[..., bool],
    ) -> tuple[CalibrationResult, float, int]:
        """Subsequent-aperture cascade for manual-aperture mode.

        Computes the target (power, iso) deterministically from the
        previous aperture's calibrated state plus the aperture-stops
        difference (typically 1.0 for a whole-stop ring step).
        Prompts the user to dial the target power, captures, runs
        a first-shot sanity check (delta from baseline within
        ``MANUAL_FLASH_BUMP_SANITY_EV``), then ISO fine-tunes.

        Returns ``(CalibrationResult, final_power_stops, final_iso)``
        so the runner can chain state into the next aperture.
        """
        cfg = self._config
        iso_floor = int(getattr(cfg, "MANUAL_ISO_FLOOR", 50))
        iso_warn = int(getattr(cfg, "ISO_WARN", 1600))
        iso_ceil = int(getattr(cfg, "ISO_CEIL", 3200))
        soft_cap = float(getattr(cfg, "MANUAL_FLASH_SOFT_CAP_STOPS", 1.0))
        hard_cap = float(getattr(cfg, "MANUAL_FLASH_HARD_CAP_STOPS", 0.0))
        bump_sanity_ev = float(
            getattr(cfg, "MANUAL_FLASH_BUMP_SANITY_EV", 0.5)
        )
        manual_tol = float(
            getattr(
                cfg, "MANUAL_FLASH_TOLERANCE_EV", cfg.TEST_SHOT_TOLERANCE_EV
            )
        )

        capture_dir = lens_dir / "calibration"
        capture_dir.mkdir(parents=True, exist_ok=True)

        # FEC inert; reset for hygiene.
        self._camera.set_fec(0.0)
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(0.0)

        target = target_for_next_aperture(
            prev_power_stops=prev_power_stops,
            prev_iso=prev_iso,
            aperture_stops_diff=aperture_stops_diff,
            soft_cap_stops=soft_cap,
            hard_cap_stops=hard_cap,
            iso_floor=iso_floor,
            iso_warn=iso_warn,
            iso_ceil=iso_ceil,
            prev_ev_delta=prev_ev_delta,
        )
        # Suppress this prompt when the new target equals what the
        # commander is already set to (carry-over from previous
        # aperture). The user already has the right power dialled.
        self._prompt_power(
            prompt_set_flash_power,
            target.power_stops_below_full,
            context=f"target for {aperture}",
            currently_set_power_stops=prev_power_stops,
        )
        self._camera.set_iso(target.iso)

        shots_taken = 0
        # First-shot sanity check.
        dest = capture_dir / _calibration_filename(aperture, shots_taken)
        self._camera.capture_to(dest)
        median = self._measure(dest)
        shots_taken += 1
        delta = ev_delta(median, baseline_median)
        self._log_shot(
            aperture=aperture,
            iso=target.iso,
            fec=0.0,
            median=median,
            ev=delta,
            filename=str(dest.name),
            lens_slug=lens_slug,
            session_id=session_id,
            notes=(
                f"subsequent-aperture sanity check, power "
                f"{power_label(target.power_stops_below_full)}"
            ),
        )
        if abs(delta) > bump_sanity_ev:
            # The first capture is significantly off the predicted
            # value. Re-prompt the user — most likely cause is the
            # commander wasn't dialled to the target power, or the
            # ring wasn't turned, or both.
            ok = prompt_set_flash_power(
                power_c_label=power_label(target.power_stops_below_full),
                power_a_label=power_label(target.power_stops_below_full + 1 / 3),
                context=(
                    f"SANITY: expected ~baseline but delta={delta:+.2f} EV. "
                    f"Verify flash power AND ring position"
                ),
            )
            if not ok:
                raise SessionAborted(
                    "user aborted at sanity-check re-prompt"
                )
            dest = capture_dir / _calibration_filename(aperture, shots_taken)
            self._camera.capture_to(dest)
            median = self._measure(dest)
            shots_taken += 1
            delta = ev_delta(median, baseline_median)
            self._log_shot(
                aperture=aperture,
                iso=target.iso,
                fec=0.0,
                median=median,
                ev=delta,
                filename=str(dest.name),
                lens_slug=lens_slug,
                session_id=session_id,
                notes="sanity check retry after user re-prompt",
            )

        # ISO fine-tune.
        shots_taken, last_median, last_delta, final_iso = self._iso_fine_tune(
            aperture=aperture,
            capture_dir=capture_dir,
            baseline=baseline_median,
            iso=target.iso,
            iso_floor=iso_floor,
            iso_ceiling=iso_ceil,
            shots_taken=shots_taken,
            power_for_log=target.power_stops_below_full,
            lens_slug=lens_slug,
            session_id=session_id,
            note_prefix=(
                f"iso fine-tune (post-target, "
                f"power {power_label(target.power_stops_below_full)})"
            ),
            initial_median=median,
            initial_delta=delta,
            tol=manual_tol,
        )

        cal = ApertureCalibration(
            aperture=aperture,
            fec_ev=0.0,
            iso=final_iso,
            final_median=last_median,
            final_ev_delta=last_delta,
            hit_tolerance=abs(last_delta) <= manual_tol,
            iso_bumped=(final_iso != iso_floor),
            test_shots_taken=shots_taken,
            flash_power_stops_below_full=target.power_stops_below_full,
        )
        result = CalibrationResult(baseline_median=baseline_median)
        result.per_aperture[aperture] = cal
        return result, target.power_stops_below_full, final_iso

    def calibrate_single_aperture(
        self,
        *,
        aperture: str,
        lens_dir: Path,
        lens_slug: str,
        session_id: str,
        baseline_median: float | None,
        set_aperture_via_camera: bool = True,
        flash_mode: str = "aa",
    ) -> CalibrationResult:
        """Calibrate one aperture and return a single-entry result.

        Used by manual-aperture mode where the runner inverts the
        loop (apertures outer, rail inner): each aperture gets one
        sweep, so calibration happens one aperture at a time. The
        return shape matches :meth:`run` so the downstream caller
        (sweep, log writer) doesn't special-case it.

        ``set_aperture_via_camera``: see
        :meth:`_calibrate_aperture`.

        ``flash_mode``: see :meth:`_calibrate_aperture`.
        """
        capture_dir = lens_dir / "calibration"
        capture_dir.mkdir(parents=True, exist_ok=True)
        cal, baseline = self._calibrate_aperture(
            aperture=aperture,
            lens_slug=lens_slug,
            session_id=session_id,
            capture_dir=capture_dir,
            baseline=baseline_median,
            set_aperture_via_camera=set_aperture_via_camera,
            flash_mode=flash_mode,
        )
        result = CalibrationResult(baseline_median=baseline or 0.0)
        result.per_aperture[aperture] = cal
        return result

    # ---- Per-aperture loop -------------------------------------------------

    def _calibrate_aperture(
        self,
        *,
        aperture: str,
        lens_slug: str,
        session_id: str,
        capture_dir: Path,
        baseline: float | None,
        set_aperture_via_camera: bool = True,
        flash_mode: str = "aa",
    ) -> tuple[ApertureCalibration, float | None]:
        """Calibrate ISO/FEC/EC at one aperture and return the result.

        ``set_aperture_via_camera``: when False, the camera ``set_aperture``
        call is skipped — used by manual-aperture mode where the lens
        ring is mechanical and the user sets it before this method
        runs.

        ``flash_mode``: ``"aa"`` (default) runs the existing AA-mode
        cascade with FEC → EC → ISO range-extension. ``"manual"``
        runs an ISO-only cascade for use when remote groups are in
        flash-M mode (which is the only working mode for non-CPU
        lenses on Z7 since the body can't communicate aperture to
        the flash). In flash-M, FEC and EC have no effect; ISO is
        the only useful exposure knob.
        """
        if flash_mode == "manual":
            return self._calibrate_aperture_manual_flash(
                aperture=aperture,
                lens_slug=lens_slug,
                session_id=session_id,
                capture_dir=capture_dir,
                baseline=baseline,
                set_aperture_via_camera=set_aperture_via_camera,
            )
        cfg = self._config
        if getattr(cfg, "LINEAR_CASCADE_ENABLED", False):
            return self._calibrate_aperture_linear(
                aperture=aperture,
                lens_slug=lens_slug,
                session_id=session_id,
                capture_dir=capture_dir,
                baseline=baseline,
                set_aperture_via_camera=set_aperture_via_camera,
            )
        ap_float = _parse_aperture(aperture)
        iso = self._initial_iso(ap_float)
        fec = 0.0
        ec = 0.0
        ec_min = float(getattr(cfg, "EC_MIN_EV", -3.0))
        ec_max = float(getattr(cfg, "EC_MAX_EV", +3.0))
        ec_step = float(getattr(cfg, "EC_STEP_EV", 1.0 / 3.0))
        plateau_threshold = float(
            getattr(cfg, "PLATEAU_DETECT_THRESHOLD", 0.5)
        )
        plateau_min_ev = float(
            getattr(cfg, "PLATEAU_DETECT_MIN_EV", 1.0)
        )
        if set_aperture_via_camera:
            self._camera.set_aperture(aperture)
        self._camera.set_iso(iso)
        self._camera.set_fec(fec)
        # Reset EC to 0 at the start of each aperture so a previous
        # aperture's bump doesn't carry over silently.
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(ec)

        last_median = 0.0
        last_delta = 0.0
        iso_bumped = False
        ec_used = False
        saturated = False
        # Plateau tracker: cumulative positive FEC+EC bumps applied
        # since the last reset, and the median we had when the
        # tracker was reset. The flash physically maxes out when
        # we keep asking for more output (FEC/EC up) and luminosity
        # stops responding — that's the only condition under which
        # ISO bump actually makes the image brighter (in AA flash
        # mode, ISO normally just makes the flash quench sooner,
        # leaving brightness unchanged). Anything below
        # ``plateau_threshold`` of expected response over at least
        # ``plateau_min_ev`` of cumulative bump is taken as plateau.
        push_start_median: float | None = None
        push_total_ev = 0.0
        converged = False
        converged_shot_index = -1
        for shot_index in range(cfg.TEST_SHOT_MAX_RETRIES + 1):
            dest = capture_dir / _calibration_filename(aperture, shot_index)
            self._camera.capture_to(dest)
            median = self._measure(dest)
            last_median = median

            if baseline is None:
                # First capture of the very first lens config of the
                # session at the widest aperture sets the baseline.
                baseline = median
                last_delta = 0.0
                self._log_shot(
                    aperture=aperture,
                    iso=iso,
                    fec=fec,
                    median=median,
                    ev=0.0,
                    filename=str(dest.name),
                    lens_slug=lens_slug,
                    session_id=session_id,
                    notes="baseline",
                )
                return ApertureCalibration(
                    aperture=aperture,
                    fec_ev=fec,
                    iso=iso,
                    final_median=median,
                    final_ev_delta=0.0,
                    hit_tolerance=True,
                    iso_bumped=False,
                    test_shots_taken=1,
                    ec_ev=ec,
                    ec_used=False,
                ), baseline

            delta = ev_delta(median, baseline)
            last_delta = delta
            self._log_shot(
                aperture=aperture,
                iso=iso,
                fec=fec,
                median=median,
                ev=delta,
                filename=str(dest.name),
                lens_slug=lens_slug,
                session_id=session_id,
                notes=f"calibration shot {shot_index}",
            )

            # Plateau check (range-extend with ISO if the flash has
            # mechanically maxed out, evidenced by luminosity not
            # tracking the cumulative FEC+EC bumps we asked for).
            if (
                push_start_median is not None
                and push_total_ev >= plateau_min_ev
            ):
                cumulative_response = ev_delta(median, push_start_median)
                if cumulative_response < push_total_ev * plateau_threshold:
                    bumped = next_iso(iso, ceiling=cfg.ISO_CEIL)
                    if bumped == iso:
                        saturated = True
                        break
                    iso = bumped
                    iso_bumped = True
                    self._camera.set_iso(iso)
                    push_start_median = None
                    push_total_ev = 0.0
                    continue

            if abs(delta) <= cfg.TEST_SHOT_TOLERANCE_EV:
                converged = True
                converged_shot_index = shot_index
                break

            # Lever priority is FEC → EC. ISO is *not* in the
            # cascade: in AA flash mode raising ISO without a
            # mechanical-max plateau just makes the flash quench
            # earlier, keeping image brightness unchanged. ISO is
            # only useful as a range-extender, triggered exclusively
            # by the plateau check above.
            new_fec = _clamp(
                _snap(fec - delta, cfg.FEC_STEP_EV),
                cfg.FEC_MIN_EV,
                cfg.FEC_MAX_EV,
            )
            # Saturation uses *current* fec / ec, not new_fec — see
            # commit history for the off-by-one this avoids.
            fec_saturated_low = (
                fec == cfg.FEC_MIN_EV and delta > cfg.TEST_SHOT_TOLERANCE_EV
            )
            fec_saturated_high = (
                fec == cfg.FEC_MAX_EV and delta < -cfg.TEST_SHOT_TOLERANCE_EV
            )
            if fec_saturated_high or fec_saturated_low:
                # FEC tapped out, fall back to EC.
                new_ec = _clamp(_snap(ec - delta, ec_step), ec_min, ec_max)
                ec_saturated_low = (
                    ec == ec_min and delta > cfg.TEST_SHOT_TOLERANCE_EV
                )
                ec_saturated_high = (
                    ec == ec_max and delta < -cfg.TEST_SHOT_TOLERANCE_EV
                )
                if ec_saturated_low or ec_saturated_high:
                    # Both FEC and EC at their respective limits with
                    # no plateau detected during the cascade. The
                    # flash *was* responding right up to the camera-
                    # side compensation max; an ISO bump now would
                    # just trigger earlier auto-quench. Accept.
                    saturated = True
                    break
                bump_ev = abs(new_ec - ec)
                ec = new_ec
                if hasattr(self._camera, "set_ec"):
                    self._camera.set_ec(ec)
                ec_used = True
            else:
                bump_ev = abs(new_fec - fec)
                fec = new_fec
                self._camera.set_fec(fec)

            # Track the bump for plateau detection. Only positive
            # under-exposure-correction bumps accumulate; over-
            # exposure correction (delta > 0, FEC/EC going down)
            # would not benefit from ISO bump anyway.
            if delta < -cfg.TEST_SHOT_TOLERANCE_EV and bump_ev > 0:
                if push_start_median is None:
                    push_start_median = last_median
                push_total_ev += bump_ev

        # Post-loop: optional flash-headroom swap. When the cascade
        # converged *and* required an ISO bump (positive plateau —
        # the flash was at its mechanical maximum), trade
        # ``ISO_HEADROOM_EV`` of FEC/EC bias for the same EV in
        # ISO. Image brightness stays the same; the flash now fires
        # below its mechanical max during the sweep, reducing
        # heating, recycle times, and battery drain.
        shots_taken = (
            converged_shot_index + 1
            if converged
            else cfg.TEST_SHOT_MAX_RETRIES + 1
        )
        iso_headroom_applied = False
        iso_headroom_attempted = False
        iso_headroom_attempt_ev: float | None = None
        if converged and iso_bumped:
            applied, attempt_ev = self._try_iso_headroom_swap(
                aperture=aperture,
                fec=fec,
                ec=ec,
                iso=iso,
                baseline=baseline,
                lens_slug=lens_slug,
                session_id=session_id,
                capture_dir=capture_dir,
                shot_index=shots_taken,
                ec_min=ec_min,
                ec_step=ec_step,
            )
            iso_headroom_attempted = attempt_ev is not None
            iso_headroom_attempt_ev = attempt_ev
            if applied is not None:
                fec, ec, iso, last_median, last_delta = applied
                iso_headroom_applied = True
                shots_taken += 1
            elif iso_headroom_attempted:
                # Verification shot was taken (and counted) even
                # though the swap was reverted.
                shots_taken += 1

        return ApertureCalibration(
            aperture=aperture,
            fec_ev=fec,
            iso=iso,
            final_median=last_median,
            final_ev_delta=last_delta,
            hit_tolerance=converged,
            iso_bumped=iso_bumped,
            test_shots_taken=shots_taken,
            saturated_at_ceiling=saturated,
            ec_ev=ec,
            ec_used=ec_used,
            iso_headroom_applied=iso_headroom_applied,
            iso_headroom_attempted=iso_headroom_attempted,
            iso_headroom_attempt_ev=iso_headroom_attempt_ev,
        ), baseline

    def _calibrate_aperture_manual_flash(
        self,
        *,
        aperture: str,
        lens_slug: str,
        session_id: str,
        capture_dir: Path,
        baseline: float | None,
        set_aperture_via_camera: bool,
    ) -> tuple[ApertureCalibration, float | None]:
        """ISO-only cascade for flash-Manual mode.

        Flash is firing at a fixed power set on the commander. FEC
        and EC have no effect because the flash isn't running an
        auto-quench threshold. The only useful knob is ISO.

        On each iteration: capture, measure central crop, compute EV
        delta. If under-exposed, bump ISO one ⅓-stop (up to
        ``ISO_CEIL``). If over-exposed, drop ISO one ⅓-stop (down to
        ``ISO_FLOOR``). Saturation at either end ends the loop.

        Conserves the existing :class:`ApertureCalibration` shape so
        downstream callers (the runner's sweep + log writer) don't
        special-case manual flash. ``fec_ev`` and ``ec_ev`` are left
        at 0; ``iso_bumped`` reflects whether *any* ISO change
        occurred during the loop.
        """
        cfg = self._config
        ap_float = _parse_aperture(aperture)
        iso = self._initial_iso(ap_float)
        iso_floor = int(getattr(cfg, "ISO_FLOOR", 100))
        iso_ceiling = int(getattr(cfg, "ISO_CEIL", 3200))
        if set_aperture_via_camera:
            self._camera.set_aperture(aperture)
        self._camera.set_iso(iso)
        # FEC stays at 0 — inert in flash-M but the camera widget
        # still wants a value to read. Some callers rely on it being
        # explicitly reset between apertures.
        self._camera.set_fec(0.0)
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(0.0)

        initial_iso = iso
        last_median = 0.0
        last_delta = 0.0
        converged = False
        saturated = False
        converged_shot_index = -1
        for shot_index in range(cfg.TEST_SHOT_MAX_RETRIES + 1):
            dest = capture_dir / _calibration_filename(aperture, shot_index)
            self._camera.capture_to(dest)
            median = self._measure(dest)
            last_median = median

            if baseline is None:
                # First capture of the very first lens config of the
                # session at the widest aperture sets the baseline.
                baseline = median
                last_delta = 0.0
                self._log_shot(
                    aperture=aperture,
                    iso=iso,
                    fec=0.0,
                    median=median,
                    ev=0.0,
                    filename=str(dest.name),
                    lens_slug=lens_slug,
                    session_id=session_id,
                    notes="baseline (manual flash)",
                )
                return ApertureCalibration(
                    aperture=aperture,
                    fec_ev=0.0,
                    iso=iso,
                    final_median=median,
                    final_ev_delta=0.0,
                    hit_tolerance=True,
                    iso_bumped=False,
                    test_shots_taken=1,
                    ec_ev=0.0,
                    ec_used=False,
                ), baseline

            delta = ev_delta(median, baseline)
            last_delta = delta
            self._log_shot(
                aperture=aperture,
                iso=iso,
                fec=0.0,
                median=median,
                ev=delta,
                filename=str(dest.name),
                lens_slug=lens_slug,
                session_id=session_id,
                notes=f"calibration shot {shot_index} (manual flash, ISO-only)",
            )

            if abs(delta) <= cfg.TEST_SHOT_TOLERANCE_EV:
                converged = True
                converged_shot_index = shot_index
                break

            # ISO is the only knob. Under-exposure → bump up; over →
            # drop down. Saturation at either end ends the loop.
            if delta < -cfg.TEST_SHOT_TOLERANCE_EV:
                bumped = next_iso(iso, ceiling=iso_ceiling)
                if bumped == iso:
                    saturated = True
                    break
                iso = bumped
            else:
                dropped = prev_iso(iso, floor=iso_floor)
                if dropped == iso:
                    saturated = True
                    break
                iso = dropped
            self._camera.set_iso(iso)

        shots_taken = (
            converged_shot_index + 1
            if converged
            else cfg.TEST_SHOT_MAX_RETRIES + 1
        )
        return ApertureCalibration(
            aperture=aperture,
            fec_ev=0.0,
            iso=iso,
            final_median=last_median,
            final_ev_delta=last_delta,
            hit_tolerance=converged,
            iso_bumped=(iso != initial_iso),
            test_shots_taken=shots_taken,
            saturated_at_ceiling=saturated,
            ec_ev=0.0,
            ec_used=False,
        ), baseline

    # ---- Linear-model cascade (AA mode, 3-5 shots) ------------------------

    def _calibrate_aperture_linear(
        self,
        *,
        aperture: str,
        lens_slug: str,
        session_id: str,
        capture_dir: Path,
        baseline: float | None,
        set_aperture_via_camera: bool = True,
    ) -> tuple[ApertureCalibration, float | None]:
        """3-5 shot linear-model AA calibration.

        Replaces the legacy iterative cascade (10-shot cap, plateau-
        detector, FEC-saturation fallback chain) with an explicit
        per-shot purpose model:

        * **Shot 1** — probe at FEC=0, ISO=floor, EC=0. Establishes
          ``delta_0`` = the rig's "natural" exposure deviation.
        * **Shot 2** — apply target bias ``-delta_0``, distributed
          across FEC then EC. Should land at baseline if the flash
          chain is linear (per the rig_flash_headroom memory's
          slope-≈-1.0 observation).
        * **Shot 3** — ALWAYS — headroom probe. Add +1 stop of bias
          in the same direction as the shot-2 adjustment. Slope =
          ``(delta_2 - delta_1) / probe_step``.
        * **Shot 4** — only if needed: either a fine-tune (if
          ``|delta_1| > tolerance`` AND headroom confirmed by shot 3)
          OR a post-ISO-bump retry (if shot 3 revealed the flash is
          mechanically maxed).
        * **Shot 5** — only if shot 4 was ISO-bump and we still need
          one more touch-up to land within tolerance.

        Slope thresholds:

        * ``slope >= 0.7 EV / stop``: linear regime, flash has
          headroom. If ``|delta_1| <= tolerance`` we're done with
          shot 2's settings (shot 3 was diagnostic). Otherwise
          fine-tune in shot 4.
        * ``slope < 0.3 EV / stop``: flash is at mechanical max.
          The +1-stop probe didn't move the needle, so adding more
          FEC/EC won't help — bump ISO.
        * ``0.3 ≤ slope < 0.7``: partial maxing; treated
          conservatively as saturated and bumps ISO.

        Replaces the legacy cascade's same-FEC-different-output
        oscillation pattern with a one-pass linear model. Combined
        with the FEC float-rounding fix in
        :meth:`NikonZ7._set_range`, the cascade is robust against
        both the float-rounding silent rejections and the
        oscillation that came from the greedy ``new_fec = fec -
        delta`` step + ⅓-stop snap interaction.
        """
        cfg = self._config
        ap_float = _parse_aperture(aperture)
        iso_floor = self._initial_iso(ap_float)
        iso = iso_floor

        fec = 0.0
        ec = 0.0
        ec_min = float(getattr(cfg, "EC_MIN_EV", -3.0))
        ec_max = float(getattr(cfg, "EC_MAX_EV", 3.0))
        ec_step = float(getattr(cfg, "EC_STEP_EV", 1.0 / 3.0))
        fec_min = float(cfg.FEC_MIN_EV)
        fec_max = float(cfg.FEC_MAX_EV)
        fec_step = float(cfg.FEC_STEP_EV)
        tolerance = float(cfg.TEST_SHOT_TOLERANCE_EV)
        iso_step_ev = float(getattr(cfg, "ISO_STEP_EV", 1.0 / 3.0))

        if set_aperture_via_camera:
            self._camera.set_aperture(aperture)
        self._camera.set_iso(iso)
        self._camera.set_fec(fec)
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(ec)

        last_median = 0.0
        last_delta = 0.0
        iso_bumped = False
        ec_used = False
        saturated = False
        converged = False
        shots = 0

        # Upper bound on shots — used by the progress renderer to
        # size the bar. The cascade actually runs 3-5 shots.
        SHOT_COUNT_MAX = 5

        def shoot(notes: str) -> tuple[float, float]:
            """Capture, measure, log. Returns (median, delta)."""
            nonlocal shots, last_median, last_delta
            # Progress BEFORE capture so the user sees what's about to
            # fire (FEC/EC/ISO currently set on the camera) and the
            # last measured delta for context.
            self._progress(
                aperture=aperture,
                aperture_index=self._cal_aperture_index,
                aperture_count=self._cal_aperture_count,
                shot_index=shots,
                shot_count_max=SHOT_COUNT_MAX,
                fec=fec, ec=ec, iso=iso,
                last_ev=last_delta if shots > 0 else None,
                label=notes,
            )
            dest = capture_dir / _calibration_filename(aperture, shots)
            self._camera.capture_to(dest)
            median = self._measure(dest)
            shots += 1
            last_median = median
            if baseline is None:
                # Caller (shot 1 of first lens config) handles
                # baseline assignment; return delta=0 placeholder.
                self._log_shot(
                    aperture=aperture, iso=iso, fec=fec,
                    median=median, ev=0.0,
                    filename=str(dest.name),
                    lens_slug=lens_slug, session_id=session_id,
                    notes="baseline",
                )
                return median, 0.0
            delta = ev_delta(median, baseline)
            last_delta = delta
            self._log_shot(
                aperture=aperture, iso=iso, fec=fec,
                median=median, ev=delta,
                filename=str(dest.name),
                lens_slug=lens_slug, session_id=session_id,
                notes=notes,
            )
            return median, delta

        # ---- Shot 1: probe at FEC=0 ----------------------------------
        median, delta_0 = shoot("linear shot 1 (probe FEC=0)")
        if baseline is None:
            # First lens config: this shot SETS the baseline; no
            # further work for this aperture.
            return ApertureCalibration(
                aperture=aperture, fec_ev=fec, iso=iso,
                final_median=median, final_ev_delta=0.0,
                hit_tolerance=True, iso_bumped=False,
                test_shots_taken=1, ec_ev=ec, ec_used=False,
            ), median

        # ---- Compute target & shot 2 ---------------------------------
        target_bias = -delta_0
        fec, ec, ec_used = _distribute_bias(
            target_bias,
            fec_min=fec_min, fec_max=fec_max, fec_step=fec_step,
            ec_min=ec_min, ec_max=ec_max, ec_step=ec_step,
        )
        self._camera.set_fec(fec)
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(ec)
        _, delta_1 = shoot("linear shot 2 (target)")

        # ---- Shot 3: headroom probe ----------------------------------
        # Probe in the same sign as the original adjustment, by one
        # whole stop. If shot 1 was exactly on target (target_bias ≈
        # 0), probe upward by convention — we still want to verify
        # linear response.
        probe_sign = 1.0 if target_bias >= 0 else -1.0
        probe_target_bias = (fec + ec) + probe_sign  # cumulative bias = current + 1 stop
        probe_fec, probe_ec, probe_ec_used = _distribute_bias(
            probe_target_bias,
            fec_min=fec_min, fec_max=fec_max, fec_step=fec_step,
            ec_min=ec_min, ec_max=ec_max, ec_step=ec_step,
        )
        # If the probe yields exactly the same settings (cascade
        # already at FEC+EC ceiling in the probe direction), we
        # can't actually probe. Skip the shot and treat as saturated
        # (the cascade has no headroom on this side anyway).
        actual_probe_step = (probe_fec + probe_ec) - (fec + ec)
        can_probe = abs(actual_probe_step) > 1e-3
        slope = 0.0
        if can_probe:
            shot2_fec, shot2_ec = fec, ec
            shot2_ec_used = ec_used
            self._camera.set_fec(probe_fec)
            if hasattr(self._camera, "set_ec"):
                self._camera.set_ec(probe_ec)
            fec, ec = probe_fec, probe_ec
            ec_used = ec_used or probe_ec_used
            _, delta_2 = shoot("linear shot 3 (headroom probe)")
            slope = (delta_2 - delta_1) / actual_probe_step

        # ---- Classify -----------------------------------------------
        slope_healthy = 0.7
        slope_maxed = 0.3
        if can_probe and slope >= slope_healthy:
            # Headroom confirmed. **⅓-stop snap-error correction** —
            # shot 2's snapped bias may have rounded the wrong way
            # off the ideal target (snap error up to ⅙ EV). If
            # |delta_1| > ⅙ EV (one half ⅓-stop), adjust the total
            # bias by ⅓-stop in the appropriate direction and snap
            # to the adjacent ⅓-stop position. This picks the better
            # of the two ⅓-stop neighbours without taking another
            # shot. Predicted delta = delta_1 ∓ 1/3 EV (slope=1
            # assumption; close enough since the slope check above
            # confirmed we're in linear regime).
            shot2_total = shot2_fec + shot2_ec
            sixth = fec_step / 2.0  # ⅙ EV when fec_step is ⅓
            if abs(delta_1) <= sixth:
                # Snap was favourable; keep shot 2's settings.
                self._camera.set_fec(shot2_fec)
                if hasattr(self._camera, "set_ec"):
                    self._camera.set_ec(shot2_ec)
                fec, ec = shot2_fec, shot2_ec
                ec_used = shot2_ec_used
                last_delta = delta_1
                converged = True
            else:
                # Snap rounded the wrong way. Adjust ±⅓ stop and
                # re-distribute. The sign of the adjustment makes
                # the image change in the direction that reduces
                # |delta|: bright image (delta > 0) → MORE negative
                # bias; dim image (delta < 0) → MORE positive bias.
                adjustment = -fec_step if delta_1 > 0 else +fec_step
                refined_total = shot2_total + adjustment
                refined_fec, refined_ec, refined_ec_used = _distribute_bias(
                    refined_total,
                    fec_min=fec_min, fec_max=fec_max, fec_step=fec_step,
                    ec_min=ec_min, ec_max=ec_max, ec_step=ec_step,
                )
                self._camera.set_fec(refined_fec)
                if hasattr(self._camera, "set_ec"):
                    self._camera.set_ec(refined_ec)
                fec, ec = refined_fec, refined_ec
                ec_used = refined_ec_used
                # Predicted delta after the ±⅓-stop adjustment.
                # More positive bias → brighter image → delta += ⅓.
                # More negative bias → darker image → delta -= ⅓.
                # The sign of ``adjustment`` matches this directly.
                last_delta = delta_1 + adjustment
                converged = abs(last_delta) <= tolerance
        elif not can_probe or slope < slope_maxed or (slope < slope_healthy and abs(delta_1) > tolerance):
            # Flash maxed (or borderline + off-target). Bump ISO by
            # enough ⅓-stops to cover the residual deficit; the
            # ISO bump gives REAL exposure gain because the flash
            # is already at mechanical max.
            deficit_ev = -delta_1 if delta_1 < 0 else 0.0
            if deficit_ev <= tolerance:
                # delta_1 was within tolerance after all (slope was
                # borderline but image is on target). Accept.
                if can_probe:
                    self._camera.set_fec(shot2_fec)
                    if hasattr(self._camera, "set_ec"):
                        self._camera.set_ec(shot2_ec)
                    fec, ec = shot2_fec, shot2_ec
                    ec_used = shot2_ec_used
                converged = True
                last_delta = delta_1
            else:
                stops_to_bump = max(1, int(round(deficit_ev / iso_step_ev)))
                new_iso = iso
                iso_gained = 0.0
                for _ in range(stops_to_bump):
                    bumped = next_iso(new_iso, ceiling=cfg.ISO_CEIL)
                    if bumped == new_iso:
                        saturated = True
                        break
                    new_iso = bumped
                    iso_gained += iso_step_ev
                if new_iso != iso_floor:
                    iso_bumped = True
                iso = new_iso
                self._camera.set_iso(iso)
                # Keep FEC/EC at shot 2's settings (the probe values
                # were diagnostic and we don't keep them).
                if can_probe and (fec, ec) != (shot2_fec, shot2_ec):
                    self._camera.set_fec(shot2_fec)
                    if hasattr(self._camera, "set_ec"):
                        self._camera.set_ec(shot2_ec)
                    fec, ec = shot2_fec, shot2_ec
                    ec_used = shot2_ec_used
                _, last_delta = shoot("linear shot 4 (post ISO bump)")
                converged = abs(last_delta) <= tolerance
                if not converged and not saturated:
                    # One more refinement after ISO bump. The flash
                    # response is now (mostly) at the new ISO; a
                    # FEC/EC adjustment by -last_delta should land us
                    # within tolerance unless the rig is genuinely
                    # nonlinear.
                    refine_bias = (fec + ec) + (-last_delta)
                    fec, ec, ec_used_rf = _distribute_bias(
                        refine_bias,
                        fec_min=fec_min, fec_max=fec_max, fec_step=fec_step,
                        ec_min=ec_min, ec_max=ec_max, ec_step=ec_step,
                    )
                    ec_used = ec_used or ec_used_rf
                    self._camera.set_fec(fec)
                    if hasattr(self._camera, "set_ec"):
                        self._camera.set_ec(ec)
                    _, last_delta = shoot("linear shot 5 (refine post-ISO)")
                    converged = abs(last_delta) <= tolerance
        else:
            # Slope healthy but |delta_1| was within tolerance and
            # can_probe was False — accept.
            converged = abs(delta_1) <= tolerance
            last_delta = delta_1

        return ApertureCalibration(
            aperture=aperture,
            fec_ev=fec,
            iso=iso,
            final_median=last_median,
            final_ev_delta=last_delta,
            hit_tolerance=converged,
            iso_bumped=iso_bumped,
            test_shots_taken=shots,
            saturated_at_ceiling=saturated,
            ec_ev=ec,
            ec_used=ec_used,
            iso_headroom_applied=False,
            iso_headroom_attempted=False,
            iso_headroom_attempt_ev=None,
        ), baseline

    def _try_iso_headroom_swap(
        self,
        *,
        aperture: str,
        fec: float,
        ec: float,
        iso: int,
        baseline: float,
        lens_slug: str,
        session_id: str,
        capture_dir: Path,
        shot_index: int,
        ec_min: float,
        ec_step: float,
    ) -> tuple[
        tuple[float, float, int, float, float] | None,
        float | None,
    ]:
        """Trade ``ISO_HEADROOM_EV`` of flash bias for ISO and verify.

        Returns ``(applied, attempt_ev)``:
        * ``applied`` is ``(fec, ec, iso, median, delta)`` when the
          verification shot landed within tolerance; ``None``
          otherwise (camera state is reverted by this method).
        * ``attempt_ev`` is the verification shot's EV delta. ``None``
          when the swap couldn't be tried at all (no FEC/EC room
          OR no ISO ceiling room); otherwise the measured ev for
          the attempt — useful for diagnosing reverts.
        """
        cfg = self._config
        headroom_ev = float(getattr(cfg, "ISO_HEADROOM_EV", 1.0))
        if headroom_ev <= 0:
            return None, None
        # Reduce FEC down to FEC_MIN, then EC down to its min, by a
        # combined ``headroom_ev``. The flash bias goes down by
        # exactly that much; ISO compensates by the same amount.
        fec_room = fec - cfg.FEC_MIN_EV
        ec_room = ec - ec_min
        total_room = fec_room + ec_room
        applicable_ev = min(headroom_ev, total_room)
        if applicable_ev <= 0:
            return None, None
        fec_reduce = min(applicable_ev, fec_room)
        ec_reduce = applicable_ev - fec_reduce
        new_fec = _snap(fec - fec_reduce, cfg.FEC_STEP_EV)
        new_ec = _snap(ec - ec_reduce, ec_step)
        # Bump ISO in ⅓-stop steps until ev_added ≈ applicable_ev or
        # ISO ceiling is reached.
        new_iso = iso
        ev_added = 0.0
        while ev_added + 1e-6 < applicable_ev:
            stepped = next_iso(new_iso, ceiling=cfg.ISO_CEIL)
            if stepped == new_iso:
                break  # at ceiling
            new_iso = stepped
            ev_added += cfg.FEC_STEP_EV
        if abs(ev_added) <= 1e-6:
            return None, None  # no ISO room to add; skip the swap
        # Apply, capture, measure.
        self._camera.set_fec(new_fec)
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(new_ec)
        self._camera.set_iso(new_iso)
        dest = capture_dir / _calibration_filename(aperture, shot_index)
        self._camera.capture_to(dest)
        median = self._measure(dest)
        delta = ev_delta(median, baseline)
        self._log_shot(
            aperture=aperture,
            iso=new_iso,
            fec=new_fec,
            median=median,
            ev=delta,
            filename=str(dest.name),
            lens_slug=lens_slug,
            session_id=session_id,
            notes=(
                f"iso headroom swap (FEC {fec:+.2f}→{new_fec:+.2f}, "
                f"EC {ec:+.2f}→{new_ec:+.2f}, ISO {iso}→{new_iso}); "
                f"verification ev={delta:+.2f}"
            ),
        )
        if abs(delta) <= cfg.TEST_SHOT_TOLERANCE_EV:
            return (new_fec, new_ec, new_iso, median, delta), delta
        # Revert: the verification shot landed outside tolerance,
        # so the swap assumption (flash at mechanical max →
        # ISO compensates 1:1 for flash reduction) didn't hold.
        self._camera.set_fec(fec)
        if hasattr(self._camera, "set_ec"):
            self._camera.set_ec(ec)
        self._camera.set_iso(iso)
        return None, delta

    # ---- Helpers ----------------------------------------------------------

    def _prompt_power(
        self,
        prompt_set_flash_power: Callable[..., bool],
        power_stops: float,
        *,
        context: str,
        currently_set_power_stops: float | None = None,
    ) -> None:
        """Render labels for groups C / A and prompt the user.

        Group A trails group C by 1/3 stop (subject-distance offset
        for the right flash, documented in
        ``rustling-tumbling-moonbeam.md``). The A label is clamped
        to the SB-5000's 1/128 minimum if C is at 1/128 itself —
        the commander can't display lower than 1/128, so groups C
        and A coincide at the floor.

        ``currently_set_power_stops``: when supplied and equal to
        the requested ``power_stops`` (within 1/6 stop, the 1/3-stop
        ladder's half-step), the prompt is suppressed — the user
        already dialled the target, no reason to ask again.

        Raises :class:`errors.SessionAborted` if the user answers `q`.
        """
        if (
            currently_set_power_stops is not None
            and abs(power_stops - currently_set_power_stops) < 1 / 6
        ):
            return
        a_stops = min(power_stops + 1 / 3, 7.0)
        ok = prompt_set_flash_power(
            power_c_label=power_label(power_stops),
            power_a_label=power_label(a_stops),
            context=context,
        )
        if not ok:
            raise SessionAborted(
                f"user aborted at flash-power prompt ({context})"
            )

    def _iso_fine_tune(
        self,
        *,
        aperture: str,
        capture_dir: Path,
        baseline: float,
        iso: int,
        iso_floor: int,
        iso_ceiling: int,
        shots_taken: int,
        power_for_log: float,
        lens_slug: str,
        session_id: str,
        note_prefix: str,
        initial_median: float | None = None,
        initial_delta: float | None = None,
        tol: float | None = None,
    ) -> tuple[int, float, float, int]:
        """Adjust ISO by 1/3-stop steps until within tolerance.

        Either evaluates an existing capture's ``initial_median /
        initial_delta`` (passed by the caller) or takes one fresh
        capture as a starting point, then iterates: adjust ISO,
        capture, evaluate, until ``|delta| <= tol`` (defaults to
        ``TEST_SHOT_TOLERANCE_EV`` when ``tol`` is None) or ISO
        saturates at floor / ceiling.

        Returns ``(shots_taken, last_median, last_delta, final_iso)``.
        """
        cfg = self._config
        if tol is None:
            tol = float(cfg.TEST_SHOT_TOLERANCE_EV)
        max_retries = int(cfg.TEST_SHOT_MAX_RETRIES)
        notes = f"{note_prefix}, power {power_label(power_for_log)}"

        # Establish the starting (median, delta).
        if initial_median is not None and initial_delta is not None:
            last_median = initial_median
            last_delta = initial_delta
        else:
            dest = capture_dir / _calibration_filename(aperture, shots_taken)
            self._camera.capture_to(dest)
            last_median = self._measure(dest)
            shots_taken += 1
            last_delta = ev_delta(last_median, baseline)
            self._log_shot(
                aperture=aperture, iso=iso, fec=0.0,
                median=last_median, ev=last_delta,
                filename=str(dest.name), lens_slug=lens_slug,
                session_id=session_id, notes=notes,
            )

        for _ in range(max_retries):
            if abs(last_delta) <= tol:
                break
            if last_delta < -tol:
                bumped = next_iso(iso, ceiling=iso_ceiling)
                if bumped == iso:
                    break
                iso = bumped
            else:
                dropped = prev_iso(iso, floor=iso_floor)
                if dropped == iso:
                    break
                iso = dropped
            self._camera.set_iso(iso)
            dest = capture_dir / _calibration_filename(aperture, shots_taken)
            self._camera.capture_to(dest)
            last_median = self._measure(dest)
            shots_taken += 1
            last_delta = ev_delta(last_median, baseline)
            self._log_shot(
                aperture=aperture, iso=iso, fec=0.0,
                median=last_median, ev=last_delta,
                filename=str(dest.name), lens_slug=lens_slug,
                session_id=session_id, notes=notes,
            )

        return shots_taken, last_median, last_delta, iso

    def _initial_iso(self, ap_float: float) -> int:
        """ISO base for an aperture, picked from the per-aperture
        ``RANGE_EXTENSION`` table; falls back to the table's smallest
        ISO if the aperture is below the smallest tabulated value."""
        table = self._config.RANGE_EXTENSION
        if ap_float in table:
            return int(table[ap_float][0])
        # Pick the largest tabulated key ≤ ap_float (i.e. nearest
        # smaller-or-equal aperture entry).
        keys = sorted(table.keys())
        candidate = keys[0]
        for k in keys:
            if k <= ap_float:
                candidate = k
        return int(table[candidate][0])

    def _log_shot(
        self,
        *,
        aperture: str,
        iso: int,
        fec: float,
        median: float,
        ev: float,
        filename: str,
        lens_slug: str,
        session_id: str,
        notes: str,
    ) -> None:
        if self._csv is None:
            return
        self._csv.append(
            ShotRow(
                timestamp=self._now().isoformat(),
                session_id=session_id,
                lens_config_slug=lens_slug,
                rail_step=0,
                rail_position_mm=0.0,
                aperture=aperture,
                iso=iso,
                shutter_speed="",
                fec=fec,
                median_central_luminance=median,
                median_ev_relative_to_baseline=ev,
                filename=filename,
                phase=Phase.CALIBRATION,
                notes=notes,
            )
        )


# ---- Free helpers ----------------------------------------------------------


def _parse_aperture(text: str) -> float:
    """``"f/4"`` -> ``4.0``; tolerates a leading space or stray decimal."""
    cleaned = text.strip().lstrip("fF").lstrip("/")
    return float(cleaned)


def _snap(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step``.

    When ``step = 1/N`` for some integer N (the common case for
    ⅓-stop, ½-stop, ¼-stop grids), the snapped value is computed as
    ``round(value * N) / N`` rather than ``round(value/step) * step``.
    The two forms are mathematically equivalent but produce slightly
    different IEEE-754 floats at specific positions: for step=1/3,
    ``-5 * (1/3) = -1.6666666666666665`` but ``-5/3 =
    -1.6666666666666667`` (off by 1 ULP). The Nikon Z7's PTP range
    widget for `flashexposurecompensation` validates against its
    internal ⅓-stop grid by what appears to be strict-equality
    comparison: the 1-ULP form is silently rejected. The
    ``camera.set_config()`` call returns success, the FEC widget
    retains its previous value, and the cascade's `set_fec(value)`
    becomes a silent no-op — masquerading as `set_fec` lag or
    "hysteresis" because subsequent shots fire at whatever the
    *previous* successfully-set FEC was.

    The bug manifested on Sigma 150 + 6T + 5T cascades that landed on
    FEC = -1.667 (k = -5); see `tools/cascade_dry_run.py` and the
    accompanying DESIGN_NOTES entry. Only k = -5 and k = -7 in the
    ⅓-stop grid suffer the 1-ULP form drift; other ⅓-stop values
    happen to round to the canonical form by coincidence.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    # Detect 1/N steps and use direct division by N to produce the
    # canonical IEEE-754 form. Threshold of 1e-9 is well below
    # ⅓-stop resolution; allows tests that pass ``1/3`` literally and
    # any caller that supplies ``1.0/3.0`` or similar.
    inv = round(1.0 / step)
    if inv > 0 and abs(inv * step - 1.0) < 1e-9:
        return round(value * inv) / inv
    return round(value / step) * step


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _distribute_bias(
    target_bias: float,
    *,
    fec_min: float,
    fec_max: float,
    fec_step: float,
    ec_min: float,
    ec_max: float,
    ec_step: float,
) -> tuple[float, float, bool]:
    """Split a target exposure bias across FEC and (if needed) EC.

    Used by the linear cascade to convert a single "I need +X EV"
    decision into a concrete (FEC, EC) setting respecting the Z7's
    asymmetric ranges (FEC: -3 to +1, EC: -3 to +3). FEC absorbs as
    much as possible (it's the "primary" lever); EC picks up any
    residual past FEC's limits. Each value is snapped to its own
    ⅓-stop grid so the camera's PTP widget accepts it.

    **Avoids k=±7 grid positions.** The Z7's PTP range widget for
    ``flashexposurecompensation`` empirically rejects every float
    representation of ±2.333 (k=±7) — see
    ``tools/fec_acceptance_grid.py``. If FEC or EC would otherwise
    snap to one of these positions, the function substitutes the
    nearest whole stop (k=±6) and puts the residual ⅓ stop in the
    other lever. Total exposure bias is preserved exactly.

    Returns ``(fec, ec, ec_used)``. ``ec_used`` is True when EC was
    actually moved away from zero — useful for the cascade's result
    summary so the operator can see whether the camera-side EC bias
    is in play.
    """
    fec = _clamp(_snap(target_bias, fec_step), fec_min, fec_max)
    # The Z7 rejects k=±7 (±2.333) for FEC specifically (empirical;
    # see tools/fec_acceptance_grid.py). Substitute k=±6 (±2.0)
    # and let the residual go to EC. EC isn't checked here —
    # ``NikonZ7._set_range``'s multi-form retry handles EC's
    # representation quirks without changing the grid; if EC's
    # k=±7 turn out to be similarly rejected, the readback there
    # will raise so we can add an explicit substitution.
    fec_k = round(fec / fec_step) if fec_step > 0 else 0
    if fec_k == -7:
        fec = -2.0
    elif fec_k == 7:
        fec = 2.0
    remaining = target_bias - fec
    if abs(remaining) < ec_step / 2.0:
        return fec, 0.0, False
    ec = _clamp(_snap(remaining, ec_step), ec_min, ec_max)
    ec_used = abs(ec) > 1e-6
    return fec, ec, ec_used


def _calibration_filename(aperture: str, shot_index: int) -> str:
    safe = aperture.replace("/", "_").replace(".", "_")
    return f"cal_{safe}_{shot_index:02d}.jpg"
