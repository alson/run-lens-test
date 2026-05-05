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

from camera.nikon_z7 import NikonZ7
from exposure.analyse import ev_delta
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
    ) -> None:
        self._camera = camera
        self._measure = measure
        self._config = config
        self._csv = shots_csv
        self._now = now

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
        for aperture in apertures:
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

    # ---- Per-aperture loop -------------------------------------------------

    def _calibrate_aperture(
        self,
        *,
        aperture: str,
        lens_slug: str,
        session_id: str,
        capture_dir: Path,
        baseline: float | None,
    ) -> tuple[ApertureCalibration, float | None]:
        cfg = self._config
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
                return ApertureCalibration(
                    aperture=aperture,
                    fec_ev=fec,
                    iso=iso,
                    final_median=median,
                    final_ev_delta=delta,
                    hit_tolerance=True,
                    iso_bumped=iso_bumped,
                    test_shots_taken=shot_index + 1,
                    ec_ev=ec,
                    ec_used=ec_used,
                ), baseline

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

        return ApertureCalibration(
            aperture=aperture,
            fec_ev=fec,
            iso=iso,
            final_median=last_median,
            final_ev_delta=last_delta,
            hit_tolerance=False,
            iso_bumped=iso_bumped,
            test_shots_taken=cfg.TEST_SHOT_MAX_RETRIES + 1,
            saturated_at_ceiling=saturated,
            ec_ev=ec,
            ec_used=ec_used,
        ), baseline

    # ---- Helpers ----------------------------------------------------------

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
    """Round ``value`` to the nearest multiple of ``step``."""
    return round(value / step) * step


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _calibration_filename(aperture: str, shot_index: int) -> str:
    safe = aperture.replace("/", "_").replace(".", "_")
    return f"cal_{safe}_{shot_index:02d}.jpg"
