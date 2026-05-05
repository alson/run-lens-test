"""Session runner — orchestrates one full session.

The runner is the bag of state that ties the StackShot controller,
the camera wrapper, calibration, droop detection, range-check, and
notifications together. It accepts every collaborator via the
constructor so tests build a runner with fakes and exercise the
whole flow without hardware.

The flow follows ``macro_lens_test_plan.md §4`` — see that file for
the design. The actual orchestration here is intentionally
straight-line: each phase reads as one method on
:class:`SessionRunner`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

from camera.nikon_z7 import NikonZ7
from errors import SessionAborted
from exposure.analyse import CropBox, crop_box, ev_delta, measure_crop
from exposure.droop import DroopDetector, DroopEvent
from exposure.range_check import RangeChecker
from notify.ntfy import (
    BATTERY_DROOP,
    LENS_CHANGE_REQUIRED,
    NtfyClient,
    Notification,
)
from session.calibrate import (
    ApertureCalibration,
    CalibrationResult,
    Calibrator,
    next_iso,
)
from session.filenames import step_filename
from session.log import (
    Phase,
    SessionMetadata,
    ShotRow,
    ShotsCSV,
    write_json,
)
from stackshot.controller import StackShotController
from stackshot.protocol import Direction
from ui.prompts import (
    confirm,
    prompt_lens_config,
    prompt_step_count,
    slugify_lens_config,
)

log = logging.getLogger(__name__)


# ---- Aperture sequence helper ---------------------------------------------


def aperture_sequence(
    available_choices: Iterable[str],
    *,
    whole_stop_scale: tuple[float, ...],
    max_fstop: float,
) -> list[str]:
    """Build the per-lens aperture sequence: widest first, then every
    whole stop in ``whole_stop_scale`` between (widest, max_fstop],
    snapped to the closest available choice. Per existing plan §1.3.
    """
    parsed: list[tuple[float, str]] = []
    for choice in available_choices:
        try:
            parsed.append((_parse_fnum(choice), choice))
        except ValueError:
            continue
    parsed.sort(key=lambda x: x[0])
    if not parsed:
        return []
    widest_value, widest_label = parsed[0]
    sequence: list[tuple[float, str]] = [(widest_value, widest_label)]
    for stop in whole_stop_scale:
        if stop <= widest_value or stop > max_fstop:
            continue
        # Closest available choice ≥ stop.
        candidate = min(parsed, key=lambda x: abs(x[0] - stop))
        if candidate not in sequence:
            sequence.append(candidate)
    # De-dup by label, preserve first-seen order.
    seen: set[str] = set()
    out: list[str] = []
    for _, label in sequence:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _parse_fnum(text: str) -> float:
    cleaned = text.strip().lstrip("fF").lstrip("/")
    return float(cleaned)


# ---- Rail / camera contracts the runner depends on ------------------------


class RailLike(Protocol):
    def get_position(self) -> float: ...

    def zero_position(self) -> None: ...

    def move_and_wait(
        self,
        direction: Direction,
        distance_mm: float,
        *,
        backlash: bool = True,
        timeout_s: float | None = None,
    ) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class CameraLike(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def verify_healthy(self, *, min_widgets: int = 100) -> None: ...

    def apply_session_config(self, **kwargs) -> None: ...

    def list_apertures(self) -> list[str]: ...

    def set_aperture(self, fnum: str) -> None: ...

    def set_iso(self, iso) -> None: ...

    def set_fec(self, ev: float) -> None: ...

    def capture_to(self, dest, *, max_retries: int = 3) -> int: ...

    def snapshot(self, label: str = "") -> None: ...


# ---- Jog / prompts dependency injection ----------------------------------


@dataclass
class JogResult:
    end_mm: float


JogStartEnd = Callable[[RailLike], JogResult]


# ---- Sharpness collector contract ----------------------------------------


class SharpnessCollectorLike(Protocol):
    """Subset of :class:`sharpness.collector.LiveCollector` the runner
    depends on. Tests can inject a no-op fake; production wires the
    real one in."""

    def record_shot(self, rail_step: int, aperture: float, jpg_path) -> object: ...

    def wait_for_pending(self) -> None: ...

    def shutdown(self, *, wait: bool = True) -> None: ...


SharpnessCollectorFactory = Callable[[Path], SharpnessCollectorLike]
"""Build a per-lens-config sharpness collector. Receives the
``cropped/`` output directory; returns an opened collector. The
default factory in :class:`SessionRunner` produces a real
:class:`LiveCollector`; tests pass a factory that returns a no-op
fake.
"""


def _real_sharpness_collector_factory(out_dir: Path) -> SharpnessCollectorLike:
    """Default factory — imports lazily so the runner module stays
    importable in environments without the sharpness package's heavier
    deps (Pillow already required, but PIL might be unavailable in a
    pure-unit-test environment)."""
    from sharpness.collector import LiveCollector
    return LiveCollector(out_dir, max_workers=4)


class _NullSharpnessCollector:
    """No-op collector used when sharpness is disabled. Methods are
    valid no-ops so the runner code never special-cases the absence
    of a collector."""

    def record_shot(self, rail_step: int, aperture: float, jpg_path) -> None:
        return None

    def wait_for_pending(self) -> None:
        return None

    def shutdown(self, *, wait: bool = True) -> None:
        return None


def _null_sharpness_collector_factory(out_dir: Path) -> SharpnessCollectorLike:
    return _NullSharpnessCollector()


# ---- Runner ---------------------------------------------------------------


@dataclass
class SessionRunner:
    """One session, one process; reused across lens configs."""

    camera: CameraLike
    rail: RailLike
    config: object
    ntfy: NtfyClient | None = None
    range_checker: RangeChecker | None = None
    output_root: Path | None = None
    measure_path: Callable[[Path], float] | None = None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc)
    prompt_lens_config: Callable[[], str | None] = field(default_factory=lambda: lambda: prompt_lens_config())
    prompt_step_count: Callable[[], int] = field(default_factory=lambda: lambda: prompt_step_count())
    confirm: Callable[[str], bool] = field(default_factory=lambda: confirm)
    jog: JogStartEnd | None = None
    droop_detector: DroopDetector | None = None
    crop_box: CropBox | None = None
    sharpness_collector_factory: SharpnessCollectorFactory = field(
        default=_null_sharpness_collector_factory
    )
    """Per-lens-config sharpness collector factory.

    Default is a no-op so existing tests stay opaque to sharpness; the
    production entry point swaps in
    :func:`_real_sharpness_collector_factory` (or constructs a
    :class:`LiveCollector` directly)."""

    # ---- internal state filled at start ----------------------------------
    session_id: str = ""
    session_dir: Path = field(default_factory=Path)
    shots_csv: ShotsCSV | None = None
    metadata: SessionMetadata | None = None
    baseline_median: float | None = None
    _sharpness_collector: SharpnessCollectorLike | None = None

    # ---- public entry points ---------------------------------------------

    def start(self) -> None:
        """Open hardware, verify, apply config, build session dir.

        Also reads the StackShot software / hardware IDs (logged in
        session.json) and runs the one-time FEC sanity check, which
        catches the SU-800 mistakenly mounted instead of the SB-5000
        commander.
        """
        self.camera.open()
        self.camera.verify_healthy(min_widgets=self.config.MIN_CONFIG_TREE_SIZE)
        self.camera.apply_session_config(
            iso=self.config.ISO_FLOOR,
            shutter_candidates=self.config.SHUTTER_SPEED_CANDIDATES,
            image_quality=self.config.IMAGE_QUALITY,
            fec=0.0,
        )
        self.session_id = self._make_session_id()
        self.session_dir = (self.output_root or Path("output")) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.shots_csv = ShotsCSV(self.session_dir / "shots.csv")

        # StackShot identification — best-effort, never aborts.
        sw_id = hw_id = None
        try:
            sw_id = self.rail.software_id()  # type: ignore[attr-defined]
            hw_id = self.rail.hardware_id()  # type: ignore[attr-defined]
        except (AttributeError, Exception):  # noqa: BLE001
            pass

        self.metadata = SessionMetadata(
            session_id=self.session_id,
            started_at=self.now().isoformat(),
            output_root=str(self.output_root or ""),
            ntfy_topic=getattr(self.config, "NTFY_TOPIC", ""),
            stackshot_software_id=sw_id,
            stackshot_hardware_id=hw_id,
        )
        self._write_metadata()

        # FEC sanity check (skip if the camera doesn't expose it —
        # e.g. fakes used in tests; production NikonZ7 always has it).
        sanity_check = getattr(self.camera, "fec_sanity_check", None)
        if sanity_check is not None:
            try:
                sanity_check(
                    baseline_fec=0.0,
                    low_fec=getattr(
                        self.config, "FEC_SANITY_LOW_EV", -2.0
                    ),
                    min_ratio=getattr(
                        self.config, "FEC_SANITY_MIN_RATIO", 0.5
                    ),
                    capture_dir=self.session_dir,
                    measure=self._measure,
                )
            except TypeError:
                # Camera fake doesn't accept these kwargs — skip.
                pass

    def stop(self, *, aborted: bool = False) -> None:
        """Move rail home, close hardware, finalise session.json.

        We don't try to move the rail home here — the per-lens-config
        flow already returns the rail to position 0 after each
        sweep. If shutdown is happening due to abort the rail might
        be anywhere; the user can re-jog when they restart.
        """
        if self.metadata is not None:
            self.metadata.ended_at = self.now().isoformat()
            self.metadata.aborted = aborted
            self._write_metadata()
        try:
            self.camera.close()
        finally:
            try:
                self.rail.close()
            except Exception:
                pass

    # ---- Lens-config loop ------------------------------------------------

    def run(self) -> None:
        """Top-level loop: prompt for each lens config, run it, repeat
        until the user enters an empty line.

        Ctrl-C and :class:`SessionAborted` both stop cleanly: the
        StackShot is told to STOP (cancels any in-flight motion), the
        camera is closed, and ``session.json`` is finalised with
        ``aborted=True``.
        """
        self.start()
        try:
            while True:
                lens_text = self.prompt_lens_config()
                if lens_text is None:
                    return
                slug = slugify_lens_config(lens_text)
                self._run_one_lens_config(slug)
                if self.metadata is not None:
                    self.metadata.lens_configs.append(slug)
        except (SessionAborted, KeyboardInterrupt):
            self._stop_rail_quietly()
            self.stop(aborted=True)
            raise
        else:
            self.stop()

    def _stop_rail_quietly(self) -> None:
        """Best-effort STOP on the rail; never raises."""
        stop = getattr(self.rail, "stop", None)
        if stop is None:
            return
        try:
            stop()
        except Exception:                            # noqa: BLE001
            pass

    # ---- One lens config -------------------------------------------------

    def _run_one_lens_config(self, slug: str) -> None:
        if self.jog is None:
            raise RuntimeError("SessionRunner.jog not configured")
        lens_dir = self.session_dir / slug
        lens_dir.mkdir(parents=True, exist_ok=True)

        jog_result = self.jog(self.rail)
        end_mm = jog_result.end_mm
        steps = self.prompt_step_count()
        step_size_mm = end_mm / (steps - 1)

        # Move rail back to position 0 with backlash compensation.
        # end_mm and step_size_mm may be negative (user jogged
        # backward to END) — use the sign-aware helper so the rail
        # gets a non-negative distance with the correct direction.
        self._return_rail_to_zero(end_mm)

        apertures = aperture_sequence(
            self.camera.list_apertures(),
            whole_stop_scale=self.config.WHOLE_STOP_SCALE,
            max_fstop=self.config.MAX_FSTOP,
        )

        # One sharpness collector per lens config. Crops land under
        # <lens>/<aperture>/cropped/<crop-name>/<source-filename>.jpg
        # — see config.SHARPNESS_CROPPED_DIRNAME and §8.3.5 of the
        # plan. The collector lives for the entire lens config so
        # probe / extension shots feed the same dataset.
        self._sharpness_collector = self.sharpness_collector_factory(lens_dir)
        if self.range_checker is not None:
            self.range_checker.set_collector(self._sharpness_collector)
        try:
            self._run_one_lens_config_inner(
                slug=slug,
                lens_dir=lens_dir,
                end_mm=end_mm,
                steps=steps,
                step_size_mm=step_size_mm,
                apertures=apertures,
            )
        finally:
            try:
                self._sharpness_collector.shutdown(wait=True)
            except Exception:                        # noqa: BLE001
                # Cropping errors must not mask a session error or
                # block lens-change recovery.
                log.exception("sharpness collector shutdown failed")
            if self.range_checker is not None:
                self.range_checker.set_collector(None)
            self._sharpness_collector = None

    def _run_one_lens_config_inner(
        self,
        *,
        slug: str,
        lens_dir: Path,
        end_mm: float,
        steps: int,
        step_size_mm: float,
        apertures: list[str],
    ) -> None:

        # Calibration at rail step 0.
        calibrator = Calibrator(
            camera=self.camera,
            measure=self._measure,
            config=self.config,
            shots_csv=self.shots_csv,
            now=self.now,
        )
        cal_result = calibrator.run(
            lens_dir=lens_dir,
            lens_slug=slug,
            session_id=self.session_id,
            apertures=apertures,
            baseline_median=self.baseline_median,
        )
        self.baseline_median = cal_result.baseline_median
        if self.metadata is not None:
            self.metadata.baseline_median_luminance = self.baseline_median
            self._write_metadata()
        write_json(
            lens_dir / "calibration.json",
            {
                "baseline_median": cal_result.baseline_median,
                "per_aperture": cal_result.per_aperture,
            },
        )

        # Sweep.
        self._run_sweep(
            lens_dir=lens_dir,
            slug=slug,
            steps=steps,
            step_size_mm=step_size_mm,
            calibration=cal_result,
            apertures=apertures,
        )

        # Range check + optional probe + optional extension sweep.
        # Returns the total number of forward steps the rail has moved
        # (original sweep + any extension), so we know how far back to
        # go to reach position 0.
        forward_steps_done = steps - 1
        if self.range_checker is not None:
            forward_steps_done = self._run_range_check_and_extend(
                lens_dir=lens_dir,
                slug=slug,
                end_mm=end_mm,
                step_size_mm=step_size_mm,
                steps=steps,
                calibration=cal_result,
                apertures=apertures,
            )

        # Return rail to 0 for the next lens config (backlash on; we
        # may have moved into negative territory if the extension was
        # on the 'near' side).
        if forward_steps_done != 0:
            self._return_rail_to_zero(forward_steps_done * step_size_mm)

        # Lens-change prompt.
        self._lens_change_recovery(slug)

    # ---- Sweep -----------------------------------------------------------

    def _run_sweep(
        self,
        *,
        lens_dir: Path,
        slug: str,
        steps: int,
        step_size_mm: float,
        calibration: CalibrationResult,
        apertures: list[str],
    ) -> None:
        droop = self.droop_detector or DroopDetector(
            window=self.config.ROLLING_WINDOW,
            criteria=(self.config.DROOP_CRIT_A, self.config.DROOP_CRIT_B),
        )
        # Sweep direction is the sign of step_size_mm. With a positive
        # end_mm (user jogged forward) sweep direction is FORWARD; with
        # negative (user jogged backward) it's BACKWARD. backlash=False
        # is correct either way as long as we never reverse direction
        # mid-sweep.
        sweep_direction = (
            Direction.FORWARD if step_size_mm >= 0 else Direction.BACKWARD
        )
        sweep_step_distance = abs(step_size_mm)
        for step_index in range(steps):
            if step_index > 0:
                self.rail.move_and_wait(
                    sweep_direction, sweep_step_distance, backlash=False
                )
            position_mm = step_index * step_size_mm
            self._run_one_step(
                lens_dir=lens_dir,
                slug=slug,
                step_index=step_index,
                position_mm=position_mm,
                apertures=apertures,
                calibration=calibration,
                droop=droop,
            )

    def _run_one_step(
        self,
        *,
        lens_dir: Path,
        slug: str,
        step_index: int,
        position_mm: float,
        apertures: list[str],
        calibration: CalibrationResult,
        droop: DroopDetector,
    ) -> None:
        aborted = False
        for aperture in apertures:
            if aborted:
                self._log_aborted_remainder(
                    lens_dir=lens_dir,
                    slug=slug,
                    step_index=step_index,
                    position_mm=position_mm,
                    aperture=aperture,
                )
                continue
            cal = calibration.per_aperture[aperture]
            self.camera.set_aperture(aperture)
            self.camera.set_iso(cal.iso)
            self.camera.set_fec(cal.fec_ev)
            dest = self._sweep_destination(lens_dir, aperture, step_index)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.camera.capture_to(dest)
            self._record_sharpness(
                step_index=step_index, aperture=aperture, dest=dest
            )
            median = self._measure(dest)
            ev = ev_delta(median, self.baseline_median or median)
            self._log_shot(
                slug=slug,
                step_index=step_index,
                position_mm=position_mm,
                aperture=aperture,
                cal=cal,
                median=median,
                ev=ev,
                filename=str(dest.relative_to(lens_dir)),
                phase=Phase.SWEEP,
            )
            event = droop.push(aperture, ev)
            if event is not None:
                self._handle_droop(event, droop)
                aborted = True
                continue
            self.sleep(self._recycle_for(aperture))

    # ---- Range check + auto-extension ----------------------------------

    def _run_range_check_and_extend(
        self,
        *,
        lens_dir: Path,
        slug: str,
        end_mm: float,
        step_size_mm: float,
        steps: int,
        calibration: CalibrationResult,
        apertures: list[str],
    ) -> int:
        """Evaluate range adequacy; probe + extend if peak is at an
        edge. Returns the total forward step count from the START
        position so the caller can drive the rail back to 0.

        Stays a no-op if ``range_checker.evaluate`` returns ``ok=True``
        (which is the stub's behaviour today).
        """
        from session.log import write_json

        assert self.range_checker is not None
        # Drain pending sharpness crops so the first evaluate() sees
        # the complete main-sweep dataset.
        if self._sharpness_collector is not None:
            self._sharpness_collector.wait_for_pending()
        result = self.range_checker.evaluate(
            lens_dir=lens_dir,
            end_mm=end_mm,
            step_size_mm=step_size_mm,
            apertures=apertures,
        )
        if (
            result.ok
            or not result.needs_extension
            or result.peak_side not in ("near", "far", "both")
        ):
            write_json(lens_dir / "range_check.json", {
                "ok": result.ok,
                "peak_side": result.peak_side,
                "extension_applied": False,
                "notes": result.notes,
            })
            return steps - 1

        extension_mm = end_mm * float(
            getattr(self.config, "RANGE_EXTENSION_FRACTION", 0.5)
        )
        # Per-side extension capacity in whole grid steps.
        extension_steps = int(extension_mm // step_size_mm)
        configured_probe_count = int(
            getattr(self.config, "RANGE_CHECK_PROBE_STEPS", 2)
        )
        if extension_steps < 1:
            # Extension wouldn't fit a single grid step — nothing to
            # probe, nothing to extend. Trust the first-call decision
            # and bail out so the user can re-jog if needed.
            write_json(lens_dir / "range_check.json", {
                "ok": False,
                "peak_side": result.peak_side,
                "extension_applied": False,
                "extension_mm": extension_mm,
                "notes": (result.notes or "")
                + " (extension shorter than one grid step; skipped)",
            })
            return steps - 1

        # Per-side probe count — split for BOTH so we shoot ~half on
        # each side; cap at the available extension slots so we never
        # try to shoot more probes than the extension has room for.
        if result.peak_side == "both":
            probe_per_side = min(
                max(1, configured_probe_count // 2 + 1), extension_steps
            )
            sides = [("far", probe_per_side), ("near", probe_per_side)]
        else:
            probe_per_side = min(configured_probe_count, extension_steps)
            sides = [(result.peak_side, probe_per_side)]

        # Walk through each requested side, shooting probes at
        # grid-aligned step indices. Probes use the same filename
        # convention as the rest of the sweep — sweep, probe, and
        # (if it follows) extension are distinguished only by the
        # phase column in shots.csv.
        current_pos = (steps - 1) * step_size_mm  # rail at end_mm
        probe_data: list = []
        probe_positions_per_side: dict[str, list[float]] = {"far": [], "near": []}
        for side, count in sides:
            for i in range(count):
                step_index = self._extension_step_index(
                    side=side, original_steps=steps, offset_in_extension=i
                )
                target = step_index * step_size_mm
                self._move_to(current_pos, target)
                current_pos = target
                self._shoot_at_position(
                    lens_dir=lens_dir,
                    slug=slug,
                    step_index=step_index,
                    position_mm=target,
                    apertures=apertures,
                    calibration=calibration,
                    phase=Phase.PROBE,
                    file_factory=lambda ap, idx=step_index:
                        self._sweep_destination(lens_dir, ap, idx),
                )
                probe_positions_per_side[side].append(target)

        # Drain pending sharpness crops so the second evaluate() sees
        # the main sweep + every probe shot just taken.
        if self._sharpness_collector is not None:
            self._sharpness_collector.wait_for_pending()

        result2 = self.range_checker.evaluate(
            lens_dir=lens_dir,
            end_mm=end_mm,
            step_size_mm=step_size_mm,
            apertures=apertures,
            probe_data=probe_data,
        )

        record_base = {
            "peak_side": result.peak_side,
            "extension_mm": extension_mm,
            "probe_positions_mm": probe_positions_per_side,
        }

        if not result2.peak_in_range_after_probe:
            if self.ntfy is not None:
                self.ntfy.with_body(
                    Notification(
                        title="Range still inadequate",
                        body="",
                        priority="high",
                        tags=("warning",),
                    ),
                    f"{slug}: probe on {result.peak_side} side did not "
                    "bracket peak; please re-jog and retry.",
                )
            self.confirm(
                f"\nRange still inadequate on {slug}. Press ENTER to "
                "continue to lens change anyway, q to abort: "
            )
            write_json(lens_dir / "range_check.json", {
                "ok": False,
                "extension_applied": False,
                **record_base,
                "notes": "probe did not bracket peak",
            })
            return self._steps_from_zero_at(current_pos, step_size_mm)

        # Probe brackets the peak — fill in the remaining grid slots
        # in the extension, on each requested side. Probes already
        # cover ``count`` slots per side; we shoot the additional
        # ``extension_steps - count`` slots.
        extension_step_count_per_side: dict[str, int] = {"far": 0, "near": 0}
        for side, probe_count_used in sides:
            for i in range(probe_count_used, extension_steps):
                step_index = self._extension_step_index(
                    side=side, original_steps=steps, offset_in_extension=i
                )
                target = step_index * step_size_mm
                self._move_to(current_pos, target)
                current_pos = target
                self._shoot_at_position(
                    lens_dir=lens_dir,
                    slug=slug,
                    step_index=step_index,
                    position_mm=target,
                    apertures=apertures,
                    calibration=calibration,
                    phase=Phase.EXTENSION,
                    file_factory=lambda ap, idx=step_index:
                        self._sweep_destination(lens_dir, ap, idx),
                )
                extension_step_count_per_side[side] += 1

        write_json(lens_dir / "range_check.json", {
            "ok": False,
            "extension_applied": True,
            **record_base,
            "extension_step_count": extension_step_count_per_side,
            "probe_step_count": {
                "far": dict(sides).get("far", 0),
                "near": dict(sides).get("near", 0),
            },
            "notes": result2.notes,
        })
        return self._steps_from_zero_at(current_pos, step_size_mm)

    @staticmethod
    def _extension_step_index(
        *, side: str, original_steps: int, offset_in_extension: int
    ) -> int:
        """Map (side, position-within-extension) to a signed step
        index. Far side continues past the main sweep; near side
        descends below zero."""
        if side == "far":
            return original_steps + offset_in_extension
        # side == "near"
        return -1 - offset_in_extension

    def _shoot_at_position(
        self,
        *,
        lens_dir: Path,
        slug: str,
        step_index: int,
        position_mm: float,
        apertures: list[str],
        calibration: CalibrationResult,
        phase: Phase,
        file_factory: Callable[[str], Path],
    ) -> None:
        for aperture in apertures:
            cal = calibration.per_aperture[aperture]
            self.camera.set_aperture(aperture)
            self.camera.set_iso(cal.iso)
            self.camera.set_fec(cal.fec_ev)
            dest = file_factory(aperture)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.camera.capture_to(dest)
            self._record_sharpness(
                step_index=step_index, aperture=aperture, dest=dest
            )
            median = self._measure(dest)
            ev = ev_delta(median, self.baseline_median or median)
            self._log_shot(
                slug=slug,
                step_index=step_index,
                position_mm=position_mm,
                aperture=aperture,
                cal=cal,
                median=median,
                ev=ev,
                filename=str(dest.relative_to(lens_dir)),
                phase=phase,
            )
            self.sleep(self._recycle_for(aperture))

    def _move_to(self, current_pos: float, target_pos: float) -> None:
        delta = target_pos - current_pos
        if delta == 0:
            return
        direction = Direction.FORWARD if delta > 0 else Direction.BACKWARD
        # Backlash on whenever we change direction; we don't track
        # the previous direction here, so default to True (safe but
        # slightly slower).
        self.rail.move_and_wait(direction, abs(delta), backlash=True)

    def _return_rail_to_zero(self, current_pos: float) -> None:
        if current_pos == 0:
            return
        direction = Direction.BACKWARD if current_pos > 0 else Direction.FORWARD
        self.rail.move_and_wait(direction, abs(current_pos), backlash=True)

    def _steps_from_zero_at(
        self, position_mm: float, step_size_mm: float
    ) -> int:
        """Forward steps from 0 to ``position_mm`` (signed)."""
        if step_size_mm == 0:
            return 0
        return int(round(position_mm / step_size_mm))

    def _handle_droop(self, event: DroopEvent, droop: DroopDetector) -> None:
        if self.ntfy is not None:
            self.ntfy.with_body(
                BATTERY_DROOP,
                f"{event.aperture}: {event.triggered_count} of last "
                f"{len(event.window_snapshot)} shots ≤ {event.threshold:+.1f} EV. "
                f"Window: {event.window_snapshot!r}",
            )
        # Pause: user swaps batteries, presses ENTER.
        ok = self.confirm(
            f"\nBattery droop on {event.aperture}. "
            "Swap flash batteries, wait for steady ready light, "
            "then press ENTER to resume. q = abort: "
        )
        if not ok:
            raise SessionAborted("user aborted at droop pause")
        droop.reset_all()

    def _log_aborted_remainder(
        self,
        *,
        lens_dir: Path,
        slug: str,
        step_index: int,
        position_mm: float,
        aperture: str,
    ) -> None:
        if self.shots_csv is None:
            return
        self.shots_csv.append(
            ShotRow(
                timestamp=self.now().isoformat(),
                session_id=self.session_id,
                lens_config_slug=slug,
                rail_step=step_index,
                rail_position_mm=position_mm,
                aperture=aperture,
                iso="",
                shutter_speed="",
                fec=0.0,
                median_central_luminance=0.0,
                median_ev_relative_to_baseline=0.0,
                filename="",
                phase=Phase.ABORTED,
                notes="skipped after droop",
            )
        )

    # ---- Lens-change recovery ------------------------------------------

    def _lens_change_recovery(self, prev_slug: str) -> None:
        if self.ntfy is not None:
            self.ntfy.with_body(
                LENS_CHANGE_REQUIRED,
                f"finished {prev_slug}; ready for next lens",
            )
        text = (
            "\nPower off camera. Change lens. Remove battery >=10s. "
            "Reinsert. Power on, wait for live view. ENTER when ready, "
            "q to quit: "
        )
        if not self.confirm(text):
            raise SessionAborted("user quit at lens change")
        self.camera.close()
        for attempt in range(3):
            try:
                self.camera.open()
                self.camera.verify_healthy(
                    min_widgets=self.config.MIN_CONFIG_TREE_SIZE
                )
                break
            except Exception as exc:
                log.warning("lens-change reconnect attempt %d failed: %s",
                            attempt + 1, exc)
                if attempt == 2:
                    if self.ntfy is not None:
                        self.ntfy.with_body(
                            Notification(
                                title="Camera reconnect failed",
                                body="",
                                priority="high",
                                tags=("warning",),
                            ),
                            f"3 attempts; last error: {exc}",
                        )
                    raise
        self.camera.apply_session_config(
            iso=self.config.ISO_FLOOR,
            shutter_candidates=self.config.SHUTTER_SPEED_CANDIDATES,
            image_quality=self.config.IMAGE_QUALITY,
            fec=0.0,
        )

    # ---- Helpers --------------------------------------------------------

    def _measure(self, jpg_path: Path) -> float:
        if self.measure_path is not None:
            return self.measure_path(jpg_path)
        if self.crop_box is None:
            from PIL import Image
            with Image.open(str(jpg_path)) as im:
                w, h = im.size
            self.crop_box = crop_box(
                w, h,
                self.config.CROP_W_FRAC, self.config.CROP_H_FRAC,
                self.config.CROP_X_FRAC, self.config.CROP_Y_FRAC,
            )
        stats = measure_crop(jpg_path, self.crop_box)
        return stats.p50

    def _sweep_destination(
        self, lens_dir: Path, aperture: str, step_index: int
    ) -> Path:
        slug = self._aperture_subdir(aperture)
        offset = getattr(self.config, "STEP_NUMBER_OFFSET", 100)
        width = getattr(self.config, "STEP_NUMBER_WIDTH", 4)
        return lens_dir / slug / step_filename(
            step_index, offset=offset, width=width
        )

    @staticmethod
    def _aperture_subdir(aperture: str) -> str:
        return aperture.replace("/", "_").replace(".", "_")

    def _record_sharpness(
        self, *, step_index: int, aperture: str, dest: Path
    ) -> None:
        """Hand a captured JPEG to the sharpness collector. Defensive —
        cropping errors are surfaced at the next ``wait_for_pending``
        call, never inline."""
        if self._sharpness_collector is None:
            return
        try:
            self._sharpness_collector.record_shot(
                rail_step=step_index,
                aperture=_parse_fnum(aperture),
                jpg_path=dest,
            )
        except RuntimeError:
            # Collector already shut down (e.g. abort path). The
            # session is winding down; skip silently.
            log.debug("sharpness record_shot rejected (collector down)")

    def _recycle_for(self, aperture: str) -> float:
        ap_float = _parse_fnum(aperture)
        table = self.config.RANGE_EXTENSION
        if ap_float in table:
            return float(table[ap_float][1])
        keys = sorted(table.keys())
        candidate = keys[0]
        for k in keys:
            if k <= ap_float:
                candidate = k
        return float(table[candidate][1])

    def _log_shot(
        self,
        *,
        slug: str,
        step_index: int,
        position_mm: float,
        aperture: str,
        cal: ApertureCalibration,
        median: float,
        ev: float,
        filename: str,
        phase: Phase,
    ) -> None:
        if self.shots_csv is None:
            return
        self.shots_csv.append(
            ShotRow(
                timestamp=self.now().isoformat(),
                session_id=self.session_id,
                lens_config_slug=slug,
                rail_step=step_index,
                rail_position_mm=position_mm,
                aperture=aperture,
                iso=cal.iso,
                shutter_speed="",
                fec=cal.fec_ev,
                median_central_luminance=median,
                median_ev_relative_to_baseline=ev,
                filename=filename,
                phase=phase,
                notes="",
            )
        )

    def _make_session_id(self) -> str:
        return self.now().strftime("%Y-%m-%d_%H%M%S")

    def _write_metadata(self) -> None:
        if self.metadata is None:
            return
        write_json(self.session_dir / "session.json", self.metadata)


# ---- Production main() ----------------------------------------------------


def main() -> None:
    """Bootstrap a real session: wires the production hardware drivers
    to the runner, with the sharpness-backed RangeChecker.

    All injectable collaborators come together here so tests can stay
    on lightweight fakes without ever touching this function.
    """
    import config
    from camera.nikon_z7 import NikonZ7
    from notify.ntfy import NtfyClient
    from sharpness.collector import LiveCollector
    from sharpness.range_check import SharpnessRangeChecker
    from stackshot.controller import open_ftdi as open_stackshot
    from stackshot.jog import jog_start_and_end as _jog

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # libusb-direct via pyusb (FtdiTransport). Bypasses ftdi_sio's
    # default DTR pulse; runs the Cognisys SET_BITMODE 0x20FF + 500ms
    # init recipe so the StackShot MCU enters USB-control mode.
    rail = open_stackshot(
        poll_interval_s=config.STACKSHOT_POLL_INTERVAL_S,
        move_timeout_s=config.STACKSHOT_MOVE_TIMEOUT_S,
    )
    camera = NikonZ7()
    ntfy = NtfyClient(config.NTFY_BASE_URL, config.NTFY_TOPIC)
    range_checker = SharpnessRangeChecker(config=config)

    def _sharpness_factory(lens_dir: Path) -> SharpnessCollectorLike:
        return LiveCollector(
            lens_dir / config.SHARPNESS_CROPPED_DIRNAME,
            max_workers=config.SHARPNESS_MAX_WORKERS,
        )

    def _camera_jog(rail) -> JogResult:
        """Drive the jog UI with camera release/reacquire wired into
        the SPACE / ENTER panel toggle. While the rail is on its
        front-panel buttons, close the gphoto2 session so the Z7's
        LCD shows live view; on ENTER, re-open and continue.
        """
        def _release_camera() -> None:
            log.info("releasing camera for live-view jog")
            camera.close()

        def _reacquire_camera() -> None:
            log.info("reacquiring camera after live-view jog")
            camera.open()
            camera.verify_healthy(min_widgets=config.MIN_CONFIG_TREE_SIZE)
            camera.apply_session_config(
                iso=config.ISO_FLOOR,
                shutter_candidates=config.SHUTTER_SPEED_CANDIDATES,
                image_quality=config.IMAGE_QUALITY,
                fec=0.0,
            )

        return JogResult(
            end_mm=_jog(
                rail,
                on_release=_release_camera,
                on_reacquire=_reacquire_camera,
            ).end_mm,
        )

    runner = SessionRunner(
        camera=camera,
        rail=rail,
        config=config,
        ntfy=ntfy,
        range_checker=range_checker,
        output_root=config.OUTPUT_ROOT,
        jog=_camera_jog,
        sharpness_collector_factory=_sharpness_factory,
    )
    runner.run()


if __name__ == "__main__":
    main()


