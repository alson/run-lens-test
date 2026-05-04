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

    # ---- internal state filled at start ----------------------------------
    session_id: str = ""
    session_dir: Path = field(default_factory=Path)
    shots_csv: ShotsCSV | None = None
    metadata: SessionMetadata | None = None
    baseline_median: float | None = None

    # ---- public entry points ---------------------------------------------

    def start(self) -> None:
        """Open hardware, verify, apply config, build session dir."""
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
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            started_at=self.now().isoformat(),
            output_root=str(self.output_root or ""),
            ntfy_topic=getattr(self.config, "NTFY_TOPIC", ""),
        )
        self._write_metadata()

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
        until the user enters an empty line."""
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
        except SessionAborted:
            self.stop(aborted=True)
            raise
        else:
            self.stop()

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
        self.rail.move_and_wait(Direction.BACKWARD, end_mm, backlash=True)

        apertures = aperture_sequence(
            self.camera.list_apertures(),
            whole_stop_scale=self.config.WHOLE_STOP_SCALE,
            max_fstop=self.config.MAX_FSTOP,
        )

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

        # Return rail to 0 for the next lens config (backlash on).
        if steps > 1:
            self.rail.move_and_wait(
                Direction.BACKWARD, step_size_mm * (steps - 1), backlash=True
            )

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
        for step_index in range(steps):
            if step_index > 0:
                self.rail.move_and_wait(
                    Direction.FORWARD, step_size_mm, backlash=False
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
        slug = aperture.replace("/", "_").replace(".", "_")
        return lens_dir / slug / f"step_{step_index:03d}.jpg"

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


