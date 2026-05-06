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
import math
import sys
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
from notify.sound import play_alert
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
    completed_slugs,
    write_json,
)
from stackshot.controller import StackShotController
from stackshot.protocol import Direction
from sharpness.planner import SamplePlan, plan_sweep
from ui.prompts import (
    OTHER,
    confirm,
    prompt_lens_config,
    prompt_lens_config_from_plan,
    prompt_magnification,
    prompt_off_target,
    prompt_range_inadequate,
    slugify_lens_config,
)
from ui.reshoot_plan import PlanEntry, load_plan


class _RejogRequested(Exception):
    """Internal signal: user picked re-jog at the all-off-target
    prompt. Caught by ``_run_one_lens_config_inner`` to redo the
    sweep with a new START / END inside the same lens-config run.
    Calibration data is preserved across re-jogs."""

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


# Aperture parsing lives in the planner (the planner's plan_sweep needs
# it for its widest-aperture float input). Re-export here so existing
# call sites in this module keep working unchanged.
from sharpness.planner import parse_aperture as _parse_fnum


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
    monotonic: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc)
    prompt_lens_config: Callable[[], str | None] = field(default_factory=lambda: lambda: prompt_lens_config())
    prompt_magnification: Callable[[], float] = field(
        default_factory=lambda: lambda: prompt_magnification()
    )
    prompt_lens_config_from_plan: Callable[[list, set[str]], object] = field(
        default_factory=lambda: lambda entries, completed: prompt_lens_config_from_plan(
            entries, completed
        )
    )
    plan_sweep: Callable[..., SamplePlan] = field(default_factory=lambda: plan_sweep)
    confirm: Callable[[str], bool] = field(default_factory=lambda: confirm)
    prompt_range_inadequate: Callable[[str], str] = field(
        default_factory=lambda: prompt_range_inadequate
    )
    prompt_off_target: Callable[[str], str] = field(
        default_factory=lambda: prompt_off_target
    )
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
    _subsample_factors: dict[str, int] = field(default_factory=dict)
    """Per-aperture skip stride for the current lens config. Built
    from ``SamplePlan.subsample_factors`` once per lens config and
    cleared on exit. Aperture A is captured at step S iff
    ``S % factor[A] == 0``; missing apertures default to factor 1
    (no skip). Skipped captures are silent — no shots.csv row, no
    droop push, no replicas."""
    _last_capture_started_at: float | None = None
    """monotonic() reading taken just before each ``camera.capture_to``.

    Used by :meth:`_wait_for_recycle` to subtract whatever wall-clock
    time has already passed (set_config, capture, sharpness queueing,
    measurement) from the configured per-aperture flash recycle. The
    user is shooting thousands of frames; sleeping ``RECYCLE_S`` on
    top of work that already exceeds it adds hours."""
    _last_capture_aperture: str | None = None
    """Aperture of the previous capture. The flash recycle is keyed on
    the *prior* shot's aperture (small apertures = more flash power
    discharged = longer recycle), not the upcoming one."""
    _plan_entries: list[PlanEntry] | None = None
    """Curated reshoot/new-shoot presets parsed from
    ``config.RESHOOT_PLAN_PATH`` at session start. ``None`` when the
    plan file is missing or disabled — the runner then falls through
    to the free-text prompt only."""
    _completed_slugs: set[str] = field(default_factory=set)
    """Slugs already successfully acquired in past sessions, plus any
    completed in the current session. Drives the ``[done]`` marker in
    the menu."""

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

        # Load the curated lens-config menu and the set of slugs that
        # past sessions have already acquired successfully. Both are
        # best-effort: a missing reshoot plan disables the menu (the
        # runner still works via the free-text prompt), and any
        # session.json that fails to parse is silently skipped.
        plan_path = getattr(self.config, "RESHOOT_PLAN_PATH", None)
        if plan_path is not None:
            try:
                self._plan_entries = load_plan(plan_path)
            except FileNotFoundError:
                self._plan_entries = None
        if self.output_root is not None:
            self._completed_slugs = completed_slugs(self.output_root)

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

        # Calibration baseline. If the user pointed config at a
        # reference JPEG we measure its central crop median and use
        # that as an absolute target — every lens config in every
        # session calibrates to the same number, so two sessions
        # with different first-lens-configs are still comparable.
        # Without a reference, fall back to first-shot-at-widest-
        # aperture-of-first-lens-config (the calibrator handles the
        # baseline_median=None path).
        ref_image = getattr(self.config, "BASELINE_REFERENCE_IMAGE", None)
        if ref_image is not None:
            from exposure.analyse import crop_box, measure_crop
            from PIL import Image
            with Image.open(str(ref_image)) as im:
                w, h = im.size
            box = crop_box(
                w, h,
                self.config.CROP_W_FRAC, self.config.CROP_H_FRAC,
                self.config.CROP_X_FRAC, self.config.CROP_Y_FRAC,
            )
            stats = measure_crop(ref_image, box)
            self.baseline_median = stats.p50
            log.info(
                "baseline median %.1f from reference image %s",
                self.baseline_median, ref_image,
            )
            if self.metadata is not None:
                self.metadata.baseline_median_luminance = self.baseline_median
                self.metadata.notes = (
                    self.metadata.notes
                    + f"\nbaseline from reference: {ref_image}"
                ).strip()
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
        until the user enters an empty line.

        Ctrl-C and :class:`SessionAborted` both stop cleanly: the
        StackShot is told to STOP (cancels any in-flight motion), the
        camera is closed, and ``session.json`` is finalised with
        ``aborted=True``.
        """
        self.start()
        try:
            while True:
                if self._plan_entries:
                    picked = self.prompt_lens_config_from_plan(
                        self._plan_entries, self._completed_slugs
                    )
                    if picked is None:
                        return
                    if picked is OTHER or picked == OTHER:
                        lens_text = self.prompt_lens_config()
                        if lens_text is None:
                            return
                        magnification: float | None = None
                    else:
                        lens_text = picked.lens_config
                        magnification = picked.magnification
                else:
                    lens_text = self.prompt_lens_config()
                    if lens_text is None:
                        return
                    magnification = None
                slug = slugify_lens_config(lens_text)
                # _run_one_lens_config records the slug into
                # metadata itself (right before the lens-change
                # prompt) so that aborting at the prompt still
                # captures the completed work.
                self._run_one_lens_config(slug, magnification=magnification)
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

    def _run_one_lens_config(
        self, slug: str, *, magnification: float | None = None
    ) -> None:
        if self.jog is None:
            raise RuntimeError("SessionRunner.jog not configured")
        lens_dir = self.session_dir / slug
        lens_dir.mkdir(parents=True, exist_ok=True)

        # Magnification is asked once per lens config; the sweep
        # plan is a function of M, the lens's widest aperture, and
        # the jogged range. When the caller picked a preset from the
        # reshoot-plan menu, the value is supplied directly; otherwise
        # we prompt as before.
        if magnification is None:
            magnification = self.prompt_magnification()

        apertures = aperture_sequence(
            self.camera.list_apertures(),
            whole_stop_scale=self.config.WHOLE_STOP_SCALE,
            max_fstop=self.config.MAX_FSTOP,
        )
        if not apertures:
            raise RuntimeError("camera reports no apertures; aborting")
        widest_aperture_str = apertures[0]
        widest_aperture = _parse_fnum(widest_aperture_str)

        # Jog → planner → check refuse loop. The user can re-jog as
        # many times as they want until the range yields the
        # configured minimum rail-position count.
        end_mm, plan = self._jog_and_plan(
            magnification=magnification,
            widest_aperture=widest_aperture,
            widest_aperture_str=widest_aperture_str,
            apertures=apertures,
        )
        steps = plan.rail_positions
        step_size_mm = math.copysign(plan.rail_step_mm, end_mm)

        # Move rail back to position 0 with backlash compensation.
        # end_mm and step_size_mm may be negative (user jogged
        # backward to END) — use the sign-aware helper so the rail
        # gets a non-negative distance with the correct direction.
        self._return_rail_to_zero(end_mm)

        # One sharpness collector per lens config. Crops land under
        # <lens>/<aperture>/cropped/<crop-name>/<source-filename>.jpg
        # — see config.SHARPNESS_CROPPED_DIRNAME and §8.3.5 of the
        # plan. The collector lives for the entire lens config so
        # probe / extension shots feed the same dataset.
        self._sharpness_collector = self.sharpness_collector_factory(lens_dir)
        self._subsample_factors = self._build_subsample_lookup(apertures, plan)
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
                magnification=magnification,
                plan=plan,
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
            self._subsample_factors = {}

    def _build_subsample_lookup(
        self, apertures: list[str], plan: SamplePlan,
    ) -> dict[str, int]:
        """Build a per-aperture skip-stride lookup from a SamplePlan.

        Walks the camera's aperture strings (``"f/4.5"`` etc.),
        parses each to a float, and reads the integer factor from
        ``plan.subsample_factors`` (which is keyed by float
        aperture). Apertures the planner doesn't list — typically
        ⅓-stops not in ``KNOWN_APERTURE_STOPS`` — default to 1
        (no skip), the safe choice.
        """
        out: dict[str, int] = {}
        for ap_str in apertures:
            try:
                ap_float = _parse_fnum(ap_str)
            except ValueError:
                out[ap_str] = 1
                continue
            out[ap_str] = max(1, int(plan.subsample_factors.get(ap_float, 1)))
        return out

    def _should_capture(self, aperture: str, step_index: int) -> bool:
        """``True`` when this (step, aperture) is on the aperture's
        sparse grid. Skipped captures are silent — no shots.csv row,
        no droop push, no replicas."""
        factor = self._subsample_factors.get(aperture, 1)
        return step_index % factor == 0

    def _jog_and_plan(
        self,
        *,
        magnification: float,
        widest_aperture: float,
        widest_aperture_str: str,
        apertures: list[str] | None = None,
    ) -> tuple[float, SamplePlan]:
        """Run the jog UI, build a sweep plan, and refuse-prompt-rejog
        if the resulting position count is below
        ``config.MIN_RAIL_POSITIONS``.

        Returns ``(end_mm, plan)``. ``end_mm`` is signed (negative
        if the user jogged backward); ``plan.rail_positions`` is
        guaranteed ≥ MIN_RAIL_POSITIONS.
        """
        min_positions = int(getattr(self.config, "MIN_RAIL_POSITIONS", 10))
        while True:
            jog_result = self.jog(self.rail)  # type: ignore[misc]
            end_mm = jog_result.end_mm
            plan = self.plan_sweep(
                magnification=magnification,
                widest_aperture=widest_aperture,
                max_aperture=self.config.MAX_FSTOP,
                rail_range_mm=abs(end_mm),
                replicas=int(getattr(self.config, "SHOTS_PER_STEP_APERTURE", 1)),
            )
            positions = plan.rail_positions or 0
            if positions >= min_positions:
                self._announce(
                    f"Plan: M={magnification:g}, widest={widest_aperture_str}, "
                    f"rail_step={plan.rail_step_mm:.3f} mm, "
                    f"positions={positions}, "
                    f"replicas={plan.replicas_per_position}"
                )
                if apertures:
                    self._announce_subsample_breakdown(
                        apertures=apertures,
                        plan=plan,
                        positions=positions,
                    )
                return end_mm, plan
            ok = self.confirm(
                f"\nJogged range ({abs(end_mm):.2f} mm) yields only "
                f"{positions} rail positions at M={magnification:g}, "
                f"widest={widest_aperture_str} (planner step="
                f"{plan.rail_step_mm:.3f} mm; need ≥ {min_positions} "
                "positions for reliable peak detection).\n"
                "[ENTER] re-jog START / END, [q] abort: "
            )
            if not ok:
                raise SessionAborted("user aborted at tiny-range prompt")
            # Loop: re-jog, recompute plan.

    def _run_one_lens_config_inner(
        self,
        *,
        slug: str,
        lens_dir: Path,
        end_mm: float,
        steps: int,
        step_size_mm: float,
        apertures: list[str],
        magnification: float,
        plan: SamplePlan,
    ) -> None:

        # Calibration at rail step 0.
        # User-facing status so the terminal isn't silent during the
        # ~30-60s of FEC closed-loop test shots that happen before
        # the first sweep frame. Without this, the next thing the
        # user sees after pressing ENTER for step count is the
        # progress bar — minutes later — and they can't tell whether
        # the rig is working or hung.
        self._announce(
            f"Calibrating {slug}: {len(apertures)} apertures "
            f"({', '.join(apertures)})..."
        )
        calibrator = Calibrator(
            camera=self.camera,
            measure=self._measure,
            config=self.config,
            shots_csv=self.shots_csv,
            now=self.now,
        )
        # Calibration retry loop. If any aperture fails to reach
        # tolerance, the user can adjust the rig and re-run
        # calibration WITHOUT re-jogging — different from the
        # off-target re-jog flow which redoes the whole sweep.
        while True:
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
            self._announce_calibration(cal_result)
            non_converged = [
                ap for ap, c in cal_result.per_aperture.items()
                if not c.hit_tolerance
            ]
            if not non_converged:
                break
            choice = self._handle_calibration_failure(
                slug=slug, cal_result=cal_result, non_converged=non_converged,
            )
            if choice == "skip":
                break
            # choice == "rerun": loop and re-run calibration.
        write_json(
            lens_dir / "calibration.json",
            {
                "baseline_median": cal_result.baseline_median,
                "per_aperture": cal_result.per_aperture,
            },
        )

        # Sweep + range-check loop. The off-target prompt may ask
        # the user to re-jog and re-sweep this lens config; that path
        # raises ``_RejogRequested`` to bring us back here with a
        # fresh range, fresh collector, and the rail back at step 0.
        # Calibration is preserved across re-jogs.
        while True:
            self._announce(
                f"Sweeping {slug}: {steps} steps × {len(apertures)} apertures..."
            )
            self._run_sweep(
                lens_dir=lens_dir,
                slug=slug,
                steps=steps,
                step_size_mm=step_size_mm,
                calibration=cal_result,
                apertures=apertures,
            )

            forward_steps_done = steps - 1
            if self.range_checker is not None:
                try:
                    forward_steps_done = self._run_range_check_and_extend(
                        lens_dir=lens_dir,
                        slug=slug,
                        end_mm=end_mm,
                        step_size_mm=step_size_mm,
                        steps=steps,
                        calibration=cal_result,
                        apertures=apertures,
                    )
                except _RejogRequested:
                    # User wants new START / END; rewind, recreate
                    # the collector, rerun the sweep loop.
                    widest_aperture_str = apertures[0]
                    end_mm, steps, step_size_mm, plan = self._rejog_for_lens_config(
                        lens_dir=lens_dir,
                        current_position_mm=(steps - 1) * step_size_mm,
                        slug=slug,
                        magnification=magnification,
                        widest_aperture=_parse_fnum(widest_aperture_str),
                        widest_aperture_str=widest_aperture_str,
                        apertures=apertures,
                    )
                    # Rebuild subsample factor lookup from the new plan.
                    self._subsample_factors = self._build_subsample_lookup(
                        apertures, plan
                    )
                    continue
            break

        # Return rail to 0 for the next lens config (backlash on; we
        # may have moved into negative territory if the extension was
        # on the 'near' side).
        if forward_steps_done != 0:
            self._return_rail_to_zero(forward_steps_done * step_size_mm)

        # Record this lens config in session metadata *before* the
        # lens-change prompt — if the user types `q` to abort the
        # session there, the work we just did still needs to show up
        # in session.json. Each entry carries the magnification, the
        # widest aperture, and the full SamplePlan (incl. the
        # subsample_factors that downstream tooling will eventually
        # honour at analysis time).
        from dataclasses import asdict as _asdict
        if self.metadata is not None and not any(
            lc.get("slug") == slug for lc in self.metadata.lens_configs
        ):
            self.metadata.lens_configs.append({
                "slug": slug,
                "magnification": float(magnification),
                "widest_aperture": float(plan.widest_aperture),
                "sample_plan": _asdict(plan),
            })
            self._write_metadata()
            self._completed_slugs.add(slug)

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
                step_count=steps,
                position_mm=position_mm,
                step_size_mm=step_size_mm,
                apertures=apertures,
                calibration=calibration,
                droop=droop,
            )
        # Newline so the next print() output doesn't overwrite the
        # final progress line.
        self._progress_done()

    def _run_one_step(
        self,
        *,
        lens_dir: Path,
        slug: str,
        step_index: int,
        step_count: int,
        position_mm: float,
        step_size_mm: float,
        apertures: list[str],
        calibration: CalibrationResult,
        droop: DroopDetector,
    ) -> None:
        aborted = False
        for ap_index, aperture in enumerate(apertures):
            if aborted:
                self._log_aborted_remainder(
                    lens_dir=lens_dir,
                    slug=slug,
                    step_index=step_index,
                    position_mm=position_mm,
                    aperture=aperture,
                )
                continue
            if not self._should_capture(aperture, step_index):
                # Off this aperture's subsample grid — silent skip.
                continue
            cal = calibration.per_aperture[aperture]
            self._progress(
                step_index=step_index,
                step_count=step_count,
                position_mm=position_mm,
                aperture=aperture,
                aperture_index=ap_index,
                aperture_count=len(apertures),
            )
            ev, event = self._capture_one(
                lens_dir=lens_dir,
                slug=slug,
                step_index=step_index,
                position_mm=position_mm,
                aperture=aperture,
                cal=cal,
                phase=Phase.SWEEP,
                droop=droop,
            )
            if event is not None:
                current_step_reshot = self._handle_droop(
                    event,
                    droop,
                    lens_dir=lens_dir,
                    slug=slug,
                    current_step_index=step_index,
                    current_position_mm=position_mm,
                    step_size_mm=step_size_mm,
                    apertures=apertures,
                    calibration=calibration,
                )
                if current_step_reshot:
                    # Reshoot already captured every aperture of
                    # the current step at fresh power; nothing more
                    # to do here. Skip the aborted-remainder rows.
                    return
                aborted = True
                continue

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
        # Off-target before any extension flow: the sharpness curves
        # are flat at every aperture, so extending the range won't
        # help — the user's START / END is on a featureless area or
        # entirely out of the lens's depth-of-field. Alert and
        # offer re-jog.
        if getattr(result, "all_off_target", False):
            choice = self._handle_off_target(slug=slug, result=result)
            if choice == "rejog":
                # Persist what we know before re-jogging so the
                # off-target sweep is at least documented on disk.
                write_json(lens_dir / "range_check.json", {
                    "ok": False,
                    "peak_side": result.peak_side,
                    "extension_applied": False,
                    "all_off_target": True,
                    "rejog_requested": True,
                    "notes": result.notes,
                })
                raise _RejogRequested()
            # choice == "skip": fall through to the normal "no
            # extension needed" early-return below, but mark the
            # off-target state in the persisted json so downstream
            # readers don't mistake this for a real OK.
            write_json(lens_dir / "range_check.json", {
                "ok": False,
                "peak_side": result.peak_side,
                "extension_applied": False,
                "all_off_target": True,
                "rejog_requested": False,
                "notes": (
                    "all apertures off-target; user accepted the "
                    "flat data and continued. " + result.notes
                ),
            })
            return steps - 1
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

        # Track total grid slots shot per side across probes +
        # any extend-more rounds. The "fill rest" branch below
        # uses this to know how far to continue from.
        shot_so_far_per_side: dict[str, int] = {"far": 0, "near": 0}
        for side, count in sides:
            shot_so_far_per_side[side] = count
        extension_step_count_per_side: dict[str, int] = {"far": 0, "near": 0}
        extend_rounds = 0

        # Extend-more loop: as long as the data still doesn't bracket
        # the peak, give the user the choice to extend further by one
        # more standard increment, accept what we have, or abort.
        while not result2.peak_in_range_after_probe:
            if self.ntfy is not None and extend_rounds == 0:
                self.ntfy.with_body(
                    Notification(
                        title="Range still inadequate",
                        body="",
                        priority="high",
                        tags=("warning",),
                    ),
                    f"{slug}: probe on {result.peak_side} side did not "
                    "bracket peak.",
                )
            extend_mm_per_side = extension_steps * abs(step_size_mm)
            choice = self.prompt_range_inadequate(
                f"\nRange still inadequate on {slug} "
                f"(peak_side={result2.peak_side or result.peak_side}). "
                f"[ENTER] extend by ~{extend_mm_per_side:.1f} mm "
                f"({extension_steps} steps) on the failing side, "
                "[s] skip and continue to lens change, [q] abort: "
            )
            if choice == "abort":
                write_json(lens_dir / "range_check.json", {
                    "ok": False,
                    "extension_applied": extend_rounds > 0,
                    **record_base,
                    "extension_step_count": extension_step_count_per_side,
                    "extend_rounds": extend_rounds,
                    "notes": "user aborted at range-inadequate prompt",
                })
                raise SessionAborted("user aborted at range-inadequate prompt")
            if choice == "skip":
                write_json(lens_dir / "range_check.json", {
                    "ok": False,
                    "extension_applied": extend_rounds > 0,
                    **record_base,
                    "extension_step_count": extension_step_count_per_side,
                    "extend_rounds": extend_rounds,
                    "notes": (
                        "probe did not bracket peak; user accepted "
                        "incomplete data"
                    ),
                })
                return self._steps_from_zero_at(current_pos, step_size_mm)
            # choice == "extend": shoot extension_steps more grid
            # slots on each originally-failing side.
            for side, _ in sides:
                for i in range(extension_steps):
                    offset = shot_so_far_per_side[side] + i
                    step_index = self._extension_step_index(
                        side=side, original_steps=steps,
                        offset_in_extension=offset,
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
                shot_so_far_per_side[side] += extension_steps
            extend_rounds += 1
            if self._sharpness_collector is not None:
                self._sharpness_collector.wait_for_pending()
            result2 = self.range_checker.evaluate(
                lens_dir=lens_dir,
                end_mm=end_mm,
                step_size_mm=step_size_mm,
                apertures=apertures,
                probe_data=probe_data,
            )

        # Probe (and any extend-more rounds) brackets the peak — fill
        # in the remaining grid slots inside the original extension on
        # each requested side, so the data is dense up to ``extension_steps``.
        for side, _ in sides:
            for i in range(shot_so_far_per_side[side], extension_steps):
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
                shot_so_far_per_side[side] += 1

        # `ok` here reflects the *original* first-call verdict — the
        # data needed extension. The fact that we extended (and now
        # bracket the peak) is captured by `extension_applied=True`.
        write_json(lens_dir / "range_check.json", {
            "ok": False,
            "extension_applied": True,
            **record_base,
            "extension_step_count": extension_step_count_per_side,
            "extend_rounds": extend_rounds,
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
            if not self._should_capture(aperture, step_index):
                # Off this aperture's subsample grid — silent skip.
                continue
            cal = calibration.per_aperture[aperture]
            self._set_capture_settings(aperture, cal)
            self._wait_for_recycle()
            dest = file_factory(aperture)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._last_capture_started_at = self.monotonic()
            self._last_capture_aperture = aperture
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

    def _capture_one(
        self,
        *,
        lens_dir: Path,
        slug: str,
        step_index: int,
        position_mm: float,
        aperture: str,
        cal: ApertureCalibration,
        phase: Phase,
        droop: DroopDetector | None = None,
    ) -> tuple[float, DroopEvent | None]:
        """Capture all configured replicas at one (step, aperture).

        Returns ``(median_ev, droop_event)``:
        * ``median_ev`` is the median EV delta across the replicas
          actually captured (one or more — possibly fewer than N if
          we short-circuited on a droop trip).
        * ``droop_event`` is non-``None`` when ``droop`` was passed
          and a replica's push tripped a criterion. The caller
          should hand it to :meth:`_handle_droop` and stop further
          captures at this (step, aperture).

        Pushing per-replica matters: a single severely under-exposed
        replica trips the severe criterion immediately (1 of last 10
        ≤ -2 EV). Without per-replica pushes the runner would take
        all N replicas, compute the median, and only then push —
        wasting two more shots after the failure was already
        evident.

        ``SHOTS_PER_STEP_APERTURE`` controls the replica count
        (default 1 for legacy behaviour).
        """
        n = max(1, int(getattr(self.config, "SHOTS_PER_STEP_APERTURE", 1)))
        self._set_capture_settings(aperture, cal)
        # The droop detector wants EV relative to the *per-aperture
        # calibration target* (cal.final_median), not relative to the
        # session-wide reference baseline. Calibration may have
        # converged a fraction of an EV under the reference; treating
        # that gap as droop signal would risk false-positive trips on
        # a perfectly stable setup. shots.csv keeps EV-vs-reference-
        # baseline as the canonical log value (good for inter-session
        # comparability); EV-vs-calibration is computed locally just
        # for the droop push.
        cal_reference = cal.final_median or self.baseline_median or 1.0
        droop_evs: list[float] = []
        droop_event: DroopEvent | None = None
        for replica in range(n):
            self._wait_for_recycle()
            dest = self._sweep_destination(
                lens_dir, aperture, step_index, replica=replica
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._last_capture_started_at = self.monotonic()
            self._last_capture_aperture = aperture
            self.camera.capture_to(dest)
            self._record_sharpness(
                step_index=step_index,
                aperture=aperture,
                dest=dest,
                replica_index=replica,
            )
            median = self._measure(dest)
            ev_baseline = ev_delta(median, self.baseline_median or median)
            ev_to_cal = ev_delta(median, cal_reference)
            self._log_shot(
                slug=slug,
                step_index=step_index,
                position_mm=position_mm,
                aperture=aperture,
                cal=cal,
                median=median,
                ev=ev_baseline,
                filename=str(dest.relative_to(lens_dir)),
                phase=phase,
            )
            droop_evs.append(ev_to_cal)
            if droop is not None:
                droop_event = droop.push(
                    aperture, ev_to_cal, step_index=step_index
                )
                if droop_event is not None:
                    break  # short-circuit: stop taking more replicas
        droop_evs.sort()
        median_ev = droop_evs[len(droop_evs) // 2]
        return median_ev, droop_event

    def _handle_droop(
        self,
        event: DroopEvent,
        droop: DroopDetector,
        *,
        lens_dir: Path,
        slug: str,
        current_step_index: int,
        current_position_mm: float,
        step_size_mm: float,
        apertures: list[str],
        calibration: CalibrationResult,
    ) -> bool:
        """Returns ``True`` if the reshoot covered the current step
        (every aperture, fresh frame); the caller can then skip
        emitting redundant aborted-remainder rows for the rest of
        the inner aperture loop. Returns ``False`` when no rewind was
        needed (droop fired on the first sample at the current step)
        — caller continues with ``aborted=True`` per the existing
        skip-the-rest-of-this-step semantics."""
        if self.ntfy is not None:
            self.ntfy.with_body(
                BATTERY_DROOP,
                f"{event.aperture}: {event.triggered_count} of last "
                f"{len(event.window_snapshot)} shots ≤ {event.threshold:+.1f} EV. "
                f"Window: {event.window_snapshot!r}",
            )
        # Local audible alert in addition to ntfy: the user is at
        # the rig with their hands on the rail when this fires;
        # waiting for a phone notification to wake them is too slow.
        # Best-effort — paplay or alert.wav missing must not block.
        wav_path = getattr(self.config, "ALERT_WAV", None)
        if wav_path is not None:
            try:
                play_alert(wav_path)
            except Exception:                            # noqa: BLE001
                log.exception("alert sound failed")
        # Pause: user swaps batteries, presses ENTER.
        ok = self.confirm(
            f"\nBattery droop on {event.aperture}. "
            "Swap flash batteries, wait for steady ready light, "
            "then press ENTER to resume. q = abort: "
        )
        if not ok:
            raise SessionAborted("user aborted at droop pause")
        # Find the earliest step (across all apertures) whose EV ≤
        # the firing trip threshold. That's the first under-exposed
        # shot we trust the user wants to overwrite with a fresh
        # capture. ``window_snapshot`` is the firing aperture's
        # window only; ``earliest_underexposed_step`` consults all.
        restart_step = droop.earliest_underexposed_step(event.threshold)
        reshot_current_step = (
            restart_step is not None and restart_step < current_step_index
        )
        if reshot_current_step:
            # Reset the droop window before recursing so the nested
            # reshoot starts from a clean slate; if its captures are
            # still under-exposed the new samples will trip droop
            # again and recurse further.
            droop.reset_all()
            self._reshoot_steps(
                lens_dir=lens_dir,
                slug=slug,
                from_step=restart_step,
                through_step=current_step_index,
                from_position_mm=current_position_mm,
                step_size_mm=step_size_mm,
                apertures=apertures,
                calibration=calibration,
                droop=droop,
            )
        droop.reset_all()
        return reshot_current_step

    def _handle_calibration_failure(
        self,
        *,
        slug: str,
        cal_result: CalibrationResult,
        non_converged: list[str],
    ) -> str:
        """Alert + 3-way prompt when calibration didn't reach
        tolerance on one or more apertures. Returns ``"rerun"`` to
        run calibration again (caller loops; user has presumably
        adjusted the rig in between) or ``"skip"`` to accept the
        non-converged values and proceed. Raises
        :class:`SessionAborted` on ``"abort"``.

        Distinct from droop: the fix is rig adjustment (lighting
        distance, commander group FEC, reference brightness), not
        battery swaps.
        """
        lines = []
        for ap in non_converged:
            c = cal_result.per_aperture[ap]
            saturated = " (lighting saturated)" if c.saturated_at_ceiling else ""
            lines.append(
                f"  {ap}: ev={c.final_ev_delta:+.2f}  "
                f"FEC={c.fec_ev:+.2f}  EC={c.ec_ev:+.2f}  ISO={c.iso}"
                f"{saturated}"
            )
        body = (
            f"{slug}: calibration didn't reach tolerance on "
            f"{len(non_converged)}/{len(cal_result.per_aperture)} apertures.\n"
            + "\n".join(lines)
        )
        if self.ntfy is not None:
            self.ntfy.with_body(
                Notification(
                    title="Calibration didn't converge",
                    body="",
                    priority="high",
                    tags=("warning",),
                ),
                body,
            )
        wav_path = getattr(self.config, "ALERT_WAV", None)
        if wav_path is not None:
            try:
                play_alert(wav_path)
            except Exception:                            # noqa: BLE001
                log.exception("alert sound failed")
        choice = self.prompt_off_target(
            f"\nCalibration didn't reach tolerance on "
            f"{', '.join(non_converged)}.\n"
            + "\n".join(lines)
            + "\n\nThis is a *lighting* / *reference* issue, not a "
            "battery droop. Adjust the rig (commander group FEC, "
            "flash distance, or pick a dimmer "
            "BASELINE_REFERENCE_IMAGE) before continuing. "
            "[ENTER] re-run calibration after adjusting, "
            "[s] proceed anyway with current values, "
            "[q] abort: "
        )
        if choice == "abort":
            raise SessionAborted("user aborted at calibration-failure prompt")
        # prompt_off_target returns "rejog" / "skip" / "abort"; map
        # "rejog" → "rerun" since calibration failure recalibrates
        # without redoing the jog.
        return "rerun" if choice == "rejog" else "skip"

    def _handle_off_target(self, *, slug: str, result) -> str:
        """Notify + prompt when every aperture's curve is flat.

        Returns ``"rejog"`` (default action — caller raises
        ``_RejogRequested``), ``"skip"`` (caller falls through to
        write the persisted json with off-target marker and
        continue), or raises :class:`SessionAborted` on ``"abort"``.
        """
        if self.ntfy is not None:
            self.ntfy.with_body(
                Notification(
                    title="Sweep off-target",
                    body="",
                    priority="high",
                    tags=("warning",),
                ),
                f"{slug}: every aperture's sharpness curve is flat. "
                "Likely the rail span doesn't cover the lens's depth "
                "of field, or the test target lacks micro-detail at "
                "the chosen position. Re-jog recommended.",
            )
        wav_path = getattr(self.config, "ALERT_WAV", None)
        if wav_path is not None:
            try:
                play_alert(wav_path)
            except Exception:                            # noqa: BLE001
                log.exception("alert sound failed")
        choice = self.prompt_off_target(
            f"\nAll apertures off-target on {slug} — every sharpness "
            "curve is flat. Either the rail span misses the focus "
            "zone or the target lacks micro-detail. "
            "[ENTER] re-jog START / END and re-sweep, "
            "[s] skip and continue with this data, "
            "[q] abort: "
        )
        if choice == "abort":
            raise SessionAborted("user aborted at off-target prompt")
        return choice  # "rejog" or "skip"

    def _rejog_for_lens_config(
        self,
        *,
        lens_dir: Path,
        current_position_mm: float,
        slug: str | None = None,
        magnification: float = 1.0,
        widest_aperture: float = 4.0,
        widest_aperture_str: str = "f/4",
        apertures: list[str] | None = None,
    ) -> tuple[float, int, float, SamplePlan]:
        """Run the jog UI again, return rail to step 0, and rebuild
        the per-lens-config sharpness collector so the next sweep's
        dataset is fresh.

        Returns ``(end_mm, steps, step_size_mm, plan)``. The new
        plan includes any updated ``rail_positions`` for the
        re-jogged range.
        """
        if self.jog is None:
            raise RuntimeError("SessionRunner.jog not configured")
        end_mm, plan = self._jog_and_plan(
            magnification=magnification,
            widest_aperture=widest_aperture,
            widest_aperture_str=widest_aperture_str,
            apertures=apertures,
        )
        steps = plan.rail_positions or 0
        step_size_mm = math.copysign(plan.rail_step_mm, end_mm)
        # Rail is wherever the jog left it; bring it back to step 0
        # via the sign-aware return helper.
        self._return_rail_to_zero(self.rail.position_mm)
        # Replace the sharpness collector so the new sweep doesn't
        # mix samples with the discarded off-target ones.
        try:
            if self._sharpness_collector is not None:
                self._sharpness_collector.shutdown(wait=True)
        except Exception:                                # noqa: BLE001
            log.exception("sharpness collector shutdown during re-jog failed")
        self._sharpness_collector = self.sharpness_collector_factory(lens_dir)
        if self.range_checker is not None:
            self.range_checker.set_collector(self._sharpness_collector)
        # Drop the discarded sweep's shots.csv rows for this slug —
        # JPEGs at those step indices are about to be overwritten by
        # the new sweep, and any phase=sweep / probe / extension /
        # reshoot / aborted entries are now stale. Calibration rows
        # are preserved (no re-calibration on re-jog).
        if slug is not None and self.shots_csv is not None:
            stale_phases = {
                Phase.SWEEP.value,
                Phase.PROBE.value,
                Phase.EXTENSION.value,
                Phase.RESHOOT.value,
                Phase.ABORTED.value,
            }
            self.shots_csv.drop_rows(
                lambda r: (
                    r.get("lens_config_slug") == slug
                    and r.get("phase") in stale_phases
                )
            )
        return end_mm, steps, step_size_mm, plan

    def _reshoot_steps(
        self,
        *,
        lens_dir: Path,
        slug: str,
        from_step: int,
        through_step: int,
        from_position_mm: float,
        step_size_mm: float,
        apertures: list[str],
        calibration: CalibrationResult,
        droop: DroopDetector,
    ) -> None:
        """Move the rail back and overwrite the captured shots in
        ``[from_step, through_step]`` (inclusive) with fresh frames.

        The original under-exposed JPEGs are overwritten in place
        (filenames are deterministic from step_index). New
        ``shots.csv`` rows are appended with ``phase=reshoot`` so
        downstream analysis can prefer them.

        EVs collected during the reshoot **are** pushed into ``droop``
        — a still-weak flash only reveals itself by firing again,
        and that's only visible during a capture. If droop trips
        again mid-reshoot, ``_handle_droop`` recurses into another
        :meth:`_reshoot_steps` call covering ``[new_earliest,
        current_step_index]``; on return this loop continues from
        ``current_step_index + 1``.
        """
        # Step numbering in user-facing messages matches the
        # progress bar's 1-based count.
        self._announce(
            f"Reshooting steps {from_step + 1}..{through_step + 1} after droop "
            "(overwriting under-exposed frames)..."
        )
        # Drop the now-stale shots.csv rows up front: the original
        # phase=sweep / aborted rows for the reshot range describe
        # files that this reshoot is about to overwrite, and any
        # phase=reshoot rows from a partial earlier-reshoot attempt
        # (in the recursive case where droop fired during reshoot)
        # are also stale. Doing the drop at the *start* — instead of
        # only at the end — keeps the CSV consistent across nested
        # reshoots: each level wipes the bad rows from the previous
        # attempt before adding fresh ones.
        if self.shots_csv is not None:
            steps_set = set(range(from_step, through_step + 1))
            stale_phases = {
                Phase.SWEEP.value,
                Phase.ABORTED.value,
                Phase.RESHOOT.value,
            }
            self.shots_csv.drop_rows(
                lambda r: (
                    r.get("lens_config_slug") == slug
                    and r.get("phase") in stale_phases
                    and self._safe_int(r.get("rail_step")) in steps_set
                )
            )
        # Drop the (already-cropped) under-exposed samples at the
        # reshoot range from the in-memory collector dataset; the
        # reshoot's record_shot calls will re-add fresh samples.
        # Without this, the next range_check evaluate() sees both
        # versions of every (step, aperture) — twice as many points
        # at those steps, with the under-exposed values dragging the
        # max/min ratio up so a genuinely flat curve no longer trips
        # the OFF_TARGET threshold.
        if self._sharpness_collector is not None:
            try:
                self._sharpness_collector.wait_for_pending()
            except Exception:                            # noqa: BLE001
                log.exception("sharpness wait_for_pending before reshoot failed")
            clear = getattr(
                self._sharpness_collector, "clear_samples_for_steps", None
            )
            if clear is not None:
                clear(range(from_step, through_step + 1))
        # Move rail backward to from_step's position with backlash
        # compensation (we're reversing direction).
        self._move_to(from_position_mm, from_step * step_size_mm)
        sweep_direction = (
            Direction.FORWARD if step_size_mm >= 0 else Direction.BACKWARD
        )
        for step_index in range(from_step, through_step + 1):
            if step_index > from_step:
                self.rail.move_and_wait(
                    sweep_direction, abs(step_size_mm), backlash=False
                )
            position_mm = step_index * step_size_mm
            skip_remainder = False
            for aperture in apertures:
                if skip_remainder:
                    # An inner reshoot already covered this step.
                    continue
                if not self._should_capture(aperture, step_index):
                    # Off this aperture's subsample grid — original
                    # sweep skipped it, so reshoot mirrors that.
                    continue
                cal = calibration.per_aperture[aperture]
                ev, event = self._capture_one(
                    lens_dir=lens_dir,
                    slug=slug,
                    step_index=step_index,
                    position_mm=position_mm,
                    aperture=aperture,
                    cal=cal,
                    phase=Phase.RESHOOT,
                    droop=droop,
                )
                if event is not None:
                    covered = self._handle_droop(
                        event,
                        droop,
                        lens_dir=lens_dir,
                        slug=slug,
                        current_step_index=step_index,
                        current_position_mm=position_mm,
                        step_size_mm=step_size_mm,
                        apertures=apertures,
                        calibration=calibration,
                    )
                    if covered:
                        # Nested reshoot covered everything up
                        # through this step; stop hitting it again
                        # and let the outer for-loop move forward.
                        skip_remainder = True

    @staticmethod
    def _safe_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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

    def _set_capture_settings(self, aperture: str, cal) -> None:
        """Configure aperture / ISO / FEC for the next capture.

        Uses the camera's batched setter when available (one PTP
        round-trip; saves several seconds per shot on the Z7) and
        falls back to three individual setters otherwise — the
        FakeCamera in the runner tests stays a drop-in.
        """
        batched = getattr(self.camera, "set_aperture_iso_fec", None)
        if batched is not None:
            batched(aperture=aperture, iso=cal.iso, fec=cal.fec_ev)
            return
        self.camera.set_aperture(aperture)
        self.camera.set_iso(cal.iso)
        self.camera.set_fec(cal.fec_ev)

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
        self, lens_dir: Path, aperture: str, step_index: int,
        *, replica: int = 0,
    ) -> Path:
        slug = self._aperture_subdir(aperture)
        offset = getattr(self.config, "STEP_NUMBER_OFFSET", 100)
        width = getattr(self.config, "STEP_NUMBER_WIDTH", 4)
        return lens_dir / slug / step_filename(
            step_index, offset=offset, width=width, replica=replica
        )

    @staticmethod
    def _aperture_subdir(aperture: str) -> str:
        return aperture.replace("/", "_").replace(".", "_")

    def _record_sharpness(
        self, *, step_index: int, aperture: str, dest: Path,
        replica_index: int = 0,
    ) -> None:
        """Hand a captured JPEG to the sharpness collector. Defensive —
        cropping errors are surfaced at the next ``wait_for_pending``
        call, never inline."""
        if self._sharpness_collector is None:
            return
        try:
            # Older fakes / null collectors don't accept
            # replica_index; fall back to the legacy 3-arg form.
            try:
                self._sharpness_collector.record_shot(
                    rail_step=step_index,
                    aperture=_parse_fnum(aperture),
                    jpg_path=dest,
                    replica_index=replica_index,
                )
            except TypeError:
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

    def _wait_for_recycle(self) -> None:
        """Sleep just long enough so the *previous* aperture's recycle
        delay has elapsed since that shutter fire.

        Called *before* the next ``capture_to`` so the time spent in
        ``_set_capture_settings`` (~2.5 s on the Z7) is counted
        against the recycle budget. The recycle is keyed on the
        previous shot's aperture — a small aperture (e.g. f/16)
        forced the flash to discharge more power, so its recycle is
        longer; that wait happens *after* the f/16 capture, before
        whatever comes next. First shot of the session is a no-op.
        """
        if (
            self._last_capture_started_at is None
            or self._last_capture_aperture is None
        ):
            return
        recycle = self._recycle_for(self._last_capture_aperture)
        elapsed = self.monotonic() - self._last_capture_started_at
        remaining = recycle - elapsed
        if remaining > 0:
            self.sleep(remaining)

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

    def _announce_subsample_breakdown(
        self,
        *,
        apertures: list[str],
        plan: SamplePlan,
        positions: int,
    ) -> None:
        """Print the per-aperture capture-count breakdown so the
        user can sanity-check before committing to the sweep. Lists
        each aperture's subsample factor, the resulting shot count,
        and a total against the unsampled baseline."""
        replicas = max(1, int(plan.replicas_per_position))
        per_ap_lines: list[str] = []
        total_shots = 0
        for ap_str in apertures:
            try:
                ap_float = _parse_fnum(ap_str)
            except ValueError:
                factor = 1
            else:
                factor = max(1, int(plan.subsample_factors.get(ap_float, 1)))
            shots = (positions + factor - 1) // factor   # ceil(positions/factor)
            per_ap_total = shots * replicas
            total_shots += per_ap_total
            per_ap_lines.append(
                f"    {ap_str:6s} ×{factor} → {shots} × {replicas} = {per_ap_total}"
            )
        unsampled_total = positions * replicas * len(apertures)
        if unsampled_total > 0:
            saved_pct = (1 - total_shots / unsampled_total) * 100
        else:
            saved_pct = 0.0
        self._announce(
            "  per-aperture (factor × shots × replicas):\n"
            + "\n".join(per_ap_lines)
            + f"\n  total: {total_shots} captures "
            f"(vs {unsampled_total} unsampled — saves {saved_pct:.0f} %)"
        )

    def _announce(self, message: str) -> None:
        """Print a stage indicator on its own line + flush.

        Used to break the silence between user input and the next
        long-running phase (calibration, sweep). Always emits a
        newline so it doesn't fight with the in-place progress bar.
        """
        print(message, flush=True)

    def _announce_calibration(self, cal_result) -> None:
        """One-line per-aperture summary printed after every
        calibration pass — succeeded or not. The user wants to see
        where each aperture landed (FEC, EC, ISO, EV deficit, and
        whether the post-convergence flash-headroom swap was applied)
        without having to wait for a failure prompt."""
        print(
            f"Calibration result (baseline median {cal_result.baseline_median:.0f}):",
            flush=True,
        )
        for ap, c in cal_result.per_aperture.items():
            status = "ok" if c.hit_tolerance else "MISS"
            if c.iso_headroom_applied:
                headroom = " +headroom"
            elif getattr(c, "iso_headroom_attempted", False):
                # Swap was tried but the verification shot landed
                # outside tolerance — revealed in the post-cal print
                # so the user knows the flash wasn't actually at
                # mechanical max despite the plateau detection.
                attempt_ev = getattr(c, "iso_headroom_attempt_ev", None)
                if attempt_ev is not None:
                    headroom = (
                        f" (headroom: reverted, verification ev="
                        f"{attempt_ev:+.2f})"
                    )
                else:
                    headroom = " (headroom: reverted)"
            else:
                headroom = ""
            ec_used = f" EC={c.ec_ev:+.2f}" if c.ec_used else ""
            iso_bumped = "*" if c.iso_bumped else ""
            print(
                f"  {ap:5s} [{status}]  ev={c.final_ev_delta:+.2f}  "
                f"FEC={c.fec_ev:+.2f}{ec_used}  "
                f"ISO={c.iso}{iso_bumped}{headroom}",
                flush=True,
            )

    def _progress(
        self,
        *,
        step_index: int,
        step_count: int,
        position_mm: float,
        aperture: str,
        aperture_index: int,
        aperture_count: int,
    ) -> None:
        """Single-line progress display, updated in place via \\r.

        Format: ``[████░░░░] step 3/10  pos +1.20mm  ap f/8 (3/5)``.
        Cleared by :meth:`_progress_done` (which writes a newline) at
        the end of the sweep so subsequent print() calls don't
        overwrite it. No-op if stdout isn't a TTY (lets pipe-to-file
        users still get the per-shot CSV without garbled output)."""
        if not sys.stdout.isatty():
            return
        bar_width = 20
        # +1 because step_index is 0-based and we want progress to
        # increment AS the step starts, not just after.
        fraction = (step_index + (aperture_index + 1) / aperture_count) / step_count
        filled = int(round(fraction * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(
            f"\r[{bar}] step {step_index + 1}/{step_count}  "
            f"pos {position_mm:+7.2f} mm  "
            f"ap {aperture} ({aperture_index + 1}/{aperture_count})"
            "\x1b[K"
        )
        sys.stdout.flush()

    def _progress_done(self) -> None:
        """Newline at end of sweep so the next print() doesn't
        overwrite the final progress line."""
        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()

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
        # crops_under_source=True writes crops next to the source
        # JPEG (lens/aperture/cropped/<crop>/step_NNNN.jpg). The
        # `out_dir` arg is unused in this mode but the API requires
        # one — pass the lens-level cropped path so it's still a
        # sensible value if the flag is ever toggled.
        return LiveCollector(
            lens_dir / config.SHARPNESS_CROPPED_DIRNAME,
            max_workers=config.SHARPNESS_MAX_WORKERS,
            crops_under_source=True,
            cropped_dirname=config.SHARPNESS_CROPPED_DIRNAME,
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
    try:
        runner.run()
    except SessionAborted as e:
        # User-initiated abort (e.g. `q` at the lens-change prompt
        # or at the range-inadequate prompt). session.json and
        # shots.csv have already been finalised by
        # SessionRunner.stop(aborted=True); a stack trace at this
        # point is noise.
        log.info("session aborted: %s", e)
    except KeyboardInterrupt:
        # Ctrl-C mid-sweep. The runner's own except clause already
        # issued CC_RAIL_STOP, closed the camera, and wrote
        # session.json with aborted=true. Just print a friendly
        # newline + status and exit cleanly.
        print()
        log.info("session aborted via Ctrl-C")


if __name__ == "__main__":
    main()


