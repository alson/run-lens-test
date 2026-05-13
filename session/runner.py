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
    CALIBRATION_NOT_CONVERGED,
    LENS_CHANGE_REQUIRED,
    MANUAL_APERTURE_READY,
    MANUAL_CASCADE_ABORTED,
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
from session.manual_aperture import (
    aperture_sequence_for_manual_mode,
    parse_lens_slug,
    ring_and_label_sequence,
)
from session.log import (
    Phase,
    SessionMetadata,
    ShotRow,
    ShotsCSV,
    completed_slugs,
    write_json,
)
from session.probe import PhaseAResult, phase_a_peaks, phase_c_window_mm
from stackshot.controller import StackShotController
from stackshot.protocol import Direction
from sharpness.planner import SamplePlan, plan_sweep
from ui.prompts import (
    OTHER,
    confirm,
    prompt_focal_length,
    prompt_lens_config,
    prompt_lens_config_from_plan,
    prompt_magnification,
    prompt_magnification_indicated,
    prompt_max_aperture,
    prompt_off_target,
    prompt_range_inadequate,
    prompt_set_flash_power,
    prompt_turn_ring,
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
    jog_freeform: Callable[[object], float] | None = None
    """Optional interactive freeform jog (no start/end set, just move
    the rail freely until the user is satisfied). Used by
    :meth:`_offer_jog_pause` to let the user verify focus/framing
    before the FEC sanity check and before each lens-config prompt
    — the StackShot retains its rail position across program crashes
    and lens swaps, so the rail is often nowhere useful when the
    runner first starts. ``None`` disables the pause prompt entirely."""
    droop_detector: DroopDetector | None = None
    crop_box: CropBox | None = None
    sharpness_collector_factory: SharpnessCollectorFactory = field(
        default=_null_sharpness_collector_factory
    )
    manual_aperture: bool = False
    """Per-session flag enabling manual-aperture mode for non-CPU
    lenses. Inverts the capture-order loop (apertures outer, rail
    inner — one rail sweep per ring turn), parses focal_length /
    f_ring from the slug, prompts for ``magnification_indicated``,
    synthesises an aperture sequence from ring whole stops, and
    drives :class:`session.calibrate.Calibrator` with the manual-
    flash (ISO-only) cascade. See
    ``~/.claude/plans/rustling-tumbling-moonbeam.md`` for context."""
    prompt_magnification_indicated: Callable[[float], float] | None = None
    """Manual-aperture prompt for ``m_indicated``. Takes the default
    (= ``m_total``) and returns the user-entered value. ``None``
    when manual_aperture is False; defaulted at start() when True."""
    prompt_focal_length: Callable[[], float] | None = None
    """Manual-aperture fallback prompt when the slug regex misses."""
    prompt_max_aperture: Callable[[], float] | None = None
    """Manual-aperture fallback prompt when the slug regex misses."""
    prompt_turn_ring: Callable[..., bool] | None = None
    """Manual-aperture per-ring-turn prompt. Signature:
    ``(aperture_label, *, f_ring, m_indicated) -> bool``."""
    prompt_set_flash_power: Callable[..., bool] | None = None
    """Manual-aperture flash-power prompt. Signature:
    ``(*, power_c_label, power_a_label, context='') -> bool``.
    Injected so tests can scripted-respond; production wires
    :func:`ui.prompts.prompt_set_flash_power`."""
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
    _eta_last_shot_at: float | None = None
    """monotonic() of the start of the previous sweep shot, used to
    measure inter-shot intervals for the ETA. ``None`` after a user
    prompt to invalidate the bridging interval (so a 5 min battery
    swap doesn't poison the average)."""
    _eta_lens_interval_sum: float = 0.0
    """Cumulative sum (seconds) of valid in-flight inter-shot
    intervals collected since the start of the current lens config.
    Reset to 0 at the start of every sweep — see the docstring on
    :meth:`_eta_record_shot` for why a per-lens cumulative mean
    beats a fast-decaying EWMA in this rig's interval distribution."""
    _eta_lens_interval_count: int = 0
    """Count of valid intervals contributing to ``_eta_lens_interval_sum``."""
    _eta_sweep_apertures: tuple[str, ...] = ()
    """Cached aperture sequence for the current sweep — needed to
    count remaining shots across the subsample grid."""
    _eta_sweep_step_count: int = 0
    """Cached step count for the current sweep."""
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
    _current_focal_length_mm: float | None = None
    _current_f_ring: float | None = None
    _current_magnification_indicated: float | None = None
    """Per-lens-config manual-aperture metadata, set in
    :meth:`_run_one_lens_config` when ``manual_aperture`` is True
    and read by :meth:`_log_shot` to populate the new shots.csv
    columns. ``None`` in auto mode (the row writer drops them as
    empty cells)."""
    _current_ring_stops: list[float] = field(default_factory=list)
    """Per-lens-config ring f-numbers walked by the manual-mode
    aperture sequence (e.g. ``[2.8, 4.0, 5.6, 8.0]`` for the 55 AIS
    at 1:1 + PK-13). Used by :meth:`_run_manual_aperture_sweep` to
    compute per-step aperture EV differences for the flash-power
    cascade. Empty in auto mode."""

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

        # Offer the user a freeform jog before the FEC sanity check.
        # The StackShot retains its rail position across crashes and
        # lens swaps, so the rail (and target) is often nowhere
        # sensible right after launch. Letting the user verify focus
        # / framing here means the sanity check fires on a real
        # subject crop, not a defocused background.
        self._offer_jog_pause(where="before FEC sanity check")

        # FEC sanity check (skip if the camera doesn't expose it —
        # e.g. fakes used in tests; production NikonZ7 always has it).
        # Also skip in manual-aperture mode: the flash is in M (manual
        # power) on the commander, so FEC has no effect on the flash
        # output — the sanity check would always fail (drop ≈ 0 EV
        # between FEC=0 and FEC=-2). Per
        # ``rustling-tumbling-moonbeam.md``.
        sanity_check = getattr(self.camera, "fec_sanity_check", None)
        if sanity_check is not None and not self.manual_aperture:
            try:
                sanity_check(
                    baseline_fec=0.0,
                    low_fec=getattr(
                        self.config, "FEC_SANITY_LOW_EV", -2.0
                    ),
                    min_drop_ev=getattr(
                        self.config, "FEC_SANITY_MIN_DROP_EV", 1.0
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

        # Clear per-lens-config manual-mode state at the start of
        # every call so leftover values from a previous lens config
        # don't contaminate this one (defensive — also cleared in the
        # finally clause below).
        self._current_focal_length_mm = None
        self._current_f_ring = None
        self._current_magnification_indicated = None

        # Magnification is asked once per lens config; the sweep
        # plan is a function of M, the lens's widest aperture, and
        # the jogged range. When the caller picked a preset from the
        # reshoot-plan menu, the value is supplied directly; otherwise
        # we prompt as before.
        if magnification is None:
            magnification = self.prompt_magnification()

        if self.manual_aperture:
            apertures, widest_aperture_str, widest_aperture = (
                self._prepare_manual_aperture_config(
                    slug=slug, magnification_total=magnification,
                )
            )
        else:
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
            self.range_checker.set_subsample_factors(
                self._float_keyed_subsample_factors()
            )
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
                self._announce("Closing sharpness pipeline (draining crops)...")
                self._sharpness_collector.shutdown(wait=True)
            except Exception:                        # noqa: BLE001
                # Cropping errors must not mask a session error or
                # block lens-change recovery.
                log.exception("sharpness collector shutdown failed")
            if self.range_checker is not None:
                self.range_checker.set_collector(None)
                self.range_checker.set_subsample_factors(None)
            self._sharpness_collector = None
            self._subsample_factors = {}
            # Clear manual-mode per-lens metadata so a follow-on
            # auto-mode lens config (or just a re-entry of this method
            # via the lens-config loop) starts fresh.
            self._current_focal_length_mm = None
            self._current_f_ring = None
            self._current_magnification_indicated = None
            self._current_ring_stops = []

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

    def _float_keyed_subsample_factors(self) -> dict[float, int]:
        """Translate the string-keyed ``_subsample_factors`` (``"f/16"``
        → 4) into the float-keyed shape the advisor expects (``16.0`` →
        4). Apertures that fail to parse are dropped — the advisor
        defaults missing keys to stride 1, which is the safe choice."""
        out: dict[float, int] = {}
        for ap_str, factor in self._subsample_factors.items():
            try:
                ap_float = _parse_fnum(ap_str)
            except ValueError:
                continue
            out[ap_float] = max(1, int(factor))
        return out

    def _prepare_manual_aperture_config(
        self,
        *,
        slug: str,
        magnification_total: float,
    ) -> tuple[list[str], str, float]:
        """Build a manual-mode aperture sequence + record per-lens-config
        metadata on ``self``.

        Returns ``(apertures, widest_aperture_str, widest_aperture)`` —
        same shape as the camera-driven auto-mode path so the caller
        can use them identically. ``apertures`` are the effective
        (lens-internal) labels rounded to the 1/3-stop scale, in
        widest-first order.
        """
        focal, f_ring = parse_lens_slug(slug)
        if focal is None:
            focal = (
                self.prompt_focal_length()
                if self.prompt_focal_length is not None
                else prompt_focal_length()
            )
        if f_ring is None:
            f_ring = (
                self.prompt_max_aperture()
                if self.prompt_max_aperture is not None
                else prompt_max_aperture()
            )
        m_indicated = (
            self.prompt_magnification_indicated(magnification_total)
            if self.prompt_magnification_indicated is not None
            else prompt_magnification_indicated(default=magnification_total)
        )

        # Record per-lens-config state so _log_shot picks it up.
        self._current_focal_length_mm = float(focal)
        self._current_f_ring = float(f_ring)
        self._current_magnification_indicated = float(m_indicated)

        pairs = ring_and_label_sequence(
            f_ring_max=float(f_ring),
            m_indicated=float(m_indicated),
            whole_stop_scale=self.config.WHOLE_STOP_SCALE,
            max_fstop=self.config.MAX_FSTOP,
            one_third_stop_scale=self.config.ONE_THIRD_STOP_SCALE,
        )
        if not pairs:
            raise RuntimeError(
                f"manual-mode aperture sequence empty for slug={slug!r} "
                f"(f_ring={f_ring}, m_indicated={m_indicated}). "
                f"Effective widest already exceeds MAX_FSTOP="
                f"{self.config.MAX_FSTOP}?"
            )
        self._current_ring_stops = [r for r, _ in pairs]
        apertures = [label for _, label in pairs]
        widest_aperture_str = apertures[0]
        widest_aperture = _parse_fnum(widest_aperture_str)
        self._announce(
            f"Manual-mode lens config: focal={focal:g}mm, "
            f"f_ring={f_ring:g}, m_total={magnification_total:g}, "
            f"m_indicated={m_indicated:g}; "
            f"ring stops to walk: "
            f"{', '.join(f'{r:g}' for r in self._current_ring_stops)}"
        )
        return apertures, widest_aperture_str, widest_aperture

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
                # In manual mode, show the ring f-number (what the
                # operator dials) for "widest" rather than the
                # effective label. Effective is metadata-only.
                if self.manual_aperture and self._current_ring_stops:
                    widest_display = f"f/{self._current_ring_stops[0]:g}"
                else:
                    widest_display = widest_aperture_str
                self._announce(
                    f"Plan: M={magnification:g}, widest={widest_display}, "
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

    def _run_manual_aperture_sweep(
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
        """Inverted loop for non-CPU lenses: one rail sweep per aperture.

        For each aperture in widest-first order:

        1. Rail returns to position 0 (skip on first iteration).
        2. Prompt the user to turn the lens ring to the target.
        3. Per-aperture flash-power cascade:
           - First aperture: coarse power search + back-calc + verify
             + ISO fine-tune
             (:meth:`Calibrator.calibrate_first_aperture_manual`).
           - Subsequent apertures: deterministic power target from
             previous (power, iso) + aperture diff + ISO fine-tune
             (:meth:`Calibrator.calibrate_subsequent_aperture_manual`).
        4. Per-aperture Phase A sharpness probe (new — auto mode only
           probes the widest aperture).
        5. Single-aperture rail sweep.

        After all apertures: rail returns to 0, ``calibration.json``
        and session metadata are written, lens-change recovery is
        triggered.
        """
        calibrator = Calibrator(
            camera=self.camera,
            measure=self._measure,
            config=self.config,
            shots_csv=self.shots_csv,
            now=self.now,
            progress=self._calibration_progress,
        )
        aggregate = CalibrationResult(
            baseline_median=self.baseline_median or 0.0
        )
        probe_enabled = bool(getattr(self.config, "PROBE_ENABLED", False))
        m_indicated = self._current_magnification_indicated or 0.0
        ring_stops = self._current_ring_stops or []
        forward_steps_done = 0
        last_step_size = step_size_mm
        last_phase_c_steps = steps
        # Per-aperture flash-power state — carries forward between
        # ring turns so the subsequent-aperture cascade can compute
        # its deterministic target.
        prev_power_stops: float | None = None
        prev_iso: int | None = None
        prev_ev_delta: float = 0.0

        set_flash_power = (
            self.prompt_set_flash_power
            if self.prompt_set_flash_power is not None
            else prompt_set_flash_power
        )

        for index, aperture in enumerate(apertures):
            if index > 0 and forward_steps_done != 0:
                self._announce(
                    f"Returning rail to start before next aperture "
                    f"(ring f/{ring_stops[index]:g})..."
                )
                self._return_rail_to_zero(
                    forward_steps_done * last_step_size
                )
                forward_steps_done = 0
            # Between apertures: ping the operator that the next
            # ring-turn / flash-power prompt is coming. Skip on
            # index==0 (operator just kicked off the session and is
            # at the rig anyway).
            if index > 0 and self.ntfy is not None:
                self.ntfy.with_body(
                    MANUAL_APERTURE_READY,
                    f"{slug}: ring f/{ring_stops[index]:g} "
                    f"({index + 1}/{len(apertures)})",
                )

            # Prompt the user to turn the ring.
            turn_ring = (
                self.prompt_turn_ring
                if self.prompt_turn_ring is not None
                else prompt_turn_ring
            )
            ring_stop = ring_stops[index] if index < len(ring_stops) else 0.0
            ok = turn_ring(
                aperture,
                ring_stop=ring_stop,
                m_indicated=m_indicated,
            )
            if not ok:
                raise SessionAborted(
                    f"user aborted at ring-turn prompt for {aperture}"
                )

            # Per-aperture cascade dispatch: first vs subsequent.
            # SessionAborted from a safety termination (coarse search
            # exhausted, etc.) gets a notification before re-raising
            # so an away-from-rig operator finds out without watching
            # the terminal.
            try:
                if index == 0 or prev_power_stops is None or prev_iso is None:
                    self._announce(
                        f"Calibrating {slug} @ ring f/{ring_stop:g} "
                        f"(first aperture: coarse search → back-calc → "
                        f"verify, flash-M)..."
                    )
                    cal_one, used_power, used_iso = (
                        calibrator.calibrate_first_aperture_manual(
                            aperture=aperture,
                            lens_dir=lens_dir,
                            lens_slug=slug,
                            session_id=self.session_id,
                            baseline_median=(
                                aggregate.baseline_median
                                if aggregate.baseline_median > 0
                                else None
                            ),
                            prompt_set_flash_power=set_flash_power,
                        )
                    )
                else:
                    # ring_stops list matches apertures 1:1 (built
                    # from the same iteration in
                    # _prepare_manual_aperture_config).
                    prev_ring = ring_stops[index - 1]
                    curr_ring = ring_stops[index]
                    # 2·log2(f_new / f_old) gives the EV stops light
                    # loss for the smaller aperture (positive).
                    aperture_stops_diff = 2.0 * math.log2(curr_ring / prev_ring)
                    self._announce(
                        f"Calibrating {slug} @ ring f/{ring_stop:g} "
                        f"(subsequent: +{aperture_stops_diff:.2f} EV "
                        f"step, flash-M)..."
                    )
                    cal_one, used_power, used_iso = (
                        calibrator.calibrate_subsequent_aperture_manual(
                            aperture=aperture,
                            lens_dir=lens_dir,
                            lens_slug=slug,
                            session_id=self.session_id,
                            baseline_median=aggregate.baseline_median,
                            prev_power_stops=prev_power_stops,
                            prev_iso=prev_iso,
                            prev_ev_delta=prev_ev_delta,
                            aperture_stops_diff=aperture_stops_diff,
                            prompt_set_flash_power=set_flash_power,
                        )
                    )
            except SessionAborted as exc:
                if self.ntfy is not None:
                    self.ntfy.with_body(
                        MANUAL_CASCADE_ABORTED,
                        f"{slug} @ ring f/{ring_stop:g}: {exc}",
                    )
                raise

            # Carry state into the next aperture: the (power, iso) the
            # cascade actually used, plus the final delta — so the
            # next call's target search can pull residual error toward
            # zero rather than just compensating for the aperture step.
            prev_power_stops = used_power
            prev_iso = used_iso
            prev_ev_delta = cal_one.per_aperture[aperture].final_ev_delta

            if (
                aggregate.baseline_median == 0
                and cal_one.baseline_median > 0
            ):
                aggregate.baseline_median = cal_one.baseline_median
            aggregate.per_aperture[aperture] = (
                cal_one.per_aperture[aperture]
            )
            self.baseline_median = aggregate.baseline_median
            self._announce_calibration(cal_one)

            # Soft-fail notification: cascade completed but ISO
            # fine-tune couldn't pull the delta inside ⅓ EV. Captures
            # at this aperture will still happen — operator can
            # decide whether to accept the data or re-run.
            ap_cal = cal_one.per_aperture[aperture]
            if not ap_cal.hit_tolerance and self.ntfy is not None:
                self.ntfy.with_body(
                    CALIBRATION_NOT_CONVERGED,
                    f"{slug} @ ring f/{ring_stop:g}: final delta "
                    f"{ap_cal.final_ev_delta:+.2f} EV, "
                    f"ISO {ap_cal.iso}",
                )

            # Per-aperture phase A + sweep.
            droop = self._make_droop_detector()
            phase_c_start = 0
            phase_c_steps = steps
            if probe_enabled:
                phase_a_result = self._run_phase_a(
                    lens_dir=lens_dir,
                    slug=slug,
                    end_mm=end_mm,
                    widest_aperture=aperture,
                    calibration=cal_one,
                    n_probes=int(
                        getattr(self.config, "PROBE_PHASE_A_POINTS", 7)
                    ),
                    droop=droop,
                )
                phase_c_plan = self._plan_phase_c(
                    phase_a_result,
                    end_mm=end_mm,
                    rail_step_mm=step_size_mm,
                    fwhm_mm=plan.estimated_fwhm_mm_widest,
                    n_probes=int(
                        getattr(self.config, "PROBE_PHASE_A_POINTS", 7)
                    ),
                    padding_x_fwhm=float(
                        getattr(
                            self.config,
                            "PROBE_PHASE_C_FWHM_PADDING",
                            1.5,
                        )
                    ),
                )
                if phase_c_plan is None:
                    self._announce(
                        f"Phase A inconclusive at {aperture} — falling "
                        f"back to uniform sweep across the jogged range."
                    )
                    self._return_rail_to_zero(end_mm)
                else:
                    phase_c_start, phase_c_steps = phase_c_plan
                    target_mm = phase_c_start * step_size_mm
                    self._announce(
                        f"Phase C at {aperture}: {phase_c_steps} steps "
                        f"from step {phase_c_start} ({target_mm:+.3f} mm)."
                    )
                    self._move_to(end_mm, target_mm)

            self._announce(
                f"Sweeping {slug} @ ring f/{ring_stop:g}: "
                f"{phase_c_steps} steps..."
            )
            self._eta_sweep_apertures = (aperture,)
            self._eta_sweep_step_count = phase_c_steps
            self._eta_lens_interval_sum = 0.0
            self._eta_lens_interval_count = 0
            self._eta_last_shot_at = None
            self._run_sweep(
                lens_dir=lens_dir,
                slug=slug,
                steps=phase_c_steps,
                step_size_mm=step_size_mm,
                calibration=cal_one,
                apertures=[aperture],
                start_step_index=phase_c_start,
                droop=droop,
            )
            forward_steps_done = phase_c_start + phase_c_steps - 1
            last_step_size = step_size_mm
            last_phase_c_steps = phase_c_steps

            # Range advisor: detect truncated peaks at this aperture
            # and extend the sweep if the corner peaks are clipped by
            # the captured rail window. Runs per-aperture in manual
            # mode (vs. auto mode's per-lens-config). _RejogRequested
            # is propagated unchanged; the cascade has no clean
            # rejog-in-place semantics since calibration is already
            # done — the runner's outer abort path will catch it.
            if self.range_checker is not None:
                self._announce(
                    f"Range check at ring f/{ring_stop:g} "
                    f"(waiting on sharpness crops)..."
                )
                range_check_end_mm = (phase_c_steps - 1) * step_size_mm
                forward_steps_done = self._run_range_check_and_extend(
                    lens_dir=lens_dir,
                    slug=slug,
                    end_mm=range_check_end_mm,
                    step_size_mm=step_size_mm,
                    steps=phase_c_steps,
                    calibration=cal_one,
                    apertures=[aperture],
                    start_step_index=phase_c_start,
                )

        # Final rail-zero return.
        if forward_steps_done != 0:
            self._announce("Returning rail to start (final)...")
            self._return_rail_to_zero(forward_steps_done * last_step_size)

        # Write calibration.json with the aggregated per-aperture result.
        write_json(
            lens_dir / "calibration.json",
            {
                "baseline_median": aggregate.baseline_median,
                "per_aperture": aggregate.per_aperture,
            },
        )

        # Record lens-config metadata.
        from dataclasses import asdict as _asdict
        if self.metadata is not None and not any(
            lc.get("slug") == slug for lc in self.metadata.lens_configs
        ):
            self.metadata.lens_configs.append({
                "slug": slug,
                "magnification": float(magnification),
                "widest_aperture": float(plan.widest_aperture),
                "sample_plan": _asdict(plan),
                "manual_aperture": True,
                "magnification_indicated": (
                    self._current_magnification_indicated
                ),
                "f_ring": self._current_f_ring,
                "focal_length_mm": self._current_focal_length_mm,
            })
            self._write_metadata()
            self._completed_slugs.add(slug)

        # Lens-change recovery.
        self._lens_change_recovery(slug)

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
        if self.manual_aperture:
            self._run_manual_aperture_sweep(
                slug=slug,
                lens_dir=lens_dir,
                end_mm=end_mm,
                steps=steps,
                step_size_mm=step_size_mm,
                apertures=apertures,
                magnification=magnification,
                plan=plan,
            )
            return

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
            progress=self._calibration_progress,
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
            self._progress_done()  # newline after the inline cal progress bar
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
        probe_enabled = bool(getattr(self.config, "PROBE_ENABLED", False))
        while True:
            # One droop detector per lens-config attempt — phase-A
            # probes and the main sweep share the same windows so a
            # flash misfire during probing isn't re-discovered from
            # scratch on the sweep's first frames.
            droop = self._make_droop_detector()
            # Phase A (when probe enabled) collapses the user's wide
            # jog to a sparse single-aperture probe pass. The
            # phase-C window is computed from the observed crop-peak
            # bracket; on inconclusive phase A we fall back to the
            # uniform sweep below.
            phase_c_start = 0
            phase_c_steps = steps
            phase_a_result: PhaseAResult | None = None
            if probe_enabled:
                widest_aperture_str = apertures[0]
                phase_a_result = self._run_phase_a(
                    lens_dir=lens_dir,
                    slug=slug,
                    end_mm=end_mm,
                    widest_aperture=widest_aperture_str,
                    calibration=cal_result,
                    n_probes=int(getattr(
                        self.config, "PROBE_PHASE_A_POINTS", 7
                    )),
                    droop=droop,
                )
                phase_c_plan = self._plan_phase_c(
                    phase_a_result,
                    end_mm=end_mm,
                    rail_step_mm=step_size_mm,
                    fwhm_mm=plan.estimated_fwhm_mm_widest,
                    n_probes=int(getattr(
                        self.config, "PROBE_PHASE_A_POINTS", 7
                    )),
                    padding_x_fwhm=float(getattr(
                        self.config, "PROBE_PHASE_C_FWHM_PADDING", 1.5
                    )),
                )
                if phase_c_plan is None:
                    self._announce(
                        "Phase A inconclusive — falling back to "
                        "uniform sweep across the user-jogged range."
                    )
                    # Retract to rail-zero, then run today's sweep.
                    self._return_rail_to_zero(end_mm)
                else:
                    phase_c_start, phase_c_steps = phase_c_plan
                    target_mm = phase_c_start * step_size_mm
                    self._announce(
                        f"Phase C: {phase_c_steps} steps × {len(apertures)} "
                        f"apertures from step {phase_c_start} "
                        f"({target_mm:+.3f} mm) — "
                        f"corner spread "
                        f"{phase_a_result.corner_spread_samples:.2f} "
                        f"phase-A samples."
                    )
                    # Single retract from end_mm to phase_c_start * step_size_mm
                    # with backlash compensation — the only direction reversal
                    # in the whole probe.
                    self._move_to(end_mm, target_mm)

            self._announce(
                f"Sweeping {slug}: {phase_c_steps} steps × "
                f"{len(apertures)} apertures..."
            )
            # Reset ETA tracking for the new sweep — earlier
            # calibration / phase-A intervals are not representative.
            self._eta_sweep_apertures = tuple(apertures)
            self._eta_sweep_step_count = phase_c_steps
            self._eta_lens_interval_sum = 0.0
            self._eta_lens_interval_count = 0
            self._eta_last_shot_at = None
            self._run_sweep(
                lens_dir=lens_dir,
                slug=slug,
                steps=phase_c_steps,
                step_size_mm=step_size_mm,
                calibration=cal_result,
                apertures=apertures,
                start_step_index=phase_c_start,
                droop=droop,
            )

            # range-check end_mm: signed extent of the *captured*
            # window. For the uniform sweep that's the user's jog;
            # for phase C it's the narrower phase-C span.
            range_check_end_mm = (phase_c_steps - 1) * step_size_mm
            forward_steps_done = phase_c_start + phase_c_steps - 1
            if self.range_checker is not None:
                self._announce("Range check (waiting on sharpness crops)...")
                try:
                    forward_steps_done = self._run_range_check_and_extend(
                        lens_dir=lens_dir,
                        slug=slug,
                        end_mm=range_check_end_mm,
                        step_size_mm=step_size_mm,
                        steps=phase_c_steps,
                        calibration=cal_result,
                        apertures=apertures,
                        start_step_index=phase_c_start,
                    )
                except _RejogRequested:
                    # User wants new START / END; rewind, recreate
                    # the collector, rerun the sweep loop.
                    widest_aperture_str = apertures[0]
                    end_mm, steps, step_size_mm, plan = self._rejog_for_lens_config(
                        lens_dir=lens_dir,
                        current_position_mm=(
                            phase_c_start + phase_c_steps - 1
                        ) * step_size_mm,
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
                    if self.range_checker is not None:
                        self.range_checker.set_subsample_factors(
                            self._float_keyed_subsample_factors()
                        )
                    continue
            break

        # Return rail to 0 for the next lens config (backlash on; we
        # may have moved into negative territory if the extension was
        # on the 'start_side' side). ``forward_steps_done`` is the absolute
        # step index from rail-zero — always correct to multiply by
        # ``step_size_mm`` to get the retract distance.
        if forward_steps_done != 0:
            self._announce("Returning rail to start...")
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
        start_step_index: int = 0,
        droop: DroopDetector | None = None,
    ) -> None:
        """Forward sweep across ``steps`` rail positions.

        ``start_step_index`` is the integer index used for the first
        captured position. The default 0 matches the user-jog flow:
        rail is at position 0, step_index runs 0..steps-1, and
        ``rail_position_mm`` recorded in shots.csv is
        ``step_index × step_size_mm`` (== absolute rail-mm from
        rail-zero). The probe (phase-C) flow passes a non-zero start
        so the file step numbers and shots.csv positions stay anchored
        to rail-zero — phase C starts at e.g. step 60, not step 0,
        so its file numbers (``step_0160.jpg`` etc.) don't collide
        with phase A's and downstream analysis sees absolute rail
        positions consistently.

        ``droop`` may be passed in by the caller so phase-A probe
        samples and the main sweep share one detector window — a
        flash misfire during probing then carries forward into the
        sweep's window. If ``None``, a fresh detector is created.
        """
        if droop is None:
            droop = self._make_droop_detector()
        # Sweep direction is the sign of step_size_mm. With a positive
        # end_mm (user jogged forward) sweep direction is FORWARD; with
        # negative (user jogged backward) it's BACKWARD. backlash=False
        # is correct either way as long as we never reverse direction
        # mid-sweep.
        sweep_direction = (
            Direction.FORWARD if step_size_mm >= 0 else Direction.BACKWARD
        )
        sweep_step_distance = abs(step_size_mm)
        end_step_index = start_step_index + steps
        for step_index in range(start_step_index, end_step_index):
            if step_index > start_step_index:
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
                # Phase-C runs at non-zero step_index (e.g. 45..96),
                # but the progress bar's "step N/M" should count
                # 1..M relative to the phase, not the absolute index.
                progress_step_display=step_index - start_step_index,
            )
        # Newline so the next print() output doesn't overwrite the
        # final progress line.
        self._progress_done()

    # ---- Coarse-to-fine focus probe ------------------------------------

    def _run_phase_a(
        self,
        *,
        lens_dir: Path,
        slug: str,
        end_mm: float,
        widest_aperture: str,
        calibration: CalibrationResult,
        n_probes: int,
        droop: DroopDetector,
    ) -> PhaseAResult:
        """Phase A: sparse forward sweep at the widest aperture only.

        Captures ``n_probes`` evenly-spaced positions across the
        user-jogged range ``[0, end_mm]``, single replica each, with
        all 17 crops cropped via a *separate* phase-A LiveCollector
        rooted at ``<lens_dir>/probe/``. Phase A samples never enter
        the per-lens collector that drives phase D, so the range
        check sees only the phase-C dataset.

        Pre: rail is at rail-position 0.
        Post: rail is at ``end_mm`` (the last probe position).
        Returns the PhaseAResult; on inconclusive results the caller
        should fall back to today's uniform-sweep flow.

        Probe captures push EV into ``droop`` so a misbehaving flash
        trips the same alert flow as the main sweep — instead of
        running all N probes against a noise-floor frame and falling
        back to "inconclusive". On a trip, the user is prompted to
        swap batteries; on resume, Phase A discards its under-exposed
        captures and restarts from probe 0 at rail-zero with fresh
        flash power.
        """
        if n_probes < 3:
            raise ValueError(f"n_probes must be ≥ 3, got {n_probes}")
        widest_ap_value = _parse_fnum(widest_aperture)
        cal = calibration.per_aperture[widest_aperture]

        # Spacing across the user's jog. Sign is preserved so we move
        # FORWARD when end_mm > 0, BACKWARD otherwise.
        spacing_mm = end_mm / (n_probes - 1)
        sweep_direction = (
            Direction.FORWARD if spacing_mm >= 0 else Direction.BACKWARD
        )
        spacing_distance = abs(spacing_mm)

        # Phase-A captures live under <lens_dir>/probe/ so they don't
        # mix with phase-C files in the same crop subtree, and the
        # phase-A collector cropping output goes to
        # <lens_dir>/probe/cropped/<crop>/. The directory is created
        # on first capture (collector handles that).
        probe_dir = lens_dir / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)

        previous_collector = self._sharpness_collector
        self._announce(
            f"Phase A: {n_probes} probes at {widest_aperture} across "
            f"{end_mm:+.2f} mm jog ..."
        )

        # Wrap the probe loop so a droop trip can jump us back here
        # after the user swaps batteries.
        while True:
            probe_collector = self.sharpness_collector_factory(probe_dir)
            self._sharpness_collector = probe_collector
            droop_tripped = False
            try:
                for probe_index in range(n_probes):
                    if probe_index > 0:
                        self.rail.move_and_wait(
                            sweep_direction, spacing_distance, backlash=False
                        )
                    position_mm = probe_index * spacing_mm
                    self._set_capture_settings(widest_aperture, cal)
                    # Phase A is single-aperture, single-replica, so
                    # the progress bar advances one notch per probe.
                    # ETA is suppressed — phase-A is too short and
                    # too unrepresentative to seed the EWMA.
                    self._progress(
                        step_index=probe_index,
                        step_count=n_probes,
                        position_mm=position_mm,
                        aperture=widest_aperture,
                        aperture_index=0,
                        aperture_count=1,
                        phase_label="probe",
                        show_eta=False,
                    )
                    self._wait_for_recycle()
                    ap_subdir = self._aperture_subdir(widest_aperture)
                    dest = probe_dir / ap_subdir / step_filename(
                        probe_index,
                        offset=int(getattr(self.config, "STEP_NUMBER_OFFSET", 100)),
                        width=int(getattr(self.config, "STEP_NUMBER_WIDTH", 4)),
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    self._last_capture_started_at = self.monotonic()
                    self._last_capture_aperture = widest_aperture
                    self.camera.capture_to(dest)
                    self._record_sharpness(
                        step_index=probe_index, aperture=widest_aperture, dest=dest,
                    )
                    median = self._measure(dest)
                    ev = ev_delta(median, self.baseline_median or median)
                    self._log_shot(
                        slug=slug,
                        step_index=probe_index,
                        position_mm=position_mm,
                        aperture=widest_aperture,
                        cal=cal,
                        median=median,
                        ev=ev,
                        filename=str(dest.relative_to(lens_dir)),
                        phase=Phase.PROBE,
                    )
                    event = droop.push(
                        widest_aperture, ev, step_index=probe_index
                    )
                    if event is not None:
                        # Pause + prompt; on user 'q' this raises.
                        self._droop_alert_and_pause(event)
                        droop_tripped = True
                        break
                if not droop_tripped:
                    # Land the progress bar on its trailing newline so
                    # subsequent prints aren't overwritten.
                    self._progress_done()
                    # Drain crops so the collector dataset is complete.
                    probe_collector.wait_for_pending()
                    dataset = probe_collector.snapshot() if hasattr(
                        probe_collector, "snapshot"
                    ) else None
                    if dataset is None:
                        # Test-fake collectors may not implement snapshot().
                        # That path can't run real phase A — return an
                        # inconclusive result and let the caller fall back.
                        return PhaseAResult(inconclusive=True)
                    # Build {crop_name: [size_at_probe_0, ...]} ordered
                    # by probe index.
                    agg = dataset.aggregated()
                    crop_names: set[str] = set()
                    for s in agg.samples:
                        crop_names.update(s.sizes.keys())
                    crop_sizes: dict[str, list[int]] = {}
                    for name in sorted(crop_names):
                        series = [
                            size for _step, size in agg.crop_series_for(
                                widest_ap_value, name
                            )
                        ]
                        if len(series) == n_probes:
                            crop_sizes[name] = series
                    if not crop_sizes:
                        return PhaseAResult(inconclusive=True)
                    return phase_a_peaks(
                        crop_sizes,
                        dynamic_range_min=getattr(
                            self.config,
                            "SHARPNESS_CROP_DYNAMIC_RANGE_MIN",
                            0.25,
                        ),
                    )
            finally:
                try:
                    probe_collector.shutdown(wait=True)
                except Exception:                        # noqa: BLE001
                    log.exception("phase-A collector shutdown failed")
                self._sharpness_collector = previous_collector

            # Trip path: clear the collector, drop the discarded
            # phase=probe rows from shots.csv, rewind to rail-zero,
            # reset the droop window, and re-enter the loop.
            droop.reset_all()
            if self.shots_csv is not None:
                self.shots_csv.drop_rows(
                    lambda r: (
                        r.get("lens_config_slug") == slug
                        and r.get("phase") == Phase.PROBE.value
                    )
                )
            # Rail is somewhere between probe 0 and probe N-1; sign-
            # aware return brings it to position 0 with backlash.
            current_pos = (probe_index) * spacing_mm  # last position visited
            self._return_rail_to_zero(current_pos)
            self._announce(
                f"Phase A: restarting probes at {widest_aperture} "
                "with fresh flash power ..."
            )

    def _plan_phase_c(
        self,
        result: PhaseAResult,
        *,
        end_mm: float,
        rail_step_mm: float,
        fwhm_mm: float,
        n_probes: int,
        padding_x_fwhm: float,
    ) -> tuple[int, int] | None:
        """Convert a PhaseAResult into phase-C step-index bounds.

        ``rail_step_mm`` here must be the *signed* step size — same
        convention as ``step_size_mm`` elsewhere in the runner: positive
        for forward jogs, negative for backward. Dividing positions
        (signed) by a signed step size gives unsigned step *indices*
        regardless of direction, since step indices count forward in
        the user's chosen direction from rail-zero.

        Returns ``(start_step_index, n_steps)`` where ``start_step_index``
        is the rail step (relative to rail-zero, in the user's
        forward direction) where phase C begins and ``n_steps`` is the
        count of steps to capture. Returns ``None`` when phase A was
        inconclusive or the resulting window collapses to zero.
        """
        window = phase_c_window_mm(
            result,
            phase_a_start_mm=0.0,
            phase_a_end_mm=end_mm,
            n_probes=n_probes,
            fwhm_mm=fwhm_mm,
            padding_x_fwhm=padding_x_fwhm,
        )
        if window is None:
            return None
        if rail_step_mm == 0:
            return None
        lo_mm, hi_mm = window
        # Both inputs are signed; dividing signed-by-signed yields a
        # non-negative step index that counts forward in the user's
        # chosen rail direction. The two computed indices may straddle
        # zero or come out reversed; min(...) is the start, max(...) is
        # the end.
        s1 = int(round(lo_mm / rail_step_mm))
        s2 = int(round(hi_mm / rail_step_mm))
        start_step = min(s1, s2)
        end_step = max(s1, s2)
        n_steps = end_step - start_step + 1
        if n_steps < 1:
            return None
        return start_step, n_steps

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
        progress_step_display: int | None = None,
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
            ev, event = self._capture_one(
                lens_dir=lens_dir,
                slug=slug,
                step_index=step_index,
                position_mm=position_mm,
                aperture=aperture,
                cal=cal,
                phase=Phase.SWEEP,
                droop=droop,
                progress_ctx=(step_count, ap_index, len(apertures)),
                progress_step_display=progress_step_display,
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
        start_step_index: int = 0,
    ) -> int:
        """Evaluate range adequacy; probe + extend if peak is at an
        edge. Returns the absolute step index from rail-zero where the
        rail currently sits; the caller multiplies by ``step_size_mm``
        to retract.

        Stays a no-op if ``range_checker.evaluate`` returns ``ok=True``
        (which is the stub's behaviour today).

        ``start_step_index`` is the rail-zero-relative step index at
        which the just-completed sweep began. The default 0 matches
        the user-jog flow. The probe (phase-C) flow passes a non-zero
        start so far/near extensions land at absolute step indices
        beyond the phase-C window, not relative to step 0.
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
            return start_step_index + steps - 1
        if (
            result.ok
            or not result.needs_extension
            or result.peak_side not in ("start_side", "end_side", "both")
        ):
            write_json(lens_dir / "range_check.json", {
                "ok": result.ok,
                "peak_side": result.peak_side,
                "extension_applied": False,
                "notes": result.notes,
            })
            return start_step_index + steps - 1

        # Per-side extension target. Prefer the advisor's per-
        # aperture extra_steps projection (max across truncated
        # apertures) when available — it's a curve-fit estimate of
        # how far past the captured edge the peak actually sits, far
        # better than the static RANGE_EXTENSION_FRACTION budget
        # whenever the truncation detector could fit something useful.
        # When the advisor returns 0 (no truncation, OFF_TARGET, or
        # the parabolic fit fell back without producing a real
        # number), fall back to the static fraction-of-range budget.
        # In either case cap at the original sweep length so a
        # runaway projection can't extend more than doubling the
        # rail range.
        configured_probe_count = int(
            getattr(self.config, "RANGE_CHECK_PROBE_STEPS", 2)
        )
        budget_steps = int(
            (end_mm * float(
                getattr(self.config, "RANGE_EXTENSION_FRACTION", 0.5)
            )) // step_size_mm
        )
        advisor_extra = int(getattr(result, "extra_steps_max", 0) or 0)
        if advisor_extra > 0:
            # Floor at probe count so we always have room for the
            # configured probes; ceiling at the original sweep length
            # to bound runaway projections.
            extension_steps = max(
                configured_probe_count, min(advisor_extra, steps)
            )
        else:
            extension_steps = budget_steps
        extension_mm = extension_steps * abs(step_size_mm)
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
            return start_step_index + steps - 1

        # Per-side probe count — split for BOTH so we shoot ~half on
        # each side; cap at the available extension slots so we never
        # try to shoot more probes than the extension has room for.
        if result.peak_side == "both":
            probe_per_side = min(
                max(1, configured_probe_count // 2 + 1), extension_steps
            )
            sides = [("end_side", probe_per_side), ("start_side", probe_per_side)]
        else:
            probe_per_side = min(configured_probe_count, extension_steps)
            sides = [(result.peak_side, probe_per_side)]

        # Walk through each requested side, shooting probes at
        # grid-aligned step indices. Probes use the same filename
        # convention as the rest of the sweep — sweep, probe, and
        # (if it follows) extension are distinguished only by the
        # phase column in shots.csv.
        current_pos = (start_step_index + steps - 1) * step_size_mm  # rail at end of sweep
        probe_data: list = []
        probe_positions_per_side: dict[str, list[float]] = {"end_side": [], "start_side": []}
        # Total probes across all sides (the progress bar counts
        # rail steps, not aperture-shots, mirroring the main-sweep
        # bar's granularity).
        probe_total = sum(c for _s, c in sides)
        probe_done = 0
        for side, count in sides:
            for i in range(count):
                step_index = self._extension_step_index(
                    side=side,
                    original_steps=steps,
                    offset_in_extension=i,
                    start_step_index=start_step_index,
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
                    progress_ctx=("probe", probe_done, probe_total),
                )
                probe_done += 1
                probe_positions_per_side[side].append(target)
        self._progress_done()

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
        shot_so_far_per_side: dict[str, int] = {"end_side": 0, "start_side": 0}
        for side, count in sides:
            shot_so_far_per_side[side] = count
        extension_step_count_per_side: dict[str, int] = {"end_side": 0, "start_side": 0}
        extend_rounds = 0
        max_extend_rounds = int(getattr(self.config, "MAX_EXTEND_ROUNDS", 8))

        # Extend-more loop: as long as the data still doesn't bracket
        # the peak, give the user the choice to extend further by one
        # more standard increment, accept what we have, or abort.
        while not result2.peak_in_range_after_probe:
            # Re-derive the failing side from the *latest* result so
            # the next round extends the side that's currently
            # truncated, not whichever was failing on the first call.
            # The original first-call ``sides`` was for probing —
            # after one or more extend rounds, the truncated side may
            # have flipped (e.g. once we capture the centre peak,
            # corners on the opposite side become the failing edges).
            current_peak_side = result2.peak_side or result.peak_side
            if current_peak_side == "both":
                current_sides = [
                    ("end_side", extension_steps),
                    ("start_side", extension_steps),
                ]
            elif current_peak_side in ("end_side", "start_side"):
                current_sides = [(current_peak_side, extension_steps)]
            else:
                # peak_side is "centered" or None: no aperture is
                # truncated, so extending rail will not help. The
                # advisor returned not-OK because some apertures are
                # OFF_TARGET (every valid crop flat / cross-compare
                # rejected) — those crops fell off the wafer entirely.
                # Rail rail extension can't conjure a peak that isn't
                # there; the cure is a different jog / target.
                # Exit cleanly instead of falling through and burning
                # extension rounds indefinitely.
                self._announce(
                    f"Range check on {slug}: peak_side="
                    f"{current_peak_side!r} but verdict not OK — "
                    "no apertures are truncated, so rail extension "
                    "won't help. Likely cause: one or more apertures' "
                    "valid crops were classified OFF_TARGET. "
                    "Continuing with the data we have."
                )
                write_json(lens_dir / "range_check.json", {
                    "ok": False,
                    "extension_applied": extend_rounds > 0,
                    **record_base,
                    "extension_step_count": extension_step_count_per_side,
                    "extend_rounds": extend_rounds,
                    "notes": (
                        f"peak_side={current_peak_side!r} after probe "
                        f"(round {extend_rounds}); no aperture is "
                        "truncated, but verdict is not OK because at "
                        "least one aperture's crops are OFF_TARGET. "
                        "Rail extension cannot recover off-wafer "
                        "crops; operator review recommended. "
                        f"Advisor notes: {result2.notes}"
                    ),
                })
                return self._steps_from_zero_at(current_pos, step_size_mm)
            # Send ntfy on every prompt round, not just the first —
            # the user may be away from the rig and only check their
            # phone periodically; missing the second prompt because
            # we suppressed it would block progress indefinitely.
            if self.ntfy is not None:
                round_label = (
                    "still inadequate"
                    if extend_rounds == 0
                    else f"still inadequate after {extend_rounds} extend round(s)"
                )
                self.ntfy.with_body(
                    Notification(
                        title=f"Range {round_label}",
                        body="",
                        priority="high",
                        tags=("warning",),
                    ),
                    f"{slug}: probe on {current_peak_side} side did not "
                    "bracket peak.",
                )
            if extend_rounds >= max_extend_rounds:
                # Stop the bleeding: parabolic extrapolations on
                # half-bell tails are unreliable, and a stuck loop
                # will burn rail and flash power without converging.
                # Fall through to the same json-write the user-skip
                # path uses so the operator can investigate offline.
                self._announce(
                    f"Hit MAX_EXTEND_ROUNDS={max_extend_rounds}; "
                    "stopping extend loop. Use a wider START / END "
                    "next time, or sample the missing tail manually."
                )
                write_json(lens_dir / "range_check.json", {
                    "ok": False,
                    "extension_applied": True,
                    **record_base,
                    "extension_step_count": extension_step_count_per_side,
                    "extend_rounds": extend_rounds,
                    "notes": (
                        "advisor did not converge after "
                        f"{max_extend_rounds} extend rounds; "
                        "edge crops likely have peaks past the parabolic "
                        "extrapolation limit. Operator review required."
                    ),
                })
                return self._steps_from_zero_at(current_pos, step_size_mm)
            extend_mm_per_side = extension_steps * abs(step_size_mm)
            sides_label = ", ".join(s for s, _ in current_sides)
            choice = self.prompt_range_inadequate(
                f"\nRange still inadequate on {slug} "
                f"(peak_side={current_peak_side}). "
                f"[ENTER] extend by ~{extend_mm_per_side:.1f} mm "
                f"({extension_steps} steps) on {sides_label}, "
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
            # slots on each currently-failing side. Use the
            # re-derived ``current_sides`` so we extend toward
            # whichever side the latest evaluate flagged.
            ext_total = extension_steps * len(current_sides)
            ext_done = 0
            for side, _ in current_sides:
                for i in range(extension_steps):
                    offset = shot_so_far_per_side[side] + i
                    step_index = self._extension_step_index(
                        side=side, original_steps=steps,
                        offset_in_extension=offset,
                        start_step_index=start_step_index,
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
                        progress_ctx=(
                            f"extend r{extend_rounds + 1}", ext_done, ext_total,
                        ),
                    )
                    ext_done += 1
                    extension_step_count_per_side[side] += 1
                shot_so_far_per_side[side] += extension_steps
            self._progress_done()
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
        fill_total = sum(
            max(0, extension_steps - shot_so_far_per_side[s]) for s, _ in sides
        )
        fill_done = 0
        for side, _ in sides:
            for i in range(shot_so_far_per_side[side], extension_steps):
                step_index = self._extension_step_index(
                    side=side, original_steps=steps, offset_in_extension=i,
                    start_step_index=start_step_index,
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
                    progress_ctx=("extend fill", fill_done, fill_total)
                        if fill_total > 0 else None,
                )
                fill_done += 1
                extension_step_count_per_side[side] += 1
                shot_so_far_per_side[side] += 1
        if fill_total > 0:
            self._progress_done()

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
                "end_side": dict(sides).get("end_side", 0),
                "start_side": dict(sides).get("start_side", 0),
            },
            "notes": result2.notes,
        })
        return self._steps_from_zero_at(current_pos, step_size_mm)

    @staticmethod
    def _extension_step_index(
        *,
        side: str,
        original_steps: int,
        offset_in_extension: int,
        start_step_index: int = 0,
    ) -> int:
        """Map (side, position-within-extension) to a signed step
        index. ``end_side`` continues past the just-completed sweep
        (step indices > original_steps); ``start_side`` descends
        below the sweep start (negative step indices).

        ``start_step_index`` is 0 for the user-jog flow and the
        phase-C window's start step for the probe flow — extensions
        always anchor to the sweep's actual rail-zero-relative start
        and end, never to step 0.
        """
        if side == "end_side":
            return start_step_index + original_steps + offset_in_extension
        # side == "start_side"
        return start_step_index - 1 - offset_in_extension

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
        progress_ctx: tuple[str, int, int] | None = None,
    ) -> None:
        """Capture one frame per aperture at the current rail position.

        Used by the probe + extension shooters (single replica). The
        main-sweep N-replica path goes through ``_capture_one``.

        ``progress_ctx`` is ``(phase_label, step_in_phase,
        total_steps_in_phase)`` — when provided, the runner shows a
        live progress bar matching the main-sweep one but counts
        within the probe/extension/etc. block.
        """
        for ap_index, aperture in enumerate(apertures):
            if not self._should_capture(aperture, step_index):
                # Off this aperture's subsample grid — silent skip.
                continue
            cal = calibration.per_aperture[aperture]
            self._set_capture_settings(aperture, cal)
            if progress_ctx is not None:
                phase_label, step_in_phase, total_in_phase = progress_ctx
                self._progress(
                    step_index=step_in_phase,
                    step_count=total_in_phase,
                    position_mm=position_mm,
                    aperture=aperture,
                    aperture_index=ap_index,
                    aperture_count=len(apertures),
                    phase_label=phase_label,
                    show_eta=False,
                )
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
        progress_ctx: tuple[int, int, int] | None = None,
        progress_step_display: int | None = None,
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
        step_count, ap_index, ap_count = progress_ctx or (0, 0, 0)
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
        is_reshoot = phase == Phase.RESHOOT
        phase_label = "RESHOOT" if is_reshoot else "step"
        bar_step_index = (
            progress_step_display
            if progress_step_display is not None
            else step_index
        )
        # ETA only meaningful during the planned sweep; reshoot is
        # off-script and adds shots the grid math doesn't account for.
        for replica in range(n):
            if progress_ctx is not None:
                if not is_reshoot:
                    self._eta_record_shot()
                self._progress(
                    step_index=bar_step_index,
                    step_count=step_count,
                    position_mm=position_mm,
                    aperture=aperture,
                    aperture_index=ap_index,
                    aperture_count=ap_count,
                    replica_index=replica,
                    replica_count=n,
                    phase_label=phase_label,
                    show_eta=not is_reshoot,
                )
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

    def _make_droop_detector(self) -> DroopDetector:
        """Return ``self.droop_detector`` if one was injected (tests),
        otherwise build a fresh one from the config-driven window +
        trip criteria. Used wherever the runner needs a detector
        instance — phase A, the main sweep, etc."""
        return self.droop_detector or DroopDetector(
            window=self.config.ROLLING_WINDOW,
            criteria=(self.config.DROOP_CRIT_A, self.config.DROOP_CRIT_B),
        )

    def _droop_alert_and_pause(self, event: DroopEvent) -> None:
        """Send the ntfy notification, play the local chirp, and
        prompt the user to swap batteries. Common to droop trips
        from any phase (sweep, reshoot, probe).

        Raises :class:`SessionAborted` if the user picks ``q``.
        """
        if self.ntfy is not None:
            self.ntfy.with_body(
                BATTERY_DROOP,
                f"{event.aperture}: {event.triggered_count} of last "
                f"{len(event.window_snapshot)} shots ≤ {event.threshold:+.1f} EV. "
                f"Window: {event.window_snapshot!r}",
            )
        wav_path = getattr(self.config, "ALERT_WAV", None)
        if wav_path is not None:
            try:
                play_alert(wav_path)
            except Exception:                            # noqa: BLE001
                log.exception("alert sound failed")
        ok = self.confirm(
            f"\nBattery droop on {event.aperture}. "
            "Swap flash batteries, wait for steady ready light, "
            "then press ENTER to resume. q = abort: "
        )
        if not ok:
            raise SessionAborted("user aborted at droop pause")
        # Don't let the multi-minute battery swap poison the ETA EWMA.
        self._eta_invalidate()

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
        self._droop_alert_and_pause(event)
        # Find the earliest step (across all apertures) whose EV ≤
        # the firing trip threshold. ``window_snapshot`` is the
        # firing aperture's window only; ``earliest_underexposed_step``
        # consults all. We always reshoot at least the current step:
        # the trip means we just captured an under-exposed frame
        # here, and leaving it on disk would pollute the sharpness
        # dataset. If the window also contains earlier under-exposed
        # steps (typical of moderate trips that drift in over a few
        # shots), rewind further to cover those too.
        restart_step = droop.earliest_underexposed_step(event.threshold)
        if restart_step is None or restart_step > current_step_index:
            # Defensive: trip implies at least one push ≤ threshold,
            # so restart_step should always be <= current_step_index.
            # Fall back to current step if the search came up empty.
            restart_step = current_step_index
        # Reset the droop window before reshooting so the nested
        # reshoot starts from a clean slate; if its captures are
        # still under-exposed the new samples will trip droop again
        # and recurse further.
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
        # Reshoot now always covers the current step's apertures,
        # so the caller can skip the rest of its inner aperture
        # loop unconditionally.
        return True

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
            self.range_checker.set_subsample_factors(
                self._float_keyed_subsample_factors()
            )
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
        reshoot_step_count = through_step - from_step + 1
        for step_index in range(from_step, through_step + 1):
            if step_index > from_step:
                self.rail.move_and_wait(
                    sweep_direction, abs(step_size_mm), backlash=False
                )
            position_mm = step_index * step_size_mm
            skip_remainder = False
            for ap_index, aperture in enumerate(apertures):
                if skip_remainder:
                    # An inner reshoot already covered this step.
                    continue
                if not self._should_capture(aperture, step_index):
                    # Off this aperture's subsample grid — original
                    # sweep skipped it, so reshoot mirrors that.
                    continue
                cal = calibration.per_aperture[aperture]
                # progress_ctx uses reshoot-local step indexing so the
                # bar fills across the reshoot range, not the sweep
                # range. _capture_one labels these "RESHOOT" and
                # suppresses the ETA (reshoots are off-script).
                ev, event = self._capture_one(
                    lens_dir=lens_dir,
                    slug=slug,
                    step_index=step_index,
                    position_mm=position_mm,
                    aperture=aperture,
                    cal=cal,
                    phase=Phase.RESHOOT,
                    droop=droop,
                    progress_ctx=(
                        reshoot_step_count, ap_index, len(apertures),
                    ),
                    progress_step_display=step_index - from_step,
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
        # Newline after the reshoot's in-place progress bar so the
        # next sweep update doesn't write back over the final reshoot
        # line on the same row.
        self._progress_done()

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
                magnification_indicated=self._current_magnification_indicated,
                f_ring=self._current_f_ring,
                focal_length_mm=self._current_focal_length_mm,
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
        """Configure aperture / ISO / FEC / EC for the next capture.

        Uses the camera's batched setter when available (one PTP
        round-trip; saves several seconds per shot on the Z7) and
        falls back to individual setters otherwise — the FakeCamera
        in the runner tests stays a drop-in.

        EC is set explicitly per-shot so different apertures' calibrated
        EC values are honoured during the sweep. Without this, sweep
        shots inherit whichever EC the camera was last left at (e.g.
        the calibrator's final aperture's EC, often 0), which makes
        any aperture that calibrated with non-zero EC fire ~that EV
        darker during the sweep and trip droop on every shot.
        """
        # In manual-aperture mode the lens diaphragm is mechanical;
        # the camera's aperture widget either errors out or is a no-op
        # for a non-CPU lens. Skip aperture-related camera writes and
        # only set ISO / FEC (FEC is inert in flash-M but the widget
        # still wants a value).
        if self.manual_aperture:
            self.camera.set_iso(cal.iso)
            self.camera.set_fec(cal.fec_ev)
            return
        batched = getattr(self.camera, "set_aperture_iso_fec", None)
        if batched is not None:
            batched(aperture=aperture, iso=cal.iso, fec=cal.fec_ev)
        else:
            self.camera.set_aperture(aperture)
            self.camera.set_iso(cal.iso)
            self.camera.set_fec(cal.fec_ev)
        set_ec = getattr(self.camera, "set_ec", None)
        if set_ec is not None:
            set_ec(getattr(cal, "ec_ev", 0.0))

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
                magnification_indicated=self._current_magnification_indicated,
                f_ring=self._current_f_ring,
                focal_length_mm=self._current_focal_length_mm,
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
        and a total against the unsampled baseline.

        In manual-aperture mode the displayed aperture is the *ring*
        f-number the operator dials (e.g. ``f/2.8``), not the
        rig-internal effective label. Effective apertures are
        capture-time metadata; the operator only cares about ring
        stops while planning.
        """
        replicas = max(1, int(plan.replicas_per_position))
        per_ap_lines: list[str] = []
        total_shots = 0
        # Manual mode: map each aperture-label (effective) to its
        # corresponding ring stop via the per-lens-config state.
        ring_for_label: dict[str, str] = {}
        if self.manual_aperture and self._current_ring_stops:
            for ap_label, ring in zip(apertures, self._current_ring_stops):
                ring_for_label[ap_label] = (
                    f"{int(round(ring))}" if abs(ring - round(ring)) < 0.05
                    else f"{ring:.1f}".rstrip("0").rstrip(".")
                )
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
            display = f"f/{ring_for_label[ap_str]}" if ap_str in ring_for_label else ap_str
            per_ap_lines.append(
                f"    {display:7s} ×{factor} → {shots} × {replicas} = {per_ap_total}"
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

    def _offer_jog_pause(self, *, where: str) -> None:
        """Optional interactive freeform jog at non-jog-and-plan
        moments — session start (before FEC sanity check) and
        before each lens-config prompt.

        ``self.jog_freeform`` is the injected freeform-jog callable;
        ``None`` disables the pause prompt entirely (test runners and
        any deployment that prefers strict prompts can leave it
        unset). The user types: empty/ENTER → continue, ``j`` → enter
        jog mode (then re-prompt on exit), ``q`` → abort the session.
        """
        if self.jog_freeform is None:
            return
        while True:
            try:
                ans = input(
                    f"\n[ENTER] continue ({where}), "
                    "[j] jog rail (verify focus / framing), "
                    "[q] abort: "
                ).strip().lower()
            except EOFError:
                return
            if ans == "" or ans == "y":
                return
            if ans == "q":
                raise SessionAborted(f"user aborted at {where}")
            if ans == "j":
                try:
                    self.jog_freeform(self.rail)
                except SessionAborted:
                    raise
                except Exception:                            # noqa: BLE001
                    log.exception("freeform jog raised")
                continue
            print(f"  unrecognised input {ans!r}", flush=True)

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

    # ---- ETA bookkeeping ----------------------------------------------------

    _ETA_OUTLIER_S = 30.0
    """Intervals longer than this are dropped from the EWMA. Real
    inter-shot intervals on this rig top out around 7 s (small
    aperture + recycle + replica). 30 s is comfortably above any
    legitimate gap and well below any user-prompt pause."""

    def _eta_record_shot(self) -> None:
        """Accumulate one inter-shot interval into the running mean.

        We use a cumulative mean over the current lens config rather
        than an EWMA because the per-shot interval distribution on
        this rig is bimodal (short intra-replica intervals vs longer
        post-flash recycle / aperture-change intervals), and the
        sweep cycles through both deterministically. A fast-decaying
        EWMA (α = 0.2 was the prior setting) over-weights whichever
        type of interval came in last, making the displayed ETA
        oscillate by ~25 % between consecutive shots even though the
        underlying schedule is steady. The cumulative mean averages
        both interval types in proportion to how often they actually
        occur, which is what the user wants for "how long until
        done?".

        The sum and count both reset at the start of each sweep
        (i.e. on every new lens config), so the average reflects the
        current setup's pace rather than carrying state from a
        previous config with different recycle settings.
        """
        now = self.monotonic()
        if self._eta_last_shot_at is not None:
            interval = now - self._eta_last_shot_at
            if 0 < interval < self._ETA_OUTLIER_S:
                self._eta_lens_interval_sum += interval
                self._eta_lens_interval_count += 1
        self._eta_last_shot_at = now

    @property
    def _eta_avg_interval_s(self) -> float | None:
        """Mean inter-shot interval in seconds for the current lens
        config, or ``None`` when no in-flight interval has been
        collected yet."""
        if self._eta_lens_interval_count <= 0:
            return None
        return self._eta_lens_interval_sum / self._eta_lens_interval_count

    def _eta_invalidate(self) -> None:
        """Discard the next interval — call after every user prompt
        so a battery-swap pause doesn't poison the average. Sum and
        count are preserved (the prior shots' data stays valid)."""
        self._eta_last_shot_at = None

    def _eta_remaining_shots(
        self, *, step_index: int, aperture_index: int, replica_index: int,
        replica_count: int,
    ) -> int:
        """Count sweep shots still to take, given the current position
        in the (step, aperture, replica) grid. Honours subsample
        factors so a wide-DOF aperture with factor 4 only contributes
        every fourth step."""
        apertures = self._eta_sweep_apertures
        step_count = self._eta_sweep_step_count
        if not apertures or step_count <= 0:
            return 0
        factors = self._subsample_factors
        remaining = 0
        # Remaining replicas of the current (step, aperture).
        remaining += max(0, replica_count - replica_index - 1)
        # Remaining apertures of the current step.
        for j, ap in enumerate(apertures):
            if j <= aperture_index:
                continue
            if step_index % factors.get(ap, 1) == 0:
                remaining += replica_count
        # All future steps in full.
        for s in range(step_index + 1, step_count):
            for ap in apertures:
                if s % factors.get(ap, 1) == 0:
                    remaining += replica_count
        return remaining

    def _format_eta(self, seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        if seconds < 3600:
            return f"{seconds // 60}:{seconds % 60:02d}"
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    def _progress(
        self,
        *,
        step_index: int,
        step_count: int,
        position_mm: float,
        aperture: str,
        aperture_index: int,
        aperture_count: int,
        replica_index: int | None = None,
        replica_count: int | None = None,
        phase_label: str = "step",
        show_eta: bool = True,
    ) -> None:
        """Single-line progress display, updated in place via \\r.

        Format without replicas:
            ``[████░░░░] step 3/10  pos +1.20mm  ap f/8 (3/5)``
        With replicas:
            ``[████░░░░] step 3/10  pos +1.20mm  ap f/8 (3/5)  rep 2/3``

        Replica progress refines the bar fraction so it advances
        smoothly between replicas instead of jumping per (step,
        aperture). Cleared by :meth:`_progress_done`. No-op if
        stdout isn't a TTY."""
        if not sys.stdout.isatty():
            return
        bar_width = 20
        # +1 because step_index is 0-based and we want progress to
        # increment AS the step starts, not just after.
        if replica_index is not None and replica_count:
            ap_progress = (
                aperture_index + (replica_index + 1) / replica_count
            ) / aperture_count
        else:
            ap_progress = (aperture_index + 1) / aperture_count
        fraction = (step_index + ap_progress) / step_count
        filled = int(round(fraction * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        line = (
            f"\r[{bar}] {phase_label} {step_index + 1}/{step_count}  "
            f"pos {position_mm:+7.2f} mm  "
            f"ap {aperture} ({aperture_index + 1}/{aperture_count})"
        )
        if replica_index is not None and replica_count:
            line += f"  rep {replica_index + 1}/{replica_count}"
        avg_interval = self._eta_avg_interval_s
        if (
            show_eta
            and avg_interval is not None
            and replica_index is not None
            and replica_count
            and self._eta_sweep_step_count
        ):
            remaining = self._eta_remaining_shots(
                step_index=step_index,
                aperture_index=aperture_index,
                replica_index=replica_index,
                replica_count=replica_count,
            )
            line += f"  eta {self._format_eta(remaining * avg_interval)}"
        sys.stdout.write(line + "\x1b[K")
        sys.stdout.flush()

    def _progress_done(self) -> None:
        """Newline at end of sweep so the next print() doesn't
        overwrite the final progress line."""
        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _calibration_progress(
        self,
        *,
        aperture: str,
        aperture_index: int,
        aperture_count: int,
        shot_index: int,
        shot_count_max: int,
        fec: float,
        ec: float,
        iso,
        last_ev: float | None,
        label: str,
    ) -> None:
        """Inline progress bar for the AA calibration cascade.

        Format::

            [████░░░░░░░░░░░░░░░░] cal 1/5(probe) ap f/5.6 (1/4) \
            FEC=+0.00 EC=+0.00 ISO=100 ev=+1.62

        The bar fraction advances smoothly within each aperture's
        shot sequence (shot_index / shot_count_max) and jumps cleanly
        when the next aperture starts. ``label`` is the cascade's
        own description of what this shot is for ("probe", "target",
        "headroom probe", "fine-tune", "post ISO bump", "refine
        post-ISO"); the renderer compresses it into a short tag.
        ``last_ev`` is ``None`` before the first shot.

        No-op when stdout isn't a TTY (e.g. logs / CI).
        """
        if not sys.stdout.isatty():
            return
        bar_width = 20
        fraction = (
            aperture_index + (shot_index / max(1, shot_count_max))
        ) / max(1, aperture_count)
        fraction = max(0.0, min(1.0, fraction))
        filled = int(round(fraction * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        # Compress the cascade's "linear shot N (label)" into a short
        # tag: ``probe``/``target``/``headroom``/``fine``/``iso``/``refine``.
        # Falls back to the raw label if no keyword matches.
        tag = label
        for keyword, short in (
            ("probe FEC=0", "probe"),
            ("target", "target"),
            ("headroom", "headroom"),
            ("fine-tune", "fine"),
            ("post ISO bump", "iso-bump"),
            ("refine post-ISO", "refine"),
            ("baseline", "baseline"),
        ):
            if keyword in label.lower():
                tag = short
                break
        ec_str = f"EC={ec:+.2f} " if abs(ec) > 1e-6 else ""
        ev_str = f"ev={last_ev:+.2f}" if last_ev is not None else "ev=    "
        line = (
            f"\r[{bar}] cal {shot_index + 1}/{shot_count_max}({tag})  "
            f"ap {aperture} ({aperture_index + 1}/{aperture_count})  "
            f"FEC={fec:+.2f} {ec_str}ISO={iso}  {ev_str}"
        )
        sys.stdout.write(line + "\x1b[K")
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

    CLI flags let the operator skip interactive prompts for headless /
    tail-extension runs:
    ``--no-probe`` overrides ``config.PROBE_ENABLED`` for this run.
    ``--lens`` / ``--magnification`` substitute the lens-config and
    magnification prompts. ``--rail-start`` / ``--rail-end`` (both
    required together when used) substitute the START / END jog with
    programmatic rail moves — interpreted as signed mm offsets from
    the rail's current position when the CLI launches. The typical
    tail-extension recipe is to manually position the rail at the
    intended START and pass ``--rail-start 0 --rail-end <signed mm>``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="macro-lens-test",
        description=(
            "Run a macro-lens sharpness sweep. With no flags the runner "
            "prompts interactively for lens config, magnification, and "
            "rail START / END. CLI flags below skip the matching prompt."
        ),
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help=(
            "Disable the Phase-A probe for this run. Equivalent to "
            "config.PROBE_ENABLED = False but scoped to this invocation. "
            "Use for tail-extension sessions where the user-jogged range "
            "intentionally lies outside the widest-aperture peak."
        ),
    )
    parser.add_argument(
        "--lens",
        metavar="TEXT",
        help=(
            "Lens config string (skips the lens prompt). Should match "
            "the verbatim text used in past sessions so the slug stays "
            "consistent for offline analysis."
        ),
    )
    parser.add_argument(
        "--magnification",
        type=float,
        metavar="M",
        help=(
            "Magnification ratio (skips the magnification prompt). "
            "E.g. 1.0 for 1:1, 0.5 for 1:2."
        ),
    )
    parser.add_argument(
        "--rail-start",
        type=float,
        metavar="MM",
        help=(
            "Signed mm offset from the rail's current hardware position "
            "to the session START. The rail moves there, and that "
            "position becomes session-zero. Usually 0 (use the rail's "
            "current position as START). Required with --rail-end."
        ),
    )
    parser.add_argument(
        "--rail-end",
        type=float,
        metavar="MM",
        help=(
            "Signed mm offset from session START to session END. "
            "Negative for backward sweeps (the common case on this rig). "
            "Required with --rail-start."
        ),
    )
    parser.add_argument(
        "--manual-aperture",
        action="store_true",
        help=(
            "Enable manual-aperture mode for non-CPU lenses (AI-S, "
            "pre-AI). Inverts the capture loop (one rail sweep per "
            "ring turn), prompts for magnification_indicated and "
            "the lens's ring max aperture (parsed from slug when "
            "possible), and switches the calibration cascade to "
            "ISO-only (flash must be in Manual mode on the SB-5000 "
            "commander since AA/TTL flash can't communicate aperture "
            "for non-CPU lenses on Z7). Phase A sharpness probe runs "
            "at every aperture (not just widest). Per-session flag — "
            "mixing CPU and non-CPU lenses needs two sessions."
        ),
    )
    args = parser.parse_args()

    if (args.rail_start is None) != (args.rail_end is None):
        parser.error(
            "--rail-start and --rail-end must be supplied together or "
            "both omitted (use --rail-start 0 for 'zero at current "
            "rail position')"
        )

    import config
    from camera.nikon_z7 import NikonZ7
    from notify.ntfy import NtfyClient
    from sharpness.collector import LiveCollector
    from sharpness.range_check import SharpnessRangeChecker
    from stackshot.controller import open_ftdi as open_stackshot
    from stackshot.jog import jog_start_and_end as _jog
    from stackshot.jog import jog_interactive as _jog_interactive

    if args.no_probe:
        # Per-invocation override; tail-extension sessions need Phase A
        # off because the user-jogged window is intentionally outside
        # the widest-aperture peak (cf. DESIGN_NOTES §6.28).
        config.PROBE_ENABLED = False

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
            executor_kind="process",
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

    def _freeform_jog(rail) -> float:
        """Single-position freeform jog used by _offer_jog_pause to
        let the user verify focus / framing or release the
        StackShot from USB-control mode (SPACE → panel-control,
        ENTER → re-acquire). Same camera release/reacquire wiring
        as the start/end jog so live view works during the SPACE
        pause."""
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

        return _jog_interactive(
            rail,
            label="Verify focus / framing — press 'e' when done",
            set_key="e",
            on_release=_release_camera,
            on_reacquire=_reacquire_camera,
        )

    # Build CLI-driven prompt / jog overrides. Each override is opt-in:
    # only the flags the user supplies replace the corresponding
    # interactive prompt; everything else stays interactive.
    runner_overrides: dict = {}
    if args.lens is not None:
        # One-shot: return the lens text on the first call, then None
        # so SessionRunner.run()'s while-True loop exits cleanly after
        # one lens config (no infinite re-prompting for headless runs).
        _lens_emitted = [False]
        def _cli_prompt_lens_config() -> str | None:
            if _lens_emitted[0]:
                return None
            _lens_emitted[0] = True
            return args.lens
        runner_overrides["prompt_lens_config"] = _cli_prompt_lens_config
        # If a reshoot plan is loaded, the plan-menu prompt fires
        # before prompt_lens_config — short-circuit it by always
        # returning OTHER so the runner falls through to the CLI
        # lens-config one-shot above.
        runner_overrides["prompt_lens_config_from_plan"] = (
            lambda entries, completed: OTHER
        )
    if args.magnification is not None:
        runner_overrides["prompt_magnification"] = lambda: args.magnification
    if args.rail_start is not None:
        # Programmatic two-leg jog. Leg 1 moves to the operator's
        # intended START position (relative to wherever the rail
        # happens to be at CLI launch); that becomes session-zero
        # implicitly because :meth:`_run_one_lens_config` reads the
        # returned end_mm as the signed offset from session-zero.
        # Leg 2 walks to END. The runner then returns the rail to
        # zero before starting the sweep, same as the interactive
        # path.
        def _cli_jog(rail_arg) -> JogResult:
            if args.rail_start != 0:
                direction = (
                    Direction.FORWARD if args.rail_start > 0
                    else Direction.BACKWARD
                )
                rail_arg.move_and_wait(
                    direction, abs(args.rail_start), backlash=True
                )
            if args.rail_end != 0:
                direction = (
                    Direction.FORWARD if args.rail_end > 0
                    else Direction.BACKWARD
                )
                rail_arg.move_and_wait(
                    direction, abs(args.rail_end), backlash=True
                )
            return JogResult(end_mm=args.rail_end)
        runner_overrides["jog"] = _cli_jog
    else:
        runner_overrides["jog"] = _camera_jog

    runner = SessionRunner(
        camera=camera,
        rail=rail,
        config=config,
        ntfy=ntfy,
        range_checker=range_checker,
        output_root=config.OUTPUT_ROOT,
        jog_freeform=_freeform_jog,
        sharpness_collector_factory=_sharpness_factory,
        manual_aperture=args.manual_aperture,
        **runner_overrides,
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


