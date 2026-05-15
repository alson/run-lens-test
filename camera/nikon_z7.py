"""Nikon Z7 wrapper around python-gphoto2.

The patterns here are lifted from ``tools/z7_ec_test.py``, which has
been proven on the rig (drain-events-before-capture, transient ``'?'``
property tolerance, autofocus-toggle recovery on capture failure).

The class accepts a ``gp_module`` and ``sleep`` callable for testing
so unit tests don't need a real camera.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Callable, Sequence

log = logging.getLogger(__name__)

import gphoto2 as gphoto2_module

from camera.widgets import (
    count_widgets,
    find,
    first_matching_choice,
    widget_choices,
)
from errors import (
    CameraCaptureError,
    CameraStateError,
    FECSanityError,
)


# ---- Module constants used by tests + production ---------------------------

EXPPROGRAM_REQUIRED = "M"

# Z7 firmware variants expose VR under several names. Try them in order
# and use the first widget that exists. Lifted from
# tools/z7_settings_and_fec_test.py.
VR_WIDGET_CANDIDATES: tuple[str, ...] = (
    "vr",
    "vrenable",
    "vibrationreduction",
    "vibrationreductiontype",
    "imagestabilization",
)
VR_OFF_CHOICES: tuple[str, ...] = ("Off", "OFF", "off", "Disabled", "0")


def _format_ec_label(ev: float) -> str:
    """Z7 widget label for an EC value in EV (⅓-stop grid).

    The Z7 EC widget reports its choices as **unsigned numeric**
    strings, with ⅓-stop fractions truncated to three decimals:
    ``"0"``, ``"1"``, ``"3"``, ``"1.333"``, ``"-1"``, ``"-1.666"``.
    Note the truncation: 2/3 → ``"0.666"`` (NOT ``"0.667"``);
    formatting with ``f"{2/3:.3f}"`` rounds up and produces a label
    that's not in the choice list, so we hard-code the two
    truncated thirds.
    """
    if abs(ev) < 1e-9:
        return "0"
    sign = "-" if ev < 0 else ""
    a = abs(ev)
    whole = int(a)
    third_idx = round((a - whole) * 3)  # 0, 1, 2, or 3
    if third_idx == 3:
        whole += 1
        third_idx = 0
    if third_idx == 0:
        return f"{sign}{whole}"
    if third_idx == 1:
        return f"{sign}{whole}.333"
    return f"{sign}{whole}.666"


def _is_question_mark(value: object) -> bool:
    """Detect the transient ``'?'`` value the Z7 sometimes returns
    immediately after a burst of DevicePropChanged events."""
    return isinstance(value, str) and value == "?"


class NikonZ7:
    """Persistent gphoto2 connection to the Nikon Z7.

    A single instance is opened at session start, kept for the entire
    session, and reopened only on lens-change battery-pull recovery.
    Shelling out per call to the gphoto2 CLI re-handshakes PTP and
    sometimes lands the Z7 in the cut-down 23-property MTP tree —
    don't.
    """

    def __init__(
        self,
        *,
        gp_module=gphoto2_module,
        sleep: Callable[[float], None] = time.sleep,
        question_mark_retries: int = 5,
        question_mark_retry_sleep_s: float = 0.2,
    ) -> None:
        self._gp = gp_module
        self._sleep = sleep
        self._qm_retries = question_mark_retries
        self._qm_sleep = question_mark_retry_sleep_s
        self._camera = None

    # ---- Lifecycle --------------------------------------------------------

    def open(self) -> None:
        if self._camera is not None:
            return
        camera = self._gp.Camera()
        camera.init()
        self._camera = camera

    def close(self) -> None:
        """Release the camera handle.

        Tolerates the camera already being gone from USB — that's the
        normal case during lens-change recovery, where the user is
        instructed to power the camera off before pressing ENTER.
        ``self._camera.exit()`` then raises GPhoto2Error -52 ("Could
        not find the requested device on the USB port"); we log it
        and clear the handle so subsequent ``open()`` can run a clean
        reconnect."""
        if self._camera is None:
            return
        try:
            self._camera.exit()
        except self._gp.GPhoto2Error as exc:                # noqa: BLE001
            log.info("camera.exit() failed (likely powered off): %s", exc)
        finally:
            self._camera = None

    @property
    def camera(self):
        if self._camera is None:
            raise RuntimeError("NikonZ7 used before open()")
        return self._camera

    @property
    def is_open(self) -> bool:
        """True between :meth:`open` and :meth:`close`. Lets callers
        guard ``close()`` so a double-close (e.g. ``stop()`` after an
        aborted lens change that already released the handle) is a
        no-op instead of triggering the GPhoto2 -52 INFO log."""
        return self._camera is not None

    # ---- Health check ----------------------------------------------------

    def verify_healthy(self, *, min_widgets: int = 100) -> None:
        """Confirm the camera is up in full Nikon-PTP mode and in M mode.

        Raises
        ------
        CameraStateError
            If the config tree is below ``min_widgets`` (camera came
            up in minimal-MTP mode) or ``expprogram`` is not ``M``.
        """
        cfg = self.camera.get_config()
        n = count_widgets(cfg)
        if n < min_widgets:
            raise CameraStateError(
                f"config tree has {n} widgets (< {min_widgets}). Camera is "
                "in minimal-MTP mode. Power off, REMOVE BATTERY 10s, reinsert, "
                "power on, wait for live view, THEN connect USB."
            )
        ep_value = self._read_property_with_qm_tolerance("expprogram")
        if ep_value != EXPPROGRAM_REQUIRED:
            raise CameraStateError(
                f"expprogram is {ep_value!r}; mode dial must be on M "
                "(or U2 if U2 is configured for M)."
            )
        # Focus mode must be Manual. The widget is readonly via gphoto2
        # (Z lenses are focus-by-wire and the camera body's focus mode
        # isn't writable over PTP), so we can only detect the wrong
        # state and tell the user to fix it on the camera. autofocus=
        # Off doesn't fully suppress AF-S behaviour during capture.
        fm_value = self._read_property_with_qm_tolerance("focusmode")
        if fm_value and fm_value != "Manual":
            raise CameraStateError(
                f"focusmode is {fm_value!r}; set the camera to Manual "
                "Focus before starting (focus-mode button + dial). "
                "AF-S/AF-C will refocus on each capture and ruin the "
                "focus-bracketed sweep."
            )

    def _read_property_with_qm_tolerance(self, name: str) -> str:
        """Re-fetch a property up to ``question_mark_retries`` times if
        gphoto2 returns ``'?'`` (transient post-DevicePropChanged state)."""
        last = "?"
        for _ in range(self._qm_retries):
            cfg = self.camera.get_config()
            widget = find(cfg, name)
            if widget is None:
                return ""
            value = widget.get_value()
            if not _is_question_mark(value):
                return value
            last = value
            self._sleep(self._qm_sleep)
        return last

    # ---- Apply session config -------------------------------------------

    def apply_session_config(
        self,
        *,
        iso: str = "100",
        shutter_candidates: Sequence[str] = ("1/200", "0.0050s", "0.005s"),
        image_quality: str = "JPEG Fine",
        capture_target: str = "Internal RAM",
        autofocus: str = "Off",
        fec: float = 0.0,
        ec: float = 0.0,
        aperture: str | None = None,
    ) -> None:
        """Push the canonical per-session settings.

        Called after :meth:`open` and after every lens-change
        reconnect. Skips any setting whose widget isn't present (Z7
        firmware variants).

        EC is explicitly reset alongside FEC so a session never
        inherits a leftover positive EC from the camera's prior
        state — that bias would otherwise corrupt the FEC sanity
        check baseline and every calibration cascade in the session.
        """
        self._set_radio("capturetarget", [capture_target])
        self._set_radio("imagequality", [image_quality])
        self._set_radio("iso", [str(iso)])
        if aperture is not None:
            self._set_radio("f-number", [aperture])
        self._set_radio("shutterspeed", list(shutter_candidates))
        self._set_radio("autofocus", [autofocus])
        self._disable_vr()
        self._set_range("flashexposurecompensation", float(fec))
        self.set_ec(float(ec))

    def _disable_vr(self) -> None:
        """Best-effort VR disable across Z7 firmware variants.

        VR introduces shake-pattern variability that swamps the
        sharpness signal at slower shutter speeds. Iterate the known
        widget names; for the first one present, set it to an
        Off-like choice. Missing-widget and missing-Off-choice are
        silently skipped — VR off is a sharpness preference, not a
        correctness requirement, and Z firmware varies.
        """
        cfg = self.camera.get_config()
        for name in VR_WIDGET_CANDIDATES:
            widget = find(cfg, name)
            if widget is None:
                continue
            chosen = first_matching_choice(widget, VR_OFF_CHOICES)
            if chosen is None:
                return
            widget.set_value(chosen)
            self.camera.set_config(cfg)
            self.drain_events()
            return

    def _set_radio(self, name: str, candidates: Sequence[str]) -> None:
        cfg = self.camera.get_config()
        widget = find(cfg, name)
        if widget is None:
            return
        chosen = first_matching_choice(widget, candidates)
        if chosen is None:
            raise CameraStateError(
                f"none of {list(candidates)!r} are valid choices for {name!r} "
                f"(choices: {widget_choices(widget)[:8]!r})"
            )
        widget.set_value(chosen)
        self.camera.set_config(cfg)
        self.drain_events()

    def _set_range(self, name: str, value: float) -> None:
        # The Z7's PTP range widget for ``flashexposurecompensation``
        # has a quirky internal grid: different ⅓-stop positions
        # accept different float representations. Empirical map (from
        # tools/fec_acceptance_grid.py against the real camera):
        #
        #   k=-7 (−2.333): NO float64 representation accepted
        #   k=-5 (−1.667): 3-decimal & 4-decimal work; k/3 & float32 don't
        #   k=-4 (−1.333): k/3 & float32 work; 3-decimal & 4-decimal don't
        #   k=-9, -8, -6, -3, etc.: all forms work
        #
        # Sending an unaccepted form returns success from
        # ``set_config`` but the widget keeps its previous value — a
        # silent no-op. Subsequent captures fire at the OLD value
        # while the caller's bookkeeping says the new one.
        #
        # Defence: try several common rounding forms in order, with
        # readback verification between each. First accepted form
        # wins. Raises CameraStateError only if ALL candidates are
        # rejected (e.g. k=-7) so the caller can snap to a nearby
        # accepted grid point.
        import struct
        v = float(value)
        candidates: list[float] = []
        seen: set[int] = set()
        for c in (
            round(v, 3),
            round(v, 4),
            struct.unpack("<f", struct.pack("<f", v))[0],
            v,
        ):
            # De-dup by bit pattern so we don't try identical floats twice.
            bits = struct.unpack("<q", struct.pack("<d", c))[0]
            if bits in seen:
                continue
            seen.add(bits)
            candidates.append(c)

        last_actual: float | None = None
        for cand in candidates:
            cfg = self.camera.get_config()
            widget = find(cfg, name)
            if widget is None:
                return
            widget.set_value(cand)
            self.camera.set_config(cfg)
            self.drain_events()
            # Retry readback a few times — the PTP property update
            # propagates asynchronously and the camera may briefly
            # continue to report the old value for ~100–300 ms.
            for _ in range(4):
                cfg2 = self.camera.get_config()
                widget2 = find(cfg2, name)
                if widget2 is None:
                    return
                last_actual = float(widget2.get_value())
                if abs(last_actual - cand) <= 1e-3:
                    return
                self._sleep(0.2)
            # This candidate's silent no-op confirmed; try next form.

        raise CameraStateError(
            f"{name} set_value({value!r}) silently rejected by all "
            f"{len(candidates)} candidate representations "
            f"({[f'{c!r}' for c in candidates]}). Camera reports "
            f"{last_actual!r}. The widget's accepted grid excludes "
            f"this ⅓-stop position — see "
            f"tools/fec_acceptance_grid.py for the empirical map."
        )
        # Readback verification. The Z7's PTP range widget for some
        # properties (notably ``flashexposurecompensation``) silently
        # rejects values that don't match its internal ⅓-stop grid by
        # strict floating-point equality. ``set_config`` returns
        # success, no exception is raised, and the widget retains its
        # previous value — causing the next capture to fire at the
        # *old* setting while the caller's bookkeeping says the new
        # one. Catching that here surfaces the failure where it
        # happens instead of letting it propagate as a phantom flash
        # response. See ``_snap`` in session/calibrate.py for the
        # original 1-ULP bug; this readback is defence-in-depth for
        # any future caller that hands us a value off the widget's
        # grid.
        cfg2 = self.camera.get_config()
        widget2 = find(cfg2, name)
        if widget2 is None:
            return
        actual = float(widget2.get_value())
        if abs(actual - float(value)) > 1e-3:
            raise CameraStateError(
                f"{name} set_value({value!r}) silently rejected: "
                f"camera reports {actual!r} after set_config. Likely "
                f"a floating-point mismatch with the widget's discrete "
                f"grid — try snapping the requested value to canonical "
                f"⅓-stop form (k/3 rather than k * (1/3))."
            )

    # ---- Convenience setters --------------------------------------------

    def list_apertures(self) -> list[str]:
        cfg = self.camera.get_config()
        widget = find(cfg, "f-number")
        if widget is None:
            return []
        return widget_choices(widget)

    def set_aperture(self, fnum: str) -> None:
        self._set_radio("f-number", [fnum])

    def set_iso(self, iso: str | int) -> None:
        self._set_radio("iso", [str(iso)])

    def set_fec(self, ev: float) -> None:
        self._set_range("flashexposurecompensation", float(ev))

    def set_ec(self, ev: float) -> None:
        """Set camera exposure compensation (in EV).

        The Z7 exposes EC as a radio widget with unsigned numeric
        choices (``"0"``, ``"1"``, ``"1.333"``, ``"-1"``); some
        firmware variants may use a range widget. Try range first,
        otherwise look up the matching ⅓-stop label and raise
        :class:`CameraStateError` if neither the requested label
        nor the rounded fallback is in the widget's choice list —
        a silent no-op here previously masked a calibrator bug
        where every positive EC bump was being dropped."""
        cfg = self.camera.get_config()
        widget = find(cfg, "exposurecompensation")
        if widget is None:
            return
        try:
            wtype = widget.get_type()
        except Exception:                                # noqa: BLE001
            wtype = None
        gp = self._gp
        if wtype == getattr(gp, "GP_WIDGET_RANGE", None):
            widget.set_value(float(ev))
            self.camera.set_config(cfg)
            self.drain_events()
            return
        choices = widget_choices(widget)
        label = _format_ec_label(ev)
        if label not in choices:
            label = _format_ec_label(round(ev))
        if label not in choices:
            raise CameraStateError(
                f"EC label {_format_ec_label(ev)!r} (or rounded "
                f"{_format_ec_label(round(ev))!r}) not in "
                f"'exposurecompensation' choices {choices[:8]!r}…"
            )
        widget.set_value(label)
        self.camera.set_config(cfg)
        self.drain_events()

    def set_aperture_iso_fec(
        self, *, aperture: str, iso: str | int, fec: float
    ) -> None:
        """Set f-number, ISO, and FEC in a single get_config / set_config /
        drain cycle.

        Each individual setter on the Z7 costs ~1.5–3 s (one full PTP
        get_config + set_config + event drain). The hot sweep loop
        previously called three of them back-to-back; this batched
        version cuts that to one round-trip, which on a 250-shot
        session saves on the order of 10 minutes of wall-clock time.

        Missing widgets (rare across firmware variants) are silently
        skipped, matching :meth:`apply_session_config`.
        """
        cfg = self.camera.get_config()

        f_widget = find(cfg, "f-number")
        if f_widget is not None:
            chosen = first_matching_choice(f_widget, [aperture])
            if chosen is None:
                raise CameraStateError(
                    f"{aperture!r} is not a valid choice for 'f-number' "
                    f"(choices: {widget_choices(f_widget)[:8]!r})"
                )
            f_widget.set_value(chosen)

        iso_widget = find(cfg, "iso")
        if iso_widget is not None:
            chosen = first_matching_choice(iso_widget, [str(iso)])
            if chosen is None:
                raise CameraStateError(
                    f"{iso!r} is not a valid choice for 'iso' "
                    f"(choices: {widget_choices(iso_widget)[:8]!r})"
                )
            iso_widget.set_value(chosen)

        fec_widget = find(cfg, "flashexposurecompensation")
        if fec_widget is not None:
            fec_widget.set_value(float(fec))

        self.camera.set_config(cfg)
        self.drain_events()

    # ---- Capture --------------------------------------------------------

    def drain_events(self, timeout_ms: int = 500) -> int:
        """Drain pending events so the next capture returns a fresh
        frame, not a stale SDRAM buffer.

        Also explicitly deletes any ``GP_EVENT_FILE_ADDED`` entries
        from the camera filesystem so the SDRAM buffer doesn't
        accumulate.
        """
        gp = self._gp
        drained = 0
        while True:
            try:
                evt_type, evt_data = self.camera.wait_for_event(timeout_ms)
            except gp.GPhoto2Error:
                break
            if evt_type == gp.GP_EVENT_TIMEOUT:
                break
            drained += 1
            if evt_type == gp.GP_EVENT_FILE_ADDED:
                try:
                    self.camera.file_delete(evt_data.folder, evt_data.name)
                except gp.GPhoto2Error:
                    pass
            if drained > 20:
                break
        return drained

    def capture_to(self, dest: Path | str, *, max_retries: int = 3) -> int:
        """Capture a fresh frame and save it to ``dest``.

        Returns the on-disk byte count. Raises
        :class:`CameraCaptureError` if all retries fail.

        Implements the recovery pattern from ``z7_ec_test.py``:
        drain → capture → if it fails, drain again, sleep, optionally
        toggle autofocus, retry.
        """
        gp = self._gp
        dest = Path(dest)
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self.drain_events(timeout_ms=200)
                fp = self.camera.capture(gp.GP_CAPTURE_IMAGE)
                cf = self.camera.file_get(
                    fp.folder, fp.name, gp.GP_FILE_TYPE_NORMAL
                )
                cf.save(str(dest))
                try:
                    self.camera.file_delete(fp.folder, fp.name)
                except gp.GPhoto2Error:
                    pass
                return os.path.getsize(dest)
            except gp.GPhoto2Error as e:
                last_err = e
                # Try to settle whatever is pending, sleep, and on the
                # second retry nudge an unrelated config to unstick the
                # state machine.
                try:
                    self.drain_events(timeout_ms=1000)
                except Exception:
                    pass
                self._sleep(1.0 * attempt)
                if attempt == 2:
                    self._toggle_autofocus_for_recovery()
        raise CameraCaptureError(
            f"capture failed after {max_retries} attempts; last error: {last_err}"
        )

    def _toggle_autofocus_for_recovery(self) -> None:
        try:
            cfg = self.camera.get_config()
            widget = find(cfg, "autofocus")
            if widget is None:
                return
            current = widget.get_value()
            choices = widget_choices(widget)
            other = next((c for c in choices if c != current), None)
            if other is None:
                return
            widget.set_value(other)
            self.camera.set_config(cfg)
            self._sleep(0.3)
            widget.set_value(current)
            self.camera.set_config(cfg)
        except Exception:
            # Best-effort recovery only; never let recovery itself
            # raise on top of the original error.
            pass

    # ---- FEC sanity check ----------------------------------------------

    def fec_sanity_check(
        self,
        *,
        baseline_fec: float = 0.0,
        low_fec: float = -2.0,
        min_drop_ev: float = 1.0,
        gamma: float = 2.2,
        capture_dir: Path | None = None,
        measure: Callable[[Path], float] | None = None,
    ) -> tuple[float, float]:
        """Take two test shots and confirm FEC reaches the flash.

        Captures one frame at ``baseline_fec`` and one at ``low_fec``,
        gamma-linearises the central-crop medians, and computes the
        scene-light EV drop between them. Raises
        :class:`FECSanityError` if the measured drop is less than
        ``min_drop_ev``.

        The check is meant to detect "FEC has no effect" failure modes
        (e.g. SU-800 mounted instead of SB-5000 commander, or the
        flashes not in remote mode). At ``low_fec=-2.0`` the rig
        should drop by close to 2 EV; the default ``min_drop_ev=1.0``
        is half the requested drop, leaving generous margin for
        ambient-light fill at the central crop while still catching
        a totally-non-responsive setup (which would show drop ≈ 0).

        The earlier pixel-ratio threshold (``min_ratio=0.5``) was
        intrinsically tighter than the requested drop: pixel ratio
        0.5 corresponds to a 2.20 EV drop in scene light at γ=2.2,
        so a perfectly responding rig that delivers exactly the
        requested -2.0 EV (pixel ratio 0.547) would falsely fail.

        Returns ``(baseline_median, low_median)``.
        """
        if measure is None:
            raise ValueError("measure callable is required")
        capture_dir = Path(capture_dir or "/tmp")

        self.set_fec(baseline_fec)
        baseline_path = capture_dir / "fec_sanity_baseline.jpg"
        self.capture_to(baseline_path)
        baseline_median = measure(baseline_path)

        self.set_fec(low_fec)
        low_path = capture_dir / "fec_sanity_low.jpg"
        self.capture_to(low_path)
        low_median = measure(low_path)

        # Restore FEC to baseline before returning.
        self.set_fec(baseline_fec)

        if baseline_median <= 0:
            raise FECSanityError(
                "baseline FEC=0 shot has zero median; flash may not have fired"
            )
        if low_median <= 0:
            raise FECSanityError(
                f"FEC={low_fec:+.1f} shot has zero median; flash may not have fired"
            )
        # Linearise via gamma and compute drop in EV.
        baseline_lin = (baseline_median / 255.0) ** gamma
        low_lin = (low_median / 255.0) ** gamma
        drop_ev = math.log2(baseline_lin / low_lin)
        if drop_ev < min_drop_ev:
            raise FECSanityError(
                f"FEC={low_fec:+.1f} dropped scene exposure by only "
                f"{drop_ev:+.2f} EV (medians {baseline_median:.0f} → "
                f"{low_median:.0f}); expected at least {min_drop_ev:.1f} EV. "
                "Most likely the SU-800 is mounted instead of the SB-5000 "
                "commander, or the flashes are not in remote mode."
            )
        return baseline_median, low_median


# ---- CLI for rig smoke-tests ---------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Nikon Z7 health check / capture",
        epilog=(
            "The Z7 is fragile about init/exit cycles (see DESIGN_NOTES). "
            "Pass --health and --capture together rather than running this "
            "tool twice — each fresh init has a non-zero chance of leaving "
            "the camera in a half-state that requires a battery pull."
        ),
    )
    parser.add_argument("--health", action="store_true",
                        help="run verify_healthy and print key properties")
    parser.add_argument("--capture", type=str, default=None,
                        help="capture one frame to the given path")
    args = parser.parse_args()

    cam = NikonZ7()
    cam.open()
    try:
        # Always print the health diagnostic so a single invocation
        # gives both. The camera is fragile about repeated init/exit
        # cycles — combining flags keeps the smoke test to one cycle.
        cam.verify_healthy()
        cfg = cam.camera.get_config()
        print(f"config tree widgets: {count_widgets(cfg)}")
        for prop in ("expprogram", "iso", "f-number", "shutterspeed",
                     "flashexposurecompensation"):
            w = find(cfg, prop)
            # Compare against None explicitly: SWIG-wrapped
            # CameraWidget proxies are falsy in some configs even
            # when they're a valid widget object.
            v = w.get_value() if w is not None else "<missing>"
            print(f"  {prop} = {v!r}")
        if args.capture:
            # Apply the same session config the production runner
            # uses — most importantly capturetarget=Internal RAM, so
            # capture() doesn't try to write to the SD card slot
            # (which can wedge the USB session if the camera flips
            # to a different storage backend mid-session).
            cam.apply_session_config()
            size = cam.capture_to(args.capture)
            print(f"captured {size} bytes to {args.capture}")
    finally:
        cam.close()


if __name__ == "__main__":
    _cli()
