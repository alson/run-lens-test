"""Nikon Z7 wrapper around python-gphoto2.

The patterns here are lifted from ``tools/z7_ec_test.py``, which has
been proven on the rig (drain-events-before-capture, transient ``'?'``
property tolerance, autofocus-toggle recovery on capture failure).

The class accepts a ``gp_module`` and ``sleep`` callable for testing
so unit tests don't need a real camera.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Sequence

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
        if self._camera is not None:
            try:
                self._camera.exit()
            finally:
                self._camera = None

    @property
    def camera(self):
        if self._camera is None:
            raise RuntimeError("NikonZ7 used before open()")
        return self._camera

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
        aperture: str | None = None,
    ) -> None:
        """Push the canonical per-session settings.

        Called after :meth:`open` and after every lens-change
        reconnect. Skips any setting whose widget isn't present (Z7
        firmware variants).
        """
        self._set_radio("capturetarget", [capture_target])
        self._set_radio("imagequality", [image_quality])
        self._set_radio("iso", [str(iso)])
        if aperture is not None:
            self._set_radio("f-number", [aperture])
        self._set_radio("shutterspeed", list(shutter_candidates))
        self._set_radio("autofocus", [autofocus])
        self._set_range("flashexposurecompensation", float(fec))

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
        cfg = self.camera.get_config()
        widget = find(cfg, name)
        if widget is None:
            return
        widget.set_value(float(value))
        self.camera.set_config(cfg)
        self.drain_events()

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
        min_ratio: float = 0.5,
        capture_dir: Path | None = None,
        measure: Callable[[Path], float] | None = None,
    ) -> tuple[float, float]:
        """Take two test shots and confirm FEC reaches the flash.

        Captures one frame at ``baseline_fec`` and one at ``low_fec``.
        ``measure(path)`` returns each frame's central-crop median
        (callers pass an :func:`exposure.analyse.measure_crop` adapter).

        Raises :class:`FECSanityError` if
        ``low_median / baseline_median > min_ratio`` (i.e. the −2 EV
        shot wasn't visibly darker, which means FEC isn't reaching the
        flash — most likely the SU-800 is mounted instead of the
        SB-5000 commander).

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
        ratio = low_median / baseline_median
        if ratio > min_ratio:
            raise FECSanityError(
                f"FEC={low_fec:+.1f} median ({low_median:.1f}) is not "
                f"sufficiently darker than FEC=0 ({baseline_median:.1f}); "
                f"ratio {ratio:.2f} > {min_ratio:.2f}. Most likely the "
                "SU-800 is mounted instead of the SB-5000 commander."
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
