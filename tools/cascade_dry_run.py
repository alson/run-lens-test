#!/usr/bin/env python3
"""Dry-run the failing cascade *without taking pictures*, querying the
camera's reported FEC/EC/ISO at every step to see if any setting
silently drifts.

Drives the EXACT cascade code path used in production (instantiates a
``Calibrator`` with a fake camera-wrapper that delegates real
get/set calls to ``NikonZ7`` but synthesises the median values from
a pre-recorded sequence). After each ``set_fec`` / ``set_ec`` /
``set_iso`` call inside the cascade, the wrapper snapshots the
camera's reported values and logs them. If at any point the camera
reports a different value than was set, we've found the bug.

Fed with the failing cascade's recorded median sequence (Sigma 150 +
6T + 5T, f/5.6, 11 shots from the live trace) so the new-fec /
plateau / ISO-bump logic takes the same path as the live run.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from camera.nikon_z7 import NikonZ7


# Median sequence from the failing 6T+5T f/5.6 cascade (shots 0..10 from
# shots.csv). The cascade is deterministic given these medians, so
# replaying them reconstructs the exact set_fec / set_iso sequence.
FAILING_MEDIANS = [216, 216, 83, 80, 82, 200, 199, 81, 82, 83, 85]


@dataclass
class Snapshot:
    label: str
    fec: float
    iso: int | str
    ec: str
    flashmode: str
    expprogram: str


def _snap(cam: NikonZ7, label: str) -> Snapshot:
    """Read the camera's CURRENT reported values via fresh get_config."""
    gp = cam._gp  # noqa: SLF001
    cfg = cam.camera.get_config()

    def _read(name: str) -> str:
        from camera.nikon_z7 import find
        w = find(cfg, name)
        if w is None:
            return "(missing)"
        try:
            v = w.get_value()
        except Exception as e:                                # noqa: BLE001
            return f"(err {e})"
        return str(v)

    fec_str = _read("flashexposurecompensation")
    try:
        fec_val = float(fec_str)
    except ValueError:
        fec_val = float("nan")

    return Snapshot(
        label=label,
        fec=fec_val,
        iso=_read("iso"),
        ec=_read("exposurecompensation"),
        flashmode=_read("flashmode"),
        expprogram=_read("expprogram"),
    )


class InstrumentedCameraWrapper:
    """Delegates ALL real get/set to NikonZ7; intercepts set_fec /
    set_ec / set_iso to snapshot the camera state immediately before
    and immediately after each call. Capture is replaced with a fake
    that returns a pre-recorded median (so we don't waste flash shots
    or wait for the rig to fire).

    The CapturingCamera protocol expected by Calibrator needs:
        - set_aperture(str)
        - set_iso(int|str)
        - set_fec(float)
        - set_ec(float)  [optional]
        - capture_to(path)
        - list_apertures()  [not used in cascade body]
    Plus the calibrator uses _measure for median.
    """

    def __init__(self, cam: NikonZ7, medians: Iterator[int]) -> None:
        self._cam = cam
        self._medians = iter(medians)
        self.snapshots: list[Snapshot] = []
        self.set_calls: list[tuple[str, float | int | str]] = []

    def _record(self, label: str) -> None:
        self.snapshots.append(_snap(self._cam, label))

    def set_aperture(self, fnum: str) -> None:
        self._cam.set_aperture(fnum)
        self.set_calls.append(("set_aperture", fnum))
        self._record(f"after set_aperture({fnum})")

    def set_iso(self, iso) -> None:  # type: ignore[no-untyped-def]
        before = _snap(self._cam, "before set_iso").iso
        self._cam.set_iso(iso)
        after = _snap(self._cam, "after set_iso").iso
        self.set_calls.append(("set_iso", iso))
        self.snapshots.append(Snapshot(
            label=f"set_iso({iso}): before={before} after={after}",
            fec=float("nan"), iso=after, ec="-", flashmode="-", expprogram="-",
        ))

    def set_fec(self, ev: float) -> None:
        before = _snap(self._cam, "before set_fec")
        self._cam.set_fec(ev)
        after = _snap(self._cam, "after set_fec")
        self.set_calls.append(("set_fec", ev))
        drift = "" if abs(after.fec - ev) < 0.01 else f"  ⚠ drift! requested {ev:+.3f}"
        self.snapshots.append(Snapshot(
            label=f"set_fec({ev:+.3f}): before_fec={before.fec:+.3f} after_fec={after.fec:+.3f}{drift}",
            fec=after.fec,
            iso=after.iso, ec=after.ec, flashmode=after.flashmode, expprogram=after.expprogram,
        ))

    def set_ec(self, ev: float) -> None:
        before = _snap(self._cam, "before set_ec")
        self._cam.set_ec(ev)
        after = _snap(self._cam, "after set_ec")
        self.set_calls.append(("set_ec", ev))
        drift = "" if before.ec == after.ec or True else "  ⚠ check"
        self.snapshots.append(Snapshot(
            label=f"set_ec({ev:+.3f}): before_ec={before.ec} after_ec={after.ec}{drift}",
            fec=after.fec, iso=after.iso, ec=after.ec, flashmode=after.flashmode, expprogram=after.expprogram,
        ))

    def capture_to(self, dest) -> int:  # type: ignore[no-untyped-def]
        # Snapshot RIGHT BEFORE the capture would have fired — this is
        # the moment that matters: what the camera *thinks* its state
        # is at trigger time.
        self._record("at-capture")
        # Write a stub file so the cascade's measure step has something
        # to "open" — actually the cascade calls measure_path which
        # we'll mock at the Calibrator level.
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"stub")
        return 4

    def list_apertures(self) -> list[str]:
        return self._cam.list_apertures()


def run_dry_run(aperture: str, baseline: float) -> None:
    cam = NikonZ7()
    cam.open()
    try:
        cam.verify_healthy(min_widgets=100)
        cam.apply_session_config(
            iso=100,
            shutter_candidates=("1/200", "0.0050s", "0.005s"),
            image_quality="JPEG Fine",
            fec=0.0,
        )
        cam.set_aperture(aperture)
        if hasattr(cam, "set_ec"):
            cam.set_ec(0.0)
        time.sleep(2)

        from session.calibrate import Calibrator
        import config

        wrapper = InstrumentedCameraWrapper(cam, FAILING_MEDIANS)

        # Calibrator needs a measure callable. Feed it the recorded
        # medians one by one so the cascade takes the exact same
        # branches as the failing run.
        median_iter = iter(FAILING_MEDIANS)

        def fake_measure(_path: Path) -> float:
            try:
                v = next(median_iter)
            except StopIteration:
                return 100.0
            return float(v)

        cal = Calibrator(
            camera=wrapper,
            config=config,
            measure=fake_measure,
        )

        print("\n--- starting cascade dry-run ---")
        result, _ = cal._calibrate_aperture(  # noqa: SLF001
            aperture=aperture,
            baseline=baseline,
            lens_slug="dry_run",
            session_id="dry_run",
            capture_dir=Path("/tmp/cascade_dry_run"),
            set_aperture_via_camera=False,
        )
        print(f"\nResult: hit_tolerance={result.hit_tolerance} fec={result.fec_ev:+.3f} "
              f"iso={result.iso} delta={result.final_ev_delta:+.2f}")

        # Report ALL snapshots
        print("\n" + "=" * 78)
        print("CAMERA STATE SNAPSHOTS (chronological)")
        print("=" * 78)
        for i, s in enumerate(wrapper.snapshots):
            print(f"{i:2d}  fec={s.fec:+.3f}  iso={s.iso:<5}  ec={s.ec:<4}  "
                  f"flashmode={s.flashmode:<10}  expprog={s.expprogram:<2}  | {s.label}")

        print("\n" + "=" * 78)
        print("SET-CALL LOG")
        print("=" * 78)
        for name, val in wrapper.set_calls:
            print(f"  {name}({val})")
    finally:
        cam.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aperture", default="f/5.6")
    p.add_argument("--baseline", type=float, default=129.0)
    args = p.parse_args()
    run_dry_run(args.aperture, args.baseline)
