#!/usr/bin/env python3
"""Probe which float representations of FEC the Z7's PTP range widget
accepts. Sends a series of candidate floats and reads back the camera's
reported value for each."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z7_ec_test import find, set_value, snapshot_state  # noqa: E402

import gphoto2 as gp


def probe(camera, name: str, candidates: list[float]) -> None:
    print(f"\n=== {name} probe ===")
    # Establish a known starting state.
    cfg = camera.get_config()
    w = find(cfg, name)
    print(f"current value: {w.get_value()!r}")

    for v in candidates:
        # Direct widget interaction without our wrapper.
        cfg = camera.get_config()
        w = find(cfg, name)
        before = w.get_value()
        w.set_value(float(v))
        camera.set_config(cfg)
        time.sleep(0.4)
        # Drain
        for _ in range(10):
            try:
                t, _ = camera.wait_for_event(200)
                if t == gp.GP_EVENT_TIMEOUT:
                    break
            except gp.GPhoto2Error:
                break
        cfg2 = camera.get_config()
        actual = find(cfg2, name).get_value()
        match = "✓" if abs(float(actual) - v) < 1e-9 else "✗"
        print(f"  set({v!r:>26s}) before={before!r:>26s} after={actual!r:>26s} {match}")


def main() -> int:
    camera = gp.Camera()
    camera.init()
    try:
        snapshot_state(camera, "initial")
        # Probe -5/3 in several float forms
        targets = [
            -5 / 3,                       # canonical: -1.6666666666666667
            -5 * (1.0 / 3.0),             # old _snap: -1.6666666666666665
            -1.6666666666666666,          # truncated 16 digits
            -1.666666666666667,           # 15 digits
            -1.66666666666667,            # 14 digits
            -1.667,                       # 3 decimals
            -1.66,                        # 2 decimals (off-grid)
            -1.7,                         # 1 decimal (off-grid)
            -2 / 3,                       # known-good (k=-2)
            -7 / 3,                       # also suspect (k=-7)
            -7 * (1.0 / 3.0),             # old _snap form of k=-7
            -2 * 0.667,                   # alternative form
            -2.0,                         # known good
            -1.0,                         # known good
            0.0,                          # known good
        ]
        # Always start from FEC=0 between probes so we know what to
        # compare "before" against.
        cfg = camera.get_config()
        w = find(cfg, "flashexposurecompensation")
        w.set_value(0.0)
        camera.set_config(cfg)
        time.sleep(0.5)

        probe(camera, "flashexposurecompensation", targets)

        # Restore
        cfg = camera.get_config()
        find(cfg, "flashexposurecompensation").set_value(0.0)
        camera.set_config(cfg)
    finally:
        camera.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
