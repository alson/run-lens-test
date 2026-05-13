#!/usr/bin/env python3
"""Probe which float representations of every ⅓-stop FEC the Z7
accepts: 3-decimal rounded, 4-decimal rounded, float32-quantized, and
the raw Python float64. Reads back after each set with a generous
settle so we know the camera has actually applied (or rejected) the
value, not just that the readback was premature."""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z7_ec_test import find, snapshot_state  # noqa: E402

import gphoto2 as gp


def to_float32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def try_set(camera, name: str, value: float) -> float:
    """Set the property, wait, return the camera-reported value."""
    cfg = camera.get_config()
    w = find(cfg, name)
    w.set_value(float(value))
    camera.set_config(cfg)
    time.sleep(2.0)  # generous settle — make absolutely sure
    # drain events
    for _ in range(10):
        try:
            t, _ = camera.wait_for_event(200)
            if t == gp.GP_EVENT_TIMEOUT:
                break
        except gp.GPhoto2Error:
            break
    cfg2 = camera.get_config()
    return float(find(cfg2, name).get_value())


def main() -> int:
    camera = gp.Camera()
    camera.init()
    try:
        snapshot_state(camera, "initial")
        print()

        # Reset to 0.
        cfg = camera.get_config()
        find(cfg, "flashexposurecompensation").set_value(0.0)
        camera.set_config(cfg)
        time.sleep(2.0)

        print(f"{'k':>3} {'k/3':>22} {'3-decimal':>10} {'4-decimal':>10} {'float32':>22}")
        print("=" * 80)

        for k in range(-9, 4):
            target = k / 3.0
            # Reset to 0 between probes to test consistent transition.
            cfg = camera.get_config()
            find(cfg, "flashexposurecompensation").set_value(0.0)
            camera.set_config(cfg)
            time.sleep(1.5)

            # Try each form, reset between.
            row = {"k": k, "expected": target}
            for form_name, form_value in [
                ("k/3",       target),
                ("3-decimal", round(target, 3)),
                ("4-decimal", round(target, 4)),
                ("float32",   to_float32(target)),
            ]:
                # Reset
                cfg = camera.get_config()
                find(cfg, "flashexposurecompensation").set_value(0.0)
                camera.set_config(cfg)
                time.sleep(1.0)
                # Send
                actual = try_set(camera, "flashexposurecompensation", form_value)
                row[form_name] = actual

            mark3 = "✓" if abs(row["3-decimal"] - target) < 0.01 else "✗"
            mark4 = "✓" if abs(row["4-decimal"] - target) < 0.01 else "✗"
            markf = "✓" if abs(row["float32"] - target) < 0.01 else "✗"
            markk = "✓" if abs(row["k/3"] - target) < 0.01 else "✗"
            print(
                f"{k:>+3d} {target:>+22.16g} "
                f"{mark3}{round(target, 3):>+9.4f} "
                f"{mark4}{round(target, 4):>+9.4f} "
                f"{markf}{to_float32(target):>+21.16g}"
            )
            # Print actuals after each
            print(f"   reported: k/3 {markk}{row['k/3']:+.10g}  "
                  f"3d {row['3-decimal']:+.10g}  "
                  f"4d {row['4-decimal']:+.10g}  "
                  f"f32 {row['float32']:+.10g}")

        # Restore
        cfg = camera.get_config()
        find(cfg, "flashexposurecompensation").set_value(0.0)
        camera.set_config(cfg)
    finally:
        camera.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
