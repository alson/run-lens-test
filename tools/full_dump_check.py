#!/usr/bin/env python3
"""Fire N shots at fixed (f/4.5, ISO 100, FEC=+1, EC=+3) with a long
settle between each so the user can listen for full-dump duration and
watch the ready-light recycle.

Usage:
  python -m tools.full_dump_check                    # 5 shots, 5 s gap
  python -m tools.full_dump_check --count 8 --gap 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import gphoto2 as gp
except ImportError:
    print("python-gphoto2 not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image  # noqa: F401
except ImportError:
    print("Pillow required.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z7_ec_test import (  # noqa: E402
    set_value,
    capture_to,
    snapshot_state,
    crop_box,
    measure_crop,
)

OUT_DIR = Path("/tmp/full_dump_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--gap", type=float, default=5.0,
                   help="seconds between shots — long enough to "
                        "hear the recycle finish")
    p.add_argument("--aperture", default="f/4.5")
    p.add_argument("--iso", default="100")
    p.add_argument("--fec", type=float, default=1.0)
    p.add_argument("--ec", default="3")
    args = p.parse_args()

    print(f"Settings: {args.aperture}, ISO {args.iso}, "
          f"FEC={args.fec:+}, EC={args.ec}")
    print(f"Will fire {args.count} shots with {args.gap:.1f} s gap.")
    print(f"Output: {OUT_DIR}")

    camera = gp.Camera()
    camera.init()
    try:
        set_value(camera, "f-number", args.aperture)
        set_value(camera, "iso", args.iso)
        set_value(camera, "shutterspeed", "0.0050s")
        set_value(camera, "imagequality", "JPEG Fine")
        set_value(camera, "autofocus", "Off")
        set_value(camera, "flashexposurecompensation", args.fec)
        set_value(camera, "exposurecompensation", args.ec)
        time.sleep(2.0)
        snapshot_state(camera, "configured")

        box = None
        results: list[tuple[int, float, float, float]] = []
        for i in range(args.count):
            print(f"\n--- shot {i+1}/{args.count} ---")
            t0 = time.time()
            dest = OUT_DIR / f"shot_{i+1:02d}.jpg"
            capture_to(camera, str(dest))
            t_capture = time.time() - t0
            if box is None:
                with Image.open(dest) as im:
                    iw, ih = im.size
                box = crop_box(iw, ih, 0.20, 0.20, 0.00, 0.15)
                print(f"    crop box: {box} on {iw}x{ih}")
            stats = measure_crop(str(dest), box)
            print(f"    capture took {t_capture:.2f}s  "
                  f"p50={stats['p50']:.0f}  "
                  f"clip-hi={stats['clip_hi_pct']:.1f}%")
            results.append(
                (i + 1, t_capture, stats["p50"], stats["clip_hi_pct"])
            )
            if i < args.count - 1:
                print(f"    waiting {args.gap:.1f}s "
                      f"(listen / watch ready light)")
                time.sleep(args.gap)
    finally:
        camera.exit()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'shot':>4s}  {'capture_s':>10s}  {'p50':>5s}  {'clip_hi%':>9s}")
    for i, dt, p50, clip in results:
        print(f"{i:>4d}  {dt:>10.2f}  {p50:>5.0f}  {clip:>8.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
