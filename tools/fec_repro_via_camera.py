#!/usr/bin/env python3
"""Reproduce the cascade's big-jump behavior using the EXACT production
camera adapter (``NikonZ7.set_fec`` + ``NikonZ7.capture_to``), not the
ad-hoc ``set_value`` helper from z7_ec_test.

If this reproduces the same-FEC-different-output pattern seen in the
failing cascade, the difference is in the production code path. If it
behaves like the clean test (smooth response, no lag), the difference
must be elsewhere in the cascade.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Pillow and numpy required.", file=sys.stderr)
    sys.exit(1)

from camera.nikon_z7 import NikonZ7

CROP_W = 0.20
CROP_H = 0.20
CROP_X = 0.00
CROP_Y = 0.15

OUT_DIR = Path("/tmp/fec_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_p50(jpg_path: Path) -> float:
    img = np.asarray(Image.open(jpg_path).convert("L"))
    h, w = img.shape
    cw, ch = int(w * CROP_W), int(h * CROP_H)
    cx = int(w / 2 + w * CROP_X - cw / 2)
    cy = int(h / 2 + h * CROP_Y - ch / 2)
    crop = img[cy:cy + ch, cx:cx + cw]
    return float(np.median(crop))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--aperture", default="f/5.6")
    p.add_argument("--iso", default=100, type=int)
    p.add_argument(
        "--fec-sequence", type=float, nargs="+",
        default=[0.0, -3.0, -1.667],
    )
    p.add_argument("--shots-per-step", type=int, default=3)
    p.add_argument(
        "--inter-step-settle", type=float, default=0.0,
        help="seconds between set_fec and first shot (default: 0 — mimics cascade)",
    )
    p.add_argument(
        "--inter-shot-recycle", type=float, default=3.0,
    )
    args = p.parse_args()

    print("=" * 78)
    print("Production-code-path FEC repro test")
    print("=" * 78)
    print(f"  uses NikonZ7.set_fec (production code path)")
    print(f"  uses NikonZ7.capture_to (production code path)")
    print(f"  inter-step settle: {args.inter_step_settle}s (cascade uses 0)")
    print(f"  inter-shot recycle: {args.inter_shot_recycle}s")
    print()

    cam = NikonZ7()
    cam.open()
    try:
        cam.verify_healthy(min_widgets=100)
        cam.apply_session_config(
            iso=args.iso,
            shutter_candidates=("1/200", "0.0050s", "0.005s"),
            image_quality="JPEG Fine",
            fec=0.0,
        )
        cam.set_aperture(args.aperture)
        time.sleep(3.0)

        results: list[tuple[float, int, float]] = []
        for fec in args.fec_sequence:
            print(f"\n--- set_fec({fec:+.3f}) ---")
            cam.set_fec(fec)
            if args.inter_step_settle > 0:
                time.sleep(args.inter_step_settle)
            for k in range(args.shots_per_step):
                if k > 0:
                    time.sleep(args.inter_shot_recycle)
                label = f"fec_{fec:+.3f}_shot{k}".replace("+", "p").replace("-", "m").replace(".", "_")
                dest = OUT_DIR / f"{label}.jpg"
                cam.capture_to(dest)
                p50 = measure_p50(dest)
                results.append((fec, k, p50))
                print(f"    shot {k}: p50={p50:6.1f}")

        cam.set_fec(0.0)
    finally:
        cam.close()

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'FEC':>+7} | " + " | ".join(f"shot{k} p50" for k in range(args.shots_per_step)))
    print("-" * 60)
    for fec in args.fec_sequence:
        cells = sorted([(k, v) for f, k, v in results if abs(f - fec) < 0.01])
        p50s = [v for _, v in cells]
        cells_str = " | ".join(f"{v:>9.1f}" for v in p50s)
        print(f"{fec:>+7.3f} | {cells_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
