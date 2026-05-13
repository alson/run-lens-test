#!/usr/bin/env python3
"""Test whether big FEC jumps + short settle times reproduce the
cascade's "same FEC, different output" bimodal behaviour.

Runs the sequence FEC=0 → −3 → −1.67 (mimicking the cascade's first
three iterations) at several settle times and captures 3 shots per
step. Compares against the smooth-sweep baseline (FEC=−1.67 ≈ p50 154).

If shot 1 at FEC=−1.67 (after coming from FEC=−3) is:
  - ~154 → big jumps fine, cascade artifact is something else
  - ~80  → big-jump + short-settle does create state (the cascade bug)

Settle range: 0.0 s (immediate trigger after set_fec returns) to 5.0 s.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import gphoto2 as gp
except ImportError:
    print("python-gphoto2 not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Pillow required.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z7_ec_test import (  # noqa: E402
    capture_to,
    crop_box,
    measure_crop,
    set_value,
    snapshot_state,
)

CROP_W = 0.20
CROP_H = 0.20
CROP_X = 0.00
CROP_Y = 0.15

OUT_DIR = Path("/tmp/fec_jump_settle")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Shot:
    settle_s: float
    fec: float
    shot_index: int
    p50: float
    mean: float


def _measure_box(camera, box_holder) -> tuple[int, int]:
    dest = OUT_DIR / "_calibrate_box.jpg"
    capture_to(camera, str(dest))
    with Image.open(dest) as im:
        iw, ih = im.size
    return crop_box(iw, ih, CROP_W, CROP_H, CROP_X, CROP_Y)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--aperture", default="f/5.6")
    p.add_argument("--iso", default="100")
    p.add_argument(
        "--settle-times", type=float, nargs="+",
        default=[0.0, 0.5, 2.0, 5.0],
        help="seconds to wait after set_fec before first shot (default: 0 0.5 2 5)",
    )
    p.add_argument(
        "--fec-sequence", type=float, nargs="+",
        default=[0.0, -3.0, -1.667],
        help="FEC values to step through (default: 0 -3 -1.667 — mimics cascade)",
    )
    p.add_argument(
        "--shots-per-step", type=int, default=3,
        help="captures per (settle, FEC) cell (default: 3)",
    )
    p.add_argument(
        "--inter-shot-recycle", type=float, default=3.0,
        help="seconds between shots within a single FEC step (default: 3)",
    )
    args = p.parse_args()

    print("=" * 78)
    print("Big-jump FEC + settle-time experiment")
    print("=" * 78)
    print(f"Aperture:           {args.aperture}")
    print(f"ISO:                {args.iso}")
    print(f"FEC sequence:       {' → '.join(f'{f:+.3f}' for f in args.fec_sequence)}")
    print(f"Settle times:       {args.settle_times}")
    print(f"Shots per cell:     {args.shots_per_step}")
    print(f"Inter-shot recycle: {args.inter_shot_recycle}s")
    print(f"Total captures:     "
          f"{len(args.settle_times) * len(args.fec_sequence) * args.shots_per_step}")
    print(f"Output:             {OUT_DIR}")
    print()

    camera = gp.Camera()
    camera.init()
    try:
        snapshot_state(camera, "initial")

        # Reset to known state
        print("\n--- baseline setup ---")
        set_value(camera, "f-number", args.aperture)
        set_value(camera, "iso", str(args.iso))
        set_value(camera, "shutterspeed", "0.0050s")
        set_value(camera, "imagequality", "JPEG Fine")
        set_value(camera, "autofocus", "Off")
        set_value(camera, "exposurecompensation", "0")
        set_value(camera, "flashexposurecompensation", 0.0)
        time.sleep(3.0)

        # Establish crop box
        box = _measure_box(camera, None)
        print(f"crop box: {box}")

        all_shots: list[Shot] = []

        for settle in args.settle_times:
            print(f"\n{'='*60}")
            print(f"Settle = {settle:.2f}s")
            print(f"{'='*60}")
            # Reset to FEC=0 between runs and give it a generous settle
            # so each settle test starts from the same state.
            set_value(camera, "flashexposurecompensation", 0.0)
            time.sleep(5.0)

            for fec in args.fec_sequence:
                set_value(camera, "flashexposurecompensation", fec)
                # Variable settle: this is the experiment variable.
                if settle > 0:
                    time.sleep(settle)
                print(f"\n  FEC = {fec:+.3f}  (after {settle:.2f}s settle)")
                for k in range(args.shots_per_step):
                    if k > 0:
                        time.sleep(args.inter_shot_recycle)
                    label = (
                        f"settle_{settle:.2f}_fec_{fec:+.3f}_shot{k}"
                        .replace("+", "p").replace("-", "m").replace(".", "_")
                    )
                    dest = OUT_DIR / f"{label}.jpg"
                    capture_to(camera, str(dest))
                    stats = measure_crop(str(dest), box)
                    s = Shot(
                        settle_s=settle, fec=fec, shot_index=k,
                        p50=stats["p50"], mean=stats["mean"],
                    )
                    all_shots.append(s)
                    print(f"    shot {k}: p50={stats['p50']:6.1f} mean={stats['mean']:6.1f}")

        # Restore
        print("\n--- restore: FEC=0 ---")
        set_value(camera, "flashexposurecompensation", 0.0)
    finally:
        camera.exit()

    # ---- analysis -------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULTS TABLE")
    print("=" * 78)
    header = f"{'settle':>7} | {'FEC':>+7} | " + " | ".join(
        f"shot{k} p50" for k in range(args.shots_per_step)
    ) + f" | {'mean(s1+)':>9} | {'shot0-mean(s1+)':>15}"
    print(header)
    print("-" * len(header))
    for settle in args.settle_times:
        for fec in args.fec_sequence:
            cell = [s for s in all_shots if s.settle_s == settle and s.fec == fec]
            if not cell:
                continue
            p50s = [s.p50 for s in cell]
            shot0 = p50s[0] if p50s else 0
            rest = p50s[1:]
            rest_mean = sum(rest) / len(rest) if rest else 0
            diff = shot0 - rest_mean if rest else 0
            p50_strs = " | ".join(f"{v:>8.1f}" for v in p50s)
            tag = ""
            if abs(diff) > 10:
                tag = " ⚠"
            print(
                f"{settle:>6.2f}s | {fec:>+7.3f} | {p50_strs} | "
                f"{rest_mean:>9.1f} | {diff:>+15.1f}{tag}"
            )

    # Cross-settle comparison at FEC=-1.667 (the target test)
    print("\n--- shot 0 at FEC=-1.667 across settles (after big jump from −3) ---")
    target = [s for s in all_shots if abs(s.fec - (-1.667)) < 0.01 and s.shot_index == 0]
    for s in target:
        print(f"  settle={s.settle_s:.2f}s → p50={s.p50:.1f}")

    print("\n--- shots 1+ at FEC=-1.667 across settles (after first stable shot) ---")
    later = [s for s in all_shots if abs(s.fec - (-1.667)) < 0.01 and s.shot_index >= 1]
    for s in later:
        print(f"  settle={s.settle_s:.2f}s shot={s.shot_index} → p50={s.p50:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
