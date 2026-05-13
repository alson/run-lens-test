#!/usr/bin/env python3
"""Static FEC sweep with N shots per step to characterise flash response
and detect any `set_fec` lag / hysteresis.

Use case: investigating why the production cascade misses at certain
(lens, aperture) combinations even when FEC + EC + ISO appear to have
headroom. The cascade reported same-FEC-different-output behaviour in a
Sigma 150 + 6T + 5T calibration; this tool isolates the variables —
fixed aperture, fixed ISO, fixed EC, fixed scene — and walks FEC in
small steps with multiple shots per step.

For each FEC step:
  - call set_fec(F)
  - settle (configurable)
  - capture N shots back-to-back (with recycle pause between)
  - record per-shot median luminance from the production crop box

If shot #1 at FEC=X consistently differs from shots #2..#N at the same
FEC=X, that's evidence of set_fec lag (the first shot fired with the
previous FEC value). If all shots vary within the group, that's
genuine flash run-to-run variance. If shots are stable within a group
but vary nonlinearly across FEC, that's the response curve.

Usage:
  python -m tools.fec_lag_sweep --aperture f/5.6 --iso 100
  python -m tools.fec_lag_sweep --aperture f/5.6 --shots 3 --step 0.333

Defaults match the failing-calibration setup: f/5.6, ISO=100, 3 shots
per step, ⅓-stop spacing from 0 down to -3.0.

The camera must NOT be claimed by a live session — abort any running
macro-lens-test (press q at the prompt) before launching this script.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import gphoto2 as gp
except ImportError:
    print("python-gphoto2 not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Pillow and numpy required.", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z7_ec_test import (  # noqa: E402
    capture_to,
    crop_box,
    linearise,
    measure_crop,
    set_value,
    snapshot_state,
)

# Match production crop box (config.py CROP_*_FRAC).
CROP_W = 0.20
CROP_H = 0.20
CROP_X = 0.00
CROP_Y = 0.15
GAMMA = 2.2

OUT_DIR = Path("/tmp/fec_lag_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Shot:
    fec: float
    shot_index: int  # 0..N-1 within a group
    p50: float
    mean: float
    p95: float
    clip_hi_pct: float
    linear_median: float
    ev_vs_ref: float = 0.0  # filled later


@dataclass
class Group:
    fec: float
    shots: list[Shot] = field(default_factory=list)


def _capture_one(camera, label: str, box_holder) -> dict:
    dest = OUT_DIR / f"{label}.jpg"
    capture_to(camera, str(dest))
    if box_holder[0] is None:
        with Image.open(dest) as im:
            iw, ih = im.size
        box_holder[0] = crop_box(iw, ih, CROP_W, CROP_H, CROP_X, CROP_Y)
        print(f"    crop box: {box_holder[0]} on {iw}x{ih}")
    return measure_crop(str(dest), box_holder[0])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Static FEC sweep with N shots per step.",
    )
    p.add_argument(
        "--aperture", default="f/5.6",
        help="aperture to set (e.g. f/5.6, f/8). Default: f/5.6.",
    )
    p.add_argument("--iso", default="100", help="fixed ISO. Default: 100.")
    p.add_argument(
        "--start", type=float, default=0.0,
        help="FEC start (inclusive). Default: 0.0.",
    )
    p.add_argument(
        "--end", type=float, default=-3.0,
        help="FEC end (inclusive). Default: -3.0.",
    )
    p.add_argument(
        "--step", type=float, default=1.0 / 3.0,
        help="FEC step magnitude (⅓-stop=0.333). Default: 0.333.",
    )
    p.add_argument(
        "--shots", type=int, default=3,
        help="shots per FEC step. Default: 3.",
    )
    p.add_argument(
        "--settle", type=float, default=2.0,
        help="seconds to wait after set_fec before first shot. Default: 2.0.",
    )
    p.add_argument(
        "--recycle", type=float, default=3.0,
        help="seconds between shots within a group (flash recycle). Default: 3.0.",
    )
    args = p.parse_args()

    # Build FEC sweep list — descending (start > end is fine).
    direction = -1 if args.end < args.start else 1
    step = args.step * direction
    fec_values: list[float] = []
    v = args.start
    # epsilon to handle float drift; ⅓-stop grid plays poorly with == checks
    eps = abs(step) * 0.05
    while (direction < 0 and v >= args.end - eps) or (direction > 0 and v <= args.end + eps):
        # Snap to ⅓-stop grid to match the production cascade.
        snapped = round(v * 3.0) / 3.0
        fec_values.append(snapped)
        v += step
    # De-duplicate consecutive (snap can collapse adjacent values).
    deduped: list[float] = []
    for f in fec_values:
        if not deduped or abs(deduped[-1] - f) > eps:
            deduped.append(f)
    fec_values = deduped

    print("=" * 78)
    print("FEC lag / response sweep")
    print("=" * 78)
    print(f"Aperture:    {args.aperture}")
    print(f"ISO:         {args.iso}")
    print(f"FEC values:  {', '.join(f'{f:+.2f}' for f in fec_values)}  "
          f"({len(fec_values)} steps × {args.shots} shots = "
          f"{len(fec_values) * args.shots} captures)")
    print(f"Settle:      {args.settle:.1f} s after set_fec")
    print(f"Recycle:     {args.recycle:.1f} s between shots within a group")
    print(f"Output:      {OUT_DIR}")
    print()

    camera = gp.Camera()
    camera.init()
    try:
        snapshot_state(camera, "initial")

        print("\n--- baseline setup ---")
        set_value(camera, "f-number", args.aperture)
        set_value(camera, "iso", str(args.iso))
        set_value(camera, "shutterspeed", "0.0050s")
        set_value(camera, "imagequality", "JPEG Fine")
        set_value(camera, "autofocus", "Off")
        set_value(camera, "exposurecompensation", "0")  # EC=0 fixed
        set_value(camera, "flashexposurecompensation", 0.0)
        time.sleep(args.settle)
        snapshot_state(camera, "after-baseline")

        box_holder: list = [None]
        groups: list[Group] = []

        for fec in fec_values:
            print(f"\n--- FEC = {fec:+.3f} ---")
            set_value(camera, "flashexposurecompensation", fec)
            time.sleep(args.settle)

            group = Group(fec=fec)
            for k in range(args.shots):
                if k > 0:
                    time.sleep(args.recycle)
                label = f"fec_{fec:+.3f}_shot{k}".replace("+", "p").replace("-", "m")
                stats = _capture_one(camera, label, box_holder)
                lin = linearise(stats["p50"], gamma=GAMMA)
                shot = Shot(
                    fec=fec,
                    shot_index=k,
                    p50=stats["p50"],
                    mean=stats["mean"],
                    p95=stats["p90"],  # measure_crop reports p90, p50, p10
                    clip_hi_pct=stats["clip_hi_pct"],
                    linear_median=lin,
                )
                group.shots.append(shot)
                print(
                    f"    shot {k}: p50={stats['p50']:6.1f} "
                    f"mean={stats['mean']:6.1f} "
                    f"clip_hi={stats['clip_hi_pct']:.1f}%"
                )
            groups.append(group)

        # Restore
        print("\n--- restore: FEC=0 ---")
        set_value(camera, "flashexposurecompensation", 0.0)

    finally:
        camera.exit()

    # ---- analysis -------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    # Use first shot of first group as reference for EV.
    ref_lin = groups[0].shots[0].linear_median if groups else 1.0
    for g in groups:
        for s in g.shots:
            s.ev_vs_ref = (
                math.log2(s.linear_median / ref_lin)
                if ref_lin > 0 and s.linear_median > 0
                else 0.0
            )

    header = (
        f"{'FEC':>+7} | "
        + " | ".join(f"shot{k:<2}p50" for k in range(args.shots))
        + " | "
        + " | ".join(f"shot{k:<2}EV " for k in range(args.shots))
        + " | first→rest"
    )
    print(header)
    print("-" * len(header))
    for g in groups:
        p50s = [f"{s.p50:>9.1f}" for s in g.shots]
        evs = [f"{s.ev_vs_ref:>+9.2f}" for s in g.shots]
        # Diagnostic: shot[0] vs mean(shots[1:])
        if len(g.shots) >= 2:
            rest = [s.p50 for s in g.shots[1:]]
            first_minus_rest = g.shots[0].p50 - (sum(rest) / len(rest))
            tag = f"{first_minus_rest:+7.1f}"
            if abs(first_minus_rest) > 10:
                tag += "  ⚠ shot0 differs"
        else:
            tag = "      —"
        print(f"{g.fec:>+7.3f} | " + " | ".join(p50s) + " | "
              + " | ".join(evs) + f" | {tag}")

    # Cross-step monotonicity check
    print("\n--- response curve (mean p50 per FEC step) ---")
    means = [(g.fec, sum(s.p50 for s in g.shots) / len(g.shots)) for g in groups]
    for fec, m in means:
        bar = "█" * int(m / 4)
        print(f"  FEC {fec:>+6.3f}  p50={m:6.1f}  {bar}")

    # Slope between consecutive steps
    print("\n--- slope (EV per ⅓-stop FEC step, mean within group) ---")
    for i in range(1, len(groups)):
        prev_mean = sum(s.linear_median for s in groups[i - 1].shots) / len(groups[i - 1].shots)
        curr_mean = sum(s.linear_median for s in groups[i].shots) / len(groups[i].shots)
        if prev_mean > 0 and curr_mean > 0:
            slope = math.log2(curr_mean / prev_mean)
            fec_delta = groups[i].fec - groups[i - 1].fec
            slope_per_stop = slope / fec_delta if fec_delta != 0 else 0
            tag = ""
            if abs(slope_per_stop - 1.0) > 0.3:
                tag = "  ← non-linear"
            print(f"  FEC {groups[i-1].fec:+6.3f} → {groups[i].fec:+6.3f}:  "
                  f"ΔEV={slope:+5.2f}  slope/stop={slope_per_stop:+5.2f}{tag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
