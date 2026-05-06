#!/usr/bin/env python3
"""Investigate the effective ceiling of FEC + EC on the Z7 with the
SB-5000 commander already biased +1.7 EV on both groups.

Hypothesis: when commander group bias is high enough that the SB-800
remotes are already firing near full power at FEC=0 / EC=0, increasing
camera-side FEC and EC has diminishing returns — eventually no effect
at all. The calibrator's cascade keeps bumping with no luminance gain,
which is what the user observed at f/4.5 (EC=+3 didn't measurably
change the wafer median).

Test plan (single camera session, fixed aperture):
  Phase 0: reset to known state (ISO=100, shutter=1/200, FEC=0, EC=0)
  Phase 1: reference at (FEC=0, EC=0)
  Phase 2: FEC sweep alone:   FEC ∈ {-3, -2, -1, 0, +1}, EC=0
  Phase 3: EC sweep at FEC=0: EC ∈ {-3, -2, -1, 0, +1, +2, +3}
  Phase 4: ceiling probe at FEC=+1: EC ∈ {0, +1, +2, +3}
  Phase 5: stacking verification at FEC=-1: EC ∈ {0, +1, +2}
  Phase 6: parity check FEC=-1+EC=+1 vs FEC=0+EC=0

Each shot:
  - capture JPEG into /tmp/fec_ec_ceiling/
  - measure central crop (same fractions as production runner)
  - record (median, linear_median, ev_vs_ref)

Final report tabulates results and flags ceilings.

Usage:
  python -m tools.fec_ec_ceiling_test
  python -m tools.fec_ec_ceiling_test --aperture f/8
"""

from __future__ import annotations

import argparse
import math
import os
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
    import numpy as np
except ImportError:
    print("Pillow and numpy required.", file=sys.stderr)
    sys.exit(1)

# Reuse helpers from the existing test script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from z7_ec_test import (  # noqa: E402
    find,
    set_value,
    capture_to,
    drain_events,
    snapshot_state,
    crop_box,
    measure_crop,
    linearise,
)

# Match production crop fractions (config.py).
CROP_W = 0.20
CROP_H = 0.20
CROP_X = 0.00
CROP_Y = 0.15
GAMMA = 2.2

OUT_DIR = Path("/tmp/fec_ec_ceiling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTLE_S = 1.5  # let flashes recycle

# ---- shot result -----------------------------------------------------------


@dataclass
class Shot:
    label: str
    fec: float
    ec: float
    iso: int
    p50: float
    p10: float
    p90: float
    mean: float
    clip_hi_pct: float
    linear_median: float
    ev_vs_ref: float | None  # filled later


def _capture_and_measure(
    camera, label: str, fec: float, ec: float, iso: int, box_holder,
) -> Shot:
    dest = OUT_DIR / f"{label}.jpg"
    capture_to(camera, str(dest))
    if box_holder[0] is None:
        with Image.open(dest) as im:
            iw, ih = im.size
        box_holder[0] = crop_box(iw, ih, CROP_W, CROP_H, CROP_X, CROP_Y)
        print(f"    crop box established: {box_holder[0]} on {iw}x{ih}")
    stats = measure_crop(str(dest), box_holder[0])
    lin = linearise(stats["p50"], gamma=GAMMA)
    print(
        f"    [{label:24s}] FEC={fec:+.2f} EC={ec:+.2f} ISO={iso:<4d}"
        f" p50={stats['p50']:6.1f}  p10={stats['p10']:6.1f}"
        f"  p90={stats['p90']:6.1f}  clip-hi={stats['clip_hi_pct']:.1f}%"
    )
    return Shot(
        label=label, fec=fec, ec=ec, iso=iso,
        p50=stats["p50"], p10=stats["p10"], p90=stats["p90"],
        mean=stats["mean"], clip_hi_pct=stats["clip_hi_pct"],
        linear_median=lin, ev_vs_ref=None,
    )


def _set_fec_ec(camera, fec: float, ec) -> None:
    set_value(camera, "flashexposurecompensation", fec)
    # EC widget on Z7 uses string choices like "+1.0", "0", etc. We
    # pass it through set_value which str()s; the widget is RADIO.
    if isinstance(ec, str):
        set_value(camera, "exposurecompensation", ec)
    else:
        set_value(camera, "exposurecompensation", _ec_str(ec))
    time.sleep(SETTLE_S)


def _ec_str(ec: float) -> str:
    """Format an EC value to match the Z7 widget's choice strings.

    Widget uses unsigned strings: "0", "1", "-1", "1.333", etc."""
    if abs(ec) < 0.05:
        return "0"
    sign = "-" if ec < 0 else ""
    mag = abs(ec)
    third = round(mag * 3) / 3
    if abs(third - round(third)) < 0.05:
        return f"{sign}{int(round(third))}"
    return f"{sign}{third:.3f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--aperture", default=None,
                   help="set f-number widget (e.g. f/4.5, f/8). "
                        "If omitted, leaves whatever is current.")
    p.add_argument("--iso", default="100")
    args = p.parse_args()

    print("=" * 78)
    print("FEC + EC ceiling investigation")
    print("=" * 78)
    print(f"Output dir: {OUT_DIR}")
    print(f"Crop fractions: w={CROP_W} h={CROP_H} x={CROP_X} y={CROP_Y}")

    camera = gp.Camera()
    camera.init()
    try:
        snapshot_state(camera, "initial")

        # ---- phase 0: reset to baseline -----------------------------------
        print("\n--- phase 0: reset baseline ---")
        if args.aperture:
            set_value(camera, "f-number", args.aperture)
        set_value(camera, "iso", str(args.iso))
        set_value(camera, "shutterspeed", "0.0050s")
        set_value(camera, "imagequality", "JPEG Fine")
        set_value(camera, "autofocus", "Off")
        _set_fec_ec(camera, 0.0, 0.0)
        snapshot_state(camera, "after-reset")

        box_holder = [None]
        shots: list[Shot] = []
        iso_int = int(args.iso)

        # ---- phase 1: reference -------------------------------------------
        print("\n--- phase 1: reference (FEC=0, EC=0) ---")
        ref = _capture_and_measure(
            camera, "00_ref_fec0_ec0", 0.0, 0.0, iso_int, box_holder
        )
        shots.append(ref)

        ref_lin = ref.linear_median

        def push(label: str, fec: float, ec: float) -> Shot:
            _set_fec_ec(camera, fec, ec)
            s = _capture_and_measure(
                camera, label, fec, ec, iso_int, box_holder
            )
            s.ev_vs_ref = math.log2(s.linear_median / ref_lin) if ref_lin > 0 else 0
            shots.append(s)
            return s

        # ---- phase 2: FEC alone -------------------------------------------
        print("\n--- phase 2: FEC sweep alone (EC=0) ---")
        for fec in [-3.0, -2.0, -1.0, 0.0, 1.0]:
            push(f"02_fec{fec:+.0f}_ec0", fec, 0.0)

        # ---- phase 3: EC alone at FEC=0 -----------------------------------
        print("\n--- phase 3: EC sweep alone (FEC=0) ---")
        for ec in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
            push(f"03_fec0_ec{ec:+.0f}", 0.0, ec)

        # ---- phase 4: ceiling at FEC=+1 -----------------------------------
        print("\n--- phase 4: ceiling probe at FEC=+1 ---")
        for ec in [0.0, 1.0, 2.0, 3.0]:
            push(f"04_fec+1_ec{ec:+.0f}", 1.0, ec)

        # ---- phase 5: stacking at FEC=-1 ----------------------------------
        print("\n--- phase 5: stacking probe at FEC=-1 ---")
        for ec in [0.0, 1.0, 2.0]:
            push(f"05_fec-1_ec{ec:+.0f}", -1.0, ec)

        # ---- phase 6: parity check ----------------------------------------
        print("\n--- phase 6: parity FEC=-1+EC=+1 vs ref ---")
        push("06_fec-1_ec+1", -1.0, 1.0)

        # ---- restore --------------------------------------------------------
        print("\n--- restore: FEC=0, EC=0 ---")
        _set_fec_ec(camera, 0.0, 0.0)

    finally:
        camera.exit()

    # ---- final report -----------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL REPORT")
    print("=" * 78)
    print(f"Reference: p50={ref.p50:.1f} linear={ref.linear_median:.4f}")
    print()
    print(f"{'label':28s} {'FEC':>5s} {'EC':>5s} {'p50':>6s} "
          f"{'EV_vs_ref':>10s} {'clip_hi%':>9s}")
    print("-" * 78)
    for s in shots:
        ev = f"{s.ev_vs_ref:+.2f}" if s.ev_vs_ref is not None else "  ref"
        print(f"{s.label:28s} {s.fec:+5.1f} {s.ec:+5.1f} {s.p50:6.1f} "
              f"{ev:>10s} {s.clip_hi_pct:8.1f}%")

    # Quick analysis
    print("\n--- analysis ---")
    fec1_shots = [s for s in shots if s.label.startswith("04_fec+1")]
    if len(fec1_shots) >= 2:
        base = fec1_shots[0].ev_vs_ref or 0
        gains = [
            (s.ec, (s.ev_vs_ref or 0) - base) for s in fec1_shots[1:]
        ]
        print(f"At FEC=+1, marginal EV per +1 EC bump:")
        for ec, gain in gains:
            print(f"  EC=0 → EC={ec:+.0f}: {gain:+.2f} EV (expected ≈ {ec:+.2f})")

    fecm1_shots = [s for s in shots if s.label.startswith("05_fec-1")]
    if len(fecm1_shots) >= 2:
        base = fecm1_shots[0].ev_vs_ref or 0
        gains = [
            (s.ec, (s.ev_vs_ref or 0) - base) for s in fecm1_shots[1:]
        ]
        print(f"At FEC=-1, marginal EV per +1 EC bump:")
        for ec, gain in gains:
            print(f"  EC=0 → EC={ec:+.0f}: {gain:+.2f} EV (expected ≈ {ec:+.2f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
