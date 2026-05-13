#!/usr/bin/env python3
"""Run the new linear AA cascade against the live rig for a single
aperture, isolated from the runner.

Validates the cascade end-to-end on real hardware:
  - the FEC float-rounding fix in NikonZ7._set_range
  - the 3-5 shot linear-model cascade

Picks up the current rig framing / lens / rail position as-is —
this is purely an exposure-calibration test, no rail movement.

Captures land under /tmp/test_linear_cascade/. Each shot is logged
to shots.csv with phase=calibration, so the rig's CSV format
matches the production runner.

Usage:
    uv run python -m tools.test_linear_cascade --aperture f/5.6
    uv run python -m tools.test_linear_cascade --aperture f/8 --baseline 129
    uv run python -m tools.test_linear_cascade --apertures f/5.6 f/8 f/11 f/16
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Pillow and numpy required.", file=sys.stderr)
    sys.exit(1)

import config  # noqa: E402
from camera.nikon_z7 import NikonZ7  # noqa: E402
from session.calibrate import Calibrator  # noqa: E402
from session.log import ShotsCSV  # noqa: E402


# Match production crop fractions.
CROP_W = 0.20
CROP_H = 0.20
CROP_X = 0.00
CROP_Y = 0.15


def _measure(jpg_path: Path) -> float:
    """Median luminance of the central crop (matches production)."""
    img = np.asarray(Image.open(jpg_path).convert("L"))
    h, w = img.shape
    cw, ch = int(w * CROP_W), int(h * CROP_H)
    cx = int(w / 2 + w * CROP_X - cw / 2)
    cy = int(h / 2 + h * CROP_Y - ch / 2)
    crop = img[cy : cy + ch, cx : cx + cw]
    return float(np.median(crop))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Live-rig test of the linear AA cascade.",
    )
    p.add_argument(
        "--apertures", nargs="+", default=["f/5.6"],
        help="aperture(s) to calibrate, in order. Default: f/5.6.",
    )
    p.add_argument(
        "--baseline", type=float, default=None,
        help=(
            "Pre-set baseline median. If omitted, the first aperture's "
            "shot 1 sets the baseline (production-like first-lens flow)."
        ),
    )
    p.add_argument(
        "--out-dir", default="/tmp/test_linear_cascade",
        help="output directory for JPEGs + shots.csv. Default: /tmp/test_linear_cascade.",
    )
    p.add_argument(
        "--legacy", action="store_true",
        help=(
            "Force the legacy iterative cascade (LINEAR_CASCADE_ENABLED "
            "= False) for A/B comparison. Default uses linear (per "
            "config.LINEAR_CASCADE_ENABLED)."
        ),
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = out_dir / "calibration"
    capture_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "shots.csv"

    # Force the cascade flag for this run only.
    config.LINEAR_CASCADE_ENABLED = not args.legacy
    flavor = "linear" if config.LINEAR_CASCADE_ENABLED else "legacy iterative"

    print("=" * 78)
    print(f"Linear-cascade live test ({flavor})")
    print("=" * 78)
    print(f"Apertures:   {args.apertures}")
    print(f"Baseline:    {args.baseline if args.baseline is not None else '(set from first shot)'}")
    print(f"Output dir:  {out_dir}")
    print()

    cam = NikonZ7()
    cam.open()
    try:
        cam.verify_healthy(min_widgets=100)
        cam.apply_session_config(
            iso=config.ISO_FLOOR,
            shutter_candidates=config.SHUTTER_SPEED_CANDIDATES,
            image_quality=config.IMAGE_QUALITY,
            fec=0.0,
        )
        # Let the rig settle.
        time.sleep(2.0)

        csv = ShotsCSV(csv_path)
        cal = Calibrator(camera=cam, config=config, measure=_measure, shots_csv=csv)

        baseline = args.baseline
        results = []
        for ap in args.apertures:
            print(f"--- {ap} ---")
            t0 = time.time()
            result, baseline = cal._calibrate_aperture(  # noqa: SLF001
                aperture=ap,
                lens_slug="test_linear_cascade",
                session_id="test_linear_cascade",
                capture_dir=capture_dir,
                baseline=baseline,
                set_aperture_via_camera=True,
            )
            dt = time.time() - t0
            results.append((ap, result, dt))
            status = "ok" if result.hit_tolerance else (
                "saturated" if result.saturated_at_ceiling else "MISS"
            )
            ec_str = f" EC={result.ec_ev:+.2f}" if result.ec_used else ""
            iso_str = f"{result.iso}*" if result.iso_bumped else f"{result.iso}"
            print(
                f"  [{status}] ev={result.final_ev_delta:+.2f} "
                f"FEC={result.fec_ev:+.2f}{ec_str} ISO={iso_str} "
                f"shots={result.test_shots_taken} "
                f"time={dt:.1f}s"
            )
            print()

        # Reset
        cam.set_fec(0.0)
        if hasattr(cam, "set_ec"):
            cam.set_ec(0.0)
        cam.set_iso(config.ISO_FLOOR)
    finally:
        cam.close()

    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'aperture':>10} {'verdict':>10} {'ev':>+8} {'FEC':>+6} {'EC':>+6} {'ISO':>5} {'shots':>6} {'time':>6}")
    print("-" * 78)
    n_ok = 0
    total_shots = 0
    for ap, r, dt in results:
        status = "OK" if r.hit_tolerance else ("SAT" if r.saturated_at_ceiling else "MISS")
        if r.hit_tolerance:
            n_ok += 1
        total_shots += r.test_shots_taken
        ec_disp = f"{r.ec_ev:+.2f}" if r.ec_used else "  -  "
        iso_disp = f"{r.iso}{'*' if r.iso_bumped else ' '}"
        print(
            f"{ap:>10} {status:>10} {r.final_ev_delta:>+8.2f} "
            f"{r.fec_ev:>+6.2f} {ec_disp:>6} {iso_disp:>5} "
            f"{r.test_shots_taken:>6} {dt:>5.1f}s"
        )
    print(f"\n{n_ok}/{len(results)} converged, total shots = {total_shots}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
