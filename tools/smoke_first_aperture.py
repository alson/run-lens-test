#!/usr/bin/env python3
"""Real-rig smoke test for the first-aperture manual-flash cascade.

Runs :meth:`session.calibrate.Calibrator.calibrate_first_aperture_manual`
against the live Z7. Prompts are auto-confirmed (no user interaction
needed beyond the initial setup) so you can see the cascade's
behaviour end-to-end:

* Phase 1 coarse search starting at MANUAL_FLASH_INITIAL_POWER_STOPS,
  full-stop adjustments based on the EV delta to baseline.
* Phase 2 ISO fine-tune at the settled power.
* Phase 3 back-calc target power (snapped to 1/3 stop, soft/hard
  capped).
* Phase 4 verify + ISO fine-tune at the target.

WHAT YOU NEED:

* Z7 attached, M mode dial, awake (half-press shutter once).
* A non-CPU lens mounted, ring on a known aperture (e.g. f/2.8 wide
  open). Camera reports f/0 in EXIF as expected.
* Flash setup ALREADY DIALLED to a sensible starting power. The
  cascade will prompt for power changes — since prompts are
  auto-confirmed, the script doesn't actually wait for the user;
  the captures happen against whatever's currently set on the
  commander.

WHAT TO READ:

The script prints each prompted (power_c, power_a) label and the
measured EV delta from baseline at that step. Convergence within
±MANUAL_FLASH_SEARCH_TOLERANCE_EV (1.0 EV) on Phase 1 then
±TEST_SHOT_TOLERANCE_EV (0.5 EV per config.py) on Phase 2/4.

Use ``--aperture-label`` to set the file-label aperture (e.g. ``4``
for the effective aperture of 55 AIS f/2.8 + PK-13 at 1:1).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from camera.nikon_z7 import NikonZ7  # noqa: E402
from exposure.analyse import crop_box, measure_crop  # noqa: E402
from session.calibrate import Calibrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aperture-label",
        default="2.8",
        help="Effective aperture label for the calibration directory. "
             "Default '2.8' assumes ring at f/2.8 with m_indicated=0 "
             "(no extension). For 55 AIS + PK-13 at 1:1 use '4'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for calibration JPEGs. Default: temp.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(
        prefix="smoke_first_aperture_"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    cam = NikonZ7()
    print("opening camera...")
    cam.open()
    cam.verify_healthy(min_widgets=config.MIN_CONFIG_TREE_SIZE)
    cam.apply_session_config(
        iso=config.ISO_FLOOR,
        shutter_candidates=config.SHUTTER_SPEED_CANDIDATES,
        image_quality=config.IMAGE_QUALITY,
        fec=0.0,
    )

    # Load baseline from config reference.
    ref = config.BASELINE_REFERENCE_IMAGE
    if ref is None or not Path(ref).exists():
        print("BASELINE_REFERENCE_IMAGE missing — using first capture as baseline")
        baseline = None
    else:
        from PIL import Image
        with Image.open(str(ref)) as im:
            w, h = im.size
        box = crop_box(
            w, h,
            config.CROP_W_FRAC, config.CROP_H_FRAC,
            config.CROP_X_FRAC, config.CROP_Y_FRAC,
        )
        baseline = measure_crop(ref, box).p50
        print(f"baseline median: {baseline:.1f}")

    # Measure: open the captured file and run measure_crop with the
    # same crop frac as the rig uses.
    def _measure(jpg: Path) -> float:
        from PIL import Image
        with Image.open(str(jpg)) as im:
            w, h = im.size
        box = crop_box(
            w, h,
            config.CROP_W_FRAC, config.CROP_H_FRAC,
            config.CROP_X_FRAC, config.CROP_Y_FRAC,
        )
        return measure_crop(jpg, box).p50

    # Auto-confirming flash-power prompt — echoes the labels so the
    # operator can see what the rig is asking for, then returns True
    # without waiting for input. The captures are taken against
    # whatever the operator already has dialled on the commander.
    def _prompt_flash(*, power_c_label, power_a_label, context=""):
        prefix = f"[{context}] " if context else ""
        print(f"{prefix}PROMPT: dial group C → {power_c_label}, "
              f"group A → {power_a_label}  (auto-confirmed)")
        return True

    calib = Calibrator(
        camera=cam,
        measure=_measure,
        config=config,
        shots_csv=None,  # smoke test; skip CSV logging
    )

    print(f"\n=== first-aperture cascade @ {args.aperture_label} ===\n")
    try:
        result, final_power, final_iso = calib.calibrate_first_aperture_manual(
            aperture=args.aperture_label,
            lens_dir=out_dir,
            lens_slug="smoke-first-aperture",
            session_id="smoke",
            baseline_median=baseline,
            prompt_set_flash_power=_prompt_flash,
        )
    finally:
        cam.close()
        print("closed camera.")

    cal = result.per_aperture[args.aperture_label]
    print("\n=== result ===")
    print(f"baseline median  : {result.baseline_median:.1f}")
    print(f"final median     : {cal.final_median:.1f}")
    print(f"final EV delta   : {cal.final_ev_delta:+.3f} EV")
    print(f"hit_tolerance    : {cal.hit_tolerance}")
    print(f"shots taken      : {cal.test_shots_taken}")
    print(f"final flash power: {final_power:.3f} stops below full "
          f"(group C label = "
          f"{__import__('session.flash_power', fromlist=['power_label']).power_label(final_power)})")
    print(f"final ISO        : {final_iso}")
    print(f"iso_bumped       : {cal.iso_bumped}")


if __name__ == "__main__":
    main()
