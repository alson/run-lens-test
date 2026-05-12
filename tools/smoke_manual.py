#!/usr/bin/env python3
"""Manual-mode smoke test.

Opens the Z7, captures one frame at ISO 100, measures the central
crop median, computes the EV delta against the BASELINE_REFERENCE_IMAGE
in config.py. Useful for verifying that a manual-aperture setup
(flash power dialled on commander, ring set to a target) produces a
sensible exposure before starting a full session.

No prompts; one capture; prints the diagnostic numbers.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Make sibling modules importable when run from the run-lens-test root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from camera.nikon_z7 import NikonZ7  # noqa: E402
from exposure.analyse import crop_box, measure_crop  # noqa: E402


def main() -> None:
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
    print(f"camera ready: ISO {config.ISO_FLOOR}, FEC=0, "
          f"shutter from {config.SHUTTER_SPEED_CANDIDATES}")

    # Baseline median from the config reference image.
    ref_image = config.BASELINE_REFERENCE_IMAGE
    if ref_image is None or not Path(ref_image).exists():
        print(f"BASELINE_REFERENCE_IMAGE missing — can't compute EV delta.")
        baseline = None
    else:
        from PIL import Image
        with Image.open(str(ref_image)) as im:
            w, h = im.size
        box = crop_box(
            w, h,
            config.CROP_W_FRAC, config.CROP_H_FRAC,
            config.CROP_X_FRAC, config.CROP_Y_FRAC,
        )
        baseline = measure_crop(ref_image, box).p50
        print(f"baseline median from {ref_image.name}: {baseline:.1f}")

    # Take one capture.
    out = Path("/tmp/smoke_manual.jpg")
    print(f"capturing -> {out}")
    cam.capture_to(out)
    time.sleep(0.5)  # tiny pause for file write

    from PIL import Image
    with Image.open(str(out)) as im:
        w, h = im.size
    box = crop_box(
        w, h,
        config.CROP_W_FRAC, config.CROP_H_FRAC,
        config.CROP_X_FRAC, config.CROP_Y_FRAC,
    )
    stats = measure_crop(out, box)
    print(f"central-crop median (p50): {stats.p50:.1f}  "
          f"(box left={box.left} top={box.top} "
          f"size={box.width}x{box.height})")

    if baseline:
        # ev_delta uses gamma 2.2 linearisation.
        from exposure.analyse import ev_delta
        delta = ev_delta(stats.p50, baseline)
        print(f"EV delta vs baseline: {delta:+.2f} EV "
              f"(positive = brighter than baseline)")
        # Sanity interpretation.
        if abs(delta) <= 0.5:
            print("→ within ±0.5 EV — calibration would converge in 0-1 ISO step")
        elif abs(delta) <= 1.0:
            print("→ within ±1.0 EV — coarse-search phase 1 would exit here")
        else:
            print(f"→ {abs(delta):.1f} EV off — coarse search would adjust power "
                  f"by {round(abs(delta))} full stop(s)")

    print("closing camera...")
    cam.close()


if __name__ == "__main__":
    main()
