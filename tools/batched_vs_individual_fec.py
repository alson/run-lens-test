"""Cal uses individual set_aperture/set_iso/set_fec; sweep uses the
batched set_aperture_iso_fec. If the batched setter applies FEC
differently (e.g. due to ordering quirks in the single set_config
PTP call), this would show up as a systematic EV gap between
otherwise-identical frames.

Test: 3× fec=-2.0 with individual setters, 3× fec=-2.0 with batched
setter, 3× fec=-2.0 with individual again. If batched and individual
readings differ by > 0.1 EV consistently, that's the bug.

Also: after the batched call, immediately call individual set_fec
again with same value and capture — does it match individual-only?
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from camera.nikon_z7 import NikonZ7
from exposure.analyse import crop_box, ev_delta, measure_crop


def main() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix="batchtest_"))
    print(f"=== Batched vs individual FEC setter ===", flush=True)
    print(f"output dir: {out_dir}", flush=True)

    camera = NikonZ7()
    camera.open()
    camera.verify_healthy(min_widgets=config.MIN_CONFIG_TREE_SIZE)
    camera.apply_session_config(
        iso=config.ISO_FLOOR,
        shutter_candidates=config.SHUTTER_SPEED_CANDIDATES,
        image_quality=config.IMAGE_QUALITY,
        fec=0.0,
    )

    box_holder: dict = {"box": None}

    def measure(p: Path) -> float:
        if box_holder["box"] is None:
            from PIL import Image
            with Image.open(str(p)) as im:
                w, h = im.size
            box_holder["box"] = crop_box(
                w, h,
                config.CROP_W_FRAC, config.CROP_H_FRAC,
                config.CROP_X_FRAC, config.CROP_Y_FRAC,
            )
        return measure_crop(p, box_holder["box"]).p50

    def shoot(label: str) -> float:
        dest = out_dir / f"{label}.jpg"
        camera.capture_to(dest)
        med = measure(dest)
        ev = ev_delta(med, 129.0)
        print(f"  {label:>30s}  med={med:6.1f}  ev={ev:+.3f}", flush=True)
        return ev

    def pause(s: float = 4.0) -> None:
        time.sleep(s)

    print()
    print("--- Round 1: individual setters, 3× fec=-2.0 ---")
    pause(10.0)
    for i in range(3):
        camera.set_aperture("f/5.6")
        camera.set_iso(100)
        camera.set_fec(-2.0)
        shoot(f"individual_{i}")
        pause()

    print()
    print("--- Round 2: batched setter, 3× fec=-2.0 ---")
    pause(10.0)
    for i in range(3):
        camera.set_aperture_iso_fec(aperture="f/5.6", iso=100, fec=-2.0)
        shoot(f"batched_{i}")
        pause()

    print()
    print("--- Round 3: individual again, 3× fec=-2.0 (control) ---")
    pause(10.0)
    for i in range(3):
        camera.set_aperture("f/5.6")
        camera.set_iso(100)
        camera.set_fec(-2.0)
        shoot(f"individual_repeat_{i}")
        pause()

    print()
    print("--- Round 4: batched then individual override ---")
    pause(10.0)
    for i in range(3):
        camera.set_aperture_iso_fec(aperture="f/5.6", iso=100, fec=-2.0)
        camera.set_fec(-2.0)  # force-reset FEC via individual setter
        shoot(f"batched_then_indiv_{i}")
        pause()

    print()
    print("=== Done. Compare ev across rounds. ===")


if __name__ == "__main__":
    main()
