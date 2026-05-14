"""Probe whether back-to-back shots at the same FEC read consistently
or drift based on the *previous* shot's energy / cap state.

Four sequences, all at f/5.6 ISO=100:

A. 5× fec=-2.0 back-to-back (3s recycle). Baseline — should be flat
   if rig is stable.
B. fec=0 (high power) → 5× fec=-2.0 (3s recycle each). Tests
   whether the FIRST fec=-2.0 shot reads dimmer than subsequent ones,
   diagnostic of cap-state asymmetry from the high-power shot.
C. fec=0 → 5× fec=-2.0 with 8s recycle each. If sequence B shows
   asymmetry and C doesn't, 3s recycle is too short after a full
   dump.
D. Cascade order: fec=0 → fec=-2.0 → fec=-3.0 (probe) → fec=-2.0
   (verification). Tests whether the post-probe fec=-2.0 matches
   the immediate post-fec=0 fec=-2.0 (sweep-condition match) or
   reads differently (sweep conditions don't match cascade shot 2).

Recovery pause between sequences (10s) so each starts from a fully-
recovered cap.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Project root on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from camera.nikon_z7 import NikonZ7
from exposure.analyse import crop_box, ev_delta, measure_crop


BASELINE_MEDIAN = 129.0


def main() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix="capstate_"))
    print(f"=== Cap-state probe ===", flush=True)
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
    camera.set_aperture("f/5.6")
    camera.set_iso(100)

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

    def shoot(fec: float, label: str) -> tuple[float, float]:
        camera.set_fec(fec)
        dest = out_dir / f"{label}.jpg"
        camera.capture_to(dest)
        med = measure(dest)
        ev = ev_delta(med, BASELINE_MEDIAN)
        print(f"  {label:>30s}  fec={fec:+.2f}  med={med:6.1f}  ev={ev:+.3f}",
              flush=True)
        return med, ev

    def pause(seconds: float, label: str = "") -> None:
        if label:
            print(f"  (pausing {seconds:.1f}s {label})", flush=True)
        time.sleep(seconds)

    print()
    print("--- Sequence A: 5× fec=-2.0, 3s recycle (baseline drift) ---")
    pause(10.0, "recover")
    for i in range(5):
        shoot(-2.0, f"A_fec-2_{i}")
        if i < 4:
            pause(3.0)

    print()
    print("--- Sequence B: fec=0 then 5× fec=-2.0, 3s recycle ---")
    pause(10.0, "recover")
    shoot(0.0, "B_fec0")
    pause(3.0)
    for i in range(5):
        shoot(-2.0, f"B_fec-2_{i}")
        if i < 4:
            pause(3.0)

    print()
    print("--- Sequence C: fec=0 then 5× fec=-2.0, 8s recycle ---")
    pause(10.0, "recover")
    shoot(0.0, "C_fec0")
    pause(8.0)
    for i in range(5):
        shoot(-2.0, f"C_fec-2_{i}")
        if i < 4:
            pause(8.0)

    print()
    print("--- Sequence D: cascade order, then post-probe verification ---")
    pause(10.0, "recover")
    shoot(0.0, "D_shot1_fec0")
    pause(3.0)
    shoot(-2.0, "D_shot2_target")
    pause(3.0)
    shoot(-3.0, "D_shot3_probe")
    pause(3.0)
    shoot(-2.0, "D_shot4_verify")
    pause(3.0)
    shoot(-2.0, "D_shot5_verify2")

    print()
    print("=== Done. Compare medians for cap-state asymmetry. ===")


if __name__ == "__main__":
    main()
