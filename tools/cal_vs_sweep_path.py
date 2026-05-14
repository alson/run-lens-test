"""Mimic the runner's CAL capture path vs SWEEP capture path at the
exact same settings and see whether they produce different readings.

Cal path (linear cascade abbreviated):
  set_aperture(f/5.6) -> set_iso(100) -> set_fec(0)  -> capture (shot 1)
  set_fec(-2.0) -> set_ec(0)                          -> capture (shot 2)
  set_fec(-3.0)                                       -> capture (shot 3 probe)
  set_fec(-2.0)                                       -> capture (post-probe, sweep-equivalent)

Sweep path:
  set_aperture_iso_fec(f/5.6, 100, -2.0) -> set_ec(0) -> capture (rep 1)
  -> capture (rep 2)
  -> capture (rep 3)

Both paths get 3s recycle between shots. The cal path also burns
~5-9s of setter time between shots; sweep replicas burn ~3s recycle
and no setter time after the first.

The diagnostic: does the post-probe fec=-2.0 (cal-style) match the
sweep replicas at fec=-2.0 (sweep-style)? Mid-EV gap would point at
the setter sequence; identical EVs say the difference must be
elsewhere (rail position, rig state, etc.).
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
    out_dir = Path(tempfile.mkdtemp(prefix="path_"))
    print(f"=== Cal-path vs sweep-path ===", flush=True)
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

    # --- CAL path (linear cascade simulated) ---
    print()
    print("--- CAL path: cascade-style sequence ---")
    pause(10.0)
    # Shot 1: fec=0
    camera.set_aperture("f/5.6")
    camera.set_iso(100)
    camera.set_fec(0.0)
    camera.set_ec(0.0)
    shoot("CAL_shot1_fec0")
    pause()
    # Shot 2: fec=-2.0 (target)
    camera.set_fec(-2.0)
    camera.set_ec(0.0)
    shoot("CAL_shot2_fec-2")
    pause()
    # Shot 3: fec=-3.0 (probe)
    camera.set_fec(-3.0)
    shoot("CAL_shot3_probe")
    pause()
    # Post-probe: back to fec=-2.0 (cascade restores shot 2 settings)
    camera.set_fec(-2.0)
    shoot("CAL_post_probe")
    pause()
    # Then a few more at -2.0 to mimic post-cascade idle
    for i in range(3):
        shoot(f"CAL_idle_{i}")
        pause()

    # --- SWEEP path (post-cal) ---
    print()
    print("--- SWEEP path: batched setter, 3 replicas ---")
    pause(10.0)
    camera.set_aperture_iso_fec(aperture="f/5.6", iso=100, fec=-2.0)
    camera.set_ec(0.0)
    for i in range(3):
        shoot(f"SWEEP_rep{i}")
        pause()

    # --- SWEEP path right after CAL (re-cal -> sweep handoff) ---
    print()
    print("--- Handoff: cal then sweep, no recovery ---")
    pause(10.0)
    # Simulate cascade end
    camera.set_aperture("f/5.6")
    camera.set_iso(100)
    camera.set_fec(0.0)
    shoot("handoff_cal_shot1_fec0")
    pause()
    camera.set_fec(-2.0)
    shoot("handoff_cal_shot2_fec-2")
    pause()
    camera.set_fec(-3.0)
    shoot("handoff_cal_shot3_probe")
    pause()
    camera.set_fec(-2.0)
    shoot("handoff_cal_post_probe")
    pause()
    # Now immediately switch to sweep path
    camera.set_aperture_iso_fec(aperture="f/5.6", iso=100, fec=-2.0)
    camera.set_ec(0.0)
    for i in range(3):
        shoot(f"handoff_sweep_rep{i}")
        pause()

    print()
    print("=== Done. ===")


if __name__ == "__main__":
    main()
