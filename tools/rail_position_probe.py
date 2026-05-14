"""Test rail-position dependence of the fec=-2.0 reading.

Move the rail in a sweep from 0 to -3 mm in 0.3-mm steps and shoot
at fec=-2.0 each time. If the reading varies systematically with
rail position, the +0.4 EV observed during the user's actual run
sweep at rail 26 is rail-position-dependent rig brightness, not a
runner-side bug.

Assumes the rail is currently at the position the user aborted at.
We add a relative offset from there.
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
from stackshot.controller import open_ftdi as open_stackshot
from stackshot.protocol import Direction


def main() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix="railprobe_"))
    print(f"=== Rail-position probe ===", flush=True)
    print(f"output dir: {out_dir}", flush=True)

    rail = open_stackshot(
        poll_interval_s=config.STACKSHOT_POLL_INTERVAL_S,
        move_timeout_s=config.STACKSHOT_MOVE_TIMEOUT_S,
    )
    start_pos = rail.get_position()
    print(f"rail starting position: {start_pos:+.3f} mm", flush=True)

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
    camera.set_fec(-2.0)

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

    def shoot(label: str, pos_mm: float) -> float:
        dest = out_dir / f"{label}.jpg"
        camera.capture_to(dest)
        med = measure(dest)
        ev = ev_delta(med, 129.0)
        print(f"  pos={pos_mm:+.3f} mm  {label:>20s}  med={med:6.1f}  ev={ev:+.3f}",
              flush=True)
        return ev

    # Sweep from start_pos to start_pos - 3.0 mm in 0.5-mm steps.
    step_mm = -0.5
    n_steps = 7
    print()
    print(f"--- Sweep at fec=-2.0, step={step_mm} mm, {n_steps} positions ---")
    current = start_pos
    for i in range(n_steps):
        if i > 0:
            rail.move_and_wait(
                Direction.FORWARD if step_mm > 0 else Direction.BACKWARD,
                abs(step_mm),
                backlash=True,
            )
            current = rail.get_position()
        time.sleep(3.0)  # recycle
        shoot(f"pos_{i:02d}", current)

    print()
    print(f"--- Reverse sweep back to start (test reversibility) ---")
    for i in range(n_steps):
        if i > 0:
            rail.move_and_wait(
                Direction.FORWARD if step_mm < 0 else Direction.BACKWARD,
                abs(step_mm),
                backlash=True,
            )
            current = rail.get_position()
        time.sleep(3.0)
        shoot(f"back_{i:02d}", current)

    # Return rail to original start.
    final = rail.get_position()
    delta = start_pos - final
    if abs(delta) > 0.001:
        rail.move_and_wait(
            Direction.FORWARD if delta > 0 else Direction.BACKWARD,
            abs(delta),
            backlash=True,
        )
    print(f"\nrail returned to {rail.get_position():+.3f} mm "
          f"(started at {start_pos:+.3f} mm)")
    print("=== Done. ===")


if __name__ == "__main__":
    main()
