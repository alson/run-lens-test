"""Single FEC=0 probe per aperture for ambient-light contribution test.

For each aperture in {f/5.6, f/8, f/11, f/16}, captures one frame at
FEC=0 / ISO from RANGE_EXTENSION, then reports the EV delta vs the
session baseline (median = 129). Compares against the two reference
cals from session 2026-05-13_230021 (150 + 5T at start, bare 150 at
end) so we can tell whether the rig has shifted, whether the LED
focus lamp is contributing, or both.

Run twice — once with the LED off, once with it on — and diff.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Project root on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from camera.nikon_z7 import NikonZ7
from exposure.analyse import crop_box, ev_delta, measure_crop


BASELINE_MEDIAN = 129.0  # session-level baseline from 2026-05-13_230021

# Reference cal data (shot 1 EVs at FEC=0).
REF_150_5T = {
    "f/5.6": 0.931,
    "f/8":   0.415,
    "f/11":  0.479,
    "f/16":  0.644,
}
REF_150_BARE = {
    "f/5.6": 1.439,
    "f/8":   0.968,
    "f/11":  1.040,
    "f/16":  1.127,
}
APERTURES = ["f/5.6", "f/8", "f/11", "f/16"]


def _iso_for(aperture: str) -> int:
    ap = float(aperture.lstrip("f/").lstrip("F/"))
    table = config.RANGE_EXTENSION
    if ap in table:
        return int(table[ap][0])
    keys = sorted(table.keys())
    pick = keys[0]
    for k in keys:
        if k <= ap:
            pick = k
    return int(table[pick][0])


def main(label: str) -> dict[str, float]:
    print(f"=== Probe (FEC=0, no cascade): {label} ===", flush=True)
    out_dir = Path(tempfile.mkdtemp(prefix="lighttest_"))
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

    # Find crop box from the first capture.
    box: object | None = None

    shot1_evs: dict[str, float] = {}
    print()
    for ap in APERTURES:
        iso = _iso_for(ap)
        camera.set_aperture(ap)
        camera.set_iso(iso)
        camera.set_fec(0.0)
        if hasattr(camera, "set_ec"):
            camera.set_ec(0.0)
        safe = ap.replace("/", "_").replace(".", "_")
        dest = out_dir / f"probe_{safe}.jpg"
        camera.capture_to(dest)
        if box is None:
            from PIL import Image
            with Image.open(str(dest)) as im:
                w, h = im.size
            box = crop_box(
                w, h,
                config.CROP_W_FRAC, config.CROP_H_FRAC,
                config.CROP_X_FRAC, config.CROP_Y_FRAC,
            )
        med = measure_crop(dest, box).p50
        ev = ev_delta(med, BASELINE_MEDIAN)
        shot1_evs[ap] = ev
        print(
            f"  {ap:6s}  iso={iso:4d}  fec=+0.00  med={med:6.1f}  ev={ev:+.3f}",
            flush=True,
        )

    print()
    print(f"=== {label} — FEC=0 probe comparison ===")
    print(
        f"  {'aperture':6s}  {'current':>8s}  {'150+5T':>8s}  {'Δ5T':>8s}  "
        f"{'bare 150':>8s}  {'Δbare':>8s}"
    )
    sum_d5t = 0.0
    sum_dbare = 0.0
    for ap in APERTURES:
        ev = shot1_evs[ap]
        r5t = REF_150_5T[ap]
        rb = REF_150_BARE[ap]
        d5t = ev - r5t
        db = ev - rb
        sum_d5t += d5t
        sum_dbare += db
        print(
            f"  {ap:6s}  {ev:+8.3f}  {r5t:+8.3f}  {d5t:+8.3f}  "
            f"{rb:+8.3f}  {db:+8.3f}"
        )
    print(
        f"  mean Δ vs 150+5T  : {sum_d5t / len(APERTURES):+.3f} EV"
    )
    print(
        f"  mean Δ vs bare 150: {sum_dbare / len(APERTURES):+.3f} EV"
    )
    return shot1_evs


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    main(label)
