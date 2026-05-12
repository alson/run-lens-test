#!/usr/bin/env python3
"""Interactive end-to-end test of the manual-aperture calibration
cascade. No rail required — captures are taken in place at the
rig's current focus.

Walks the full aperture sequence for a non-CPU lens config:

* First aperture: coarse power search → back-calc → verify with
  user-dialed flash power (group C / A on SB-5000 commander).
* Subsequent apertures: deterministic per-step power target computed
  from previous aperture's calibrated state.

You'll be prompted to:

1. Turn the lens aperture ring to each marked stop in turn.
2. Dial the flash power on the SB-5000 commander to match the
   rig's calculated target (the rig says "1/16 -1/3" etc. — match
   it exactly on the commander).

For each aperture the rig reports the final (power, ISO) it
converged on. Per-aperture captures land in ``<out>/calibration/``.

Usage::

    python tools/manual_cascade_live.py \
        --slug "Nikon Micro-Nikkor 55mm f_2.8 AIS 287318" \
        --f-ring 2.8 --m-indicated 0.5
"""

from __future__ import annotations

import argparse
import math
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
from session.flash_power import power_label  # noqa: E402
from session.manual_aperture import ring_and_label_sequence  # noqa: E402
from ui.prompts import (
    confirm,
    prompt_set_flash_power,
    prompt_turn_ring,
)  # noqa: E402


def _measure_factory():
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
    return _measure


def _baseline_median(measure) -> float | None:
    ref = config.BASELINE_REFERENCE_IMAGE
    if ref is None or not Path(ref).exists():
        return None
    return measure(ref)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug", required=True,
        help="Lens-config slug (used for the file path).",
    )
    parser.add_argument(
        "--f-ring", type=float, required=True,
        help="Lens ring max aperture (e.g. 2.8 for the 55 AIS).",
    )
    parser.add_argument(
        "--m-indicated", type=float, required=True,
        help="Magnification indicated on the lens barrel "
             "(0 for infinity, 0.5 for 1:2, etc.).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output dir for captures. Default: tmpdir.",
    )
    parser.add_argument(
        "--max-fstop", type=float, default=None,
        help="Stop the sequence at this effective f-number. "
             f"Default: config.MAX_FSTOP ({config.MAX_FSTOP}).",
    )
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(
        prefix="manual_cascade_live_"
    ))
    out.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out}")

    max_fstop = args.max_fstop or config.MAX_FSTOP

    pairs = ring_and_label_sequence(
        f_ring_max=args.f_ring,
        m_indicated=args.m_indicated,
        whole_stop_scale=config.WHOLE_STOP_SCALE,
        max_fstop=max_fstop,
        one_third_stop_scale=config.ONE_THIRD_STOP_SCALE,
    )
    if not pairs:
        print("ERROR: aperture sequence empty (effective widest > max_fstop?)")
        sys.exit(1)

    print(f"\nSequence ({len(pairs)} apertures, ring stops):")
    print("  " + ", ".join(f"f/{ring:g}" for ring, _ in pairs))
    if not confirm("\n[ENTER] to start, [q] to abort: "):
        print("aborted")
        sys.exit(0)

    cam = NikonZ7()
    print("\nopening camera...")
    cam.open()
    cam.verify_healthy(min_widgets=config.MIN_CONFIG_TREE_SIZE)
    cam.apply_session_config(
        iso=config.ISO_FLOOR,
        shutter_candidates=config.SHUTTER_SPEED_CANDIDATES,
        image_quality=config.IMAGE_QUALITY,
        fec=0.0,
    )
    measure = _measure_factory()
    baseline = _baseline_median(measure)
    if baseline is not None:
        print(f"baseline median: {baseline:.1f}")
    else:
        print("baseline reference not found — first capture sets baseline")

    calib = Calibrator(
        camera=cam,
        measure=measure,
        config=config,
        shots_csv=None,
    )

    prev_power: float | None = None
    prev_iso: int | None = None
    prev_ev_delta: float = 0.0
    results: list[tuple[str, float, int, float]] = []  # (label, power, iso, delta)
    aborted = False

    try:
        for index, (ring, label) in enumerate(pairs):
            print(f"\n=== {index + 1}/{len(pairs)}  ring f/{ring:g} ===")
            if not prompt_turn_ring(
                label, ring_stop=ring, m_indicated=args.m_indicated,
            ):
                print("aborted at ring-turn prompt")
                aborted = True
                break

            if index == 0:
                print("first aperture: coarse power search + back-calc")
                cal_result, power, iso = (
                    calib.calibrate_first_aperture_manual(
                        aperture=label,
                        lens_dir=out,
                        lens_slug=args.slug,
                        session_id="live",
                        baseline_median=baseline,
                        prompt_set_flash_power=prompt_set_flash_power,
                    )
                )
                if baseline is None:
                    baseline = cal_result.baseline_median
            else:
                stops_diff = 2.0 * math.log2(ring / pairs[index - 1][0])
                print(f"subsequent aperture: +{stops_diff:.2f} EV step")
                cal_result, power, iso = (
                    calib.calibrate_subsequent_aperture_manual(
                        aperture=label,
                        lens_dir=out,
                        lens_slug=args.slug,
                        session_id="live",
                        baseline_median=baseline,
                        prev_power_stops=prev_power,
                        prev_iso=prev_iso,
                        prev_ev_delta=prev_ev_delta,
                        aperture_stops_diff=stops_diff,
                        prompt_set_flash_power=prompt_set_flash_power,
                    )
                )

            ap_cal = cal_result.per_aperture[label]
            print(f"  → power = {power_label(power)} ({power:.3f} stops below full)")
            print(f"  → ISO   = {iso}")
            print(f"  → final delta = {ap_cal.final_ev_delta:+.3f} EV  "
                  f"(hit_tolerance={ap_cal.hit_tolerance})")
            print(f"  → shots taken = {ap_cal.test_shots_taken}")
            results.append((ring, power, iso, ap_cal.final_ev_delta))
            prev_power = power
            prev_iso = iso
            prev_ev_delta = ap_cal.final_ev_delta
    finally:
        cam.close()
        print("\ncamera closed.")

    print("\n=== SUMMARY ===")
    print(f"{'ring':<10} {'power':<12} {'ISO':<6} {'delta EV':<10}")
    for ring, power, iso, delta in results:
        print(f"f/{ring:<8g} {power_label(power):<12} {iso:<6} {delta:+.3f}")
    if aborted:
        print("(session aborted before completion)")


if __name__ == "__main__":
    main()
