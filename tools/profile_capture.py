"""Per-shot timing profiler for the Z7 capture pipeline.

Drives a few real captures through both the unbatched (set_aperture +
set_iso + set_fec) and batched (set_aperture_iso_fec) code paths and
prints a phase breakdown so we can quantify where the wall-clock time
goes between shots — and confirm what `set_aperture_iso_fec` saves on
the rig.

Usage::

    uv run python -m tools.profile_capture          # 3 shots, default apertures
    uv run python -m tools.profile_capture --shots 5
    uv run python -m tools.profile_capture --apertures f/4,f/8,f/16

The script assumes the camera is already powered on, in M mode, with
the SB-5000 commander mounted, and connected over USB. It does **not**
talk to the StackShot — focus position is whatever you left it at.
JPEGs are written to ``/tmp/profile_capture_*.jpg`` and overwritten
each run.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import time
from pathlib import Path
from typing import Iterator

from camera.nikon_z7 import NikonZ7


@contextlib.contextmanager
def _timed(samples: list[float], label: str) -> Iterator[None]:
    t0 = time.monotonic()
    yield
    samples.append(time.monotonic() - t0)
    # Realtime feedback so the user can see something is happening.
    print(f"    {label:18s} {samples[-1]*1000:7.0f} ms")


def _run_pass(
    cam: NikonZ7,
    *,
    label: str,
    apertures: list[str],
    iso: str,
    fec: float,
    shots: int,
    batched: bool,
    out_dir: Path,
) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {
        "set_config_total": [],
        "capture_to": [],
        "shot_total": [],
    }
    print()
    print(f"=== {label} (batched={batched}, shots/aperture={shots}) ===")
    for ap in apertures:
        print(f"  aperture {ap}")
        for i in range(shots):
            print(f"  shot {i + 1}/{shots}")
            t_shot_start = time.monotonic()

            t0 = time.monotonic()
            if batched:
                cam.set_aperture_iso_fec(aperture=ap, iso=iso, fec=fec)
            else:
                cam.set_aperture(ap)
                cam.set_iso(iso)
                cam.set_fec(fec)
            timings["set_config_total"].append(time.monotonic() - t0)
            print(f"    {'set_config_total':18s} "
                  f"{timings['set_config_total'][-1]*1000:7.0f} ms")

            ap_slug = ap.replace("/", "_").replace(".", "_")
            dest = out_dir / f"profile_{label.lower()}_{ap_slug}_{i:02d}.jpg"
            with _timed(timings["capture_to"], "capture_to"):
                cam.capture_to(dest)

            timings["shot_total"].append(time.monotonic() - t_shot_start)
            print(f"    {'shot_total':18s} "
                  f"{timings['shot_total'][-1]*1000:7.0f} ms")
    return timings


def _summarise(name: str, timings: dict[str, list[float]]) -> None:
    print()
    print(f"--- {name} summary ---")
    for k, vals in timings.items():
        if not vals:
            continue
        med = statistics.median(vals)
        lo, hi = min(vals), max(vals)
        print(f"  {k:18s} n={len(vals):3d}  "
              f"median={med:6.2f}s  min={lo:6.2f}s  max={hi:6.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shots", type=int, default=3,
        help="shots per aperture (default 3)",
    )
    parser.add_argument(
        "--apertures", default="f/4,f/8,f/16",
        help="comma-separated aperture choices (default f/4,f/8,f/16)",
    )
    parser.add_argument(
        "--iso", default="100",
        help="ISO to set (default 100; uses 400 if your widest aperture "
             "is f/16 to avoid range warnings)",
    )
    parser.add_argument(
        "--fec", type=float, default=0.0,
        help="FEC EV to set (default 0)",
    )
    parser.add_argument(
        "--out-dir", default="/tmp",
        help="where to drop profile JPEGs",
    )
    parser.add_argument(
        "--mode", choices=("both", "individual", "batched"), default="both",
        help="which path to exercise",
    )
    args = parser.parse_args()

    apertures = [a.strip() for a in args.apertures.split(",") if a.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = NikonZ7()
    cam.open()
    try:
        cam.verify_healthy()
        # Set the per-session defaults so capturetarget=Internal RAM,
        # autofocus=Off, VR=Off, etc. The first call also incurs the
        # apply_session_config cost itself — print it for context.
        t0 = time.monotonic()
        cam.apply_session_config(iso=args.iso, fec=args.fec)
        print(f"apply_session_config (warmup): "
              f"{(time.monotonic() - t0)*1000:.0f} ms")

        # Warm-up shot (first capture often pays a one-off PTP cost
        # we don't want to skew the median).
        warm_dest = out_dir / "profile_warmup.jpg"
        t0 = time.monotonic()
        cam.capture_to(warm_dest)
        print(f"warmup capture: {(time.monotonic() - t0)*1000:.0f} ms "
              f"({os.path.getsize(warm_dest) // 1024} KB)")

        results: dict[str, dict[str, list[float]]] = {}
        if args.mode in ("both", "individual"):
            results["individual"] = _run_pass(
                cam,
                label="INDIVIDUAL",
                apertures=apertures,
                iso=args.iso,
                fec=args.fec,
                shots=args.shots,
                batched=False,
                out_dir=out_dir,
            )
        if args.mode in ("both", "batched"):
            results["batched"] = _run_pass(
                cam,
                label="BATCHED",
                apertures=apertures,
                iso=args.iso,
                fec=args.fec,
                shots=args.shots,
                batched=True,
                out_dir=out_dir,
            )

        for name, timings in results.items():
            _summarise(name, timings)

        if "individual" in results and "batched" in results:
            ind_med = statistics.median(results["individual"]["shot_total"])
            bat_med = statistics.median(results["batched"]["shot_total"])
            print()
            print(f"shot_total median: individual={ind_med:.2f}s  "
                  f"batched={bat_med:.2f}s  delta={ind_med - bat_med:+.2f}s "
                  f"({(ind_med - bat_med) / ind_med * 100:+.1f}%)")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
