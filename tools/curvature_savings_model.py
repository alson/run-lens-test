"""Project savings across the curvature corpus at different padding levels.

Reads the per-scenario CSV from ``curvature_survey.py`` and, for each
row, models the implied phase-C window:

    window_frames = spread + 2 × padding × FWHM

We then compute the projected savings against an assumed user-jog of
``user_jog_factor × FWHM`` raw frames — a single multiplier captures
"how wide does the user jog relative to the bell". 30× FWHM is a
reasonable default (matches the typical 5-15 × DOF jog at f/4.5 → DOF ≈
2 × FWHM, so 30 × FWHM = 15 × DOF). The savings projection is

    savings = 1 − window_frames / user_jog_frames

and reports percentiles + counts of scenarios where the window
*exceeds* the user's jog (phase D fires / the algorithm is no faster
than today).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                spread = float(r["crop_spread_frames"])
                fwhm = float(r["centre_fwhm_frames"]) if r["centre_fwhm_frames"] else None
            except (ValueError, KeyError):
                continue
            if fwhm is None or fwhm <= 0:
                continue
            rows.append({
                "lens_config": r["lens_config"],
                "aperture": float(r["aperture"]),
                "spread": spread,
                "fwhm": fwhm,
                "ratio": spread / fwhm,
            })
    return rows


def project_savings(
    rows: list[dict],
    *,
    padding: float,
    user_jog_factor: float,
) -> tuple[np.ndarray, int]:
    """Return (savings_array, n_phase_d_fires).

    savings_array[i] = 1 - window/jog for row i (clamped to ≥ 0). When
    window ≥ jog, savings is 0 and phase_d_fires increments.
    """
    spread = np.array([r["spread"] for r in rows], dtype=float)
    fwhm = np.array([r["fwhm"] for r in rows], dtype=float)
    window = spread + 2 * padding * fwhm
    jog = user_jog_factor * fwhm
    fires = int(np.sum(window >= jog))
    savings = np.where(window >= jog, 0.0, 1.0 - window / jog)
    return savings, fires


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="output CSV from curvature_survey.py")
    p.add_argument("--user-jog", type=float, default=30.0,
                   help="user-jog factor × FWHM (default 30 ≈ 15 × DOF)")
    args = p.parse_args(argv)

    rows = load_csv(args.csv)
    if not rows:
        print(f"no usable rows in {args.csv}", file=sys.stderr)
        return 1
    print(f"loaded {len(rows)} scenarios from {args.csv}\n")

    print(f"projected savings at user_jog = {args.user_jog} × FWHM "
          f"(matches a typical 15 × DOF user jog at f/4.5):\n")
    print(f"{'pad':>5} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} "
          f"{'p99':>7} {'mean':>7} {'phaseD':>8}")
    for padding in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        savings, fires = project_savings(rows, padding=padding, user_jog_factor=args.user_jog)
        pcts = np.percentile(savings, [10, 25, 50, 75, 90, 99])
        print(f"{padding:>5.1f} "
              f"{pcts[0]*100:>6.0f}% "
              f"{pcts[1]*100:>6.0f}% "
              f"{pcts[2]*100:>6.0f}% "
              f"{pcts[3]*100:>6.0f}% "
              f"{pcts[4]*100:>6.0f}% "
              f"{pcts[5]*100:>6.0f}% "
              f"{savings.mean()*100:>6.0f}% "
              f"{fires:>3d}/{len(rows):>3d}")

    print("\n(phaseD = scenarios where the projected phase-C window is ≥ "
          "user jog; on those, the algorithm reverts to a full sweep + "
          "phase D extension, no savings.)")

    # Variant: model spread × scale factor (sensitivity). What if we
    # under-estimate spread by 20%? The bracket extends as observed,
    # but real corners may sit outside. Compute "phase D fires when
    # observed-window doesn't cover the true spread" too. (Not modelled
    # here; would need a second-pass estimator.)
    return 0


if __name__ == "__main__":
    sys.exit(main())
