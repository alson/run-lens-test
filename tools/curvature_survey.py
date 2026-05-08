"""Field-curvature survey across the offline rail-coverage corpus.

For every ``.../cropped`` scenario under a root, compute the per-crop
smoothed-argmax peak position (in raw-frame units) and report the
**corner spread** (max-peak − min-peak across the 9 crops) and an
estimate of the centre crop's **FWHM** in the same units.

Rationale: the live coarse-to-fine probe sizes its phase-C window
based on the observed corner spread. If most lens configurations have
a much smaller corner spread than the worst-case 793 µm we measured
on MC 105 + 61 mm, we can run with a tighter padding constant for the
typical case and let phase D catch the high-FC outliers.

Usage::

    cd run-lens-test
    uv run python -m tools.curvature_survey \\
        --root ~/shared/files/pics/digikam/test-raw-data \\
        --out /tmp/curvature_survey.csv

The script picks the **widest** aperture of each (lens, magnification)
group — that's where FWHM is smallest, so spread/FWHM is most punishing.
It writes one row per scenario; aggregate analysis is in the report
that follows.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# rail-coverage lives in a sister venv; add it to sys.path so we can
# reuse its scan/detect primitives without spawning a second uv project.
RAIL_COV_SRC = Path.home() / "shared/projects/personal/fotografie/analyse-lens-test/rail-coverage/src"
sys.path.insert(0, str(RAIL_COV_SRC))

from rail_coverage.scan import read_scenario  # noqa: E402
from lens_test_detect_core import smooth  # noqa: E402


_APERTURE_RE = re.compile(r"^F(\d+(?:\.\d+)?)$")


def parse_aperture(name: str) -> float | None:
    """Extract the f-number from a folder name like 'F5.6'."""
    m = _APERTURE_RE.match(name)
    return float(m.group(1)) if m else None


def smoothed_peak_and_fwhm(sizes: list[int]) -> tuple[float | None, float | None]:
    """Return (peak_frame_idx, fwhm_frames) using a 3-window smooth.

    Peak is the smoothed argmax in raw-frame coordinates (offset back
    by 1 to undo the centred 3-window). FWHM is the width at
    half max-above-baseline; baseline is the smoothed series minimum.
    Both ``None`` when the series is too short or has no dynamic range.
    """
    if len(sizes) < 5:
        return None, None
    sm = smooth(sizes, window=3)
    if sm.size == 0:
        return None, None
    pk_sm = int(np.argmax(sm))
    pk_raw = pk_sm + 1  # centre of 3-window
    base = float(sm.min())
    peak = float(sm.max())
    if peak <= base:
        return float(pk_raw), None
    if (peak - base) / peak < 0.10:
        # Effectively flat — no usable peak.
        return None, None
    half = base + 0.5 * (peak - base)
    # Walk left and right from the smoothed argmax to find half-max
    # crossings; linearly interpolate the crossing index.
    def crossing(start: int, step: int) -> float | None:
        i = start
        while 0 <= i + step < sm.size:
            y0, y1 = sm[i], sm[i + step]
            if (y0 - half) * (y1 - half) <= 0 and y0 != y1:
                # Linear interpolation: y0 + t*(y1-y0) = half
                t = (half - y0) / (y1 - y0)
                return float(i + t * step)
            i += step
        return None

    left = crossing(pk_sm, -1)
    right = crossing(pk_sm, +1)
    if left is None or right is None:
        return float(pk_raw), None
    return float(pk_raw), float(right - left)


@dataclass
class ScenarioMeasurement:
    date: str           # yyyy-mm-dd
    lens_config: str    # lens + magnification
    aperture: float
    n_frames: int       # of the centre crop's primary chunk
    n_valid_crops: int
    crop_spread_frames: float
    centre_fwhm_frames: float | None
    spread_over_fwhm: float | None
    crop_peaks: dict[str, float]  # crop_name → frame index


def _scenario_id(cropped_dir: Path) -> tuple[str, str, float | None]:
    """Pull (date, lens_config, aperture) from a cropped path layout.

    Expected: ``test-raw-data/<date>/<lens>/<mag>/F<ap>/cropped``.
    """
    parts = cropped_dir.parts
    # Walk back from 'cropped' to find date / lens / mag / aperture.
    cropped_idx = parts.index("cropped")
    aperture = parse_aperture(parts[cropped_idx - 1])
    mag = parts[cropped_idx - 2]
    lens = parts[cropped_idx - 3]
    date = parts[cropped_idx - 4]
    return date, f"{lens}/{mag}", aperture


def measure_scenario(cropped_dir: Path) -> ScenarioMeasurement | None:
    """Read a scenario, compute per-crop smoothed peaks, and the
    corner spread + centre FWHM."""
    try:
        scenario = read_scenario(cropped_dir)
    except Exception:
        return None
    date, lens_config, aperture = _scenario_id(cropped_dir)
    if aperture is None:
        return None

    centre_series = next((c for c in scenario.crops if c.name == "center"), None)
    if centre_series is None or len(centre_series.frames) < 5:
        return None
    centre_sizes = [f.size for f in centre_series.frames]
    centre_pk, centre_fwhm = smoothed_peak_and_fwhm(centre_sizes)
    if centre_pk is None:
        return None

    # Per-crop peak.
    crop_peaks: dict[str, float] = {}
    for crop in scenario.crops:
        if crop.name == "center":
            crop_peaks[crop.name] = centre_pk
            continue
        sizes = [f.size for f in crop.frames]
        if len(sizes) < 5:
            continue
        # Dynamic-range gate, like the live detector: drop crops whose
        # series doesn't swing enough to be a real bell.
        mx = max(sizes)
        if mx <= 0 or (mx - min(sizes)) / mx < 0.25:
            continue
        pk, _ = smoothed_peak_and_fwhm(sizes)
        if pk is None:
            continue
        crop_peaks[crop.name] = pk

    if len(crop_peaks) < 3:
        return None

    spread = max(crop_peaks.values()) - min(crop_peaks.values())
    ratio = (spread / centre_fwhm) if centre_fwhm and centre_fwhm > 0 else None
    return ScenarioMeasurement(
        date=date,
        lens_config=lens_config,
        aperture=aperture,
        n_frames=len(centre_sizes),
        n_valid_crops=len(crop_peaks),
        crop_spread_frames=spread,
        centre_fwhm_frames=centre_fwhm,
        spread_over_fwhm=ratio,
        crop_peaks=crop_peaks,
    )


def survey(root: Path, *, csv_fh=None, csv_writer=None) -> list[ScenarioMeasurement]:
    """Walk all cropped subdirs and measure each scenario.

    If ``csv_writer`` is provided, each successful measurement is also
    appended (and the file flushed) — so a kill mid-survey doesn't lose
    partial results. NFS round-trips make the full walk slow.
    """
    measurements: list[ScenarioMeasurement] = []
    print("walking tree for cropped dirs ...", file=sys.stderr, flush=True)
    cropped_dirs = sorted(root.rglob("cropped"))
    print(f"found {len(cropped_dirs)} cropped dirs", file=sys.stderr, flush=True)
    for i, cropped_dir in enumerate(cropped_dirs, 1):
        if not cropped_dir.is_dir():
            continue
        try:
            m = measure_scenario(cropped_dir)
        except Exception as e:
            print(f"  ! {cropped_dir}: {e}", file=sys.stderr, flush=True)
            continue
        if i % 25 == 0 or i == len(cropped_dirs):
            print(f"  [{i}/{len(cropped_dirs)}] {cropped_dir.parent.name}",
                  file=sys.stderr, flush=True)
        if m is not None:
            measurements.append(m)
            if csv_writer is not None:
                csv_writer.writerow([
                    m.date, m.lens_config, m.aperture, m.n_frames,
                    m.n_valid_crops,
                    f"{m.crop_spread_frames:.2f}",
                    f"{m.centre_fwhm_frames:.2f}" if m.centre_fwhm_frames else "",
                    f"{m.spread_over_fwhm:.3f}" if m.spread_over_fwhm else "",
                ])
                if csv_fh is not None:
                    csv_fh.flush()
    return measurements


def pick_widest_aperture_per_config(
    measurements: list[ScenarioMeasurement],
) -> list[ScenarioMeasurement]:
    """Group by (date, lens_config) and keep only the widest aperture.

    Widest = smallest f-number = where the bell is narrowest, so the
    spread/FWHM ratio is most demanding."""
    groups: dict[tuple[str, str], list[ScenarioMeasurement]] = defaultdict(list)
    for m in measurements:
        groups[(m.date, m.lens_config)].append(m)
    result: list[ScenarioMeasurement] = []
    for ms in groups.values():
        result.append(min(ms, key=lambda m: m.aperture))
    return result


def report_distribution(
    label: str, ms: list[ScenarioMeasurement], *, attr: str
) -> None:
    """Print percentiles for a numeric attribute across a measurement set."""
    values = [getattr(m, attr) for m in ms if getattr(m, attr) is not None]
    if not values:
        print(f"{label}: no values")
        return
    arr = np.asarray(values, dtype=float)
    pcts = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    print(
        f"{label} (n={len(values)}): "
        f"p10={pcts[0]:.2f}  p25={pcts[1]:.2f}  p50={pcts[2]:.2f}  "
        f"p75={pcts[3]:.2f}  p90={pcts[4]:.2f}  p95={pcts[5]:.2f}  p99={pcts[6]:.2f}  "
        f"max={arr.max():.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, type=Path,
        help="root of the test-raw-data tree (containing dated subdirs)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="optional CSV output path (one row per widest-aperture scenario)",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"root {args.root} not found", file=sys.stderr)
        return 1

    print(f"surveying {args.root} ...", file=sys.stderr)
    csv_fh = None
    csv_writer = None
    if args.out:
        csv_fh = args.out.open("w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["date", "lens_config", "aperture", "n_frames",
                             "n_valid_crops", "crop_spread_frames",
                             "centre_fwhm_frames", "spread_over_fwhm"])
    all_ms = survey(args.root, csv_fh=csv_fh, csv_writer=csv_writer)
    if csv_fh is not None:
        csv_fh.close()
    print(f"  measured {len(all_ms)} scenarios", file=sys.stderr)

    widest = pick_widest_aperture_per_config(all_ms)
    print(f"  → {len(widest)} unique (date, lens_config) groups (widest aperture per group)\n")

    print("Distribution of crop_spread / centre_FWHM across widest-aperture scenarios:")
    report_distribution("  spread/FWHM", widest, attr="spread_over_fwhm")

    print("\nDistribution of crop_spread (in raw-frame units):")
    report_distribution("  spread", widest, attr="crop_spread_frames")

    print("\nDistribution of centre FWHM (in raw-frame units):")
    report_distribution("  fwhm", widest, attr="centre_fwhm_frames")

    # Top 10 worst-case configs by spread/FWHM.
    rated = [m for m in widest if m.spread_over_fwhm is not None]
    rated.sort(key=lambda m: m.spread_over_fwhm, reverse=True)
    print("\nTop 15 worst-case spread/FWHM configurations:")
    print(f"  {'date':<10} {'lens_config':<70} {'f':>5} {'spread':>7} {'fwhm':>5} {'ratio':>6}")
    for m in rated[:15]:
        print(f"  {m.date:<10} {m.lens_config[:70]:<70} f/{m.aperture:<3g} "
              f"{m.crop_spread_frames:>7.1f} {m.centre_fwhm_frames:>5.1f} {m.spread_over_fwhm:>6.2f}")

    print("\nBottom 10 (best-case, lowest spread/FWHM):")
    print(f"  {'date':<10} {'lens_config':<70} {'f':>5} {'spread':>7} {'fwhm':>5} {'ratio':>6}")
    for m in rated[-10:]:
        print(f"  {m.date:<10} {m.lens_config[:70]:<70} f/{m.aperture:<3g} "
              f"{m.crop_spread_frames:>7.1f} {m.centre_fwhm_frames:>5.1f} {m.spread_over_fwhm:>6.2f}")

    if args.out:
        print(f"\nwrote {args.out} ({len(all_ms)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
