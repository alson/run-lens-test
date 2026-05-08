"""Replay the coarse-to-fine probe on the offline rail-coverage corpus.

The live ``probe_simulator`` validates the algorithm on three live
calibration sessions. Those three sessions are at p95-max of the
field-curvature distribution — they aren't representative of the
typical lens config. To validate phase-A density and padding choices
across a *broader* sample, this tool replays the probe algorithm on
each scenario in the offline corpus (``test-raw-data/`` tree).

Algorithm per scenario:

1. Read all frames per crop at the *widest* aperture of the
   (date, lens, magnification) group, sorted by capture sequence.
2. Pick ``N`` evenly-spaced frames as the phase-A samples.
3. Run :func:`session.probe.phase_a_peaks` to get per-crop peak
   positions (in frame units).
4. Compute the phase-C window in frame units:
   ``[min_peak − padding × FWHM_widest, max_peak + padding × FWHM_widest]``.
5. Count how many of the original frames fall in the window =
   phase-C captures. Savings = ``1 − (phase_a_n + phase_c_n) / total``.
6. For each crop, compare its full-sweep peak file size vs its
   peak file size *within the phase-C window* = rel-loss %.

Key limitation vs the live simulator:

- The older corpus has each aperture captured as a separate
  scenario (not aperture-cycled at each rail step), so we can only
  validate the algorithm at the widest aperture. Cross-aperture
  consistency must come from the live three-session simulator.
- Sub-frame phase-A sample selection: the older corpus has
  variable replica counts per rail position (mostly ~3). We pick
  evenly-spaced ``frame indices`` from the sorted seq list, which
  may oversample one rail position when replicas cluster. That's
  conservative — denser phase-A is harder, not easier.

Use::

    cd run-lens-test
    uv run python -m tools.probe_replay_corpus \\
        --root ~/shared/files/pics/digikam/test-raw-data \\
        --phase-a-points 9 --padding 1.5

    # Or sweep the phase-A density:
    uv run python -m tools.probe_replay_corpus \\
        --root ~/shared/files/pics/digikam/test-raw-data \\
        --phase-a-sweep
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# rail-coverage's scan module reads the older filename layout.
RAIL_COV_SRC = Path.home() / "shared/projects/personal/fotografie/analyse-lens-test/rail-coverage/src"
sys.path.insert(0, str(RAIL_COV_SRC))

from rail_coverage.scan import read_scenario  # noqa: E402

from session.probe import phase_a_peaks, phase_c_window_mm  # noqa: E402

from tools.curvature_survey import (  # noqa: E402
    parse_aperture,
    smoothed_peak_and_fwhm,
)


@dataclass
class ReplayResult:
    date: str
    lens_config: str
    aperture: float
    total_frames: int
    phase_a_frames: int
    phase_c_frames: int
    inconclusive: bool
    fell_back: bool   # phase A inconclusive → fall back to full sweep
    worst_rel_loss_pct: float
    crops_outside_window: int
    n_valid_crops: int
    fwhm_frames: float | None
    spread_frames: float

    @property
    def savings_pct(self) -> float:
        """Per-aperture savings ratio.

        This *overstates* phase-A overhead vs the live algorithm, where
        phase A runs once across the lens config (1 ap × 1 replica)
        but phase C captures all K apertures × R replicas. The older
        corpus has each aperture as a separate scenario, so the
        per-aperture replay can't see that amortization. Use
        :meth:`amortized_savings_pct` for a like-for-like compare.
        """
        used = self.phase_a_frames + self.phase_c_frames
        if self.total_frames <= 0:
            return 0.0
        return 100 * (1 - used / self.total_frames)

    def amortized_savings_pct(self, *, k_apertures: int = 5) -> float:
        """Cross-aperture savings ratio with phase-A amortized over K.

        Models the live-algorithm economics on a corpus that's only
        per-aperture: ``savings = 1 − (N/K + phase_c) / total``.
        Phase-A captures (1 aperture × 1 replica) divide by K; phase-C
        captures (K apertures × R replicas) ratio with the per-aperture
        full sweep.
        """
        if self.total_frames <= 0 or k_apertures < 1:
            return 0.0
        used = self.phase_a_frames / k_apertures + self.phase_c_frames
        return 100 * (1 - used / self.total_frames)


def _scenario_id(cropped_dir: Path) -> tuple[str, str, float | None]:
    parts = cropped_dir.parts
    cropped_idx = parts.index("cropped")
    aperture = parse_aperture(parts[cropped_idx - 1])
    mag = parts[cropped_idx - 2]
    lens = parts[cropped_idx - 3]
    date = parts[cropped_idx - 4]
    return date, f"{lens}/{mag}", aperture


def _per_crop_frame_sizes(cropped_dir: Path) -> dict[str, list[int]]:
    """Return ``{crop_name: [size_seq_0, size_seq_1, ...]}``.

    Frames are sorted by capture sequence number, which is monotonic
    in time and ~monotonic in rail position within a single sweep.
    """
    scenario = read_scenario(cropped_dir)
    out: dict[str, list[int]] = {}
    for crop in scenario.crops:
        if not crop.frames:
            continue
        out[crop.name] = [f.size for f in crop.frames]
    return out


def replay_scenario(
    cropped_dir: Path,
    *,
    phase_a_points: int,
    padding_x_fwhm: float,
) -> ReplayResult | None:
    date, lens_config, aperture = _scenario_id(cropped_dir)
    if aperture is None:
        return None
    crop_sizes = _per_crop_frame_sizes(cropped_dir)
    if "center" not in crop_sizes:
        return None
    centre_sizes = crop_sizes["center"]
    n = len(centre_sizes)
    if n < phase_a_points + 3:
        return None  # not enough frames for the algorithm to make sense

    centre_pk_full, centre_fwhm = smoothed_peak_and_fwhm(centre_sizes)
    if centre_fwhm is None or centre_fwhm <= 0:
        return None

    # Phase A: pick N evenly-spaced frame indices from the full sweep.
    pick_idx = np.linspace(0, n - 1, phase_a_points).round().astype(int)
    phase_a_per_crop: dict[str, list[int]] = {}
    for name, sizes in crop_sizes.items():
        if len(sizes) != n:
            continue  # mismatched length — skip
        phase_a_per_crop[name] = [sizes[i] for i in pick_idx]

    pa = phase_a_peaks(phase_a_per_crop)
    if pa.inconclusive:
        # Fall back to the full sweep.
        return ReplayResult(
            date=date, lens_config=lens_config, aperture=aperture,
            total_frames=n, phase_a_frames=phase_a_points,
            phase_c_frames=n,  # full uniform sweep
            inconclusive=True, fell_back=True,
            worst_rel_loss_pct=0.0, crops_outside_window=0,
            n_valid_crops=len(pa.crop_peaks),
            fwhm_frames=centre_fwhm,
            spread_frames=pa.corner_spread_samples,
        )

    # Phase C window. Convert phase-A sample coordinates back to
    # full-sweep frame indices via the linspace inverse: sample i ↔
    # frame index pick_idx[i].
    sample_to_frame = pick_idx.astype(float)
    # Linear interp the sub-sample peak position in frame units.
    def _sample_to_frame_idx(sample_pos: float) -> float:
        # pick_idx maps integer phase-A sample → frame index. For a
        # fractional sample position ``x``, return the linear
        # interpolation between adjacent pick_idx values.
        lo = int(np.floor(sample_pos))
        hi = lo + 1
        if lo < 0:
            return float(sample_to_frame[0])
        if hi >= len(sample_to_frame):
            return float(sample_to_frame[-1])
        frac = sample_pos - lo
        return float(
            sample_to_frame[lo] + frac * (sample_to_frame[hi] - sample_to_frame[lo])
        )

    min_frame = _sample_to_frame_idx(pa.min_peak_sample or 0.0)
    max_frame = _sample_to_frame_idx(pa.max_peak_sample or 0.0)
    pad_frames = padding_x_fwhm * centre_fwhm
    window_lo = int(round(min_frame - pad_frames))
    window_hi = int(round(max_frame + pad_frames))
    # Clamp to capture range.
    window_lo = max(0, window_lo)
    window_hi = min(n - 1, window_hi)
    phase_c_n = max(0, window_hi - window_lo + 1)

    # Per-crop rel-loss: full-sweep peak vs in-window peak.
    worst_loss = 0.0
    outside = 0
    for name, sizes in crop_sizes.items():
        if len(sizes) != n:
            continue
        full_peak = max(sizes)
        if full_peak <= 0:
            continue
        full_peak_idx = int(np.argmax(sizes))
        if not (window_lo <= full_peak_idx <= window_hi):
            outside += 1
        window_sizes = sizes[window_lo: window_hi + 1] or [0]
        window_peak = max(window_sizes)
        if window_peak <= 0:
            continue
        rel_loss = 100 * (full_peak - window_peak) / full_peak
        if rel_loss > worst_loss:
            worst_loss = rel_loss

    return ReplayResult(
        date=date, lens_config=lens_config, aperture=aperture,
        total_frames=n,
        phase_a_frames=phase_a_points,
        phase_c_frames=phase_c_n,
        inconclusive=False, fell_back=False,
        worst_rel_loss_pct=worst_loss,
        crops_outside_window=outside,
        n_valid_crops=len(pa.crop_peaks),
        fwhm_frames=centre_fwhm,
        spread_frames=pa.corner_spread_samples,
    )


_widest_cache: list[Path] | None = None


def _widest_dirs(root: Path) -> list[Path]:
    """Cached corpus walk: one widest-aperture cropped dir per
    (date, lens, magnification) group. The walk is the slow part on
    NFS (5+ min), so the sweep modes reuse it across iterations."""
    global _widest_cache
    if _widest_cache is not None:
        return _widest_cache
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for cropped_dir in sorted(root.rglob("cropped")):
        if not cropped_dir.is_dir():
            continue
        date, lens_config, aperture = _scenario_id(cropped_dir)
        if aperture is None:
            continue
        groups[(date, lens_config)].append(cropped_dir)

    def _ap_of(d: Path) -> float:
        _, _, ap = _scenario_id(d)
        return ap if ap is not None else 999.0

    widest = [min(dirs, key=_ap_of) for dirs in groups.values()]
    _widest_cache = widest
    return widest


def replay_corpus(
    root: Path,
    *,
    phase_a_points: int,
    padding_x_fwhm: float,
) -> list[ReplayResult]:
    """Replay every cropped/ scenario at its widest aperture."""
    results: list[ReplayResult] = []
    for widest_dir in _widest_dirs(root):
        try:
            r = replay_scenario(
                widest_dir,
                phase_a_points=phase_a_points,
                padding_x_fwhm=padding_x_fwhm,
            )
        except Exception as e:
            print(f"  ! {widest_dir}: {e}", file=sys.stderr)
            continue
        if r is not None:
            results.append(r)
    return results


def report_summary(
    label: str, results: list[ReplayResult], *, k_apertures: int = 5,
) -> None:
    if not results:
        print(f"{label}: no usable scenarios")
        return
    per_ap = np.array([r.savings_pct for r in results])
    amort = np.array([r.amortized_savings_pct(k_apertures=k_apertures)
                      for r in results])
    losses = np.array([r.worst_rel_loss_pct for r in results])
    inconclusive = sum(1 for r in results if r.inconclusive)
    pcts = np.percentile(per_ap, [10, 50, 90])
    apcts = np.percentile(amort, [10, 50, 90])
    loss_pcts = np.percentile(losses, [50, 90, 99])
    n_loss_over_2 = int(np.sum(losses > 2.0))
    n_loss_over_10 = int(np.sum(losses > 10.0))
    print(f"{label}  n={len(results):3d}  "
          f"per-ap p10/p50/p90 = {pcts[0]:>3.0f}/{pcts[1]:>3.0f}/{pcts[2]:>3.0f}%  "
          f"amort(K={k_apertures}) p10/p50/p90 = {apcts[0]:>3.0f}/{apcts[1]:>3.0f}/{apcts[2]:>3.0f}%  "
          f"rel-loss p50/p90/p99 = {loss_pcts[0]:.1f}/{loss_pcts[1]:.1f}/{loss_pcts[2]:.1f}%  "
          f"loss>2%: {n_loss_over_2:2d}, >10%: {n_loss_over_10:2d}  "
          f"inconc: {inconclusive}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path,
                   help="root of the test-raw-data tree")
    p.add_argument("--phase-a-points", type=int, default=9)
    p.add_argument("--padding", type=float, default=1.5,
                   help="phase-C padding × FWHM_widest")
    p.add_argument("--phase-a-sweep", action="store_true",
                   help="sweep phase-A points in {3,5,7,9,11,15,21}")
    p.add_argument("--padding-sweep", action="store_true",
                   help="sweep padding in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}")
    p.add_argument("--top-losses", type=int, default=0,
                   help="show the N worst-rel-loss scenarios at the chosen point")
    args = p.parse_args(argv)

    if not args.root.is_dir():
        print(f"root {args.root} not found", file=sys.stderr)
        return 1

    if args.phase_a_sweep:
        for n in (3, 5, 7, 9, 11, 15, 21):
            print(f"--- phase-A points = {n}, padding = {args.padding} × FWHM ---",
                  file=sys.stderr, flush=True)
            results = replay_corpus(args.root, phase_a_points=n, padding_x_fwhm=args.padding)
            report_summary(f"N={n:2d}", results)
        return 0

    if args.padding_sweep:
        for pad in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            print(f"--- phase-A points = {args.phase_a_points}, padding = {pad} × FWHM ---",
                  file=sys.stderr, flush=True)
            results = replay_corpus(
                args.root, phase_a_points=args.phase_a_points, padding_x_fwhm=pad
            )
            report_summary(f"pad={pad:.1f}", results)
        return 0

    results = replay_corpus(
        args.root,
        phase_a_points=args.phase_a_points,
        padding_x_fwhm=args.padding,
    )
    report_summary(
        f"N={args.phase_a_points}, padding={args.padding}", results
    )

    if args.top_losses > 0:
        results.sort(key=lambda r: -r.worst_rel_loss_pct)
        print(f"\nTop {args.top_losses} worst rel-loss scenarios:")
        for r in results[:args.top_losses]:
            print(f"  {r.date} {r.lens_config[:60]:<60} "
                  f"f/{r.aperture:<3g}  loss={r.worst_rel_loss_pct:5.1f}%  "
                  f"savings={r.savings_pct:5.1f}%  "
                  f"FWHM={r.fwhm_frames:.1f} spread={r.spread_frames:.1f} "
                  f"({r.n_valid_crops} valid crops)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
