"""Dry-run simulator for the coarse-to-fine focus probe algorithm.

Replays a previously captured live-test session through the
phase-A → phase-C pipeline described in DESIGN_NOTES §6.26 and
reports:

- Shot count savings vs the actual uniform sweep that captured the
  data (phase A captures + phase C captures, vs original sweep).
- Per (aperture, crop) peak file size in the proposed phase-C window
  compared to the ground-truth peak in the full captured sweep —
  i.e. relative sharpness loss caused by the narrower window.
- Whether the existing post-phase-C ``RangeChecker.evaluate`` flow
  (phase D) would have fired because a corner peak fell outside the
  predicted phase-C window.

Usage::

    cd run-lens-test
    uv run python -m tools.probe_simulator /path/to/session_dir
    # or omit the arg to run all three known reference sessions

The simulator does NOT touch hardware or write new JPEGs. It only
reads file sizes from the existing ``cropped/<crop>/step_NNNN*.jpg``
files and the ``session.json`` ``sample_plan`` block.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharpness.detect import (
    DEFAULT_DYNAMIC_RANGE_MIN,
    DEFAULT_FLAT_RATIO,
    CropStatus,
    analyze_crops_for_aperture,
    peak_index,
    smooth,
)


# Phase A density across the user's jogged range. Calibrated by
# ``probe_simulator --phase-a-points`` sweep (see config.py docstring):
# N=9 strictly dominates 7 on the max-FC case (44 % vs 38 % savings)
# at the cost of 2 extra phase-A captures (~10 s).
PHASE_A_POINTS = 9
# Padding on each side of observed corner-peak band, in FWHM_widest units.
# Calibrated against the curvature corpus survey (327 scenarios across
# 100+ lens configs from 2025-07 → 2026-02): the median config has
# spread/FWHM = 0.88, p90 = 2.2, p99 = 8.8. Padding = 1.5 catches
# every config in the corpus's p99 (0.3 % phase-D-fire rate); the live
# calibration sessions are at the p95-max tail, so the simulator's
# 7-34 % savings are pessimistic for typical configs (corpus model
# projects ~74 % median savings at user_jog = 15 × FWHM).
PHASE_C_FWHM_PADDING = 1.5
# Phase B trigger threshold: phase-A peak inside this fraction of the
# samples skips phase B.
PHASE_B_INTERIOR_FRACTION = 0.6


_STEP_RE = re.compile(r"^step_(\d+)(?:_[a-z])?\.jpg$")
DEFAULT_REFERENCE_SESSIONS = [
    "2026-05-06_175727",  # MC 105 + 61mm extension, truncated right side
    "2026-05-06_231805",  # MC 105 + 61mm extension, clean
    "2026-05-06_200150",  # MC 105 + 5T+6T diopter
]
ALL_CROPS = (
    "center",
    "top-left", "top-right", "bottom-left", "bottom-right",
    "left-edge", "right-edge", "top-edge", "bottom-edge",
    "dx-top-left", "dx-top-right", "dx-bottom-left", "dx-bottom-right",
    "dx-left-edge", "dx-right-edge", "dx-top-edge", "dx-bottom-edge",
)


@dataclass
class CropSeries:
    """Per-crop captured data: master_step_idx → median replica size."""

    name: str
    by_step: dict[int, int] = field(default_factory=dict)


@dataclass
class ApertureData:
    aperture_text: str          # e.g. "f/4.5"
    aperture_value: float       # 4.5
    crops: dict[str, CropSeries]  # crop_name → series


@dataclass
class SessionData:
    path: Path
    lens_config: str
    rail_step_mm: float
    fwhm_widest_mm: float
    widest_aperture: float
    subsample_factors: dict[float, int]
    apertures: dict[str, ApertureData]   # aperture_text → ApertureData


def load_session(session_dir: Path) -> SessionData:
    """Read sample_plan and all captured crop file sizes."""
    sj = session_dir / "session.json"
    payload = json.loads(sj.read_text())
    cfgs = payload.get("lens_configs") or []
    if not cfgs or not isinstance(cfgs[0], dict):
        raise SystemExit(f"no lens_configs in {sj}")
    cfg = cfgs[0]
    plan = cfg.get("sample_plan", {})
    lens_config = cfg.get("slug") or cfg.get("name") or "?"
    rail_step_mm = plan["rail_step_mm"]
    fwhm = plan["estimated_fwhm_mm_widest"]
    widest = plan["widest_aperture"]
    subsample = {float(k): int(v) for k, v in plan.get("subsample_factors", {}).items()}

    # Locate the lens directory by name.
    lens_dirs = [d for d in session_dir.iterdir() if d.is_dir() and d.name not in {"calibration"}]
    if not lens_dirs:
        raise SystemExit(f"no lens directory in {session_dir}")
    lens_dir = lens_dirs[0]

    apertures: dict[str, ApertureData] = {}
    for ap_dir in sorted(d for d in lens_dir.iterdir() if d.is_dir() and d.name.startswith("f_")):
        ap_text = ap_dir.name.replace("f_", "f/").replace("_", ".")
        ap_value = float(ap_text.removeprefix("f/"))
        cropped = ap_dir / "cropped"
        if not cropped.is_dir():
            continue
        crops: dict[str, CropSeries] = {}
        for crop_name in ALL_CROPS:
            d = cropped / crop_name
            if not d.is_dir():
                continue
            by_step: dict[int, list[int]] = defaultdict(list)
            for f in d.iterdir():
                m = _STEP_RE.match(f.name)
                if not m:
                    continue
                by_step[int(m.group(1))].append(f.stat().st_size)
            if not by_step:
                continue
            series = CropSeries(name=crop_name)
            for step, sizes in by_step.items():
                series.by_step[step] = int(median(sizes))
            crops[crop_name] = series
        if crops:
            apertures[ap_text] = ApertureData(
                aperture_text=ap_text,
                aperture_value=ap_value,
                crops=crops,
            )

    return SessionData(
        path=session_dir,
        lens_config=lens_config,
        rail_step_mm=rail_step_mm,
        fwhm_widest_mm=fwhm,
        widest_aperture=widest,
        subsample_factors=subsample,
        apertures=apertures,
    )


def _crop_peak_step(series: CropSeries) -> int | None:
    """Smoothed-argmax peak in master-step coordinates, or None when no data."""
    if not series.by_step:
        return None
    sorted_steps = sorted(series.by_step)
    sizes = [series.by_step[s] for s in sorted_steps]
    if len(sizes) < 3:
        return None
    sm = smooth(sizes, window=3)
    if sm.size == 0:
        return None
    return sorted_steps[int(np.argmax(sm)) + 1]


def _crop_dr(series: CropSeries) -> float:
    if not series.by_step:
        return 0.0
    vals = list(series.by_step.values())
    mx = max(vals)
    if mx <= 0:
        return 0.0
    return (mx - min(vals)) / mx


@dataclass
class PhaseAResult:
    sample_steps: list[int]                # master-step indices probed
    centre_peak_step: float | None         # sub-sample resolution
    corner_spread_steps: float             # max - min across crops, sub-sample
    min_peak_step: float | None            # smallest peak across valid crops
    max_peak_step: float | None            # largest peak across valid crops
    valid_crop_peaks: dict[str, float]     # crop_name → peak_step (sub-sample)
    inconclusive: bool


def simulate_phase_a(
    sess: SessionData,
    n_points: int | None = None,
) -> PhaseAResult:
    """Pick `n_points` evenly spaced master-step indices from the
    widest aperture's captured range, compute a peak per crop using
    just those samples, and return the spread between peaks.
    """
    if n_points is None:
        n_points = PHASE_A_POINTS
    widest_text = f"f/{sess.widest_aperture:g}"
    if widest_text not in sess.apertures:
        # Fall back to whatever aperture has the most steps captured.
        widest_text = max(sess.apertures, key=lambda k: len(sess.apertures[k].crops["center"].by_step))
    ap = sess.apertures[widest_text]

    # Master-step indices captured at this aperture.
    centre = ap.crops["center"]
    captured_steps = sorted(centre.by_step)
    if len(captured_steps) < n_points:
        n_points = len(captured_steps)

    # Pick `n_points` indices into `captured_steps`.
    pick_idx = np.linspace(0, len(captured_steps) - 1, n_points).round().astype(int)
    sample_steps = [captured_steps[i] for i in pick_idx]

    # For each crop, restrict to phase-A samples and find the peak.
    # We use a parabolic fit through the sample triplet around the
    # discrete argmax for sub-sample resolution — phase A's spacing
    # is coarse (range / n_points), and integer-step peaks would
    # quantize the corner spread by a factor of that spacing.
    valid_peaks: dict[str, float] = {}
    for crop_name, series in ap.crops.items():
        sizes = [series.by_step.get(s, 0) for s in sample_steps]
        if min(sizes) <= 0 or max(sizes) <= 0:
            continue
        dr = (max(sizes) - min(sizes)) / max(sizes)
        if dr < DEFAULT_DYNAMIC_RANGE_MIN:
            continue
        arr = np.asarray(sizes, dtype=float)
        argmax_idx = int(np.argmax(arr))
        # Parabolic fit on (idx-1, idx, idx+1) when not at boundary;
        # otherwise treat the boundary sample as the peak (flag as
        # potentially truncated — phase D will catch it post phase C).
        if 0 < argmax_idx < len(arr) - 1:
            y0, y1, y2 = arr[argmax_idx - 1], arr[argmax_idx], arr[argmax_idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-9:
                shift = 0.5 * (y0 - y2) / denom
                # Clamp shift to ±1 sample so a wonky parabola can't
                # jump us across a neighbour.
                shift = max(-1.0, min(1.0, shift))
            else:
                shift = 0.0
            # Sub-sample peak in master-step units: sample_steps is
            # the integer master-step index per phase-A sample.
            sample_lo = sample_steps[argmax_idx - 1] if argmax_idx > 0 else sample_steps[argmax_idx]
            sample_mid = sample_steps[argmax_idx]
            sample_hi = sample_steps[argmax_idx + 1] if argmax_idx < len(sample_steps) - 1 else sample_mid
            spacing = (sample_hi - sample_lo) / 2 if sample_hi != sample_lo else 1.0
            peak_master_step = sample_mid + shift * spacing
        else:
            peak_master_step = float(sample_steps[argmax_idx])
        valid_peaks[crop_name] = peak_master_step

    centre_pk = valid_peaks.get("center")
    if not valid_peaks:
        return PhaseAResult(
            sample_steps=sample_steps,
            centre_peak_step=None,
            corner_spread_steps=0.0,
            min_peak_step=None,
            max_peak_step=None,
            valid_crop_peaks={},
            inconclusive=True,
        )

    pmin = min(valid_peaks.values())
    pmax = max(valid_peaks.values())
    spread = pmax - pmin
    inconclusive = len(valid_peaks) < 3
    return PhaseAResult(
        sample_steps=sample_steps,
        centre_peak_step=centre_pk,
        corner_spread_steps=spread,
        min_peak_step=pmin,
        max_peak_step=pmax,
        valid_crop_peaks=valid_peaks,
        inconclusive=inconclusive,
    )


@dataclass
class PerCropVerdict:
    crop_name: str
    full_sweep_peak_size: int
    phase_c_peak_size: int
    rel_loss_pct: float
    in_phase_c_window: bool
    truncated_in_phase_c: bool


@dataclass
class ApertureSimResult:
    aperture_text: str
    aperture_value: float
    phase_c_master_steps: list[int]
    phase_c_capture_count: int
    full_sweep_capture_count: int
    crop_verdicts: list[PerCropVerdict]
    phase_d_would_fire: bool


@dataclass
class SessionSimResult:
    sess: SessionData
    phase_a: PhaseAResult
    phase_c_window_mm: float
    phase_c_window_steps: int
    by_aperture: dict[str, ApertureSimResult]
    total_simulated_captures: int
    total_actual_captures: int

    @property
    def savings_pct(self) -> float:
        if self.total_actual_captures <= 0:
            return 0.0
        return 100 * (1 - self.total_simulated_captures / self.total_actual_captures)


def simulate_session(sess: SessionData) -> SessionSimResult:
    pa = simulate_phase_a(sess)
    fwhm_steps = sess.fwhm_widest_mm / sess.rail_step_mm
    pad_steps = PHASE_C_FWHM_PADDING * fwhm_steps
    if pa.min_peak_step is None or pa.inconclusive:
        # Fall back to phase-A samples as the phase-C window (degenerate).
        if pa.sample_steps:
            window_lo = pa.sample_steps[0]
            window_hi = pa.sample_steps[-1]
        else:
            window_lo = window_hi = 0
    else:
        # Bracket [min_peak − pad, max_peak + pad]. Crucially, the
        # window is NOT centred on the centre-crop peak — on
        # high-field-curvature configs (MC 105 + 61 mm at 1.92×) all
        # the corner peaks land on one side of the centre, so a
        # symmetric window misses half the corners.
        window_lo = int(round(pa.min_peak_step - pad_steps))
        window_hi = int(round(pa.max_peak_step + pad_steps))
    window_steps = window_hi - window_lo
    window_mm = window_steps * sess.rail_step_mm

    by_aperture: dict[str, ApertureSimResult] = {}
    total_simulated = len(pa.sample_steps)  # phase A captures (single aperture)
    total_actual = 0
    for ap_text, ap in sess.apertures.items():
        # Captured steps at this aperture.
        captured = sorted(ap.crops["center"].by_step) if "center" in ap.crops else []
        full_count = len(captured)
        total_actual += full_count
        in_window = [s for s in captured if window_lo <= s <= window_hi]
        phase_c_count = len(in_window)
        total_simulated += phase_c_count

        verdicts: list[PerCropVerdict] = []
        phase_d_fire = False
        # For each crop: full-sweep peak vs phase-C window peak.
        for crop_name, series in ap.crops.items():
            full_steps = sorted(series.by_step)
            full_sizes = [series.by_step[s] for s in full_steps]
            if not full_sizes:
                continue
            # Peak file size across the full sweep.
            full_peak_size = max(full_sizes)
            # Same crop within phase-C window.
            window_sizes = [series.by_step[s] for s in in_window]
            if not window_sizes:
                continue
            window_peak_size = max(window_sizes)
            rel_loss = 100 * (full_peak_size - window_peak_size) / full_peak_size
            full_pk_step = full_steps[int(np.argmax(full_sizes))]
            in_window_flag = window_lo <= full_pk_step <= window_hi

            # Phase D would fire if the bell within phase C is truncated.
            sm = smooth(window_sizes, window=3)
            if sm.size:
                pk_in_w = int(np.argmax(sm)) + 1
                truncated = pk_in_w < 2 or pk_in_w > len(window_sizes) - 1 - 2
            else:
                truncated = False
            if truncated and (max(window_sizes) - min(window_sizes)) / max(window_sizes) >= DEFAULT_DYNAMIC_RANGE_MIN:
                phase_d_fire = True

            verdicts.append(PerCropVerdict(
                crop_name=crop_name,
                full_sweep_peak_size=full_peak_size,
                phase_c_peak_size=window_peak_size,
                rel_loss_pct=rel_loss,
                in_phase_c_window=in_window_flag,
                truncated_in_phase_c=truncated,
            ))

        by_aperture[ap_text] = ApertureSimResult(
            aperture_text=ap_text,
            aperture_value=ap.aperture_value,
            phase_c_master_steps=in_window,
            phase_c_capture_count=phase_c_count,
            full_sweep_capture_count=full_count,
            crop_verdicts=verdicts,
            phase_d_would_fire=phase_d_fire,
        )

    return SessionSimResult(
        sess=sess,
        phase_a=pa,
        phase_c_window_mm=window_mm,
        phase_c_window_steps=window_steps,
        by_aperture=by_aperture,
        total_simulated_captures=total_simulated,
        total_actual_captures=total_actual,
    )


def report(result: SessionSimResult, *, verbose: bool = False) -> None:
    s = result.sess
    print(f"\n=== {s.path.name} — {s.lens_config} ===")
    print(f"  rail_step={s.rail_step_mm*1000:.2f} µm   FWHM_widest={s.fwhm_widest_mm*1000:.0f} µm")
    print(f"  Phase A: {len(result.phase_a.sample_steps)} probes at f/{s.widest_aperture:g}; "
          f"{len(result.phase_a.valid_crop_peaks)} valid crop peaks; "
          f"corner spread {result.phase_a.corner_spread_steps:.1f} steps "
          f"({result.phase_a.corner_spread_steps * s.rail_step_mm * 1000:.0f} µm)")
    if result.phase_a.inconclusive:
        print(f"  Phase A: INCONCLUSIVE (<3 valid crops)")
    if result.phase_a.min_peak_step is not None:
        # Window bracket from PhaseA min/max peak ± padding (computed in simulate_session).
        pad = PHASE_C_FWHM_PADDING * (s.fwhm_widest_mm / s.rail_step_mm)
        lo = int(round(result.phase_a.min_peak_step - pad))
        hi = int(round(result.phase_a.max_peak_step + pad))
        print(f"  Phase C window: {result.phase_c_window_steps} steps "
              f"({result.phase_c_window_mm * 1000:.0f} µm) bracket [step {lo}, step {hi}]; "
              f"centre-crop peak at step {result.phase_a.centre_peak_step:.1f}")
    if verbose:
        # Phase A peaks per crop, sorted by position.
        by_pos = sorted(result.phase_a.valid_crop_peaks.items(), key=lambda kv: kv[1])
        print("  Phase-A per-crop peaks:")
        for name, pk in by_pos:
            print(f"    {name:<18} {pk:7.1f}")

    print(f"  {'aperture':<8} {'full':>5} {'phase-C':>8} {'savings':>8} {'worst rel-loss %':>17} {'phase-D?':>10} {'crops outside':>15}")
    for ap_text, r in result.by_aperture.items():
        worst_loss = max((v.rel_loss_pct for v in r.crop_verdicts), default=0.0)
        outside = sum(1 for v in r.crop_verdicts if not v.in_phase_c_window)
        savings = 100 * (1 - r.phase_c_capture_count / r.full_sweep_capture_count) if r.full_sweep_capture_count else 0
        print(f"  {ap_text:<8} {r.full_sweep_capture_count:>5} {r.phase_c_capture_count:>8} "
              f"{savings:>7.0f}% {worst_loss:>16.2f}% {'YES' if r.phase_d_would_fire else 'no':>10} "
              f"{outside:>15}")
        if verbose:
            for v in sorted(r.crop_verdicts, key=lambda v: -v.rel_loss_pct)[:5]:
                if v.rel_loss_pct > 0.5 or not v.in_phase_c_window:
                    flag = "OUT" if not v.in_phase_c_window else "in "
                    trunc = " trunc" if v.truncated_in_phase_c else ""
                    print(f"      {flag} {v.crop_name:<18} loss {v.rel_loss_pct:5.1f}%{trunc}")
    print()
    print(f"  TOTAL captures: actual={result.total_actual_captures}  simulated={result.total_simulated_captures}  "
          f"savings={result.savings_pct:.0f}%")


def main(argv: list[str] | None = None) -> int:
    global PHASE_A_POINTS, PHASE_C_FWHM_PADDING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", help="path to session dir")
    parser.add_argument("--root", default=str(Path.home() / "macro-lens-test-output"),
                        help="root of session dirs (default ~/macro-lens-test-output)")
    parser.add_argument("-v", "--verbose", action="store_true", help="per-crop diagnostics")
    parser.add_argument(
        "--padding-sweep",
        action="store_true",
        help="instead of a single run, sweep PHASE_C_FWHM_PADDING over [1.0..3.0]",
    )
    parser.add_argument(
        "--phase-a-points",
        type=int,
        default=PHASE_A_POINTS,
        help=f"phase-A probe count (default {PHASE_A_POINTS})",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=PHASE_C_FWHM_PADDING,
        help=f"phase-C padding × FWHM_widest (default {PHASE_C_FWHM_PADDING})",
    )
    args = parser.parse_args(argv)
    PHASE_A_POINTS = args.phase_a_points
    PHASE_C_FWHM_PADDING = args.padding

    if args.session:
        targets = [Path(args.session)]
    else:
        root = Path(args.root)
        targets = [root / s for s in DEFAULT_REFERENCE_SESSIONS]

    sessions: list[SessionData] = []
    for t in targets:
        if not t.is_dir():
            print(f"  skipping {t} (not a directory)")
            continue
        try:
            sessions.append(load_session(t))
        except SystemExit as e:
            print(f"  skipping {t}: {e}")
            continue

    if args.padding_sweep:
        print(f"\nPadding sweep — phase A points: {PHASE_A_POINTS}, padding × FWHM:")
        print(f"{'session':<48} {'pad':>4} {'savings':>8} {'worst loss%':>11} {'phase-D fires':>14}")
        for pad in (1.0, 1.5, 2.0, 2.5, 3.0):
            PHASE_C_FWHM_PADDING = pad
            for sess in sessions:
                result = simulate_session(sess)
                worst = max(
                    (max((v.rel_loss_pct for v in r.crop_verdicts), default=0.0)
                     for r in result.by_aperture.values()),
                    default=0.0,
                )
                fires = sum(1 for r in result.by_aperture.values() if r.phase_d_would_fire)
                print(f"{sess.path.name:<48} {pad:>4.1f} {result.savings_pct:>7.0f}% "
                      f"{worst:>10.2f}% {fires}/{len(result.by_aperture):>14}")
        return 0

    for sess in sessions:
        result = simulate_session(sess)
        report(result, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
