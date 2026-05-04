"""Tests for sharpness.advisor — top-level recommendation aggregation."""

from __future__ import annotations

from sharpness.advisor import (
    Direction,
    Recommendation,
    Verdict,
    advise,
)
from sharpness.collector import CollectedDataset, ShotSample
from sharpness.detect import CropAnalysis, CropStatus


# Synthetic stepped bell modelled on a clean rail sweep
BELL = [
    80, 90, 100, 110, 120, 128,
    132, 130, 124, 116, 106, 96,
    87, 80, 75, 72, 70, 69,
    68,
]
RISING = [int(50 + i * 5) for i in range(15)]   # truncated end
FALLING = [int(120 - i * 5) for i in range(15)]  # truncated start


def _dataset(crop_sizes_by_aperture: dict[float, dict[str, list[int]]]) -> CollectedDataset:
    """Build a CollectedDataset where each aperture has the given per-crop series."""
    samples = []
    for aperture, crop_sizes in crop_sizes_by_aperture.items():
        n = len(next(iter(crop_sizes.values())))
        for step in range(n):
            sizes = {name: series[step] for name, series in crop_sizes.items()}
            samples.append(ShotSample(rail_step=step, aperture=aperture, sizes=sizes))
    return CollectedDataset(samples=samples)


# --- _build_recommendation via advise -------------------------------------


class TestAdvise:
    def test_clean_bell_is_ok(self) -> None:
        ds = _dataset({8.0: {"center": BELL, "left-edge": BELL}})
        report = advise(ds)
        rec = report[8.0]
        assert rec.verdict == Verdict.OK
        assert rec.direction == Direction.NONE
        assert rec.extra_steps == 0

    def test_truncated_end_yields_extend_forward(self) -> None:
        ds = _dataset({11.0: {"center": RISING, "left-edge": RISING}})
        rec = advise(ds)[11.0]
        assert rec.verdict == Verdict.NEEDS_RESHOOT
        assert rec.direction == Direction.EXTEND_FORWARD
        assert rec.extra_steps >= 2

    def test_truncated_start_yields_extend_backward(self) -> None:
        ds = _dataset({4.0: {"center": FALLING, "left-edge": FALLING}})
        rec = advise(ds)[4.0]
        assert rec.verdict == Verdict.NEEDS_RESHOOT
        assert rec.direction == Direction.EXTEND_BACKWARD
        assert rec.extra_steps >= 2

    def test_mixed_truncation_yields_both(self) -> None:
        ds = _dataset({8.0: {"center": RISING, "left-edge": FALLING}})
        rec = advise(ds)[8.0]
        assert rec.verdict == Verdict.NEEDS_RESHOOT
        assert rec.direction == Direction.BOTH

    def test_extra_steps_uses_worst_case_across_crops(self) -> None:
        # One crop's vertex is just past the end (small gap), another is far past.
        gentle = list(range(50, 65))  # vertex slightly past the end
        steep = [int(50 + 100 * i) for i in range(15)]  # vertex very far past
        ds = _dataset({8.0: {"center": gentle, "left-edge": steep}})
        rec = advise(ds)[8.0]
        assert rec.verdict == Verdict.NEEDS_RESHOOT
        # The worst-case crop should drive extra_steps
        assert rec.extra_steps >= 2

    def test_off_target_alone_does_not_trigger_reshoot(self) -> None:
        # 4 on-target bells + 1 off-target flat crop ⇒ scenario is OK
        ds = _dataset(
            {
                8.0: {
                    "center": BELL,
                    "left-edge": BELL,
                    "right-edge": BELL,
                    "bottom-edge": BELL,
                    "top-left": [70] * len(BELL),  # flat / off-target
                }
            }
        )
        rec = advise(ds)[8.0]
        assert rec.verdict == Verdict.OK
        assert rec.direction == Direction.NONE
        assert "off-target" in rec.summary()

    def test_independent_apertures(self) -> None:
        ds = _dataset(
            {
                4.0: {"center": BELL, "left-edge": BELL},
                11.0: {"center": RISING, "left-edge": RISING},
            }
        )
        report = advise(ds)
        assert report[4.0].verdict == Verdict.OK
        assert report[11.0].verdict == Verdict.NEEDS_RESHOOT
        assert report[11.0].direction == Direction.EXTEND_FORWARD

    def test_empty_dataset_returns_empty_dict(self) -> None:
        assert advise(CollectedDataset(samples=[])) == {}

    def test_summary_includes_truncated_crop_names(self) -> None:
        ds = _dataset({8.0: {"center": RISING, "left-edge": BELL}})
        rec = advise(ds)[8.0]
        s = rec.summary()
        assert "center" in s
        assert "f/8" in s
        assert "rail step" in s


# --- Recommendation public surface ---------------------------------------


class TestRecommendation:
    def test_ok_summary_is_concise(self) -> None:
        r = Recommendation(
            aperture=5.6,
            verdict=Verdict.OK,
            direction=Direction.NONE,
            extra_steps=0,
        )
        assert r.summary() == "f/5.6: OK"

    def test_ok_summary_notes_off_target_count(self) -> None:
        r = Recommendation(
            aperture=5.6,
            verdict=Verdict.OK,
            direction=Direction.NONE,
            extra_steps=0,
            crop_analyses=(
                CropAnalysis(
                    crop_name="top-left",
                    status=CropStatus.OFF_TARGET,
                    n_steps=10,
                    peak_step=None,
                    peak_pct=None,
                    estimated_peak_step=None,
                    extra_steps_needed=None,
                    max_size=70,
                    min_size=68,
                ),
            ),
        )
        assert "1 off-target" in r.summary()

    def test_truncated_helpers(self) -> None:
        ok = CropAnalysis(
            crop_name="center",
            status=CropStatus.OK,
            n_steps=10,
            peak_step=5,
            peak_pct=0.5,
            estimated_peak_step=None,
            extra_steps_needed=None,
            max_size=100,
            min_size=80,
        )
        trunc = CropAnalysis(
            crop_name="left-edge",
            status=CropStatus.TRUNCATED_END,
            n_steps=10,
            peak_step=9,
            peak_pct=1.0,
            estimated_peak_step=15.0,
            extra_steps_needed=7,
            max_size=100,
            min_size=80,
        )
        off = CropAnalysis(
            crop_name="top-left",
            status=CropStatus.OFF_TARGET,
            n_steps=10,
            peak_step=None,
            peak_pct=None,
            estimated_peak_step=None,
            extra_steps_needed=None,
            max_size=70,
            min_size=68,
        )
        r = Recommendation(
            aperture=8.0,
            verdict=Verdict.NEEDS_RESHOOT,
            direction=Direction.EXTEND_FORWARD,
            extra_steps=7,
            crop_analyses=(ok, trunc, off),
        )
        assert r.truncated_crops == (trunc,)
        assert r.off_target_crops == (off,)
