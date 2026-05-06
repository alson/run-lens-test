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

    def test_all_off_target_yields_off_target_verdict(self) -> None:
        # Every crop is flat. The aperture has no truncated crops AND
        # no real-peak crops, so the verdict is OFF_TARGET (not OK).
        flat = [70] * len(BELL)
        ds = _dataset(
            {
                8.0: {
                    "center": flat,
                    "left-edge": flat,
                    "right-edge": flat,
                    "bottom-edge": flat,
                }
            }
        )
        rec = advise(ds)[8.0]
        assert rec.verdict == Verdict.OFF_TARGET
        assert rec.direction == Direction.NONE
        assert rec.extra_steps == 0
        assert "OFF_TARGET" in rec.summary()
        assert "no focus signal" in rec.summary()

    def test_off_target_alone_does_not_trigger_reshoot(self) -> None:
        # 4 on-target bells + 1 invalid flat crop ⇒ scenario is OK.
        # The flat crop fell off the wafer (dynamic range ≈ 0%) so the
        # advisor flags it invalid and excludes it from the verdict.
        ds = _dataset(
            {
                8.0: {
                    "center": BELL,
                    "left-edge": BELL,
                    "right-edge": BELL,
                    "bottom-edge": BELL,
                    "top-left": [70] * len(BELL),  # flat / off-wafer
                }
            }
        )
        rec = advise(ds)[8.0]
        assert rec.verdict == Verdict.OK
        assert rec.direction == Direction.NONE
        assert "invalid" in rec.summary()
        assert "top-left" in rec.summary()
        assert {c.crop_name for c in rec.invalid_crops} == {"top-left"}

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


# --- Multi-crop validity & field-curvature aggregation -------------------


# Bell, rising, and falling all sized to match BELL (length 19) so the
# _dataset() helper, which keys all crops at the same aperture off the
# *first* crop's length, doesn't IndexError when crops are mixed.
N = len(BELL)
RISING_19 = [int(50 + i * 5) for i in range(N)]   # truncated end, 19 steps
FALLING_19 = [int(120 - i * 5) for i in range(N)]  # truncated start, 19 steps
# Off-wafer flat-noise series. (max=104, min=100 → ~4% range, well below
# the 25% validity floor.) Used for "invalid" tests so the gate trips.
OFF_WAFER_NOISE = [
    100, 101, 102, 103, 104, 103, 102, 101, 100, 102, 101,
    100, 102, 103, 102, 101, 100, 102, 100,
]
assert len(OFF_WAFER_NOISE) == N


class TestMultiCropAggregation:
    def test_all_valid_all_ok_yields_aperture_ok(self) -> None:
        # Centre and corners all see the same in-range bell ⇒ OK.
        crops = {
            "center": BELL,
            "top-left": BELL,
            "top-right": BELL,
            "bottom-left": BELL,
            "bottom-right": BELL,
        }
        rec = advise(_dataset({8.0: crops}))[8.0]
        assert rec.verdict == Verdict.OK
        assert rec.direction == Direction.NONE
        assert rec.invalid_crops == ()
        assert all(c.valid for c in rec.crop_analyses)

    def test_field_curvature_with_valid_corners_yields_both(self) -> None:
        # Centre peaks fine, one corner truncates forward, another
        # truncates backward — classic field-curvature signature. The
        # aperture aggregate must collapse to BOTH so the runner
        # extends bilaterally.
        crops = {
            "center": BELL,
            "top-left": RISING_19,     # peak past the far end
            "bottom-right": FALLING_19,  # peak before the start
        }
        rec = advise(_dataset({8.0: crops}))[8.0]
        assert rec.verdict == Verdict.NEEDS_RESHOOT
        assert rec.direction == Direction.BOTH
        assert {c.crop_name for c in rec.truncated_crops} == {
            "top-left", "bottom-right"
        }
        # All three crops are valid in this scenario.
        assert rec.invalid_crops == ()

    def test_field_curvature_centre_ok_corners_forward_yields_extend(
        self,
    ) -> None:
        # Centre peaks fine but the corners want more rail past the
        # far end. The aperture aggregate must promote to NEEDS_RESHOOT
        # / EXTEND_FORWARD because *any* valid truncated crop wins.
        crops = {
            "center": BELL,
            "top-left": RISING_19,
            "top-right": RISING_19,
        }
        rec = advise(_dataset({8.0: crops}))[8.0]
        assert rec.verdict == Verdict.NEEDS_RESHOOT
        assert rec.direction == Direction.EXTEND_FORWARD

    def test_invalid_corners_excluded_and_called_out(self) -> None:
        # Centre + one corner are good; the other corners fell off the
        # wafer (low dynamic range). Aperture should be OK and the
        # rationale should mention the invalid corners by name.
        crops = {
            "center": BELL,
            "top-left": BELL,
            "top-right": OFF_WAFER_NOISE,
            "bottom-right": OFF_WAFER_NOISE,
        }
        rec = advise(_dataset({8.0: crops}))[8.0]
        assert rec.verdict == Verdict.OK
        assert rec.direction == Direction.NONE
        invalid_names = {c.crop_name for c in rec.invalid_crops}
        assert invalid_names == {"top-right", "bottom-right"}
        # All invalid crops have dynamic range below the floor and a
        # value plausibly close to the noise level.
        for c in rec.invalid_crops:
            assert c.valid is False
            assert c.dynamic_range < 0.10
        # Rationale calls them out individually so the user knows
        # *which* corners fell off the wafer.
        s = rec.summary()
        assert "invalid" in s
        assert "top-right" in s
        assert "bottom-right" in s

    def test_invalid_corners_do_not_drive_direction(self) -> None:
        # Centre + valid corner are bell-shaped (would normally be OK);
        # but a "rising" series in an *invalid* corner must NOT
        # promote the aperture to NEEDS_RESHOOT / EXTEND_FORWARD.
        # Build a low-dynamic-range "rising" series so it's both
        # rising and invalid (max ≈ 118 min = 100 → dr ≈ 0.15).
        n = len(BELL)
        invalid_rising = [100 + i for i in range(n)]
        # Confirm the synthetic series stays under the validity floor.
        assert (max(invalid_rising) - min(invalid_rising)) / max(
            invalid_rising
        ) < 0.25
        crops = {
            "center": BELL,
            "left-edge": BELL,
            "top-left": invalid_rising,
        }
        rec = advise(_dataset({8.0: crops}))[8.0]
        assert rec.verdict == Verdict.OK
        assert rec.direction == Direction.NONE
        assert {c.crop_name for c in rec.invalid_crops} == {"top-left"}

    def test_all_crops_invalid_yields_off_target(self) -> None:
        # Every crop fell off the wafer / has no focus signal. The
        # aperture verdict is OFF_TARGET and the rationale flags every
        # crop as invalid.
        crops = {
            "center": OFF_WAFER_NOISE,
            "left-edge": OFF_WAFER_NOISE,
            "top-left": OFF_WAFER_NOISE,
            "bottom-right": OFF_WAFER_NOISE,
        }
        rec = advise(_dataset({8.0: crops}))[8.0]
        assert rec.verdict == Verdict.OFF_TARGET
        assert rec.direction == Direction.NONE
        assert rec.extra_steps == 0
        assert len(rec.invalid_crops) == len(crops)
        assert rec.valid_crops == ()
        s = rec.summary()
        assert "OFF_TARGET" in s
        # Every invalid crop appears in the rationale.
        for name in crops:
            assert name in s

    def test_per_crop_accessor_returns_dict_keyed_by_name(self) -> None:
        crops = {
            "center": BELL,
            "left-edge": BELL,
            "top-left": OFF_WAFER_NOISE,
        }
        rec = advise(_dataset({8.0: crops}))[8.0]
        per = rec.per_crop()
        assert set(per) == set(crops)
        assert per["center"].valid is True
        assert per["top-left"].valid is False
        assert per["top-left"].dynamic_range < 0.25


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
