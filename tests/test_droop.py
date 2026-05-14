"""Tests for exposure.droop.DroopDetector (session-level)."""

from __future__ import annotations

import pytest

from exposure.droop import DroopDetector


# ---- Configuration --------------------------------------------------------


# Mirrors config.py defaults for the loosened, session-pooled
# detector.
WINDOW = 10
CRIT_SEVERE = (1, -2.0)   # 1 shot ≤ -2 EV → trip
CRIT_MODERATE = (2, -0.7)  # 2 of last 10 ≤ -0.7 EV → trip


def make_detector():
    return DroopDetector(window=WINDOW, criteria=(CRIT_SEVERE, CRIT_MODERATE))


# ---- Severe (one-shot) trip -----------------------------------------------


class TestSevereCriterion:
    def test_does_not_fire_on_clean_shots(self) -> None:
        d = make_detector()
        for v in [-0.1, 0.0, +0.1, -0.2, -0.05]:
            assert d.push("f/4", v) is None

    def test_fires_on_single_severe_shot(self) -> None:
        d = make_detector()
        d.push("f/4", 0.0)
        evt = d.push("f/4", -2.5)
        assert evt is not None
        assert evt.aperture == "f/4"
        assert evt.threshold == pytest.approx(-2.0)
        assert evt.triggered_count == 1


# ---- Moderate (cumulative) trip --------------------------------------------


class TestModerateCriterion:
    def test_does_not_fire_on_one_moderate_shot(self) -> None:
        d = make_detector()
        d.push("f/4", 0.0)
        assert d.push("f/4", -0.8) is None

    def test_fires_after_two_moderate_shots(self) -> None:
        d = make_detector()
        d.push("f/4", 0.0)
        d.push("f/4", -0.8)
        evt = d.push("f/4", -0.9)
        assert evt is not None
        assert evt.threshold == pytest.approx(-0.7)
        assert evt.triggered_count == 2


# ---- Session-level pooling (the user's requested behaviour) ---------------


class TestSessionLevelPooling:
    def test_two_apertures_one_low_each_trips_moderate(self) -> None:
        # Single aperture would need TWO low shots to trip moderate;
        # session-level pooling means one f/5.6 + one f/8 below
        # threshold ALSO trips, even though no single aperture has
        # been pushed against twice.
        d = make_detector()
        d.push("f/4", 0.0)
        assert d.push("f/5.6", -0.8) is None    # 1 of 2 ≤ -0.7
        evt = d.push("f/8", -0.9)               # 2 of 3 ≤ -0.7 → trip
        assert evt is not None
        assert evt.aperture == "f/8"            # most recent push

    def test_severe_trips_regardless_of_history(self) -> None:
        d = make_detector()
        d.push("f/4", 0.0)
        d.push("f/5.6", 0.0)
        evt = d.push("f/8", -3.0)               # severe in 1 shot
        assert evt is not None
        assert evt.threshold == pytest.approx(-2.0)


# ---- reset_all -------------------------------------------------------------


class TestResetAll:
    def test_reset_clears_window(self) -> None:
        d = make_detector()
        d.push("f/4", -2.5)                     # would have tripped
        d.reset_all()
        assert d.window_for("f/4") == ()
        assert d.session_window_entries() == ()

    def test_reset_lets_post_pause_shots_not_immediately_retrip(self) -> None:
        d = make_detector()
        evt = d.push("f/4", -2.5)
        assert evt is not None

        # User swaps batteries, presses ENTER → reset_all().
        d.reset_all()

        # Subsequent normal shots should not trip again from the
        # stale low values.
        for _ in range(5):
            assert d.push("f/4", 0.0) is None


# ---- window_for snapshot --------------------------------------------------


class TestWindowFor:
    def test_filters_session_window_by_aperture(self) -> None:
        # window_for is now a *filter* over the shared session
        # window, not an isolated per-aperture series.
        d = make_detector()
        d.push("f/4", -0.1)
        d.push("f/8", 0.2)
        d.push("f/4", -0.3)
        assert d.window_for("f/4") == (-0.1, -0.3)
        assert d.window_for("f/8") == (0.2,)

    def test_caps_at_window_size(self) -> None:
        d = DroopDetector(window=3, criteria=())
        for v in [1.0, 2.0, 3.0, 4.0]:
            d.push("f/4", v)
        # Session-level deque drops oldest → (2,3,4).
        assert d.window_for("f/4") == (2.0, 3.0, 4.0)

    def test_empty_for_unknown_aperture(self) -> None:
        d = make_detector()
        d.push("f/4", 0.0)
        assert d.window_for("f/22") == ()


# ---- session_window_entries -----------------------------------------------


class TestSessionWindow:
    def test_returns_full_pooled_window(self) -> None:
        d = make_detector()
        d.push("f/4", -0.1, step_index=0)
        d.push("f/8", 0.2, step_index=0)
        d.push("f/16", -0.3, step_index=0)
        entries = d.session_window_entries()
        assert [e.aperture for e in entries] == ["f/4", "f/8", "f/16"]
        assert [e.ev_value for e in entries] == [
            pytest.approx(-0.1),
            pytest.approx(0.2),
            pytest.approx(-0.3),
        ]


# ---- earliest_underexposed_step (cross-aperture) --------------------------


class TestEarliestUnderexposedStep:
    def test_finds_min_step_below_threshold_across_apertures(self) -> None:
        d = make_detector()
        d.push("f/4", -0.1, step_index=0)
        d.push("f/16", -0.8, step_index=2)      # earliest ≤ -0.7
        d.push("f/8", -0.9, step_index=3)
        assert d.earliest_underexposed_step(-0.7) == 2

    def test_none_when_nothing_below_threshold(self) -> None:
        d = make_detector()
        d.push("f/4", -0.1, step_index=0)
        d.push("f/8", -0.2, step_index=1)
        assert d.earliest_underexposed_step(-0.7) is None


# ---- Two-sided detection (over-exposure) -----------------------------------


class TestOverExposureCriteria:
    """Mirror criteria with positive thresholds. Over-trips fire on
    rig-state shifts (e.g. flash bumped during a battery swap) and
    carry ``direction="over"`` so the runner can branch to silent
    re-calibration."""

    def test_severe_over_trips_on_single_shot(self) -> None:
        d = DroopDetector(
            window=10,
            criteria=((1, -2.0), (2, -0.7), (1, +2.0), (2, +0.7)),
        )
        d.push("f/4", 0.0)
        evt = d.push("f/4", +2.5)
        assert evt is not None
        assert evt.direction == "over"
        assert evt.threshold == pytest.approx(+2.0)

    def test_moderate_over_trips_on_two_shots(self) -> None:
        d = DroopDetector(
            window=10,
            criteria=((1, -2.0), (2, -0.7), (1, +2.0), (2, +0.7)),
        )
        d.push("f/4", 0.0)
        assert d.push("f/4", +0.8) is None
        evt = d.push("f/4", +0.9)
        assert evt is not None
        assert evt.direction == "over"
        assert evt.threshold == pytest.approx(+0.7)

    def test_under_and_over_can_coexist(self) -> None:
        d = DroopDetector(
            window=10,
            criteria=((1, -2.0), (2, -0.7), (1, +2.0), (2, +0.7)),
        )
        # Mix of under and over readings — over crit fires when 2 ≥
        # +0.7 accumulate, regardless of negative samples interspersed.
        assert d.push("f/4", 0.0) is None
        assert d.push("f/4", -0.5) is None
        assert d.push("f/4", +0.8) is None
        evt = d.push("f/4", +0.9)
        assert evt is not None
        assert evt.direction == "over"

    def test_explicit_direction_overrides_sign_inference(self) -> None:
        from exposure.droop import TripCriterion
        # Zero threshold with explicit direction.
        crit = TripCriterion(count=1, ev_threshold=0.0, direction="over")
        assert crit.direction == "over"
        crit2 = TripCriterion(count=1, ev_threshold=0.0, direction="under")
        assert crit2.direction == "under"


# ---- earliest_anomalous_step (two-sided rewind search) --------------------


class TestEarliestAnomalousStep:
    def test_under_direction_matches_legacy_helper(self) -> None:
        d = DroopDetector(window=10, criteria=())
        d.push("f/4", -0.1, step_index=0)
        d.push("f/4", -0.8, step_index=2)
        d.push("f/4", -0.9, step_index=3)
        assert d.earliest_anomalous_step(-0.7, direction="under") == 2
        # Legacy alias returns the same result.
        assert d.earliest_underexposed_step(-0.7) == 2

    def test_over_direction_finds_min_step_above_threshold(self) -> None:
        d = DroopDetector(window=10, criteria=())
        d.push("f/4", +0.1, step_index=0)
        d.push("f/4", +0.8, step_index=2)
        d.push("f/4", +0.5, step_index=3)
        d.push("f/4", +0.9, step_index=5)
        assert d.earliest_anomalous_step(+0.7, direction="over") == 2

    def test_invalid_direction_raises(self) -> None:
        d = DroopDetector(window=10, criteria=())
        with pytest.raises(ValueError):
            d.earliest_anomalous_step(0.0, direction="sideways")


# ---- Construction validation ----------------------------------------------


class TestConstruction:
    def test_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DroopDetector(window=0, criteria=())

    def test_three_tuple_criterion_with_explicit_direction(self) -> None:
        d = DroopDetector(
            window=5,
            criteria=((1, 0.0, "over"),),
        )
        # Zero threshold + over → any non-negative push trips.
        evt = d.push("f/4", +0.5)
        assert evt is not None
        assert evt.direction == "over"

    def test_legacy_two_tuple_auto_derives_direction(self) -> None:
        d = DroopDetector(window=5, criteria=((1, -1.0),))
        evt = d.push("f/4", -1.5)
        assert evt is not None
        assert evt.direction == "under"
