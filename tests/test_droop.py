"""Tests for exposure.droop.DroopDetector."""

from __future__ import annotations

import pytest

from exposure.droop import DroopDetector


# ---- Configuration --------------------------------------------------------


# Mirrors config.py defaults.
WINDOW = 10
CRIT_A = (3, -0.7)   # 3 of last 10 ≤ -0.7 EV
CRIT_B = (2, -2.0)   # 2 of last 10 ≤ -2.0 EV


def make_detector():
    return DroopDetector(window=WINDOW, criteria=(CRIT_A, CRIT_B))


# ---- Trip criterion fires --------------------------------------------------


class TestCriterionA:
    def test_does_not_fire_with_clean_series(self) -> None:
        d = make_detector()
        for v in [-0.1, 0.0, +0.1, -0.2, -0.05, +0.05, -0.15, 0.0, -0.1, 0.0]:
            assert d.push("f/4", v) is None

    def test_fires_after_three_low_shots(self) -> None:
        d = make_detector()
        # Two bad shots interleaved with good ones — should not fire yet.
        assert d.push("f/4", 0.0) is None
        assert d.push("f/4", -0.8) is None        # 1 of 2 ≤ -0.7
        assert d.push("f/4", 0.0) is None
        assert d.push("f/4", -0.9) is None        # 2 of 4 ≤ -0.7
        # Third low one in the last 10 — fires.
        evt = d.push("f/4", -0.75)
        assert evt is not None
        assert evt.aperture == "f/4"
        assert evt.triggered_count == 3
        assert evt.threshold == pytest.approx(-0.7)


class TestCriterionB:
    def test_fires_with_two_very_low_shots(self) -> None:
        d = make_detector()
        # Two ≤ -2.0 EV shots with normal in between.
        assert d.push("f/4", 0.0) is None
        assert d.push("f/4", -2.5) is None
        assert d.push("f/4", 0.0) is None
        evt = d.push("f/4", -2.1)
        assert evt is not None
        # criterion B should fire (2 of 4 ≤ -2.0); criterion A also
        # would fire (2 of 4 ≤ -0.7 = 2 of 4… but A needs 3 ≤ -0.7).
        # So B fires first because we evaluate them in tuple order
        # and B's count is reached at 2.
        assert evt.threshold == pytest.approx(-2.0)
        assert evt.triggered_count == 2


# ---- Per-aperture isolation -----------------------------------------------


class TestPerApertureIsolation:
    def test_one_aperture_droops_others_dont(self) -> None:
        d = make_detector()
        # Hit f/8 with three bad shots; f/4 and f/16 stay clean.
        d.push("f/4", 0.0)
        d.push("f/16", 0.0)
        d.push("f/8", -0.8)
        d.push("f/4", -0.1)
        d.push("f/8", -0.9)
        d.push("f/16", 0.0)
        evt = d.push("f/8", -0.75)
        assert evt is not None and evt.aperture == "f/8"
        # f/4 window only has -0.1, 0.0 — well clear.
        assert d.push("f/4", -0.1) is None

    def test_multiple_apertures_can_each_trip(self) -> None:
        d = make_detector()
        # Drive both f/8 and f/16 to the trip point alternately; each
        # should fire when its own window crosses.
        d.push("f/8", -0.8)
        d.push("f/16", -2.1)
        d.push("f/8", -0.9)
        d.push("f/16", -2.5)  # criterion B (2 of 2 ≤ -2.0) — fires here
        # Now push another low shot at f/8 — third ≤ -0.7 in window.
        evt8 = d.push("f/8", -0.75)
        assert evt8 is not None and evt8.aperture == "f/8"


# ---- reset_all -------------------------------------------------------------


class TestResetAll:
    def test_reset_clears_windows(self) -> None:
        d = make_detector()
        d.push("f/4", -2.5)
        d.push("f/4", -2.1)  # would have tripped on the previous push
        d.reset_all()
        assert d.window_for("f/4") == ()

    def test_reset_lets_post_pause_shots_not_immediately_retrip(self) -> None:
        d = make_detector()
        # Trip the detector.
        d.push("f/4", -2.5)
        evt = d.push("f/4", -2.1)
        assert evt is not None

        # User swaps batteries, presses ENTER → reset_all().
        d.reset_all()

        # Subsequent normal shots should not trip again from the
        # stale low values.
        for _ in range(5):
            assert d.push("f/4", 0.0) is None


# ---- window_for snapshot --------------------------------------------------


class TestWindowFor:
    def test_returns_tuple_in_push_order(self) -> None:
        d = make_detector()
        for v in [-0.1, 0.2, -0.3]:
            d.push("f/4", v)
        assert d.window_for("f/4") == (-0.1, 0.2, -0.3)

    def test_caps_at_window_size(self) -> None:
        d = DroopDetector(window=3, criteria=())
        for v in [1.0, 2.0, 3.0, 4.0]:
            d.push("f/4", v)
        # Oldest dropped: should be (2,3,4).
        assert d.window_for("f/4") == (2.0, 3.0, 4.0)

    def test_empty_for_unknown_aperture(self) -> None:
        d = make_detector()
        assert d.window_for("f/22") == ()


# ---- Construction validation ----------------------------------------------


class TestConstruction:
    def test_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DroopDetector(window=0, criteria=())
