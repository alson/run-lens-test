"""Tests for session.calibrate.Calibrator.

The camera is a fake that records each set/capture and returns
scripted medians from ``measure``. Time is frozen so logged
timestamps are deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pytest

from session.calibrate import (
    Calibrator,
    next_iso,
    _parse_aperture,
    _snap,
)
from session.log import Phase, ShotsCSV


# ---- Test config ----------------------------------------------------------


@pytest.fixture
def cfg():
    return SimpleNamespace(
        RANGE_EXTENSION={
            2.0: (100, 3.0),
            4.0: (100, 3.0),
            8.0: (100, 4.0),
            11.0: (200, 5.0),
            16.0: (400, 6.0),
        },
        ISO_FLOOR=100,
        ISO_WARN=1600,
        ISO_CEIL=3200,
        FEC_MIN_EV=-3.0,
        FEC_MAX_EV=+1.0,
        FEC_STEP_EV=1.0 / 3.0,
        EC_MIN_EV=-3.0,
        EC_MAX_EV=+3.0,
        EC_STEP_EV=1.0 / 3.0,
        TEST_SHOT_TOLERANCE_EV=1.0 / 3.0,
        TEST_SHOT_MAX_RETRIES=6,
    )


# ---- Fake camera ----------------------------------------------------------


@dataclass
class FakeCamera:
    """Records every set/capture call. Tests can introspect ``.history``."""

    history: list[tuple[str, object]] = field(default_factory=list)
    last_dest: Path | None = None

    def set_aperture(self, fnum: str) -> None:
        self.history.append(("aperture", fnum))

    def set_iso(self, iso) -> None:
        self.history.append(("iso", iso))

    def set_fec(self, ev: float) -> None:
        self.history.append(("fec", float(ev)))

    def set_ec(self, ev: float) -> None:
        self.history.append(("ec", float(ev)))

    def capture_to(self, dest) -> int:
        self.last_dest = Path(dest)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"x")
        self.history.append(("capture", str(dest)))
        return 1


# ---- Helpers --------------------------------------------------------------


@pytest.fixture
def camera():
    return FakeCamera()


@pytest.fixture
def now():
    fixed = datetime(2026, 5, 4, 18, 35, tzinfo=timezone.utc)
    return lambda: fixed


def make_calibrator(camera, cfg, *, medians, csv=None, now=None):
    queue = deque(medians)

    def _measure(_p):
        return queue.popleft()

    if now is None:
        fixed = datetime(2026, 5, 4, 18, 35, tzinfo=timezone.utc)
        now = lambda: fixed
    return Calibrator(
        camera=camera,
        measure=_measure,
        config=cfg,
        shots_csv=csv,
        now=now,
    )


# ---- Free helpers ---------------------------------------------------------


class TestNextIso:
    def test_climbs_one_step(self) -> None:
        assert next_iso(100, ceiling=3200) == 125

    def test_capped_by_ceiling(self) -> None:
        assert next_iso(3200, ceiling=3200) == 3200

    def test_jumps_past_an_unusual_value(self) -> None:
        assert next_iso(150, ceiling=3200) == 160


class TestParseAperture:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("f/4", 4.0),
            ("f/2.8", 2.8),
            ("f/16", 16.0),
            (" F/5.6 ", 5.6),
        ],
    )
    def test_parses_common_formats(self, text: str, expected: float) -> None:
        assert _parse_aperture(text) == pytest.approx(expected)


class TestSnap:
    def test_rounds_to_third_stop(self) -> None:
        assert _snap(0.5, 1 / 3) == pytest.approx(2 / 3)
        assert _snap(-0.4, 1 / 3) == pytest.approx(-1 / 3)


# ---- Baseline establishment -----------------------------------------------


class TestBaseline:
    def test_first_lens_first_aperture_sets_baseline(
        self, camera, cfg, tmp_path
    ) -> None:
        calib = make_calibrator(camera, cfg, medians=[123.0])
        result = calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=None,
        )
        assert result.baseline_median == pytest.approx(123.0)
        ap = result.per_aperture["f/4"]
        assert ap.fec_ev == 0.0
        assert ap.test_shots_taken == 1
        assert ap.hit_tolerance is True

    def test_subsequent_calls_reuse_provided_baseline(
        self, camera, cfg, tmp_path
    ) -> None:
        # Second lens config: baseline already known; first capture
        # is +0.0 EV (in tolerance) so we accept immediately.
        calib = make_calibrator(camera, cfg, medians=[123.0])
        result = calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=120.0,
        )
        assert result.baseline_median == pytest.approx(120.0)


# ---- Closed-loop convergence ----------------------------------------------


class TestClosedLoop:
    def test_converges_in_two_shots(self, camera, cfg, tmp_path) -> None:
        # Baseline 100; first shot 200 → measured +2.2 EV via gamma
        # 2.2 (200/100 in linear-display); FEC drops by that;
        # second shot returns to baseline → hit tolerance.
        calib = make_calibrator(
            camera, cfg, medians=[200.0, 100.0],
        )
        result = calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=100.0,
        )
        ap = result.per_aperture["f/4"]
        assert ap.test_shots_taken == 2
        assert ap.hit_tolerance is True
        # FEC should be in the negative EV range, snapped to 1/3 stop.
        assert ap.fec_ev < -1.0
        # And exactly representable as a 1/3-stop multiple.
        assert (ap.fec_ev / cfg.FEC_STEP_EV) == pytest.approx(
            round(ap.fec_ev / cfg.FEC_STEP_EV)
        )

    def test_first_dim_shot_pushes_fec_to_max_before_fallback(
        self, camera, cfg, tmp_path
    ) -> None:
        # Regression for the FEC-saturation off-by-one: when the
        # *first* test shot lands so dim that the requested
        # correction exceeds FEC_MAX_EV, the calibrator used to
        # misread the clamped target as "FEC already at max" and
        # bump ISO instead of FEC, leaving FEC stuck at 0 forever.
        # Now FEC reaches FEC_MAX before any fallback (EC then ISO)
        # kicks in.
        medians = [25.0, 50.0, 100.0]  # ev: -2.0, -1.0, 0.0
        calib = make_calibrator(camera, cfg, medians=medians)
        result = calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=100.0,
        )
        ap = result.per_aperture["f/4"]
        # FEC was actually pushed to FEC_MAX, not stuck at 0.
        assert ap.fec_ev == cfg.FEC_MAX_EV
        # FEC was set on the camera at least once with FEC_MAX_EV.
        fec_history = [v for k, v in camera.history if k == "fec"]
        assert cfg.FEC_MAX_EV in fec_history
        # Tolerance hit via FEC + EC dance (not ISO yet — EC has
        # plenty of headroom).
        assert ap.hit_tolerance is True
        assert ap.ec_used is True
        assert ap.iso_bumped is False

    def test_iso_bumps_on_plateau_not_on_fec_ec_saturation(
        self, camera, cfg, tmp_path
    ) -> None:
        # Constant medians = flash is mechanically maxed and the
        # luminosity doesn't track the FEC/EC bumps the calibrator
        # is asking for. Plateau detection should fire and the
        # calibrator should bump ISO as a range-extender. (In AA
        # flash mode, ISO bumping at FEC/EC saturation *without*
        # plateau just causes earlier flash quench and image
        # brightness stays the same — the user's physical insight,
        # so the cascade no longer auto-bumps ISO at FEC+EC max.)
        medians = [6.0] * 8
        calib = make_calibrator(camera, cfg, medians=medians)
        result = calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=100.0,
        )
        ap = result.per_aperture["f/4"]
        assert ap.iso_bumped is True

    def test_no_iso_bump_when_flash_responds_through_full_cascade(
        self, camera, cfg, tmp_path
    ) -> None:
        # Medians track the FEC bumps faithfully (each step roughly
        # doubles in linear-light terms) — flash responds. Even
        # when both FEC and EC reach their camera-side limits without
        # closing the gap, ISO must NOT bump: the flash was
        # mechanically able to deliver every requested EV; an ISO
        # bump would just shift the flash's auto-quench earlier and
        # leave brightness unchanged.
        # Shot 0 at median 6 (~-9 EV vs baseline 100) → FEC bumped
        # to FEC_MAX=+1. Shot 1 at median 12 (~-6 EV) shows the
        # flash *responded* (median doubled per +1 EV) → not plateau.
        # FEC at MAX, fall back to EC; bump straight to EC_MAX=+3.
        # Shot 2 hits baseline → tolerance hit. ec_used=True,
        # iso_bumped=False (no plateau ever detected; the flash was
        # responding stop-for-stop the whole way through).
        medians = [6.0, 12.0, 100.0]
        calib = make_calibrator(camera, cfg, medians=medians)
        result = calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=100.0,
        )
        ap = result.per_aperture["f/4"]
        assert ap.iso_bumped is False
        assert ap.hit_tolerance is True

    def test_iso_bumps_when_fec_saturates_high(self, camera, cfg, tmp_path) -> None:
        # Lens is too dim: every shot is well below baseline (median
        # 25 vs baseline 100 = -2 EV). FEC will saturate at +1.0 EV
        # without closing the gap. Then ISO should bump and we keep
        # trying.
        # We script enough medians so that we eventually reach
        # max_retries; verify iso_bumped flag is set.
        medians = [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
        calib = make_calibrator(camera, cfg, medians=medians)
        result = calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=100.0,
        )
        ap = result.per_aperture["f/4"]
        assert ap.iso_bumped is True
        # ISO should have climbed from 100.
        iso_history = [v for k, v in camera.history if k == "iso"]
        assert iso_history[-1] > 100


# ---- Initial ISO from RANGE_EXTENSION -------------------------------------


class TestInitialIso:
    def test_uses_table_value_when_aperture_present(
        self, camera, cfg, tmp_path
    ) -> None:
        calib = make_calibrator(camera, cfg, medians=[120.0])
        calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/16"],
            baseline_median=120.0,
        )
        # ISO 400 is the table value at f/16.
        iso_calls = [v for k, v in camera.history if k == "iso"]
        assert iso_calls[0] == 400

    def test_falls_back_to_nearest_smaller_key(
        self, camera, cfg, tmp_path
    ) -> None:
        calib = make_calibrator(camera, cfg, medians=[120.0])
        calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/5.6"],     # not in table; nearest <= is f/4 -> 100
            baseline_median=120.0,
        )
        iso_calls = [v for k, v in camera.history if k == "iso"]
        assert iso_calls[0] == 100


# ---- Logging via ShotsCSV --------------------------------------------------


class TestCsvLogging:
    def test_appends_one_row_per_test_shot(self, camera, cfg, tmp_path) -> None:
        csv = ShotsCSV(tmp_path / "shots.csv")
        calib = make_calibrator(
            camera, cfg, medians=[200.0, 100.0], csv=csv,
        )
        calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=100.0,
        )
        rows = csv.read_rows()
        assert len(rows) == 2
        assert all(r["phase"] == Phase.CALIBRATION.value for r in rows)
        assert all(r["lens_config_slug"] == "lens" for r in rows)

    def test_baseline_shot_has_zero_ev_and_baseline_note(
        self, camera, cfg, tmp_path
    ) -> None:
        csv = ShotsCSV(tmp_path / "shots.csv")
        calib = make_calibrator(camera, cfg, medians=[123.0], csv=csv)
        calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4"],
            baseline_median=None,
        )
        row = csv.read_rows()[0]
        assert float(row["median_ev_relative_to_baseline"]) == 0.0
        assert row["notes"] == "baseline"


# ---- Multi-aperture order -------------------------------------------------


class TestMultiAperture:
    def test_walks_apertures_in_given_order(self, camera, cfg, tmp_path) -> None:
        calib = make_calibrator(
            camera, cfg, medians=[100.0, 100.0, 100.0],
        )
        calib.run(
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            apertures=["f/4", "f/8", "f/11"],
            baseline_median=100.0,
        )
        aperture_calls = [v for k, v in camera.history if k == "aperture"]
        assert aperture_calls == ["f/4", "f/8", "f/11"]
