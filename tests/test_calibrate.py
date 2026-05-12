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
        # Manual-aperture flash-M cascade constants.
        MANUAL_ISO_FLOOR=100,
        MANUAL_FLASH_SOFT_CAP_STOPS=1.0,
        MANUAL_FLASH_HARD_CAP_STOPS=0.0,
        MANUAL_FLASH_INITIAL_POWER_STOPS=5.0,
        MANUAL_FLASH_SEARCH_TOLERANCE_EV=1.0,
        MANUAL_FLASH_BUMP_SANITY_EV=0.5,
        MANUAL_FLASH_MAX_SEARCH_ITERATIONS=15,
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


# ---- Manual-aperture mode (non-CPU lens) ----------------------------------


class TestCalibrateSingleAperture:
    def test_returns_single_entry_result(self, camera, cfg, tmp_path) -> None:
        calib = make_calibrator(camera, cfg, medians=[100.0])
        result = calib.calibrate_single_aperture(
            aperture="f/4",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=100.0,
        )
        assert list(result.per_aperture.keys()) == ["f/4"]
        assert result.per_aperture["f/4"].hit_tolerance is True

    def test_set_aperture_via_camera_false_skips_camera_setter(
        self, camera, cfg, tmp_path
    ) -> None:
        """For non-CPU lenses the diaphragm is mechanical — the camera's
        aperture widget either errors or is a no-op. Calling
        :meth:`set_aperture` in that case is wrong (or, at best,
        wastes a PTP round-trip). With ``set_aperture_via_camera=False``
        the Calibrator must not touch ``camera.set_aperture``."""
        calib = make_calibrator(camera, cfg, medians=[100.0])
        calib.calibrate_single_aperture(
            aperture="f/4",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=100.0,
            set_aperture_via_camera=False,
            flash_mode="manual",
        )
        aperture_calls = [v for k, v in camera.history if k == "aperture"]
        assert aperture_calls == [], (
            f"manual mode must not call set_aperture, got {aperture_calls}"
        )

    def test_set_aperture_via_camera_true_still_calls_camera(
        self, camera, cfg, tmp_path
    ) -> None:
        """Default behaviour (auto / CPU lens) is unchanged."""
        calib = make_calibrator(camera, cfg, medians=[100.0])
        calib.calibrate_single_aperture(
            aperture="f/4",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=100.0,
            # set_aperture_via_camera defaults True; flash_mode='aa'.
        )
        aperture_calls = [v for k, v in camera.history if k == "aperture"]
        assert aperture_calls == ["f/4"]


class TestManualFlashCascade:
    def test_iso_only_no_fec_writes_for_underexposure(
        self, camera, cfg, tmp_path
    ) -> None:
        """Manual flash cascade should NEVER touch FEC for adjustment;
        only ISO bumps. We seed an under-exposed median sequence so
        the cascade must do something — and verify it bumped ISO."""
        # First shot: -1 EV under (median 50 vs baseline 100). Second
        # shot: in tolerance. FEC writes only happen at the initial
        # reset (one set_fec(0.0) per aperture); no per-step FEC moves.
        calib = make_calibrator(
            camera, cfg, medians=[50.0, 100.0],
        )
        result = calib.calibrate_single_aperture(
            aperture="f/4",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=100.0,
            set_aperture_via_camera=False,
            flash_mode="manual",
        )
        # Aperture not set: see above.
        # FEC writes: exactly one — the initial reset to 0.0 at the
        # start of the aperture. No cascade-driven FEC bumps.
        fec_writes = [v for k, v in camera.history if k == "fec"]
        assert fec_writes == [0.0], (
            f"manual mode should leave FEC at 0, got writes {fec_writes}"
        )
        # ISO writes: initial set_iso, then a bump after the under-
        # exposed shot. Order: [100, bumped].
        iso_writes = [v for k, v in camera.history if k == "iso"]
        assert iso_writes[0] == 100, f"initial ISO should be 100, got {iso_writes[0]}"
        assert iso_writes[-1] > 100, (
            f"manual cascade must bump ISO on under-exposure, "
            f"got writes {iso_writes}"
        )
        # Converged on the second capture.
        ap = result.per_aperture["f/4"]
        assert ap.hit_tolerance is True
        assert ap.fec_ev == 0.0
        assert ap.iso_bumped is True

    def test_iso_drop_on_overexposure(self, camera, cfg, tmp_path) -> None:
        """Over-exposed (median > baseline) should drop ISO, not
        bump it. Confirms the manual cascade goes both directions
        unlike the AA cascade where ISO only goes up as a range
        extender."""
        # Use an aperture whose initial ISO is > floor so we have
        # somewhere to drop to. f/8 → ISO 100 per fixture; need an
        # aperture with initial ISO 200 to test drop. f/11.
        calib = make_calibrator(
            camera, cfg, medians=[400.0, 100.0],
        )
        result = calib.calibrate_single_aperture(
            aperture="f/11",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=100.0,
            set_aperture_via_camera=False,
            flash_mode="manual",
        )
        iso_writes = [v for k, v in camera.history if k == "iso"]
        assert iso_writes[0] == 200, (
            f"f/11 starts at ISO 200 per RANGE_EXTENSION; got {iso_writes[0]}"
        )
        assert iso_writes[-1] < 200, (
            f"manual cascade must drop ISO on over-exposure, "
            f"got writes {iso_writes}"
        )
        ap = result.per_aperture["f/11"]
        assert ap.hit_tolerance is True

    def test_baseline_setting_on_first_capture(self, camera, cfg, tmp_path) -> None:
        """First call with baseline=None should set the baseline from
        the first capture's median (same behaviour as AA cascade)."""
        calib = make_calibrator(camera, cfg, medians=[123.0])
        result = calib.calibrate_single_aperture(
            aperture="f/4",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=None,
            set_aperture_via_camera=False,
            flash_mode="manual",
        )
        assert result.baseline_median == pytest.approx(123.0)
        ap = result.per_aperture["f/4"]
        assert ap.hit_tolerance is True
        assert ap.test_shots_taken == 1
        assert ap.fec_ev == 0.0


# ---- First-aperture manual-flash cascade ----------------------------------


def _power_prompt_recorder():
    """Returns (callable, list-of-recorded-args)."""
    recorded: list[dict] = []

    def _prompt(*, power_c_label, power_a_label, context=""):
        recorded.append(
            dict(
                power_c_label=power_c_label,
                power_a_label=power_a_label,
                context=context,
            )
        )
        return True

    return _prompt, recorded


class TestCalibrateFirstApertureManual:
    def test_converges_with_phase_1_in_one_iter(
        self, camera, cfg, tmp_path
    ) -> None:
        """Initial power 1/32 (5.0 stops), first capture lands within
        ±1 EV of baseline → Phase 1 exits in 1 iter. Phase 2 ISO
        fine-tune skipped (already within ±1/3). Phase 3 back-calc
        keeps power. Phase 4 verifies in 1 iter."""
        # Medians: phase 1 capture (close to baseline) + phase 4
        # verification (close to baseline).
        calib = make_calibrator(camera, cfg, medians=[105.0, 102.0])
        prompt, recorded = _power_prompt_recorder()
        result, final_power, final_iso = (
            calib.calibrate_first_aperture_manual(
                aperture="f/2.8",
                lens_dir=tmp_path,
                lens_slug="lens",
                session_id="s",
                baseline_median=100.0,
                prompt_set_flash_power=prompt,
            )
        )
        ap = result.per_aperture["f/2.8"]
        assert ap.hit_tolerance is True
        # Phase 1 settled at 5.0 (1/32) with delta = +0.157 EV
        # (median 105 vs baseline 100). Phase 3 back-calc with the
        # new brute-force solver picks (power=5.33, iso=125) which
        # nudges the residual +0.157 EV closer to baseline (light
        # change ≈ -0.008 EV). Phase 4 verifies at the new target.
        assert ap.flash_power_stops_below_full == pytest.approx(16 / 3)
        assert final_power == pytest.approx(16 / 3)
        assert final_iso == 125
        # Two prompts: phase 1 coarse search step 1 (1/32), phase 4
        # verify at the back-calc target (1/32 -0.3).
        assert len(recorded) == 2
        assert recorded[0]["power_c_label"] == "1/32"
        assert recorded[0]["power_a_label"] == "1/32 -0.3"
        assert recorded[1]["power_c_label"] == "1/32 -0.3"

    def test_phase_1_bumps_up_on_underexposure(
        self, camera, cfg, tmp_path
    ) -> None:
        """First capture is heavily under-exposed → Phase 1 bumps
        power one stop up (e.g. 1/32 → 1/16). Second capture is
        within ±1 EV → exit Phase 1."""
        # Medians:
        #   [0] phase 1 step 1 (power 1/32): 30 → -3.82 EV, bump
        #   [1] phase 1 step 2 (power 1/16): 90 → -0.33 EV, exit phase 1
        #   [2] phase 2 ISO bump 50→100 verification: 95 → in tol
        #   [3] phase 4 verification (back-calc may bump power): 100
        calib = make_calibrator(
            camera, cfg, medians=[30.0, 90.0, 95.0, 100.0],
        )
        prompt, recorded = _power_prompt_recorder()
        result, final_power, _ = calib.calibrate_first_aperture_manual(
            aperture="f/2.8",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=100.0,
            prompt_set_flash_power=prompt,
        )
        # Phase 1 settled at 4.0 (1/16) after a +1-stop bump. Phase 2
        # ISO fine-tune needed +1/3 stop (ISO 100→125) to pull the
        # -0.33 EV under into tolerance. Phase 3 back-calc absorbs
        # the 1/3-stop ISO overhang into power: 4.0 - 1/3 = 11/3 ≈
        # 3.667 (= "1/8 -2/3"). Phase 4 verifies at the new power
        # with ISO back to floor 100.
        assert final_power == pytest.approx(11 / 3)
        labels = [r["power_c_label"] for r in recorded]
        # phase 1 step 1, phase 1 step 2, phase 4 verify.
        assert labels == ["1/32", "1/16", "1/8 -0.7"]

    def test_phase_1_safety_termination_too_dim(
        self, camera, cfg, tmp_path
    ) -> None:
        """If even 1/1 power is still too dim, abort with
        SessionAborted instead of looping forever or returning
        garbage."""
        # All captures very dim — even at 1/1 the search won't
        # converge.
        calib = make_calibrator(camera, cfg, medians=[5.0] * 20)
        prompt, _ = _power_prompt_recorder()
        from errors import SessionAborted
        with pytest.raises(SessionAborted, match="still under-exposed"):
            calib.calibrate_first_aperture_manual(
                aperture="f/2.8",
                lens_dir=tmp_path,
                lens_slug="lens",
                session_id="s",
                baseline_median=100.0,
                prompt_set_flash_power=prompt,
            )

    def test_phase_1_safety_termination_too_bright(
        self, camera, cfg, tmp_path
    ) -> None:
        """If even 1/128 is still too bright, abort."""
        calib = make_calibrator(camera, cfg, medians=[250.0] * 20)
        prompt, _ = _power_prompt_recorder()
        from errors import SessionAborted
        with pytest.raises(SessionAborted, match="still over-exposed"):
            calib.calibrate_first_aperture_manual(
                aperture="f/2.8",
                lens_dir=tmp_path,
                lens_slug="lens",
                session_id="s",
                baseline_median=100.0,
                prompt_set_flash_power=prompt,
            )


# ---- Subsequent-aperture manual-flash cascade -----------------------------


class TestCalibrateSubsequentApertureManual:
    def test_uses_target_for_next_aperture(self, camera, cfg, tmp_path) -> None:
        """Calls the deterministic target_for_next_aperture helper
        and prompts the user once with the resulting label."""
        # Prev: 1/4 (2.0 stops), ISO 100 (= 1 stop above MANUAL_ISO_FLOOR).
        # diff = 1 stop aperture.
        # iso_above_floor = log2(100/50) = 1.
        # total_boost = 1 + 1 = 2.
        # naive_power = 2 - 2 = 0 (= 1/1).
        # 0 < soft_cap=1.0, hold at soft_cap. iso_stops = 1.
        # iso = 50 * 2^1 = 100. So target = (1/2, 100).
        calib = make_calibrator(camera, cfg, medians=[100.0])
        prompt, recorded = _power_prompt_recorder()
        result, final_power, final_iso = (
            calib.calibrate_subsequent_aperture_manual(
                aperture="f/4",
                lens_dir=tmp_path,
                lens_slug="lens",
                session_id="s",
                baseline_median=100.0,
                prev_power_stops=2.0,    # 1/4
                prev_iso=100,
                prev_ev_delta=0.0,
                aperture_stops_diff=1.0,
                prompt_set_flash_power=prompt,
            )
        )
        # Target should be power=1/2 (1.0 stops), ISO 100.
        assert final_power == pytest.approx(1.0)
        assert final_iso == 100
        assert len(recorded) == 1
        assert recorded[0]["power_c_label"] == "1/2"
        ap = result.per_aperture["f/4"]
        assert ap.flash_power_stops_below_full == pytest.approx(1.0)
        assert ap.hit_tolerance is True

    def test_sanity_check_fires_on_large_initial_delta(
        self, camera, cfg, tmp_path
    ) -> None:
        """If the first capture is more than
        MANUAL_FLASH_BUMP_SANITY_EV off baseline, the cascade
        re-prompts the user to verify the power / ring setting."""
        # First capture: median 30 → delta = -3.43 EV (huge).
        # Re-prompt fires. Retry capture: 100 (in tol).
        calib = make_calibrator(camera, cfg, medians=[30.0, 100.0])
        prompt, recorded = _power_prompt_recorder()
        calib.calibrate_subsequent_aperture_manual(
            aperture="f/4",
            lens_dir=tmp_path,
            lens_slug="lens",
            session_id="s",
            baseline_median=100.0,
            prev_power_stops=2.0,
            prev_iso=100,
            prev_ev_delta=0.0,
            aperture_stops_diff=1.0,
            prompt_set_flash_power=prompt,
        )
        # Two prompts: initial target, then SANITY re-prompt.
        assert len(recorded) == 2
        assert "SANITY" in recorded[1]["context"]
