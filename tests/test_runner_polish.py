"""Tests for the session.runner polish (Ctrl-C, FEC sanity, IDs,
baseline reference image)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from errors import FECSanityError
from session.runner import SessionRunner
from tests.test_runner import FakeCamera, FakeRail, cfg, make_runner


# ---- Ctrl-C -----------------------------------------------------------------


class TestCtrlC:
    def test_keyboard_interrupt_stops_rail_and_finalises_session(
        self, cfg, tmp_path
    ) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 5,
            inputs={"steps": 3, "end_mm": 4.0},
        )

        # Trigger a KeyboardInterrupt during the prompt.
        def _prompt() -> str | None:
            raise KeyboardInterrupt

        runner.prompt_lens_config = _prompt

        with pytest.raises(KeyboardInterrupt):
            runner.run()

        # rail.stop() was called.
        assert any(k == "stop" for k, _ in rail.history)
        # session.json shows aborted = True and ended_at set.
        meta = json.loads((runner.session_dir / "session.json").read_text())
        assert meta["aborted"] is True
        assert meta["ended_at"] is not None
        assert camera.closed is True


# ---- FEC sanity check at session start ------------------------------------


class TestFECSanityCheckAtStart:
    def test_called_when_camera_supports_it(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        sanity_calls: list[dict] = []

        def _sanity(**kwargs):
            sanity_calls.append(kwargs)

        camera.fec_sanity_check = _sanity                # type: ignore[attr-defined]
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        assert len(sanity_calls) == 1
        # Defaults populated from config.
        assert sanity_calls[0]["baseline_fec"] == 0.0

    def test_skipped_when_camera_lacks_method(self, cfg, tmp_path) -> None:
        # FakeCamera has no fec_sanity_check; start() must not error.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()  # should be a no-op for the sanity step

    def test_failure_propagates(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()

        def _sanity(**kwargs):
            raise FECSanityError("SU-800 detected")

        camera.fec_sanity_check = _sanity                # type: ignore[attr-defined]
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        with pytest.raises(FECSanityError):
            runner.start()


# ---- StackShot IDs in session.json ----------------------------------------


class TestStackShotIDs:
    def test_ids_recorded_when_rail_exposes_them(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        rail.software_id = lambda: 0x12345678            # type: ignore[attr-defined]
        rail.hardware_id = lambda: 0xABCDEF01            # type: ignore[attr-defined]
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        meta = json.loads((runner.session_dir / "session.json").read_text())
        assert meta["stackshot_software_id"] == 0x12345678
        assert meta["stackshot_hardware_id"] == 0xABCDEF01

    def test_ids_left_null_when_rail_lacks_them(self, cfg, tmp_path) -> None:
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        meta = json.loads((runner.session_dir / "session.json").read_text())
        assert meta["stackshot_software_id"] is None
        assert meta["stackshot_hardware_id"] is None


# ---- Baseline reference image ------------------------------------------


class TestBaselineFromReference:
    """If config.BASELINE_REFERENCE_IMAGE points at a JPEG, the
    runner reads its central-crop median once at session start and
    uses that as the absolute calibration target — instead of the
    per-session "first shot at widest aperture" baseline."""

    def _write_uniform_jpeg(self, path: Path, value: int) -> None:
        arr = np.full((600, 800), value, dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(path, format="JPEG", quality=95)

    def test_baseline_set_from_reference(self, cfg, tmp_path) -> None:
        ref = tmp_path / "ref.jpg"
        self._write_uniform_jpeg(ref, 137)
        cfg.BASELINE_REFERENCE_IMAGE = ref

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        # Median of a uniform 137 image is 137 (JPEG quantisation can
        # nudge it a little — allow ±2).
        assert runner.baseline_median == pytest.approx(137.0, abs=2.0)

    def test_no_reference_falls_back_to_first_shot(self, cfg, tmp_path) -> None:
        # Default config has no BASELINE_REFERENCE_IMAGE attribute,
        # OR it's None. Either way, runner.baseline_median stays None
        # at session start; the calibrator picks it up from the first
        # widest-aperture shot of the first lens config (validated
        # in test_calibrate.TestBaseline).
        cfg.BASELINE_REFERENCE_IMAGE = None
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        assert runner.baseline_median is None

    def test_session_json_records_reference_path(self, cfg, tmp_path) -> None:
        ref = tmp_path / "ref.jpg"
        self._write_uniform_jpeg(ref, 100)
        cfg.BASELINE_REFERENCE_IMAGE = ref

        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0],
        )
        runner.start()
        meta = json.loads((runner.session_dir / "session.json").read_text())
        assert meta["baseline_median_luminance"] == pytest.approx(100.0, abs=2.0)
        assert "baseline from reference" in meta["notes"]
        assert str(ref) in meta["notes"]
