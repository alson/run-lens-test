"""Tests for the session.runner polish (Ctrl-C, FEC sanity, IDs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
