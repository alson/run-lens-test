"""Tests covering the streaming-sharpness wiring in session.runner.

These don't go through the real LiveCollector — they substitute a
recording fake so we can assert the runner makes the right calls in
the right order: record_shot per capture, wait_for_pending before
each evaluate, shutdown at the end of every lens config.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exposure.range_check import RangeChecker, RangeResult
from session.runner import SessionRunner, SharpnessCollectorLike
from tests.test_runner import FakeCamera, FakeRail, cfg, make_runner


# ---- Fake sharpness collector ----------------------------------------------


@dataclass
class FakeSharpnessCollector:
    """Records every call so tests can assert the call order."""

    out_dir: Path
    events: list[tuple] = field(default_factory=list)
    shutdown_called: bool = False

    def record_shot(self, rail_step: int, aperture: float, jpg_path) -> None:
        # Verify the runner passes the source filename — the cropper
        # uses source.name to derive the per-crop output filename.
        self.events.append(("record", rail_step, aperture, str(jpg_path)))

    def wait_for_pending(self) -> None:
        self.events.append(("wait", None, None, None))

    def shutdown(self, *, wait: bool = True) -> None:
        self.shutdown_called = True
        self.events.append(("shutdown", None, None, None))


# ---- Scripted RangeChecker (so we can drive evaluate path) ----------------


class _OkChecker(RangeChecker):
    def __init__(self) -> None:
        super().__init__(config=None)

    def evaluate(self, *args, **kwargs) -> RangeResult:                # type: ignore[override]
        return RangeResult(
            ok=True, peak_side="centered", needs_extension=False,
        )


# ---- Helper: assemble a runner with a FakeSharpnessCollector --------------


def _build_runner(cfg, tmp_path: Path):
    camera = FakeCamera(apertures=["f/4"])
    rail = FakeRail()
    runner = make_runner(
        camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
        medians=[100.0] * 20,
        inputs={"steps": 3, "end_mm": 4.0},
    )
    runner.range_checker = _OkChecker()
    collectors: list[FakeSharpnessCollector] = []

    def _factory(out_dir: Path) -> SharpnessCollectorLike:
        c = FakeSharpnessCollector(out_dir=out_dir)
        collectors.append(c)
        return c

    runner.sharpness_collector_factory = _factory

    def _prompt():
        if not hasattr(_prompt, "called"):
            _prompt.called = True
            return "L"
        return None

    runner.prompt_lens_config = _prompt
    return runner, camera, collectors


# ---- Tests -----------------------------------------------------------------


class TestStreamingPipeline:
    def test_record_shot_after_each_capture(self, cfg, tmp_path: Path) -> None:
        runner, camera, collectors = _build_runner(cfg, tmp_path)
        runner.run()

        assert len(collectors) == 1
        col = collectors[0]
        # 1 calibration + 3 sweep = 4 captures; record_shot is called
        # for each of the SWEEP captures (the calibrator doesn't go
        # through the collector — calibration shots are exposure
        # references, not sharpness data).
        records = [e for e in col.events if e[0] == "record"]
        assert len(records) == 3
        # rail_step is 0..2 in order; aperture is the float 4.0.
        assert [e[1] for e in records] == [0, 1, 2]
        assert all(e[2] == pytest.approx(4.0) for e in records)
        # Each record carries the on-disk path of the source JPEG so
        # the cropper can use its filename verbatim.
        for _, _, _, path_str in records:
            assert path_str.endswith(".jpg")
            assert "f_4" in path_str

    def test_wait_before_each_evaluate(self, cfg, tmp_path: Path) -> None:
        runner, camera, collectors = _build_runner(cfg, tmp_path)
        runner.run()

        col = collectors[0]
        # _OkChecker returns ok=True so only the first evaluate is
        # invoked → exactly one wait before it.
        waits = [e for e in col.events if e[0] == "wait"]
        assert len(waits) == 1

    def test_shutdown_at_end_of_lens_config(self, cfg, tmp_path: Path) -> None:
        runner, camera, collectors = _build_runner(cfg, tmp_path)
        runner.run()

        col = collectors[0]
        assert col.shutdown_called is True

    def test_one_collector_per_lens_config(self, cfg, tmp_path: Path) -> None:
        # Run two lens configs back to back; each should get its own
        # collector instance.
        camera = FakeCamera(apertures=["f/4"])
        rail = FakeRail()
        runner = make_runner(
            camera=camera, rail=rail, cfg=cfg, tmp_path=tmp_path,
            medians=[100.0] * 20,
            inputs={"steps": 3, "end_mm": 4.0},
        )
        runner.range_checker = _OkChecker()
        collectors: list[FakeSharpnessCollector] = []

        def _factory(out_dir: Path) -> SharpnessCollectorLike:
            c = FakeSharpnessCollector(out_dir=out_dir)
            collectors.append(c)
            return c

        runner.sharpness_collector_factory = _factory
        prompts = deque(["lensA", "lensB", None])
        runner.prompt_lens_config = lambda: prompts.popleft()
        runner.run()

        assert len(collectors) == 2
        assert all(c.shutdown_called for c in collectors)
        # Each collector is rooted under its own lens dir.
        assert collectors[0].out_dir.name == "lensA"
        assert collectors[1].out_dir.name == "lensB"

    def test_capture_and_record_are_paired_in_order(
        self, cfg, tmp_path: Path
    ) -> None:
        # The runner records each shot immediately after capture, so
        # in the captured / recorded interleaved log the ordering must
        # be: capture(N), record(N), capture(N+1), record(N+1), …
        runner, camera, collectors = _build_runner(cfg, tmp_path)
        runner.run()

        col = collectors[0]
        # Build a unified timeline: take the sweep captures (those land
        # under <lens>/f_4/, NOT under <lens>/calibration/) and the
        # record calls — they should match 1:1 by path.
        capture_paths = [
            v for k, v in camera.history
            if k == "capture" and "/f_4/" in v
        ]
        record_paths = [e[3] for e in col.events if e[0] == "record"]
        assert capture_paths == record_paths
