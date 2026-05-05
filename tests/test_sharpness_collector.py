"""Tests for sharpness.collector.

The cropper is mocked with a fake callable so we exercise the threading
and aggregation logic without going through Pillow on every test.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sharpness.collector import (
    CollectedDataset,
    LiveCollector,
    ShotSample,
    collect_from_dir,
)
from sharpness.cropper import CropResult


def _fake_crop(sizes: dict[str, int]):
    """Make a crop function that returns a fixed sizes map."""

    def fn(source: Path, out_dir: Path) -> CropResult:
        return CropResult(source=source, sizes=dict(sizes))

    return fn


# --- ShotSample / CollectedDataset ----------------------------------------


class TestCollectedDataset:
    def test_by_aperture_groups_and_sorts(self) -> None:
        ds = CollectedDataset(
            samples=[
                ShotSample(rail_step=2, aperture=8.0, sizes={"center": 100}),
                ShotSample(rail_step=0, aperture=8.0, sizes={"center": 90}),
                ShotSample(rail_step=1, aperture=11.0, sizes={"center": 80}),
            ]
        )
        out = ds.by_aperture()
        assert sorted(out.keys()) == [8.0, 11.0]
        # 8.0 group sorted by rail_step
        assert [s.rail_step for s in out[8.0]] == [0, 2]

    def test_crop_series_for_returns_sorted_pairs(self) -> None:
        ds = CollectedDataset(
            samples=[
                ShotSample(rail_step=2, aperture=8.0, sizes={"center": 100, "x": 5}),
                ShotSample(rail_step=0, aperture=8.0, sizes={"center": 90, "x": 5}),
                ShotSample(rail_step=1, aperture=11.0, sizes={"center": 80, "x": 5}),
            ]
        )
        assert ds.crop_series_for(8.0, "center") == [(0, 90), (2, 100)]
        assert ds.crop_series_for(11.0, "center") == [(1, 80)]

    def test_crop_series_skips_missing_crops(self) -> None:
        # If a sample doesn't have the crop key (shouldn't happen in
        # practice but defensive), it's excluded rather than KeyError.
        ds = CollectedDataset(
            samples=[
                ShotSample(rail_step=0, aperture=8.0, sizes={"center": 100}),
                ShotSample(rail_step=1, aperture=8.0, sizes={"top-left": 80}),
            ]
        )
        assert ds.crop_series_for(8.0, "center") == [(0, 100)]


# --- LiveCollector --------------------------------------------------------


class TestLiveCollector:
    def test_records_shots_in_order(self, tmp_path: Path) -> None:
        sizes = {"center": 1234, "top-left": 5678}
        with LiveCollector(tmp_path, crop_fn=_fake_crop(sizes), max_workers=1) as c:
            c.record_shot(0, 8.0, tmp_path / "step_000.jpg")
            c.record_shot(1, 8.0, tmp_path / "step_001.jpg")

        ds = c.dataset
        assert [s.rail_step for s in ds.samples] == [0, 1]
        assert all(s.sizes == sizes for s in ds.samples)

    def test_records_multiple_apertures(self, tmp_path: Path) -> None:
        with LiveCollector(
            tmp_path, crop_fn=_fake_crop({"center": 100}), max_workers=1
        ) as c:
            for ap in (4.0, 5.6, 8.0):
                for step in range(3):
                    c.record_shot(step, ap, tmp_path / f"x_{step}_{ap}.jpg")

        by_ap = c.dataset.by_aperture()
        assert sorted(by_ap.keys()) == [4.0, 5.6, 8.0]
        assert all(len(v) == 3 for v in by_ap.values())

    def test_threading_preserves_all_samples_under_concurrency(
        self, tmp_path: Path
    ) -> None:
        # Make the crop function deliberately slow so multiple workers
        # are in flight simultaneously; assert no samples are dropped.
        latch = threading.Event()

        def slow_crop(source: Path, out_dir: Path) -> CropResult:
            latch.wait(0.05)
            return CropResult(source=source, sizes={"center": 1})

        with LiveCollector(tmp_path, crop_fn=slow_crop, max_workers=4) as c:
            for i in range(50):
                c.record_shot(i, 8.0, tmp_path / f"x_{i}.jpg")
            latch.set()

        assert len(c.dataset.samples) == 50
        assert {s.rail_step for s in c.dataset.samples} == set(range(50))

    def test_propagates_cropper_errors_at_shutdown(self, tmp_path: Path) -> None:
        def bad_crop(source: Path, out_dir: Path) -> CropResult:
            raise IOError("disk full")

        c = LiveCollector(tmp_path, crop_fn=bad_crop, max_workers=1)
        c.record_shot(0, 8.0, tmp_path / "x.jpg")
        with pytest.raises(RuntimeError, match="failed"):
            c.shutdown()

    def test_record_after_shutdown_raises(self, tmp_path: Path) -> None:
        c = LiveCollector(tmp_path, crop_fn=_fake_crop({}), max_workers=1)
        c.shutdown()
        with pytest.raises(RuntimeError, match="after shutdown"):
            c.record_shot(0, 8.0, tmp_path / "x.jpg")

    def test_idempotent_shutdown(self, tmp_path: Path) -> None:
        c = LiveCollector(tmp_path, crop_fn=_fake_crop({}), max_workers=1)
        c.shutdown()
        c.shutdown()  # must not raise

    def test_dataset_returns_a_snapshot_copy(self, tmp_path: Path) -> None:
        with LiveCollector(
            tmp_path, crop_fn=_fake_crop({"center": 1}), max_workers=1
        ) as c:
            c.record_shot(0, 8.0, tmp_path / "x.jpg")
        snap = c.dataset
        # Mutating the snapshot's samples must not bleed into the
        # collector's internal list.
        snap.samples.clear()
        assert len(c.dataset.samples) == 1

    def test_uses_provided_crop_fn(self, tmp_path: Path) -> None:
        mock = MagicMock(return_value=CropResult(source=tmp_path / "x.jpg", sizes={}))
        with LiveCollector(tmp_path, crop_fn=mock, max_workers=1) as c:
            c.record_shot(7, 11.0, tmp_path / "x.jpg")
        # First positional arg is the source path; second is the out dir.
        called_args = mock.call_args.args
        assert called_args[0] == tmp_path / "x.jpg"
        assert called_args[1] == tmp_path

    def test_clear_samples_for_steps_drops_matching_rail_steps(
        self, tmp_path: Path
    ) -> None:
        # Build a 3-step × 2-aperture dataset, then drop steps {1, 2}.
        # Step 0 survives at both apertures; samples at steps 1 and 2
        # are gone.
        with LiveCollector(
            tmp_path,
            crop_fn=_fake_crop({"center": 1}),
            max_workers=1,
        ) as c:
            for step in range(3):
                for ap in (4.0, 8.0):
                    c.record_shot(step, ap, tmp_path / f"x_{step}_{ap}.jpg")
            c.wait_for_pending()
            assert len(c.dataset.samples) == 6
            c.clear_samples_for_steps([1, 2])
            survivors = c.dataset.samples
            assert len(survivors) == 2
            assert {s.rail_step for s in survivors} == {0}
            assert {s.aperture for s in survivors} == {4.0, 8.0}

    def test_clear_samples_for_steps_empty_is_noop(self, tmp_path: Path) -> None:
        with LiveCollector(
            tmp_path,
            crop_fn=_fake_crop({"center": 1}),
            max_workers=1,
        ) as c:
            c.record_shot(0, 4.0, tmp_path / "x.jpg")
            c.wait_for_pending()
            c.clear_samples_for_steps([])
            assert len(c.dataset.samples) == 1

    def test_crops_under_source_routes_per_aperture(self, tmp_path: Path) -> None:
        # In production the runner sets crops_under_source=True so the
        # cropper writes to <jpg.parent>/cropped/<crop>/<source>.
        # Without this every aperture's "step_0100.jpg" overwrites the
        # previous one in <lens>/cropped/<crop>/.
        out_dirs: list[Path] = []

        def capture(source: Path, out_dir: Path) -> CropResult:
            out_dirs.append(out_dir)
            return CropResult(source=source, sizes={"center": 1})

        f4 = tmp_path / "f_4"
        f4.mkdir()
        f8 = tmp_path / "f_8"
        f8.mkdir()
        with LiveCollector(
            tmp_path,
            crop_fn=capture,
            max_workers=1,
            crops_under_source=True,
            cropped_dirname="cropped",
        ) as c:
            c.record_shot(0, 4.0, f4 / "step_0100.jpg")
            c.record_shot(0, 8.0, f8 / "step_0100.jpg")
        assert sorted(out_dirs) == sorted([
            f4 / "cropped",
            f8 / "cropped",
        ])


class TestWaitForPending:
    def test_blocks_until_in_flight_crops_complete(self, tmp_path: Path) -> None:
        # Slow crop function holds each call until we set the latch.
        latch = threading.Event()

        def slow_crop(source: Path, out_dir: Path) -> CropResult:
            latch.wait(timeout=1.0)
            return CropResult(source=source, sizes={"center": 1})

        c = LiveCollector(tmp_path, crop_fn=slow_crop, max_workers=2)
        try:
            for i in range(4):
                c.record_shot(i, 8.0, tmp_path / f"x_{i}.jpg")

            # Before the latch fires, dataset is incomplete.
            assert len(c.dataset.samples) < 4
            latch.set()
            c.wait_for_pending()
            # After the barrier, every sample is in.
            assert len(c.dataset.samples) == 4
        finally:
            c.shutdown()

    def test_subsequent_record_shot_still_works(self, tmp_path: Path) -> None:
        with LiveCollector(
            tmp_path, crop_fn=_fake_crop({"center": 7}), max_workers=2
        ) as c:
            for i in range(3):
                c.record_shot(i, 8.0, tmp_path / f"a_{i}.jpg")
            c.wait_for_pending()
            assert len(c.dataset.samples) == 3
            # Pool is still alive; we can record more.
            for i in range(3, 6):
                c.record_shot(i, 8.0, tmp_path / f"b_{i}.jpg")
            c.wait_for_pending()
            assert len(c.dataset.samples) == 6

    def test_errors_surface_at_barrier(self, tmp_path: Path) -> None:
        def bad_crop(source: Path, out_dir: Path) -> CropResult:
            raise IOError("boom")

        c = LiveCollector(tmp_path, crop_fn=bad_crop, max_workers=1)
        try:
            c.record_shot(0, 8.0, tmp_path / "x.jpg")
            with pytest.raises(RuntimeError, match="failed"):
                c.wait_for_pending()
            # After surfacing, the error list is cleared so a follow-up
            # successful crop + wait works.
            c._crop_fn = _fake_crop({"center": 1})  # type: ignore[attr-defined]
            c.record_shot(1, 8.0, tmp_path / "y.jpg")
            c.wait_for_pending()  # must not re-raise
        finally:
            c.shutdown()

    def test_idempotent_when_no_pending_work(self, tmp_path: Path) -> None:
        with LiveCollector(
            tmp_path, crop_fn=_fake_crop({"center": 1}), max_workers=1
        ) as c:
            c.wait_for_pending()  # nothing in flight; must not block
            c.record_shot(0, 8.0, tmp_path / "x.jpg")
            c.wait_for_pending()
            c.wait_for_pending()  # already drained


class TestCollectFromDir:
    def test_drives_collector_synchronously(self, tmp_path: Path) -> None:
        items = [
            (0, 8.0, tmp_path / "0.jpg"),
            (1, 8.0, tmp_path / "1.jpg"),
            (2, 11.0, tmp_path / "2.jpg"),
        ]
        ds = collect_from_dir(
            items, tmp_path / "out", crop_fn=_fake_crop({"center": 42})
        )
        assert len(ds.samples) == 3
        assert all(s.sizes["center"] == 42 for s in ds.samples)
