"""Tests for exposure.analyse — pure pixel math + JPEG round-trip."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from exposure.analyse import (
    CropBox,
    CropStats,
    _measure_array,
    crop_box,
    ev_delta,
    linearise,
    measure_crop,
    save_overlay,
)


# ---- crop_box --------------------------------------------------------------


class TestCropBox:
    def test_centred_full_image(self) -> None:
        box = crop_box(100, 100, 1.0, 1.0)
        assert box == CropBox(left=0, top=0, right=100, bottom=100)

    def test_centred_quarter(self) -> None:
        box = crop_box(100, 100, 0.5, 0.5)
        # Centred 50x50 in a 100x100 image: 25..75 on each axis.
        assert box == CropBox(left=25, top=25, right=75, bottom=75)

    def test_y_offset_moves_down(self) -> None:
        box = crop_box(100, 100, 0.2, 0.2, y_off_frac=0.1)
        # Crop centre is image centre + 10 px = (50, 60); 20x20 around that.
        assert box == CropBox(left=40, top=50, right=60, bottom=70)

    def test_clamped_to_image_bounds(self) -> None:
        # Negative offset that would push crop off the top; should clamp.
        box = crop_box(100, 100, 0.5, 0.5, y_off_frac=-1.0)
        assert box.top == 0
        assert box.bottom > 0


# ---- _measure_array (faster than going through a JPEG) ---------------------


class TestMeasureArray:
    def test_uniform_grey(self) -> None:
        arr = np.full((50, 50), 128, dtype=np.uint8)
        stats = _measure_array(arr, CropBox(0, 0, 50, 50))
        assert stats.p50 == pytest.approx(128.0)
        assert stats.mean == pytest.approx(128.0)
        assert stats.clip_hi_pct == 0.0
        assert stats.clip_lo_pct == 0.0
        assert stats.shape == (50, 50)

    def test_high_clip_detected(self) -> None:
        arr = np.full((10, 10), 255, dtype=np.uint8)
        stats = _measure_array(arr, CropBox(0, 0, 10, 10))
        assert stats.clip_hi_pct == 100.0

    def test_low_clip_detected(self) -> None:
        arr = np.zeros((10, 10), dtype=np.uint8)
        stats = _measure_array(arr, CropBox(0, 0, 10, 10))
        assert stats.clip_lo_pct == 100.0

    def test_empty_crop_raises(self) -> None:
        arr = np.full((10, 10), 100, dtype=np.uint8)
        with pytest.raises(ValueError, match="empty"):
            _measure_array(arr, CropBox(5, 5, 5, 5))


# ---- linearise / ev_delta --------------------------------------------------


class TestLinearise:
    def test_floor_protects_against_log_zero(self) -> None:
        # value 0 → linearised at the floor (1/255)^gamma > 0.
        assert linearise(0.0) > 0.0

    def test_white_is_one(self) -> None:
        assert linearise(255.0) == pytest.approx(1.0)

    def test_relative_doubling_is_one_stop(self) -> None:
        # Pixel ratio 2:1 in linear light is one EV.
        # In sRGB-style gamma 2.2, displayed values 100 vs 143 should
        # be ≈ 1 stop apart (143/100)^2.2 ≈ 2.16. Close to 1 EV.
        assert math.isclose(
            ev_delta(143, 100), 1.0, abs_tol=0.2
        )


class TestEvDelta:
    def test_zero_when_equal(self) -> None:
        assert ev_delta(120, 120) == pytest.approx(0.0)

    def test_brighter_is_positive(self) -> None:
        assert ev_delta(200, 100) > 0

    def test_dimmer_is_negative(self) -> None:
        assert ev_delta(50, 100) < 0

    def test_floor_keeps_zero_baseline_finite(self) -> None:
        # The floor in linearise() means ev_delta never sees 0 base
        # via the public API — it falls back to (1/255)**gamma. So
        # ev_delta(0, 0) is well-defined and equal to 0.
        assert ev_delta(0, 0) == pytest.approx(0.0)

    def test_known_pair(self) -> None:
        # FEC sweep on the rig produced 100 vs 143 ~= +1 EV. Use the
        # same datapoint as a sanity peg.
        assert ev_delta(143, 100) == pytest.approx(1.0, abs=0.15)


# ---- JPEG round-trip via measure_crop --------------------------------------


def _write_jpg(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="L").save(path, format="JPEG", quality=95)


class TestMeasureCropRoundTrip:
    def test_uniform_grey_jpeg(self, tmp_path: Path) -> None:
        arr = np.full((100, 100), 128, dtype=np.uint8)
        path = tmp_path / "grey.jpg"
        _write_jpg(path, arr)
        stats = measure_crop(path, CropBox(0, 0, 100, 100))
        # JPEG quantisation can shift the median slightly; allow 2.
        assert stats.p50 == pytest.approx(128.0, abs=2.0)

    def test_brightness_difference_through_jpeg(self, tmp_path: Path) -> None:
        bright = np.full((50, 50), 200, dtype=np.uint8)
        dim = np.full((50, 50), 50, dtype=np.uint8)
        bp = tmp_path / "bright.jpg"
        dp = tmp_path / "dim.jpg"
        _write_jpg(bp, bright)
        _write_jpg(dp, dim)
        bs = measure_crop(bp, CropBox(0, 0, 50, 50))
        ds = measure_crop(dp, CropBox(0, 0, 50, 50))
        assert bs.p50 > ds.p50
        # 200/50 = 4x linear-display, ≈ +4 stops of 8-bit perceptual.
        # In linear-light terms after gamma, well over +2 EV.
        assert ev_delta(bs.p50, ds.p50) > 2.0


# ---- save_overlay ----------------------------------------------------------


class TestSaveOverlay:
    def test_writes_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src.jpg"
        dst = tmp_path / "ov.jpg"
        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(src, format="JPEG")
        save_overlay(src, CropBox(50, 50, 150, 150), dst, max_width=200)
        assert dst.exists()
        with Image.open(dst) as im:
            # Should have been scaled to width 200 (200/400 = 0.5).
            assert im.size[0] == 200
            assert im.size[1] == 150
