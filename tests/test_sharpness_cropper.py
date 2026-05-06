"""Tests for sharpness.cropper.

``crop_to_bytes`` operates on a decoded ndarray (BGR, uint8 — the
turbojpeg native layout). ``crop_image_file`` is exercised
end-to-end with a real JPEG round-trip to confirm the
decode → slice → encode → file pipeline holds together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sharpness.cropper import (
    CropResult,
    crop_image_file,
    crop_to_bytes,
)
from sharpness.crops import CROP_NAMES, CropBox


# --- helpers ---------------------------------------------------------------


def _flat_bgr(w: int, h: int, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noisy_bgr(w: int, h: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype="uint8")


# --- pure crop_to_bytes ----------------------------------------------------


class TestCropToBytes:
    def test_returns_jpeg_bytes(self) -> None:
        img = _flat_bgr(1000, 1000)
        out = crop_to_bytes(img, CropBox(0, 0, 100, 100))
        assert out[:3] == b"\xff\xd8\xff"  # JPEG SOI marker

    def test_uniform_image_compresses_small(self) -> None:
        img = _flat_bgr(1000, 1000)
        small = crop_to_bytes(img, CropBox(0, 0, 500, 500))
        # libjpeg-turbo (no extra optimization pass) lands a 500×500
        # uniform grey at ~5–6 KB; the proxy here is "trivially small
        # vs textured content", so 8 KB is a comfortable ceiling.
        assert len(small) < 8000

    def test_textured_image_compresses_larger(self) -> None:
        textured = _noisy_bgr(500, 500)
        big = crop_to_bytes(textured, CropBox(0, 0, 500, 500))
        flat = _flat_bgr(500, 500)
        small = crop_to_bytes(flat, CropBox(0, 0, 500, 500))
        assert len(big) > 10 * len(small)

    def test_quality_changes_size(self) -> None:
        textured = _noisy_bgr(500, 500, seed=0)
        hi = crop_to_bytes(textured, CropBox(0, 0, 500, 500), quality=95)
        lo = crop_to_bytes(textured, CropBox(0, 0, 500, 500), quality=40)
        assert len(hi) > len(lo)

    def test_handles_non_contiguous_slice(self) -> None:
        # Box that lands away from origin produces a non-contiguous
        # numpy slice; the encoder needs C-contiguous input.
        img = _noisy_bgr(1000, 1000)
        out = crop_to_bytes(img, CropBox(100, 200, 500, 800))
        assert out[:3] == b"\xff\xd8\xff"


# --- crop_image_file (one real-JPEG round-trip) -----------------------------


class TestCropImageFile:
    def test_writes_seventeen_crops_with_recorded_sizes(
        self, tmp_path: Path
    ) -> None:
        # 1500x1500 noisy JPEG → 1000 mid-band wide, well above 500x500
        # so DX crops fit.
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 256, size=(1500, 1500, 3), dtype="uint8")
        Image.fromarray(arr).save(tmp_path / "source.jpg", quality=92)

        out_dir = tmp_path / "cropped"
        result = crop_image_file(tmp_path / "source.jpg", out_dir)

        assert isinstance(result, CropResult)
        assert set(result.sizes.keys()) == set(CROP_NAMES)
        for name in CROP_NAMES:
            target = out_dir / name / "source.jpg"
            assert target.is_file(), f"missing {target}"
            assert result.size_for(name) == target.stat().st_size

    def test_uses_source_filename_under_crop_subdir(self, tmp_path: Path) -> None:
        Image.new("RGB", (1500, 1500), color=(50, 50, 50)).save(
            tmp_path / "step_005.jpg"
        )
        out_dir = tmp_path / "out"
        crop_image_file(tmp_path / "step_005.jpg", out_dir)
        # Each crop subdir holds a file named after the source.
        for name in CROP_NAMES:
            assert (out_dir / name / "step_005.jpg").is_file()

    def test_rejects_image_too_small(self, tmp_path: Path) -> None:
        Image.new("RGB", (300, 300), color=(0, 0, 0)).save(tmp_path / "small.jpg")
        with pytest.raises(ValueError):
            crop_image_file(tmp_path / "small.jpg", tmp_path / "out")

    def test_size_for_returns_recorded_value(self) -> None:
        result = CropResult(source=Path("/x.jpg"), sizes={"center": 12345})
        assert result.size_for("center") == 12345
