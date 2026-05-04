"""Tests for sharpness.cropper.

The pure ``crop_to_bytes`` is exercised with synthetic in-memory images;
``crop_image_file`` is exercised once end-to-end with a real Pillow JPEG
to confirm the box→file→size pipeline holds together.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sharpness.cropper import (
    CropResult,
    crop_image_file,
    crop_to_bytes,
)
from sharpness.crops import CROP_NAMES, CropBox, all_crop_boxes


# --- pure crop_to_bytes ----------------------------------------------------


class TestCropToBytes:
    def test_returns_jpeg_bytes(self) -> None:
        img = Image.new("RGB", (1000, 1000), color=(128, 128, 128))
        out = crop_to_bytes(img, CropBox(0, 0, 100, 100))
        assert out[:3] == b"\xff\xd8\xff"  # JPEG SOI marker

    def test_uniform_image_compresses_small(self) -> None:
        img = Image.new("RGB", (1000, 1000), color=(128, 128, 128))
        small = crop_to_bytes(img, CropBox(0, 0, 500, 500))
        assert len(small) < 5000  # uniform grey JPEGs compress trivially

    def test_textured_image_compresses_larger(self) -> None:
        # High-frequency noise gives much larger JPEGs than uniform grey.
        import numpy as np

        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(500, 500, 3), dtype="uint8")
        textured = Image.fromarray(arr)
        big = crop_to_bytes(textured, CropBox(0, 0, 500, 500))

        flat = Image.new("RGB", (500, 500), color=(128, 128, 128))
        small = crop_to_bytes(flat, CropBox(0, 0, 500, 500))

        assert len(big) > 10 * len(small)

    def test_quality_changes_size(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(500, 500, 3), dtype="uint8")
        textured = Image.fromarray(arr)
        hi = crop_to_bytes(textured, CropBox(0, 0, 500, 500), quality=95)
        lo = crop_to_bytes(textured, CropBox(0, 0, 500, 500), quality=40)
        assert len(hi) > len(lo)


# --- crop_image_file (one real-JPEG round-trip) -----------------------------


class TestCropImageFile:
    def test_writes_seventeen_crops_with_recorded_sizes(self, tmp_path: Path) -> None:
        # Build a 1500x1500 PIL image, save as JPEG, then crop it.
        # 1500 / 1.5 = 1000, well above 500x500 so DX crops fit.
        import numpy as np

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
