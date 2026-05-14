"""Tests for sharpness.cropper.

``crop_image_file`` extracts 17 lossless sub-JPEGs from a source
file via ``TurboJPEG.crop_multiple``; the decode → re-encode pipeline
that the previous implementation used is gone, so all assertions
exercise the lossless-transform path end-to-end with real JPEGs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from turbojpeg import TurboJPEG

from lens_test_detect_core.mcu import MCU_SIZE
from sharpness.cropper import CropResult, crop_image_file
from sharpness.crops import CROP_NAMES, all_crop_boxes


# --- helpers ---------------------------------------------------------------


def _write_textured_jpeg(path: Path, w: int, h: int, seed: int = 1) -> None:
    """Write a textured JPEG that decodes cleanly through libjpeg-turbo.

    Uses 4:2:0 subsampling to match the capture-rig sources; without
    that the crop_multiple MCU is 8x8 rather than 16x16 and the
    alignment math in :mod:`lens_test_detect_core.mcu` is off.
    """
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype="uint8")
    Image.fromarray(arr).save(path, "JPEG", quality=92, subsampling=2)


# --- crop_image_file (lossless round-trip) ---------------------------------


class TestCropImageFile:
    def test_writes_seventeen_crops_with_recorded_sizes(
        self, tmp_path: Path
    ) -> None:
        # The DX inner region needs to be >= crop_size. With the default
        # 512 crop, image / 1.5 >= 512 → image >= 768. Round up to 800
        # (also MCU-aligned).
        src = tmp_path / "source.jpg"
        _write_textured_jpeg(src, 800, 800)

        out_dir = tmp_path / "cropped"
        result = crop_image_file(src, out_dir)

        assert isinstance(result, CropResult)
        assert set(result.sizes.keys()) == set(CROP_NAMES)
        for name in CROP_NAMES:
            target = out_dir / name / "source.jpg"
            assert target.is_file(), f"missing {target}"
            assert result.size_for(name) == target.stat().st_size

    def test_uses_source_filename_under_crop_subdir(self, tmp_path: Path) -> None:
        src = tmp_path / "step_005.jpg"
        _write_textured_jpeg(src, 800, 800, seed=2)
        out_dir = tmp_path / "out"
        crop_image_file(src, out_dir)
        for name in CROP_NAMES:
            assert (out_dir / name / "step_005.jpg").is_file()

    def test_rejects_image_too_small(self, tmp_path: Path) -> None:
        src = tmp_path / "small.jpg"
        _write_textured_jpeg(src, 400, 400)
        with pytest.raises(ValueError):
            crop_image_file(src, tmp_path / "out")

    def test_size_for_returns_recorded_value(self) -> None:
        result = CropResult(source=Path("/x.jpg"), sizes={"center": 12345})
        assert result.size_for("center") == 12345

    def test_crops_are_lossless_at_interior(self, tmp_path: Path) -> None:
        """A lossless JPEG crop preserves the encoded MCU blocks exactly,
        which means decoded interior pixels match the corresponding
        source sub-region bit-for-bit. The first/last few rows and
        columns may differ slightly because 4:2:0 chroma upsampling at
        the new edges uses different neighbour MCUs than the source
        had — that's an inherent feature of the format, not a re-encode.
        Interior pixels (a 16-px stride away from each border) must
        match exactly."""
        src = tmp_path / "lossless.jpg"
        # Z7 II-sized image so the centre crop lands at (3872, 2496) —
        # the natural-alignment case the design targets.
        _write_textured_jpeg(src, 8256, 5504, seed=3)
        crop_image_file(src, tmp_path / "out")

        tj = TurboJPEG()
        source_arr = tj.decode(src.read_bytes())  # (H, W, 3) BGR
        boxes = all_crop_boxes(8256, 5504)
        for name, box in boxes.items():
            crop_path = tmp_path / "out" / name / "lossless.jpg"
            crop_arr = tj.decode(crop_path.read_bytes())
            expected = source_arr[box.top : box.bottom, box.left : box.right]
            assert crop_arr.shape == expected.shape, name
            # Interior region: trim 16 px on every side. Anything
            # losing data here would mean a real re-encode happened.
            interior_crop = crop_arr[16:-16, 16:-16]
            interior_expected = expected[16:-16, 16:-16]
            assert np.array_equal(interior_crop, interior_expected), (
                f"{name}: interior pixels differ — not a lossless crop"
            )

    def test_crop_files_smaller_than_full_source(self, tmp_path: Path) -> None:
        """Sanity: each cropped file must be smaller than the source.
        A re-encode at default quality would also satisfy this, so it
        isn't a tight lossless check, but it catches outright bugs
        (e.g. accidentally writing the whole source to every crop)."""
        src = tmp_path / "size.jpg"
        _write_textured_jpeg(src, 2048, 2048, seed=4)
        result = crop_image_file(src, tmp_path / "out")
        source_size = src.stat().st_size
        for name, size in result.sizes.items():
            assert size < source_size, f"{name} not smaller than source"

    def test_all_origins_must_be_mcu_aligned(self, tmp_path: Path) -> None:
        """Crop boxes the cropper feeds into ``crop_multiple`` must all
        be MCU-aligned (otherwise libjpeg-turbo errors out). This is a
        contract test on the upstream box generator, exercised through
        the cropper to confirm the integration."""
        src = tmp_path / "aligned.jpg"
        _write_textured_jpeg(src, 8256, 5504)
        boxes = all_crop_boxes(8256, 5504)
        for name, box in boxes.items():
            assert box.left % MCU_SIZE == 0, name
            assert box.top % MCU_SIZE == 0, name
        # And the cropper succeeds end-to-end with these origins.
        crop_image_file(src, tmp_path / "out")
