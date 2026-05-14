"""Tests for sharpness.crops — pure crop geometry."""

from __future__ import annotations

import pytest

from lens_test_detect_core.mcu import DEFAULT_CROP_SIZE, MCU_SIZE
from sharpness.crops import (
    CROP_NAMES,
    CROP_NAMES_DX,
    CROP_NAMES_FULL_FRAME,
    CropBox,
    all_crop_boxes,
)


# Z7 native JPEG dimensions from the rig.
Z7_W, Z7_H = 8256, 5504


class TestAllCropBoxes:
    def test_returns_seventeen_named_boxes(self) -> None:
        boxes = all_crop_boxes(Z7_W, Z7_H)
        assert len(boxes) == 17
        assert set(boxes.keys()) == set(CROP_NAMES)
        assert len(CROP_NAMES_FULL_FRAME) == 9
        assert len(CROP_NAMES_DX) == 8

    def test_every_box_is_default_crop_size(self) -> None:
        for name, box in all_crop_boxes(Z7_W, Z7_H).items():
            assert box.width == DEFAULT_CROP_SIZE, name
            assert box.height == DEFAULT_CROP_SIZE, name

    def test_every_box_is_inside_image(self) -> None:
        for name, box in all_crop_boxes(Z7_W, Z7_H).items():
            assert 0 <= box.left, name
            assert 0 <= box.top, name
            assert box.right <= Z7_W, name
            assert box.bottom <= Z7_H, name

    def test_every_origin_is_mcu_aligned(self) -> None:
        """Lossless JPEG cropping requires every origin to be a multiple
        of the MCU size (16 for 4:2:0 sources)."""
        for name, box in all_crop_boxes(Z7_W, Z7_H).items():
            assert box.left % MCU_SIZE == 0, f"{name} left={box.left}"
            assert box.top % MCU_SIZE == 0, f"{name} top={box.top}"

    def test_full_frame_corner_anchors_z7ii(self) -> None:
        """Z7 II's 8256x5504 sensor is the design target — every crop
        origin lands naturally without snapping for crop_size=512.
        """
        boxes = all_crop_boxes(Z7_W, Z7_H)
        # center: ((W - crop)/2, (H - crop)/2) = (3872, 2496)
        assert boxes["center"] == CropBox(3872, 2496, 4384, 3008)
        # top-left: (0, 0)
        assert boxes["top-left"] == CropBox(0, 0, 512, 512)
        # top-right: (W - 512, 0)
        assert boxes["top-right"] == CropBox(Z7_W - 512, 0, Z7_W, 512)
        # bottom-right: (W - 512, H - 512)
        assert boxes["bottom-right"] == CropBox(
            Z7_W - 512, Z7_H - 512, Z7_W, Z7_H
        )
        # bottom-left: (0, H - 512)
        assert boxes["bottom-left"] == CropBox(0, Z7_H - 512, 512, Z7_H)

    def test_full_frame_edge_anchors_z7ii(self) -> None:
        boxes = all_crop_boxes(Z7_W, Z7_H)
        # left-edge: x=0, y=(H-512)/2
        assert boxes["left-edge"] == CropBox(0, 2496, 512, 3008)
        # right-edge: x=W-512, y=(H-512)/2
        assert boxes["right-edge"] == CropBox(Z7_W - 512, 2496, Z7_W, 3008)
        # top-edge: x=(W-512)/2, y=0
        assert boxes["top-edge"] == CropBox(3872, 0, 4384, 512)
        # bottom-edge: x=(W-512)/2, y=H-512
        assert boxes["bottom-edge"] == CropBox(3872, Z7_H - 512, 4384, Z7_H)

    def test_dx_corners_form_smaller_inner_rectangle(self) -> None:
        boxes = all_crop_boxes(Z7_W, Z7_H)
        # The four DX corners should sit strictly inside the full-frame corners.
        assert boxes["dx-top-left"].left > boxes["top-left"].left
        assert boxes["dx-top-left"].top > boxes["top-left"].top
        assert boxes["dx-bottom-right"].right < boxes["bottom-right"].right
        assert boxes["dx-bottom-right"].bottom < boxes["bottom-right"].bottom

    def test_dx_inner_box_dimensions_within_mcu_drift(self) -> None:
        # The DX corner-to-corner span should be image / 1.5 up to MCU
        # alignment slop (the bottom/right corners snap down up to
        # MCU_SIZE-1 px from the ideal 1.5x split).
        boxes = all_crop_boxes(Z7_W, Z7_H)
        dx_w = int(Z7_W / 1.5)
        dx_h = int(Z7_H / 1.5)
        spanned_w = boxes["dx-top-right"].right - boxes["dx-top-left"].left
        spanned_h = boxes["dx-bottom-left"].bottom - boxes["dx-top-left"].top
        assert dx_w - MCU_SIZE <= spanned_w <= dx_w
        assert dx_h - MCU_SIZE <= spanned_h <= dx_h

    def test_rejects_image_smaller_than_crop(self) -> None:
        with pytest.raises(ValueError, match="smaller than crop"):
            all_crop_boxes(400, 400)

    def test_rejects_image_too_small_for_dx_region(self) -> None:
        # Just barely big enough for a default crop, but the DX inner
        # region (image / 1.5) ends up smaller than crop — must raise.
        with pytest.raises(ValueError, match="DX inner region"):
            all_crop_boxes(700, 700)

    def test_custom_crop_size_must_be_mcu_multiple_for_alignment(self) -> None:
        # 256 is a clean MCU multiple, so the alignment property holds
        # for it too.
        boxes = all_crop_boxes(Z7_W, Z7_H, crop_w=256, crop_h=256)
        assert all(b.width == 256 and b.height == 256 for b in boxes.values())
        assert all(
            b.left % MCU_SIZE == 0 and b.top % MCU_SIZE == 0
            for b in boxes.values()
        )


class TestCropBoxDataclass:
    def test_width_height(self) -> None:
        b = CropBox(10, 20, 60, 90)
        assert b.width == 50
        assert b.height == 70

    def test_frozen(self) -> None:
        b = CropBox(0, 0, 1, 1)
        with pytest.raises(Exception):
            b.left = 5  # type: ignore[misc]
