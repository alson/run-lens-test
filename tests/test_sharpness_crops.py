"""Tests for sharpness.crops — pure crop geometry."""

from __future__ import annotations

import pytest

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

    def test_every_box_is_500x500(self) -> None:
        for name, box in all_crop_boxes(Z7_W, Z7_H).items():
            assert box.width == 500, name
            assert box.height == 500, name

    def test_every_box_is_inside_image(self) -> None:
        for name, box in all_crop_boxes(Z7_W, Z7_H).items():
            assert 0 <= box.left, name
            assert 0 <= box.top, name
            assert box.right <= Z7_W, name
            assert box.bottom <= Z7_H, name

    def test_corner_anchors_match_bash(self) -> None:
        # Verify against the explicit calculations in make-17-crops.sh
        boxes = all_crop_boxes(Z7_W, Z7_H)
        # center: (W/2 - 250, H/2 - 250)
        assert boxes["center"].left == Z7_W // 2 - 250
        assert boxes["center"].top == Z7_H // 2 - 250
        # top-left: (0, 0)
        assert boxes["top-left"] == CropBox(0, 0, 500, 500)
        # top-right: (W - 500, 0)
        assert boxes["top-right"].left == Z7_W - 500
        assert boxes["top-right"].top == 0
        # bottom-right: (W - 500, H - 500)
        assert boxes["bottom-right"].left == Z7_W - 500
        assert boxes["bottom-right"].top == Z7_H - 500
        # bottom-left: (0, H - 500)
        assert boxes["bottom-left"].left == 0
        assert boxes["bottom-left"].top == Z7_H - 500

    def test_edge_anchors_match_bash(self) -> None:
        boxes = all_crop_boxes(Z7_W, Z7_H)
        # left-edge: x=0, y=H/2-250
        assert boxes["left-edge"].left == 0
        assert boxes["left-edge"].top == Z7_H // 2 - 250
        # right-edge: x=W-500, y=H/2-250
        assert boxes["right-edge"].left == Z7_W - 500
        assert boxes["right-edge"].top == Z7_H // 2 - 250
        # top-edge: x=W/2-250, y=0
        assert boxes["top-edge"].left == Z7_W // 2 - 250
        assert boxes["top-edge"].top == 0
        # bottom-edge: x=W/2-250, y=H-500
        assert boxes["bottom-edge"].left == Z7_W // 2 - 250
        assert boxes["bottom-edge"].top == Z7_H - 500

    def test_dx_corners_form_smaller_inner_rectangle(self) -> None:
        boxes = all_crop_boxes(Z7_W, Z7_H)
        # The four DX corners should sit strictly inside the full-frame corners.
        assert boxes["dx-top-left"].left > boxes["top-left"].left
        assert boxes["dx-top-left"].top > boxes["top-left"].top
        assert boxes["dx-bottom-right"].right < boxes["bottom-right"].right
        assert boxes["dx-bottom-right"].bottom < boxes["bottom-right"].bottom

    def test_dx_inner_box_dimensions(self) -> None:
        # The DX corner-to-corner span should be image / 1.5 minus one crop.
        boxes = all_crop_boxes(Z7_W, Z7_H)
        dx_w = int(Z7_W / 1.5)
        dx_h = int(Z7_H / 1.5)
        # span between left edge of dx-top-left and right edge of dx-top-right
        spanned_w = boxes["dx-top-right"].right - boxes["dx-top-left"].left
        spanned_h = boxes["dx-bottom-left"].bottom - boxes["dx-top-left"].top
        assert spanned_w == dx_w
        assert spanned_h == dx_h

    def test_rejects_image_smaller_than_crop(self) -> None:
        with pytest.raises(ValueError, match="smaller than crop"):
            all_crop_boxes(400, 400)

    def test_rejects_image_too_small_for_dx_region(self) -> None:
        # Just barely big enough for a 500×500 full-frame crop, but the DX inner
        # region (image / 1.5) ends up smaller than 500 — must raise.
        with pytest.raises(ValueError, match="DX inner region"):
            all_crop_boxes(600, 600)

    def test_custom_crop_size(self) -> None:
        boxes = all_crop_boxes(Z7_W, Z7_H, crop_w=200, crop_h=200)
        assert all(b.width == 200 and b.height == 200 for b in boxes.values())


class TestCropBoxDataclass:
    def test_width_height(self) -> None:
        b = CropBox(10, 20, 60, 90)
        assert b.width == 50
        assert b.height == 70

    def test_frozen(self) -> None:
        b = CropBox(0, 0, 1, 1)
        with pytest.raises(Exception):
            b.left = 5  # type: ignore[misc]
