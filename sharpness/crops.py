"""Geometry for the 17 crop windows.

Direct port of ``analyse-lens-test/make-17-crops.sh``: nine 500×500 crops
covering centre / edges / corners of the full frame, plus eight crops on
an APS-C-equivalent inner region (``image / 1.5`` per axis, centred).
The DX inner region's centre crop is intentionally omitted in the bash
script — keeping that here so file-size comparisons stay backwards-
compatible with the offline rail-coverage tool.

All functions are pure: image dimensions in, crop boxes out. No PIL,
no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

# Crop dimensions — must match make-17-crops.sh exactly so the live and
# offline analyses are directly comparable.
DEFAULT_CROP_W = 500
DEFAULT_CROP_H = 500

# DX (Nikon APS-C) crop factor. Used to define the inner sub-frame on
# which the dx-* crops are anchored.
DX_FACTOR = 1.5

CROP_NAMES_FULL_FRAME = (
    "center",
    "left-edge",
    "right-edge",
    "top-edge",
    "bottom-edge",
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
)

CROP_NAMES_DX = (
    "dx-left-edge",
    "dx-top-left",
    "dx-top-edge",
    "dx-top-right",
    "dx-right-edge",
    "dx-bottom-right",
    "dx-bottom-edge",
    "dx-bottom-left",
)

CROP_NAMES = CROP_NAMES_FULL_FRAME + CROP_NAMES_DX


@dataclass(frozen=True)
class CropBox:
    """Pixel rectangle (left, top, right, bottom). Half-open at right
    and bottom to match Pillow / numpy slicing conventions.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _full_frame_boxes(
    image_w: int, image_h: int, crop_w: int, crop_h: int
) -> dict[str, CropBox]:
    """Nine 500×500 crops on the full image: centre, four edges, four corners."""
    cx = image_w // 2 - crop_w // 2
    cy = image_h // 2 - crop_h // 2
    rx = image_w - crop_w  # right column
    by = image_h - crop_h  # bottom row
    return {
        "center": _box(cx, cy, crop_w, crop_h),
        "left-edge": _box(0, cy, crop_w, crop_h),
        "right-edge": _box(rx, cy, crop_w, crop_h),
        "top-edge": _box(cx, 0, crop_w, crop_h),
        "bottom-edge": _box(cx, by, crop_w, crop_h),
        "top-left": _box(0, 0, crop_w, crop_h),
        "top-right": _box(rx, 0, crop_w, crop_h),
        "bottom-right": _box(rx, by, crop_w, crop_h),
        "bottom-left": _box(0, by, crop_w, crop_h),
    }


def _dx_boxes(
    image_w: int, image_h: int, crop_w: int, crop_h: int
) -> dict[str, CropBox]:
    """Eight crops on the DX (APS-C) inner region.

    Matches the integer arithmetic in the bash script: zsh evaluates
    ``image_width / 1.5`` as integer division (truncation) when the
    operands are integers — but 1.5 is floating-point, so zsh promotes
    and truncates the result. We replicate that with ``int(...)``.
    """
    dxw = int(image_w / DX_FACTOR)
    dxh = int(image_h / DX_FACTOR)
    left_margin = (image_w - dxw) // 2
    top_margin = (image_h - dxh) // 2

    dx_cx = left_margin + (dxw // 2) - crop_w // 2
    dx_cy = top_margin + (dxh // 2) - crop_h // 2
    dx_rx = left_margin + dxw - crop_w
    dx_by = top_margin + dxh - crop_h

    return {
        "dx-left-edge": _box(left_margin, dx_cy, crop_w, crop_h),
        "dx-top-left": _box(left_margin, top_margin, crop_w, crop_h),
        "dx-top-edge": _box(dx_cx, top_margin, crop_w, crop_h),
        "dx-top-right": _box(dx_rx, top_margin, crop_w, crop_h),
        "dx-right-edge": _box(dx_rx, dx_cy, crop_w, crop_h),
        "dx-bottom-right": _box(dx_rx, dx_by, crop_w, crop_h),
        "dx-bottom-edge": _box(dx_cx, dx_by, crop_w, crop_h),
        "dx-bottom-left": _box(left_margin, dx_by, crop_w, crop_h),
    }


def _box(x: int, y: int, w: int, h: int) -> CropBox:
    return CropBox(left=x, top=y, right=x + w, bottom=y + h)


def all_crop_boxes(
    image_w: int,
    image_h: int,
    *,
    crop_w: int = DEFAULT_CROP_W,
    crop_h: int = DEFAULT_CROP_H,
) -> dict[str, CropBox]:
    """All 17 crop windows for an image of the given size.

    Raises ``ValueError`` if the image is smaller than the crop size or
    smaller than the DX-inner region.
    """
    if image_w < crop_w or image_h < crop_h:
        raise ValueError(
            f"image {image_w}x{image_h} smaller than crop {crop_w}x{crop_h}"
        )
    dxw = int(image_w / DX_FACTOR)
    dxh = int(image_h / DX_FACTOR)
    if dxw < crop_w or dxh < crop_h:
        raise ValueError(
            f"DX inner region {dxw}x{dxh} smaller than crop {crop_w}x{crop_h}"
        )
    boxes = _full_frame_boxes(image_w, image_h, crop_w, crop_h)
    boxes.update(_dx_boxes(image_w, image_h, crop_w, crop_h))
    return boxes
