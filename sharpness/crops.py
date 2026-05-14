"""Geometry for the 17 crop windows.

Direct port of ``analyse-lens-test/make-17-crops.sh``: nine 512x512 crops
covering centre / edges / corners of the full frame, plus eight crops on
an APS-C-equivalent inner region (``image / 1.5`` per axis, centred).
The DX inner region's centre crop is intentionally omitted in the bash
script — keeping that here so file-size comparisons stay backwards-
compatible with the offline rail-coverage tool.

Crop size is 512x512 px (formerly 500x500) so every crop origin lands
on a 16-pixel MCU boundary, enabling lossless JPEG transforms in
:mod:`sharpness.cropper`. See :mod:`lens_test_detect_core.mcu` for the
shared alignment helpers; the constants here just re-export the
defaults so the public API stays stable.

All functions are pure: image dimensions in, crop boxes out. No PIL,
no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from lens_test_detect_core.mcu import (
    DEFAULT_CROP_SIZE,
    DX_FACTOR as _DX_FACTOR,
    compute_crop_origins,
    compute_dx_origins,
)

# Crop dimensions — must match make-17-crops.sh exactly so the live and
# offline analyses are directly comparable. Both currently 512 (the
# smallest 16-multiple at or above the legacy 500 px default).
DEFAULT_CROP_W = DEFAULT_CROP_SIZE
DEFAULT_CROP_H = DEFAULT_CROP_SIZE

# DX (Nikon APS-C) crop factor. Re-exported from the shared module so
# pre-existing callers of ``sharpness.crops.DX_FACTOR`` keep working.
DX_FACTOR = _DX_FACTOR

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

# Peer groups for absolute-size cross-compare (partial-frame detection).
# Each group lists crops at the same *position class* — corners-with-
# corners, edges-with-edges — so the comparison is apples-to-apples.
# Centre has no peer group (singleton) and is omitted. See
# ``lens_test_detect_core.partial_frame_by_peer_compare`` for the
# detection logic.
#
# **Full-frame only.** The DX inner corners and edges are deliberately
# excluded. By geometry, any DX inner crop sits *inside* the
# corresponding full-frame outer crop's strip — so if a DX inner crop
# were partial-frame (wafer edge cutting through it), the full-frame
# outer crop would necessarily be partial-or-worse. The full-frame
# peer groups catch every real partial-frame condition; adding DX
# peer groups only contributes false positives from wafer pattern-
# density asymmetry. Empirical case: the 24-120 + 45 mm extension
# session (2026-05-11_161129) showed dx-right-edge at 0.76-0.79 of
# its 4-member DX-edge peer median across all five apertures, while
# the corresponding full-frame right-edge sat at 0.82-0.94 (normal
# range). The asymmetry was the DX peer group's two vertical edges
# (dx-top-edge, dx-bottom-edge) carrying ~20 % more detail than the
# horizontal edges, dragging the median up. Same ratio at both 35 mm
# and 45 mm magnifications confirmed the signal was pattern density,
# not geometry.
CROP_PEER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("top-left", "top-right", "bottom-left", "bottom-right"),
    ("top-edge", "bottom-edge", "left-edge", "right-edge"),
)


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
    """Nine 512x512 crops on the full image: centre, four edges, four corners.

    Origins come from
    :func:`lens_test_detect_core.mcu.compute_crop_origins` so they're
    always 16-aligned for lossless JPEG cropping. Only valid when
    ``crop_w == crop_h``; that's the case for every caller in this
    project.
    """
    if crop_w != crop_h:
        raise ValueError(
            f"full-frame boxes require square crops; got {crop_w}x{crop_h}"
        )
    origins = compute_crop_origins(image_w, image_h, crop_w)
    return {name: _box(x, y, crop_w, crop_h) for name, (x, y) in origins.items()}


def _dx_boxes(
    image_w: int, image_h: int, crop_w: int, crop_h: int
) -> dict[str, CropBox]:
    """Eight crops on the DX (APS-C) inner region.

    Origins come from
    :func:`lens_test_detect_core.mcu.compute_dx_origins` so they're
    always 16-aligned for lossless JPEG cropping. Only valid when
    ``crop_w == crop_h``; that's the case for every caller.
    """
    if crop_w != crop_h:
        raise ValueError(f"dx boxes require square crops; got {crop_w}x{crop_h}")
    origins = compute_dx_origins(image_w, image_h, crop_w)
    return {name: _box(x, y, crop_w, crop_h) for name, (x, y) in origins.items()}


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

    Origins are MCU-aligned (every ``left`` and ``top`` is a multiple
    of 16) so the boxes feed directly into
    :func:`turbojpeg.TurboJPEG.crop_multiple` for lossless JPEG
    extraction.

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
