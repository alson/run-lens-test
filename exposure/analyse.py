"""Central-crop exposure analysis.

The session uses a small offset crop on the wafer test target, takes
the median of the cropped pixels in 8-bit grayscale, and converts to
an EV difference relative to a session baseline. The numbers feed
both the per-aperture FEC fine-tune loop and the battery-droop
detector.

Functions are pure: they take a JPEG path or numpy array in and
return numbers. No camera, no I/O state beyond the JPEG read itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# 8-bit pixel floor used to avoid log2(0) when linearising.
_PIXEL_FLOOR = 1.0


@dataclass(frozen=True)
class CropBox:
    """Pixel rectangle (left, top, right, bottom). Half-open at right
    and bottom to match Pillow / numpy slicing conventions."""

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


def crop_box(
    image_w: int,
    image_h: int,
    crop_w_frac: float,
    crop_h_frac: float,
    x_off_frac: float = 0.0,
    y_off_frac: float = 0.0,
) -> CropBox:
    """Compute a centred crop box, optionally offset from image centre.

    All ``*_frac`` arguments are fractions of the corresponding image
    dimension. Positive ``y_off_frac`` moves the crop *down* in pixel
    space, matching the wafer layout convention used in
    ``tools/z7_ec_test.py``.
    """
    cw = int(image_w * crop_w_frac)
    ch = int(image_h * crop_h_frac)
    cx = image_w // 2 + int(image_w * x_off_frac)
    cy = image_h // 2 + int(image_h * y_off_frac)
    left = max(0, cx - cw // 2)
    top = max(0, cy - ch // 2)
    right = min(image_w, left + cw)
    bottom = min(image_h, top + ch)
    return CropBox(left=left, top=top, right=right, bottom=bottom)


@dataclass(frozen=True)
class CropStats:
    """Summary statistics for the cropped grayscale region."""

    p10: float
    p50: float  # median, also used as the canonical exposure metric
    p90: float
    mean: float
    clip_hi_pct: float  # percent of pixels at >= 254
    clip_lo_pct: float  # percent of pixels at <= 1
    shape: tuple[int, int]


def measure_crop(jpg_path: Path | str, box: CropBox) -> CropStats:
    """Compute stats over the grayscale crop of a JPEG."""
    img = Image.open(str(jpg_path)).convert("L")
    arr = np.asarray(img)
    return _measure_array(arr, box)


def _measure_array(arr: np.ndarray, box: CropBox) -> CropStats:
    crop = arr[box.top:box.bottom, box.left:box.right]
    if crop.size == 0:
        raise ValueError(f"crop is empty for box {box!r}; image shape {arr.shape}")
    p10, p50, p90 = np.percentile(crop, [10, 50, 90])
    return CropStats(
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        mean=float(np.mean(crop)),
        clip_hi_pct=float(np.mean(crop >= 254)) * 100.0,
        clip_lo_pct=float(np.mean(crop <= 1)) * 100.0,
        shape=(int(crop.shape[0]), int(crop.shape[1])),
    )


def linearise(value_0_255: float, *, gamma: float = 2.2) -> float:
    """Convert an 8-bit display value to relative linear light.

    Uses a power-law approximation; close enough for the ±1.5 EV
    range we operate in. See DESIGN_NOTES §3.7.
    """
    v = max(value_0_255, _PIXEL_FLOOR)
    return (v / 255.0) ** gamma


def ev_delta(
    value_0_255: float,
    baseline_0_255: float,
    *,
    gamma: float = 2.2,
) -> float:
    """EV difference between ``value`` and ``baseline``.

    Positive = brighter than baseline. Returns ``0.0`` when both are
    equal even at the floor (avoids ``log2(1.0)`` underflow noise).
    """
    lin_value = linearise(value_0_255, gamma=gamma)
    lin_base = linearise(baseline_0_255, gamma=gamma)
    if lin_base <= 0:
        raise ValueError("baseline must be positive")
    return math.log2(lin_value / lin_base)


def save_overlay(
    jpg_path: Path | str,
    box: CropBox,
    overlay_path: Path | str,
    *,
    max_width: int = 1200,
    border_px: int = 3,
    colour: tuple[int, int, int] = (255, 0, 0),
    quality: int = 85,
) -> None:
    """Save a downscaled copy of the JPEG with the crop box drawn on top.

    Lifted from ``tools/z7_ec_test.py``: useful when iterating on the
    crop fractions to confirm the box lands on the wafer.
    """
    img = Image.open(str(jpg_path)).convert("RGB")
    iw, ih = img.size
    scale = min(1.0, max_width / iw) if iw > 0 else 1.0
    if scale < 1.0:
        img = img.resize((int(iw * scale), int(ih * scale)))
    draw = ImageDraw.Draw(img)
    sb = (
        int(box.left * scale),
        int(box.top * scale),
        int(box.right * scale),
        int(box.bottom * scale),
    )
    for w in range(border_px):
        draw.rectangle(
            (sb[0] - w, sb[1] - w, sb[2] + w, sb[3] + w),
            outline=colour,
        )
    img.save(str(overlay_path), quality=quality)
