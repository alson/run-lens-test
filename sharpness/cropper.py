"""Apply crop boxes to a JPEG and report file sizes.

The cropper opens an image once, slices it into 17 sub-images, saves
each to ``<out>/cropped/<crop_name>/<source_name>``, and returns a
``CropResult`` with the file size of every produced JPEG. File size is
the sharpness proxy used downstream — same metric as
``analyse-lens-test/select-crops.sh``.

The pure cropping function (``crop_to_bytes``) takes a ``PIL.Image``
plus a box and returns JPEG bytes; the I/O wrapper (``crop_image_file``)
opens / saves on top of that. This split keeps unit tests fast (no JPEG
encode round-trip needed for the geometry) while keeping a small
real-image integration test honest.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from sharpness.crops import CropBox, all_crop_boxes


# Quality that matches the camera output. ``select-crops.sh`` uses file
# size as a proxy for sharpness; for that proxy to remain meaningful we
# need to re-encode at high quality so detail in the crop survives.
DEFAULT_JPEG_QUALITY = 92


@dataclass(frozen=True)
class CropResult:
    """File sizes per crop, returned by ``crop_image_file``.

    ``sizes`` maps crop name -> JPEG file size in bytes (post-save).
    """

    source: Path
    sizes: dict[str, int]

    def size_for(self, crop_name: str) -> int:
        return self.sizes[crop_name]


def crop_to_bytes(
    image: Image.Image,
    box: CropBox,
    *,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Return JPEG-encoded bytes for one crop. Pure: no disk I/O."""
    sub = image.crop((box.left, box.top, box.right, box.bottom))
    buf = io.BytesIO()
    # ``optimize`` shaves a percent or two but adds noticeable CPU; skip it
    # since the bytes go straight to disk and we care about throughput.
    sub.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def crop_image_file(
    source: Path,
    out_dir: Path,
    *,
    crop_w: int | None = None,
    crop_h: int | None = None,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> CropResult:
    """Open ``source``, write 17 crops under ``out_dir/<crop_name>/``.

    The output filename matches the source filename, so a sweep's worth
    of crops in the same crop subdirectory naturally sort by capture
    time (matching ``analyse-lens-test`` conventions).
    """
    img = Image.open(source)
    img.load()
    boxes_kw: dict[str, int] = {}
    if crop_w is not None:
        boxes_kw["crop_w"] = crop_w
    if crop_h is not None:
        boxes_kw["crop_h"] = crop_h
    boxes = all_crop_boxes(img.width, img.height, **boxes_kw)

    sizes: dict[str, int] = {}
    for name, box in boxes.items():
        target_dir = out_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        data = crop_to_bytes(img, box, quality=quality)
        target.write_bytes(data)
        sizes[name] = len(data)
    return CropResult(source=source, sizes=sizes)
