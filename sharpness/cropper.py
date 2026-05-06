"""Apply crop boxes to a JPEG and report file sizes.

The cropper decodes the source JPEG once into an ndarray, slices it
into 17 sub-images, encodes each back to JPEG, writes them to
``<out>/<crop_name>/<source_name>``, and returns a ``CropResult``
with the file size of every produced JPEG. File size is the
sharpness proxy used downstream — same metric as
``analyse-lens-test/select-crops.sh``.

Backed by ``libjpeg-turbo`` via the :mod:`turbojpeg` ctypes wrapper.
The previous PIL/Pillow path linked the same libjpeg-turbo on this
system but went through more Python dispatch and an Image-object
allocation per crop. Direct turbojpeg saves ~13 % per shot
single-threaded; the gain stacks with the ProcessPoolExecutor
worker pool the live collector uses in production.

Quality note: at the same nominal ``quality=92`` setting, turbojpeg
produces slightly larger files (~10–17 %) than PIL because PIL
applies a small extra optimization step that turbojpeg's default
encode doesn't. The file-size-as-sharpness proxy stays valid as
long as every shot in a session uses the same encoder + quality —
the comparison is monotonic in scene detail, not calibrated to an
absolute byte count.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from turbojpeg import TurboJPEG

from sharpness.crops import CropBox, all_crop_boxes


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


# A single :class:`TurboJPEG` instance owns a libturbojpeg handle
# (a few KB of native state). Constructing one per call costs a
# dlopen + tjInitCompress / tjInitDecompress, which is wasteful when
# the cropper runs hundreds of times per session. One handle per
# (process, thread) is plenty — the underlying tj* APIs are
# thread-safe per handle.
_tj_local = threading.local()


def _tj() -> TurboJPEG:
    instance = getattr(_tj_local, "instance", None)
    if instance is None:
        instance = TurboJPEG()
        _tj_local.instance = instance
    return instance


def crop_to_bytes(
    image: np.ndarray,
    box: CropBox,
    *,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Return JPEG-encoded bytes for one crop slice. Pure: no disk I/O.

    ``image`` is a decoded ndarray in BGR order (turbojpeg's native
    layout). The slice is forced contiguous if it isn't already so
    libturbojpeg's encoder gets a clean buffer.
    """
    sub = image[box.top:box.bottom, box.left:box.right]
    if not sub.flags["C_CONTIGUOUS"]:
        sub = np.ascontiguousarray(sub)
    return _tj().encode(sub, quality=quality)


def crop_image_file(
    source: Path,
    out_dir: Path,
    *,
    crop_w: int | None = None,
    crop_h: int | None = None,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> CropResult:
    """Decode ``source``, write 17 crops under ``out_dir/<crop_name>/``.

    The output filename matches the source filename, so a sweep's
    worth of crops in the same crop subdirectory naturally sort by
    capture time (matching ``analyse-lens-test`` conventions).
    """
    raw = Path(source).read_bytes()
    arr = _tj().decode(raw)  # (H, W, 3) BGR, uint8
    h, w = arr.shape[:2]
    boxes_kw: dict[str, int] = {}
    if crop_w is not None:
        boxes_kw["crop_w"] = crop_w
    if crop_h is not None:
        boxes_kw["crop_h"] = crop_h
    boxes = all_crop_boxes(w, h, **boxes_kw)

    sizes: dict[str, int] = {}
    for name, box in boxes.items():
        target_dir = out_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(source).name
        data = crop_to_bytes(arr, box, quality=quality)
        target.write_bytes(data)
        sizes[name] = len(data)
    return CropResult(source=Path(source), sizes=sizes)
