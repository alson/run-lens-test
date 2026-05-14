"""Apply crop boxes to a JPEG losslessly and report file sizes.

The cropper reads the source JPEG bytes, calls
``TurboJPEG.crop_multiple`` to extract 17 sub-images **without decoding
or re-encoding** (lossless JPEG transform), writes them to
``<out>/<crop_name>/<source_name>``, and returns a ``CropResult`` with
the file size of every produced JPEG. File size is the sharpness proxy
used downstream — same metric as ``analyse-lens-test/select-crops.sh``.

Backed by ``libjpeg-turbo`` via the :mod:`turbojpeg` ctypes wrapper.
The previous implementation decoded the source to BGR, sliced the
ndarray 17 times, and re-encoded each slice at ``quality=92`` — two
generations of JPEG compression on every crop. ``crop_multiple``
re-packs encoded MCU blocks into the new JPEGs directly, so the crops
inherit the source's encoded quality exactly and add zero generations
of loss. This works only when the crop origins are on MCU boundaries
(16 px for the 4:2:0-subsampled sources this pipeline ships); the
origins emitted by :func:`sharpness.crops.all_crop_boxes` are
guaranteed MCU-aligned via :mod:`lens_test_detect_core.mcu`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from turbojpeg import TurboJPEG

from sharpness.crops import all_crop_boxes


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
# dlopen + tjInitTransform, which is wasteful when the cropper runs
# hundreds of times per session. One handle per (process, thread) is
# plenty — the underlying tj* APIs are thread-safe per handle.
_tj_local = threading.local()


def _tj() -> TurboJPEG:
    instance = getattr(_tj_local, "instance", None)
    if instance is None:
        instance = TurboJPEG()
        _tj_local.instance = instance
    return instance


def crop_image_file(
    source: Path,
    out_dir: Path,
    *,
    crop_w: int | None = None,
    crop_h: int | None = None,
) -> CropResult:
    """Lossless-crop ``source`` into 17 sub-JPEGs under ``out_dir``.

    The output filename matches the source filename, so a sweep's
    worth of crops in the same crop subdirectory naturally sort by
    capture time (matching ``analyse-lens-test`` conventions).

    Crops are produced by ``TurboJPEG.crop_multiple`` — no decode and
    no re-encode. The output crops inherit the source's encoded
    JPEG quality exactly.
    """
    raw = Path(source).read_bytes()
    w, h, _subsample, _colorspace = _tj().decode_header(raw)

    boxes_kw: dict[str, int] = {}
    if crop_w is not None:
        boxes_kw["crop_w"] = crop_w
    if crop_h is not None:
        boxes_kw["crop_h"] = crop_h
    boxes = all_crop_boxes(w, h, **boxes_kw)

    names = list(boxes.keys())
    crop_params = [
        (b.left, b.top, b.width, b.height) for b in (boxes[n] for n in names)
    ]
    crops = _tj().crop_multiple(raw, crop_params)

    sizes: dict[str, int] = {}
    for name, data in zip(names, crops, strict=True):
        target_dir = out_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(source).name
        target.write_bytes(data)
        sizes[name] = len(data)
    return CropResult(source=Path(source), sizes=sizes)
