"""Filename helpers used by the runner, calibrator, and adapter.

Centralised here so the same naming convention is applied across
sweep / probe / extension shots and the cropped output. The single
source of truth is the source JPEG's basename — the cropper writes
crops under ``<aperture>/cropped/<crop-name>/<source-basename>``.
"""

from __future__ import annotations


def step_filename(
    step_index: int,
    *,
    offset: int,
    width: int = 4,
    extension: str = ".jpg",
) -> str:
    """Build a ``step_NNNN.jpg`` filename for ``step_index``.

    The signed ``step_index`` is shifted by ``offset`` so the *filename*
    is always non-negative. With the default offset of 100, the main
    sweep starts at ``step_0100.jpg``, far-side extensions continue
    upward (``step_0130.jpg`` …), and near-side extensions go below
    the offset (``step_0099.jpg`` …).

    Raises ``ValueError`` if the result would be negative or wouldn't
    fit in the requested width.
    """
    n = offset + step_index
    if n < 0:
        raise ValueError(
            f"step_index {step_index} + offset {offset} = {n} < 0; "
            "increase STEP_NUMBER_OFFSET in config.py"
        )
    formatted = f"{n:0{width}d}"
    if len(formatted) > width:
        raise ValueError(
            f"step_index {step_index} + offset {offset} = {n} exceeds "
            f"width {width}; increase STEP_NUMBER_WIDTH"
        )
    return f"step_{formatted}{extension}"


def parse_step_filename(filename: str, *, offset: int) -> int:
    """Extract the signed ``step_index`` from a filename produced by
    :func:`step_filename`.

    Returns the ``step_index`` (signed). Raises ``ValueError`` if the
    filename doesn't match the convention.
    """
    if not filename.startswith("step_"):
        raise ValueError(f"not a step filename: {filename!r}")
    stem = filename[len("step_"):]
    # Strip extension.
    if "." in stem:
        stem = stem.split(".", 1)[0]
    try:
        return int(stem) - offset
    except ValueError as exc:
        raise ValueError(f"non-integer step in {filename!r}") from exc
